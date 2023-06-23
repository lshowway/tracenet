import torch, math
from torch import nn
from sparsemax import Sparsemax

from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.xlnet.modeling_xlnet import XLNetPreTrainedModel, XLNetModel
from transformers.models.roberta.modeling_roberta import RobertaModel


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"relu": torch.nn.functional.relu, "gelu": _gelu_python}
XLNetLayerNorm = nn.LayerNorm


class AdditiveAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.output_feature, config.output_feature, bias=False) # q: batch*d*1  => batch*1*20
        self.w2 = nn.Linear(config.output_feature, config.output_feature, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, config.output_feature), requires_grad=True)
        self.activation = nn.Tanh()
        self.v = nn.Linear(config.output_feature, 1, bias=False)

    def forward(self, query, values):
        query = query.permute(0, 2, 1) # batch*1*128
        t1 = self.w1(query) + self.w2(values) # batch*len*d, 前者广播
        energy = self.activation(t1 + self.bias) # batch*len*d
        alpha = self.v(energy) # batch*len*d = >batch*len*1

        return alpha


class TraceNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_select_prob = config.seq_select_prob
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.C = nn.Linear(config.hidden_size, config.output_feature)
        self.A = nn.Linear(config.hidden_size, config.output_feature)
        self.activation = ACT2FN['relu']
        self.drop = nn.Dropout(config.dropout_prob)
        self.sparsemax = Sparsemax(dim=1)
        self.add_attention = AdditiveAttention(config)

        self.w_1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w_2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w_3 = nn.Parameter(torch.ones(1), requires_grad=True)

    def attentd_to_real_token(self, attention_mask, xlnet_output):
        input_shape = xlnet_output.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = torch.squeeze(extended_attention_mask)
        extended_attention_mask = torch.unsqueeze(extended_attention_mask, dim=-1)

        return extended_attention_mask

    def proactive_mask(self, C_or_A, item_weights):
        # 输出已经加了
        batch, length = item_weights.shape[:2]
        # 生成一个batch*1的向量，值为seq_select_prob
        prob_matrix = torch.full([batch, 1, 1], self.seq_select_prob).to(self.device)  # batch*1*1
        # 输出的张量，每个元素有seq_select_prob的概率为1（被选择），有1-seq_select_prob的概率为0
        prob_matrix = torch.bernoulli(prob_matrix)  # .bool() # batch*1*1
        # p=2表示二范式, dim=1表示按行归一化
        t1 = torch.mul(item_weights, prob_matrix)  # batch*len*1

        # 以item weights归一化之后的概率输出为1（该token被mask）
        prob_matrix_2 = torch.bernoulli(t1)  # .bool()
        prob_matrix_2 = torch.ones(prob_matrix_2.shape).to(self.device) - prob_matrix_2  # 取反, 1->0, 0->1
        C_or_A = torch.mul(C_or_A, prob_matrix_2)

        return C_or_A

    def forward(self, layer_i, attention_mask, xlnet_output, item_weights, proactive_masking, A_hat=None):
        # N layer, Encoder-Decoder in each layer smooth regularization
        # if not attention_mask is None:
        attention_mask = self.attentd_to_real_token(attention_mask, xlnet_output).to(self.device)
        if layer_i == 0:
            C = self.C(xlnet_output)
            A = self.A(xlnet_output)  # batch*length*128
        else:
            C = A_hat
            A = self.A(xlnet_output)
        C = self.activation(C)
        A = self.activation(A)
        C = self.drop(C)
        A = self.drop(A)
        if proactive_masking:
            C = self.proactive_mask(C, item_weights)
            A = self.proactive_mask(A, item_weights)

        # 计算q
        L = item_weights  # batch*length*1
        C_T = torch.transpose(C, dim0=-2, dim1=-1)  # batch*128*length
        q = torch.matmul(C_T, L)  # batch*128*1

        alpha = self.add_attention(query=q, values=C)  # ((batch*d*1), batch*length*d) => (batch*length*1)
        alpha = alpha + attention_mask
        alpha = nn.Softmax(dim=1)(alpha)
        h = torch.matmul(C.permute(0, 2, 1), alpha)  # (batch*d*length) * (batch*length*1) => (batch*d*1)

        # Locator

        L = self.add_attention(query=h, values=A)  # (batch*d*1)， (batch*length*d) => (batch*length*1)
        if layer_i == 0:
            x = self.w_3 ** 2 - self.w_2 ** 2 - self.w_1 ** 2
            L = L * nn.Sigmoid()(x)
        elif layer_i == 1:
            x = self.w_3 ** 2 - self.w_2 ** 2
            L = L * nn.Sigmoid()(x)
        elif layer_i == 2:
            x = self.w_3 ** 2
            L = L * nn.Sigmoid()(x)
        else:
            return
        L = L + attention_mask
        L = self.sparsemax(L)

        output = (h, L, A)  # hidden_state, item_weight, A
        return output


class TraceNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_hidden_states = config.output_hidden_states
        self.output_item_weights = config.output_item_weights
        self.tracenetLayer = nn.ModuleList([TraceNetLayer(config) for _ in range(config.num_hubo_layers)])

    def forward(self, attention_mask, hidden_states, item_weights, proactive_masking):
        all_hidden_states = ()
        all_item_weights = ()
        all_A_hat = ()
        outputs = ()
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if self.output_item_weights:
            all_item_weights = all_item_weights + (item_weights,)
        A_hat = None
        for i, layer_module in enumerate(self.tracenetLayer):
            layer_outputs = layer_module(i, attention_mask, hidden_states, item_weights, proactive_masking, A_hat)
            hidden_states_i = layer_outputs[0]
            item_weights = layer_outputs[1]
            A_hat = layer_outputs[2]
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_i,)
                all_A_hat = all_A_hat + (A_hat,)
            if self.output_item_weights:
                all_item_weights = all_item_weights + (item_weights,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
            outputs = outputs + (all_A_hat,)
        if self.output_item_weights:
            outputs = outputs + (all_item_weights,)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.num_labels = config.num_labels
        self.output = torch.nn.Linear(config.output_feature, self.num_labels)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.output(hidden_states)
        return logits


class RobertTraceNetForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.method = config.method
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.focusnet = TraceNetModel(config)
        self.discriminator = Discriminator(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        item_weights=None, proactive_masking=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0]
        all_item_weights = focus_outputs[1] # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            hidden_states = torch.squeeze(all_hidden_states[1])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '2':
            hidden_states = torch.squeeze(all_hidden_states[2])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '3':
            hidden_states = torch.squeeze(all_hidden_states[3])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            hidden_states = torch.squeeze(t1)
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        else:
            print('============wrong============', flush=True)

        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        final_outputs = final_outputs + (all_item_weights, )
        return final_outputs


class XLNetTraceNetForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.method = config.method

        self.transformer = XLNetModel(config)
        self.focusnet = TraceNetModel(config)
        self.discriminator = Discriminator(config)
        # self.xlnetLayer = TraceNetLayer(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            item_weights=None, proactive_masking=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = transformer_outputs[0]  # token_level的output：output batch*len*768 ?
        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0]
        all_item_weights = focus_outputs[1]  # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            hidden_states = torch.squeeze(all_hidden_states[1])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '2':
            hidden_states = torch.squeeze(all_hidden_states[2])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '3':
            hidden_states = torch.squeeze(all_hidden_states[3])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            hidden_states = torch.squeeze(t1)
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        else:
            print('============wrong============', flush=True)

        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        final_outputs = final_outputs + (all_item_weights,)
        return final_outputs


class BertTraceNetForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.focusnet = TraceNetModel(config)
        self.discriminator = Discriminator(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, item_weights = None, proactive_masking = None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0]
        all_item_weights = focus_outputs[1]  # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            hidden_states = torch.squeeze(all_hidden_states[1])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '2':
            hidden_states = torch.squeeze(all_hidden_states[2])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '3':
            hidden_states = torch.squeeze(all_hidden_states[3])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            hidden_states = torch.squeeze(t1)
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        else:
            print('============wrong============', flush=True)

        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        final_outputs = final_outputs + (all_item_weights,)
        return final_outputs


class RobertAttnForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.method = config.method
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)

        # additive attn
        from attentions import AdditiveAttentionLayer as Attn

        # dot attn 论文中使用了dot attn了嘛？？

        # scale dot product attn
        # from attentions import ScaledDotProductAttention

        # stack N-layer attentions (additive)
        # from attentions import AdditiveAttentionModel as Attn

        # stack N-layer attentions (scaled dot product)
        # from attentions import ScaledDotProductAttentionModel as Attn
        self.focusnet = Attn(config)

        self.discriminator = Discriminator(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        item_weights=None, proactive_masking=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0]
        all_item_weights = focus_outputs[1] # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            hidden_states = torch.squeeze(all_hidden_states[1])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '2':
            hidden_states = torch.squeeze(all_hidden_states[2])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '3':
            hidden_states = torch.squeeze(all_hidden_states[3])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            hidden_states = torch.squeeze(t1)
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        else:
            print('============wrong============', flush=True)

        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        final_outputs = final_outputs + (all_item_weights, )
        return final_outputs


class XLNetAttnForSequenceClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.method = config.method

        self.transformer = XLNetModel(config)
        # additive attn
        from attentions import AdditiveAttentionLayer as Attn

        # dot attn 论文中使用了dot attn了嘛？？

        # scale dot product attn
        # from attentions import ScaledDotProductAttention

        # stack N-layer attentions (additive)
        # from attentions import AdditiveAttentionModel as Attn

        # stack N-layer attentions (scaled dot product)
        # from attentions import ScaledDotProductAttentionModel as Attn
        self.focusnet = Attn(config)
        self.discriminator = Discriminator(config)
        # self.xlnetLayer = TraceNetLayer(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            item_weights=None, proactive_masking=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = transformer_outputs[0]  # token_level的output：output batch*len*768 ?
        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0]
        all_item_weights = focus_outputs[1]  # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            hidden_states = torch.squeeze(all_hidden_states[1])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '2':
            hidden_states = torch.squeeze(all_hidden_states[2])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '3':
            hidden_states = torch.squeeze(all_hidden_states[3])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            hidden_states = torch.squeeze(t1)
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        else:
            print('============wrong============', flush=True)

        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        final_outputs = final_outputs + (all_item_weights,)
        return final_outputs


class BertAttnNetForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # additive attn
        from attentions import AdditiveAttentionLayer as Attn

        # dot attn 论文中使用了dot attn了嘛？？

        # scale dot product attn
        # from attentions import ScaledDotProductAttention

        # stack N-layer attentions (additive)
        # from attentions import AdditiveAttentionModel as Attn

        # stack N-layer attentions (scaled dot product)
        # from attentions import ScaledDotProductAttentionModel as Attn
        self.focusnet = Attn(config)
        self.discriminator = Discriminator(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, item_weights = None, proactive_masking = None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0]
        all_item_weights = focus_outputs[1]  # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            hidden_states = torch.squeeze(all_hidden_states[1])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '2':
            hidden_states = torch.squeeze(all_hidden_states[2])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == '3':
            hidden_states = torch.squeeze(all_hidden_states[3])
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            hidden_states = torch.squeeze(t1)
            logits = self.discriminator(hidden_states)
            final_outputs = final_outputs + (logits,)
        else:
            print('============wrong============', flush=True)

        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        final_outputs = final_outputs + (all_item_weights,)
        return final_outputs
