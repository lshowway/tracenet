from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import \
    RobertaModel, RobertaPreTrainedModel
from tracenet import TraceNetModel

logger = logging.getLogger(__name__)


class RoBERTaForABSA(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBERTaForABSA, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels, False)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        start_ids = start_ids.unsqueeze(1)  # batch 1 L
        aspect_vec = torch.bmm(start_ids, sequence_output).squeeze(1)
        logits = self.typing(aspect_vec)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels)
            return loss, logits
        else:
            return logits


class RoBERTaTraceNetForABSA(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBERTaTraceNetForABSA, self).__init__(config)
        self.method = config.method
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.focusnet = TraceNetModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.output_feature, self.num_labels, True)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_ids=None, labels=None,
                item_weights=None, proactive_masking=None,):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        # sequence_output = self.dropout(outputs[0]) # BLD
        sequence_output = outputs[0] # BLD
        start_ids = start_ids.unsqueeze(1)  # batch 1 L
        aspect_vec = torch.bmm(start_ids, sequence_output).squeeze()  # batch 1 d
        sequence_output = aspect_vec.unsqueeze(1).repeat(1, sequence_output.size(1), 1)  # batch L d

        focus_outputs = self.focusnet(attention_mask=attention_mask, hidden_states=sequence_output,
                                      item_weights=item_weights, proactive_masking=proactive_masking)
        all_hidden_states = focus_outputs[0] # bach L d
        all_item_weights = focus_outputs[2]  # batch*len*1

        method = self.method
        final_outputs = ()
        if method == '1':
            sequence_output = torch.squeeze(all_hidden_states[1])
        elif method == '2':
            sequence_output = torch.squeeze(all_hidden_states[2])
        elif method == '3':
            sequence_output = torch.squeeze(all_hidden_states[3])
        elif method == 'mean':
            t1 = torch.stack(all_hidden_states[1:])
            t1 = torch.mean(t1, dim=0, keepdim=False)
            sequence_output = torch.squeeze(t1) # Dd'
        else:
            print('============wrong============', flush=True)

        logits = self.typing(sequence_output)
        final_outputs = final_outputs + (logits,)
        L2_loss = torch.tensor(0.0)
        final_outputs = final_outputs + (L2_loss,)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels)
            final_outputs = final_outputs + (loss,)
        final_outputs = final_outputs + (all_item_weights,)
        return final_outputs