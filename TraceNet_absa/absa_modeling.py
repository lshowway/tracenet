from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from torch import nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import \
    RobertaModel, RobertaPreTrainedModel

logger = logging.getLogger(__name__)


class RoBERTaForABSA(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBERTaForABSA, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels, False)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask, token_type_ids)
        sequence_output = self.dropout(outputs[0])
        start_ids = start_ids.unsqueeze(1)  # batch 1 L
        entity_vec = torch.bmm(start_ids, sequence_output).squeeze(1)
        logits = self.typing(entity_vec)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits

