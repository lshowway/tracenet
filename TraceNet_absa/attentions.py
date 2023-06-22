import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()

        self.query = nn.Linear(config.output_feature, config.output_feature)
        self.key = nn.Linear(config.output_feature, config.output_feature)
        self.value = nn.Linear(config.output_feature, config.output_feature)

        self.hidden_size = config.output_feature

        # self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, hidden_states):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)

        return output


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim):
        super(DotProductAttention, self).__init__()
        self.normalize = nn.LayerNorm(hidden_dim)
        self.out_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context.squeeze(1), attn


class AdditiveAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.output_feature, config.output_feature, bias=False) # q: batch*d*1  => batch*1*20
        self.w2 = nn.Linear(config.output_feature, config.output_feature, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, config.output_feature), requires_grad=True)
        self.activation = nn.Tanh()
        self.v = nn.Linear(config.output_feature, 1, bias=False)

    def forward(self, query, values):
        t1 = self.w1(query) + self.w2(values) # batch*len*d, 前者广播
        energy = self.activation(t1 + self.bias) # batch*len*d
        alpha = self.v(energy) # batch*len*d = >batch*len*1
        output = torch.matmul(alpha.permute(0, 2, 1), values)

        return output


class AdditiveAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([AdditiveAttentionLayer(config) for _ in range(config.num_hubo_layers)])

    def forward(self, query, values):
        layer_outputs = None
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(query, values)
            query = layer_outputs
            values = layer_outputs
        return layer_outputs


class ScaledDotProductAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([ScaledDotProductAttention(config) for _ in range(config.num_hubo_layers)])

    def forward(self, query, values=None):
        layer_outputs = None
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(query)
            query = layer_outputs
        return layer_outputs