# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.nn.functional as F

from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
      """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow modelings in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def mix_act(x):
    signature = (x >= 0).type_as(x)
    t = 1.12643 * x - math.pi
    small_mask = torch.abs(t) < 1e-5
    pos_mask = t >= 0
    t.masked_fill_(small_mask & pos_mask, 1e-5)
    t.masked_fill_(small_mask & ~pos_mask, -1e-5)
    return signature * (x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))) + (1.0 - signature) * torch.sin(t) / t
    # return signature * gelu(x) + (1.0 - signature) * torch.sin(1.12643 * x - math.pi) / (1.12643 * x - math.pi - 1e-6)


def mish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu,
          "swish": swish, "mix_act": mix_act, "mish": mish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
      """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 task_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 rel_pos_type=0,
                 max_rel_pos=128,
                 rel_pos_bins=32,
                 fast_qkv=False,
                 initializer_range=0.02):
        """Constructs BertConfig.

            Args:
                vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
                hidden_size: Size of the encoder layers and the pooler layer.
                num_hidden_layers: Number of hidden layers in the Transformer encoder.
                num_attention_heads: Number of attention heads for each attention layer in
                    the Transformer encoder.
                intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                    layer in the Transformer encoder.
                hidden_act: The non-linear activation function (function or string) in the
                    encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
                hidden_dropout_prob: The dropout probabilitiy for all fully connected
                    layers in the embeddings, encoder, and pooler.
                attention_probs_dropout_prob: The dropout ratio for the attention
                    probabilities.
                max_position_embeddings: The maximum sequence length that this model might
                    ever be used with. Typically set this to something large just in case
                    (e.g., 512 or 1024 or 2048).
                type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                    `BertModel`.
                initializer_range: The sttdev of the truncated_normal_initializer for
                    initializing all weight matrices.
            """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.task_dropout_prob = task_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.rel_pos_type = rel_pos_type
            self.max_rel_pos = max_rel_pos
            self.rel_pos_bins = rel_pos_bins
            self.fast_qkv = fast_qkv
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model self file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
                    """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x_in):
            x = x_in.float()
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            x = x.type_as(x_in)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
      """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None
        if hasattr(config, 'max_task_num') and config.max_task_num > 0:
            self.task_embeddings = nn.Embedding(
                config.max_task_num, config.hidden_size
            )
        else:
            self.task_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, task_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            embeddings = embeddings + \
                self.token_type_embeddings(token_type_ids)

        if (self.task_embeddings is not None) and (task_ids is not None):
            embeddings = embeddings + \
                self.task_embeddings(task_ids.unsqueeze(-1).expand_as(input_ids))

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.fast_qkv = config.fast_qkv
        if config.fast_qkv:
            self.qkv_linear = nn.Linear(
                config.hidden_size, 3*self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rel_pos_type = config.rel_pos_type

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q += self.q_bias
                v += self.v_bias
            else:
                _sz = (1,) * (q.ndimension()-1) + (-1,)
                q += self.q_bias.view(*_sz)
                v += self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    def forward(self, hidden_states, attention_mask, rel_pos=None):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        if self.rel_pos_type in (1, 2):
            attention_scores = attention_scores + rel_pos
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, rel_pos=None):
        self_output, attention_probs = self.self(input_tensor, attention_mask,
                                                 rel_pos=rel_pos)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, input_tensor):
        hidden_states = F.linear(
            hidden_states, self.dense.weight, self.dense.bias)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = F.linear(
            hidden_states, self.dense.weight, self.dense.bias)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, input_tensor):
        hidden_states = F.linear(
            hidden_states, self.dense.weight, self.dense.bias)
        hidden_states = self.dropout(hidden_states)
        # B = 1000
        # th = hidden_states * ((hidden_states < B).type_as(hidden_states)
        #                       * (hidden_states > -B).type_as(hidden_states))
        # hidden_states = th + (hidden_states > B).type_as(hidden_states) * torch.unsqueeze(th.max(dim=2)[0], dim=2) + (hidden_states < -B).type_as(hidden_states) * torch.unsqueeze(th.min(dim=2)[0], dim=2)
        return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, rel_pos=None):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask,
            rel_pos=rel_pos)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance /
                                                    max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])
        self.rel_pos_type = config.rel_pos_type
        self.max_rel_pos = config.max_rel_pos
        self.rel_pos_bins = config.rel_pos_bins
        if config.rel_pos_type in (1, 2):
            self.rel_pos_onehot_size = config.rel_pos_bins
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, output_attention=False, position_ids=None):
        # if self.training:
        #     print(self.rel_pos_bias.weight.data[0, -2:].tolist())
        rel_pos, predict_rel_pos = None, None
        if self.rel_pos_type in (1, 2):
            # (B,L,L)
            rel_pos_mat = position_ids.unsqueeze(-2) - \
                position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(
                rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
            rel_pos = F.one_hot(
                rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
            # (B,H,L,L)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        all_encoder_layers = []
        all_encoder_attention_probs = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(
                hidden_states, attention_mask,
                rel_pos=rel_pos)
            if output_attention:
                all_encoder_attention_probs.append(attention_probs)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attention_probs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# class BertPredictionHeadTransform(nn.Module):
#     def __init__(self, config):
#         super(BertPredictionHeadTransform, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
#             self.transform_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.transform_act_fn = config.hidden_act
#         self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
#
#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.transform_act_fn(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
          a simple interface for dowloading and loading pretrained modelings.
      """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            # numpy.truncnorm() would take a long time in philly clusters
            # module.weight = torch.nn.Parameter(torch.Tensor(
            #     truncnorm.rvs(-1, 1, size=list(module.weight.data.shape)) * self.config.initializer_range))
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        warmup_checkpoint=None, hub_path=None, from_tf=False, no_segment=False, rel_pos_type=0,
                        max_rel_pos=128, rel_pos_bins=32, fast_qkv=False, hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1, task_dropout_prob=0.1,
                        remove_task_specifical_layers=False, keep_cls=True, *inputs, **kwargs):
        """
            Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
            Download and cache the pre-trained model file if needed.

            Params:
                pretrained_model_name_or_path: either:
                    - a str with the name of a pre-trained model to load selected in the list of:
                        . `bert-base-uncased`
                        . `bert-large-uncased`
                        . `bert-base-cased`
                        . `bert-large-cased`
                        . `bert-base-multilingual-uncased`
                        . `bert-base-multilingual-cased`
                        . `bert-base-chinese`
                    - a path or url to a pretrained model archive containing:
                        . `bert_config.json` a configuration file for the model
                        . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                    - a path or url to a pretrained model archive containing:
                        . `bert_config.json` a configuration file for the model
                        . `model.chkpt` a TensorFlow checkpoint
                from_tf: should we load the weights from a locally saved TensorFlow checkpoint
                cache_dir: an optional path to a folder in which the pre-trained modelings will be cached.
                state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained modelings
                *inputs, **kwargs: additional input for the specific Bert class
                    (ex: num_labels for BertForSequenceClassification)
            """
        is_roberta = 'roberta' in pretrained_model_name_or_path
        if is_roberta:
            pretrained_model_name_or_path = "bert-" + \
                pretrained_model_name_or_path.split('-')[-1] + "-cased"
        if hub_path is not None:
            import file_util
            config_file = file_util.get_model_config_path(
                pretrained_model_name_or_path, hub_path)
            weights_path = file_util.get_model_path(
                pretrained_model_name_or_path, hub_path)
        else:
            if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
                archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
            else:
                archive_file = pretrained_model_name_or_path
            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file, cache_dir=cache_dir)
                logger.info("Find resolved_vocab_file = {}".format(
                    resolved_archive_file))
            except EnvironmentError:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file))
                return None
            if resolved_archive_file == archive_file:
                logger.info("loading archive file {}".format(archive_file))
            else:
                logger.info("loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
            tempdir = None
            if os.path.isdir(resolved_archive_file) or from_tf:
                serialization_dir = resolved_archive_file
            else:
                # Extract archive to temp dir
                tempdir = tempfile.mkdtemp()
                logger.info("extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir))
                with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(archive, tempdir)
                serialization_dir = tempdir
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
        # Load config
        logger.info("Load model config from: {}".format(config_file))
        if warmup_checkpoint is not None:
            weights_path = warmup_checkpoint
        logger.info("Load model weight from: {}".format(weights_path))
        config = BertConfig.from_json_file(config_file)
        if no_segment:
            config.type_vocab_size = 0
            logger.info("Set config to no segment embedding !")
        if is_roberta:
            config.vocab_size = 50265
        config.rel_pos_type = rel_pos_type
        config.max_rel_pos = max_rel_pos
        config.rel_pos_bins = rel_pos_bins
        config.fast_qkv = fast_qkv
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.task_dropout_prob = task_dropout_prob
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(
                weights_path, map_location='cpu')
        if hub_path is None and tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if hub_path is None and from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            # logger.info("Find key: {}, shape = {}".format(key, state_dict[key].shape))
            if key == 'bert.embeddings.token_type_embeddings.weight' and config.type_vocab_size < state_dict[key].shape[0]:
                state_dict[key].data = state_dict[key].data[:config.type_vocab_size, :]
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('discriminator.'):
                new_key = key[len("discriminator."):]
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        _all_head_size = config.num_attention_heads * \
            int(config.hidden_size / config.num_attention_heads)
        n_config_num_qkv = 1

        _k = 'bert.embeddings.position_embeddings.weight'
        n_config_pos_emb = 1
        if (_k in state_dict) and (n_config_pos_emb * config.hidden_size != state_dict[_k].shape[1]):
            logger.info(
                "n_config_pos_emb*config.hidden_size != state_dict[bert.embeddings.position_embeddings.weight] ({0}*{1} != {2})".format(
                    n_config_pos_emb, config.hidden_size, state_dict[_k].shape[1]))
            assert state_dict[_k].shape[1] % config.hidden_size == 0
            n_state_pos_emb = int(state_dict[_k].shape[1] / config.hidden_size)
            assert (n_state_pos_emb == 1) != (n_config_pos_emb ==
                                              1), "!!!!n_state_pos_emb == 1 xor n_config_pos_emb == 1!!!!"
            if n_state_pos_emb == 1:
                state_dict[_k].data = state_dict[_k].data.unsqueeze(1).repeat(
                    1, n_config_pos_emb, 1).reshape((config.max_position_embeddings, n_config_pos_emb * config.hidden_size))
            elif n_config_pos_emb == 1:
                _task_idx = 0
                state_dict[_k].data = state_dict[_k].data.view(
                    config.max_position_embeddings, n_state_pos_emb, config.hidden_size).select(1, _task_idx)
        if _k in state_dict and state_dict[_k].shape[0] != config.max_position_embeddings:
            logger.info("Adjust max_position_embeddings from {} to {}".format(
                state_dict[_k].shape[0], config.max_position_embeddings))
            state_dict[_k] = state_dict[_k][:config.max_position_embeddings, :]

        if remove_task_specifical_layers:
            to_remove_keys = []
            for key in state_dict.keys():
                if not (key.startswith("bert.embeddings.") or
                        key.startswith("bert.encoder.") or
                        key.startswith("bert.pooler.dense.")):
                    if keep_cls and (key.startswith("cls.predictions.") or key.startswith("cls.seq_relationship.")):
                        continue

                    to_remove_keys.append(key)

            for key in to_remove_keys:
                logger.info("Drop parameter: {}".format(key))
                state_dict.pop(key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model, config


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

      Params:
          config: a BertConfig class instance with the configuration to build a new model

      Inputs:
          `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
              with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
              `extract_features.py`, `run_classifier.py` and `run_squad.py`)
          `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
              types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
              a `sentence B` token (see BERT paper for more details).
          `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
              selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
              input sequence length in the current batch. It's the mask that we typically use for attention when
              a batch has varying length sentences.
          `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

      Outputs: Tuple of (encoded_layers, pooled_output)
          `encoded_layers`: controled by `output_all_encoded_layers` argument:
              - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                  of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                  encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
              - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                  to the last attention block of shape [batch_size, sequence_length, hidden_size],
          `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
              classifier pretrained on top of the hidden state associated to the first character of the
              input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

      Example usage:
      ```python
      # Already been converted into WordPiece token ids
      input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
      input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
      token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

      config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
          num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

      model = modeling.BertModel(config=config)
      all_encoder_layers, pooled_output = model(
          input_ids, token_type_ids, input_mask)
      ```
      """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, task_ids=None, output_attention=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        position_ids = torch.arange(input_ids.size(
            1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embedding_output = self.embeddings(
            input_ids, token_type_ids,
            position_ids=position_ids, task_ids=task_ids)
        encoded_layers, all_layers_attention_probs = self.encoder(embedding_output, extended_attention_mask,
                                                                  output_attention=output_attention,
                                                                  output_all_encoded_layers=output_all_encoded_layers,
                                                                  position_ids=position_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if output_attention:
            return encoded_layers, pooled_output, all_layers_attention_probs
        else:
            return encoded_layers, pooled_output


class UniLMForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
      This module is composed of the BERT model with a linear layer on top of
      the pooled output.

      Params:
          `config`: a BertConfig class instance with the configuration to build a new model.
          `num_labels`: the number of classes for the classifier. Default = 2.

      Inputs:
          `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
              with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
              `extract_features.py`, `run_classifier.py` and `run_squad.py`)
          `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
              types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
              a `sentence B` token (see BERT paper for more details).
          `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
              selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
              input sequence length in the current batch. It's the mask that we typically use for attention when
              a batch has varying length sentences.
          `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
              with indices selected in [0, ..., num_labels].

      Outputs:
          if `labels` is not `None`:
              Outputs the CrossEntropy classification loss of the output with the labels.
          if `labels` is `None`:
              Outputs the classification logits of shape [batch_size, num_labels].

      Example usage:
      ```python
      # Already been converted into WordPiece token ids
      input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
      input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
      token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

      config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
          num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

      num_labels = 2

      model = BertForSequenceClassification(config, num_labels)
      logits = model(input_ids, token_type_ids, input_mask)
      ```
      """

    def __init__(self, config, num_labels):
        super(UniLMForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.task_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.float().view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class UniLMTraceNetForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(UniLMTraceNetForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        from TraceNet.tracenet import TraceNetModel, Discriminator
        self.focusnet = TraceNetModel(config)
        self.discriminator = Discriminator(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                item_weights=None, proactive_masking=None,):

        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

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


class UniLMAttnForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(UniLMAttnForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        from TraceNet.tracenet import Discriminator
        # additive attn
        from TraceNet.attentions import AdditiveAttentionLayer as Attn

        # dot attn 论文中使用了dot attn了嘛？？

        # scale dot product attn
        # from attentions import ScaledDotProductAttention

        # stack N-layer attentions (additive)
        # from attentions import AdditiveAttentionModel as Attn

        # stack N-layer attentions (scaled dot product)
        # from attentions import ScaledDotProductAttentionModel as Attn
        self.focusnet = Attn(config)
        self.discriminator = Discriminator(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                item_weights=None, proactive_masking=None,):

        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

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

