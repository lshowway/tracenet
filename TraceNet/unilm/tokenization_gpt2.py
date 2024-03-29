# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
												unicode_literals)

import sys
import logging
import os
import json
import six
import copy
from io import open
import regex as re
from hashlib import sha256
import tempfile

try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse

import requests
from tqdm import tqdm
import shutil

# from .file_utils import cached_path
# , is_tf_available, is_torch_available

# if is_tf_available():
#     import tensorflow as tf
# if is_torch_available():
#     import torch
import torch

logger = logging.getLogger(__name__)

try:
	from torch.hub import _get_torch_home

	torch_cache_home = _get_torch_home()
except ImportError:
	torch_cache_home = os.path.expanduser(
		os.getenv('TORCH_HOME', os.path.join(
			os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'transformers')

try:
	from pathlib import Path

	PYTORCH_PRETRAINED_BERT_CACHE = Path(
		os.getenv('PYTORCH_TRANSFORMERS_CACHE', os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
except (AttributeError, ImportError):
	PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_TRANSFORMERS_CACHE',
																						os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
																											default_cache_path))

PYTORCH_TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility
TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility

SPECIAL_TOKENS_MAP_FILE = 'special_tokens_map.json'
ADDED_TOKENS_FILE = 'added_tokens.json'
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'


def http_get(url, temp_file, proxies=None):
	req = requests.get(url, stream=True, proxies=proxies)
	content_length = req.headers.get('Content-Length')
	total = int(content_length) if content_length is not None else None
	progress = tqdm(unit="B", total=total)
	for chunk in req.iter_content(chunk_size=1024):
		if chunk:  # filter out keep-alive new chunks
			progress.update(len(chunk))
			temp_file.write(chunk)
	progress.close()


def url_to_filename(url, etag=None):
	"""
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) ands '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
	url_bytes = url.encode('utf-8')
	url_hash = sha256(url_bytes)
	filename = url_hash.hexdigest()

	if etag:
		etag_bytes = etag.encode('utf-8')
		etag_hash = sha256(etag_bytes)
		filename += '.' + etag_hash.hexdigest()

	if url.endswith('.h5'):
		filename += '.h5'

	return filename


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None, etag_timeout=10):
	"""
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
	if cache_dir is None:
		cache_dir = TRANSFORMERS_CACHE
	if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
		cache_dir = str(cache_dir)
	if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
		cache_dir = str(cache_dir)

	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	# Get eTag to add to filename, if it exists.
	if url.startswith("s3://"):
		etag = s3_etag(url, proxies=proxies)
	else:
		try:
			response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
			if response.status_code != 200:
				etag = None
			else:
				etag = response.headers.get("ETag")
		except (EnvironmentError, requests.exceptions.Timeout):
			etag = None

	if sys.version_info[0] == 2 and etag is not None:
		etag = etag.decode('utf-8')
	filename = url_to_filename(url, etag)

	# get cache path to put the file
	cache_path = os.path.join(cache_dir, filename)

	# If we don't have a connection (etag is None) and can't identify the file
	# try to get the last downloaded one
	if not os.path.exists(cache_path) and etag is None:
		matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
		matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
		if matching_files:
			cache_path = os.path.join(cache_dir, matching_files[-1])

	if not os.path.exists(cache_path) or force_download:
		# Download to temporary file, then copy to cache dir once finished.
		# Otherwise you get corrupt cache entries if the download gets interrupted.
		with tempfile.NamedTemporaryFile() as temp_file:
			logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

			# GET file object
			if url.startswith("s3://"):
				s3_get(url, temp_file, proxies=proxies)
			else:
				http_get(url, temp_file, proxies=proxies)

			# we are copying the file before closing it, so flush to avoid truncation
			temp_file.flush()
			# shutil.copyfileobj() starts at the current position, so go to the start
			temp_file.seek(0)

			logger.info("copying %s to cache at %s", temp_file.name, cache_path)
			with open(cache_path, 'wb') as cache_file:
				shutil.copyfileobj(temp_file, cache_file)

			logger.info("creating metadata file for %s", cache_path)
			meta = {'url': url, 'etag': etag}
			meta_path = cache_path + '.json'
			with open(meta_path, 'w') as meta_file:
				output_string = json.dumps(meta)
				if sys.version_info[0] == 2 and isinstance(output_string, str):
					output_string = unicode(output_string, 'utf-8')  # The beauty of python 2
				meta_file.write(output_string)

			logger.info("removing temp file %s", temp_file.name)

	return cache_path


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None):
	"""
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
    """
	if cache_dir is None:
		cache_dir = TRANSFORMERS_CACHE
	if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
		url_or_filename = str(url_or_filename)
	if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
		cache_dir = str(cache_dir)

	parsed = urlparse(url_or_filename)

	if parsed.scheme in ('http', 'https', 's3'):
		# URL, so get it from the cache (downloading if necessary)
		return get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
	elif os.path.exists(url_or_filename):
		# File, and it exists.
		return url_or_filename
	elif parsed.scheme == '':
		# File, but it doesn't exist.
		raise EnvironmentError("file {} not found".format(url_or_filename))
	else:
		# Something unknown
		raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


class PreTrainedTokenizer(object):
	""" Base class for all tokenizers.
    Handle all the shared methods for tokenization and special tokens as well as methods dowloading/caching/loading pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

    Class attributes (overridden by derived classes):

        - ``vocab_files_names``: a python ``dict`` with, as keys, the ``__init__`` keyword name of each vocabulary file required by the model, and as associated values, the filename for saving the associated file (string).
        - ``pretrained_vocab_files_map``: a python ``dict of dict`` the high-level keys being the ``__init__`` keyword name of each vocabulary file required by the model, the low-level being the `short-cut-names` (string) of the pretrained modelings with, as associated values, the `url` (string) to the associated pretrained vocabulary file.
        - ``max_model_input_sizes``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained modelings, and as associated values, the maximum length of the sequence inputs of this model, or None if the model has no maximum input size.
        - ``pretrained_init_configuration``: a python ``dict`` with, as keys, the `short-cut-names` (string) of the pretrained modelings, and as associated values, a dictionnary of specific arguments to pass to the ``__init__``method of the tokenizer class for this pretrained model when loading the tokenizer with the ``from_pretrained()`` method.

    Parameters:

        - ``bos_token``: (`Optional`) string: a beginning of sentence token. Will be associated to ``self.bos_token`` and ``self.bos_token_id``

        - ``eos_token``: (`Optional`) string: an end of sentence token. Will be associated to ``self.eos_token`` and ``self.eos_token_id``

        - ``unk_token``: (`Optional`) string: an unknown token. Will be associated to ``self.unk_token`` and ``self.unk_token_id``

        - ``sep_token``: (`Optional`) string: a separation token (e.g. to separate context and query in an input sequence). Will be associated to ``self.sep_token`` and ``self.sep_token_id``

        - ``pad_token``: (`Optional`) string: a padding token. Will be associated to ``self.pad_token`` and ``self.pad_token_id``

        - ``cls_token``: (`Optional`) string: a classification token (e.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model). Will be associated to ``self.cls_token`` and ``self.cls_token_id``

        - ``mask_token``: (`Optional`) string: a masking token (e.g. when training a model with masked-language modeling). Will be associated to ``self.mask_token`` and ``self.mask_token_id``

        - ``additional_special_tokens``: (`Optional`) list: a list of additional special tokens. Adding all special tokens here ensure they won't be split by the tokenization process. Will be associated to ``self.additional_special_tokens`` and ``self.additional_special_tokens_ids``
    """
	vocab_files_names = {}
	pretrained_vocab_files_map = {}
	pretrained_init_configuration = {}
	max_model_input_sizes = {}

	SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token",
															 "pad_token", "cls_token", "mask_token",
															 "additional_special_tokens"]

	@property
	def bos_token(self):
		""" Beginning of sentence token (string). Log an error if used while not having been set. """
		if self._bos_token is None:
			logger.error("Using bos_token, but it is not set yet.")
		return self._bos_token

	@property
	def eos_token(self):
		""" End of sentence token (string). Log an error if used while not having been set. """
		if self._eos_token is None:
			logger.error("Using eos_token, but it is not set yet.")
		return self._eos_token

	@property
	def unk_token(self):
		""" Unknown token (string). Log an error if used while not having been set. """
		if self._unk_token is None:
			logger.error("Using unk_token, but it is not set yet.")
		return self._unk_token

	@property
	def sep_token(self):
		""" Separation token (string). E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
		if self._sep_token is None:
			logger.error("Using sep_token, but it is not set yet.")
		return self._sep_token

	@property
	def pad_token(self):
		""" Padding token (string). Log an error if used while not having been set. """
		if self._pad_token is None:
			logger.error("Using pad_token, but it is not set yet.")
		return self._pad_token

	@property
	def cls_token(self):
		""" Classification token (string). E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
		if self._cls_token is None:
			logger.error("Using cls_token, but it is not set yet.")
		return self._cls_token

	@property
	def mask_token(self):
		""" Mask token (string). E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
		if self._mask_token is None:
			logger.error("Using mask_token, but it is not set yet.")
		return self._mask_token

	@property
	def additional_special_tokens(self):
		""" All the additional special tokens you may want to use (list of strings). Log an error if used while not having been set. """
		if self._additional_special_tokens is None:
			logger.error("Using additional_special_tokens, but it is not set yet.")
		return self._additional_special_tokens

	@bos_token.setter
	def bos_token(self, value):
		self._bos_token = value

	@eos_token.setter
	def eos_token(self, value):
		self._eos_token = value

	@unk_token.setter
	def unk_token(self, value):
		self._unk_token = value

	@sep_token.setter
	def sep_token(self, value):
		self._sep_token = value

	@pad_token.setter
	def pad_token(self, value):
		self._pad_token = value

	@cls_token.setter
	def cls_token(self, value):
		self._cls_token = value

	@mask_token.setter
	def mask_token(self, value):
		self._mask_token = value

	@additional_special_tokens.setter
	def additional_special_tokens(self, value):
		self._additional_special_tokens = value

	@property
	def bos_token_id(self):
		""" Id of the beginning of sentence token in the vocabulary. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.bos_token)

	@property
	def eos_token_id(self):
		""" Id of the end of sentence token in the vocabulary. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.eos_token)

	@property
	def unk_token_id(self):
		""" Id of the unknown token in the vocabulary. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.unk_token)

	@property
	def sep_token_id(self):
		""" Id of the separation token in the vocabulary. E.g. separate context and query in an input sequence. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.sep_token)

	@property
	def pad_token_id(self):
		""" Id of the padding token in the vocabulary. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.pad_token)

	@property
	def cls_token_id(self):
		""" Id of the classification token in the vocabulary. E.g. to extract a summary of an input sequence leveraging self-attention along the full depth of the model. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.cls_token)

	@property
	def mask_token_id(self):
		""" Id of the mask token in the vocabulary. E.g. when training a model with masked-language modeling. Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.mask_token)

	@property
	def additional_special_tokens_ids(self):
		""" Ids of all the additional special tokens in the vocabulary (list of integers). Log an error if used while not having been set. """
		return self.convert_tokens_to_ids(self.additional_special_tokens)

	def __init__(self, max_len=None, **kwargs):
		self._bos_token = None
		self._eos_token = None
		self._unk_token = None
		self._sep_token = None
		self._pad_token = None
		self._cls_token = None
		self._mask_token = None
		self._additional_special_tokens = []

		self.max_len = max_len if max_len is not None else int(1e12)

		# Added tokens
		self.added_tokens_encoder = {}
		self.added_tokens_decoder = {}

		# inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
		self.init_inputs = ()
		self.init_kwargs = {}

		for key, value in kwargs.items():
			if key in self.SPECIAL_TOKENS_ATTRIBUTES:
				if key == 'additional_special_tokens':
					assert isinstance(value, (list, tuple)) and all(
						isinstance(t, str) or (six.PY2 and isinstance(t, unicode)) for t in value)
				else:
					assert isinstance(value, str) or (six.PY2 and isinstance(value, unicode))
				setattr(self, key, value)

	@classmethod
	def from_pretrained(cls, *inputs, **kwargs):
		r"""
        Instantiate a :class:`~transformers.PreTrainedTokenizer` (or a derived class) from a predefined tokenizer.

        Args:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a predefined tokenizer to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing vocabulary files required by the tokenizer, for instance saved using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - (not applicable to all derived classes) a path or url to a single saved vocabulary file if and only if the tokenizer only requires a single vocabulary file (e.g. Bert, XLNet), e.g.: ``./my_model_directory/vocab.txt``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the vocabulary files and override the cached versions if they exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            inputs: (`optional`) positional arguments: will be passed to the Tokenizer ``__init__`` method.

            kwargs: (`optional`) keyword arguments: will be passed to the Tokenizer ``__init__`` method. Can be used to set special tokens like ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``, ``additional_special_tokens``. See parameters in the doc string of :class:`~transformers.PreTrainedTokenizer` for details.

        Examples::

            # We can't instantiate directly the base class `PreTrainedTokenizer` so let's show our examples on a derived class: BertTokenizer

            # Download vocabulary from S3 and cache.
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/')

            # If the tokenizer uses a single vocabulary file, you can point directly to this file
            tokenizer = BertTokenizer.from_pretrained('./test/saved_model/my_vocab.txt')

            # You can link tokens to special vocabulary when instantiating
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')
            # You should be sure '<unk>' is in the vocabulary when doing that.
            # Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'}) instead)
            assert tokenizer.unk_token == '<unk>'

        """
		return cls._from_pretrained(*inputs, **kwargs)

	@classmethod
	def _from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
		cache_dir = kwargs.pop('cache_dir', None)
		force_download = kwargs.pop('force_download', False)
		proxies = kwargs.pop('proxies', None)

		s3_models = list(cls.max_model_input_sizes.keys())
		vocab_files = {}
		init_configuration = {}
		if pretrained_model_name_or_path in s3_models:
			# Get the vocabulary from AWS S3 bucket
			for file_id, map_list in cls.pretrained_vocab_files_map.items():
				vocab_files[file_id] = map_list[pretrained_model_name_or_path]
			if cls.pretrained_init_configuration and pretrained_model_name_or_path in cls.pretrained_init_configuration:
				init_configuration = cls.pretrained_init_configuration[pretrained_model_name_or_path]
		else:
			# Get the vocabulary from local files
			logger.info(
				"Model name '{}' not found in model shortcut name list ({}). "
				"Assuming '{}' is a path or url to a directory containing tokenizer files.".format(
					pretrained_model_name_or_path, ', '.join(s3_models),
					pretrained_model_name_or_path))

			# Look for the tokenizer main vocabulary files
			for file_id, file_name in cls.vocab_files_names.items():
				if os.path.isdir(pretrained_model_name_or_path):
					# If a directory is provided we look for the standard filenames
					full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
				else:
					# If a path to a file is provided we use it (will only work for non-BPE tokenizer using a single vocabulary file)
					full_file_name = pretrained_model_name_or_path
				if not os.path.exists(full_file_name):
					logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
					full_file_name = None
				vocab_files[file_id] = full_file_name

			# Look for the additional tokens files
			additional_files_names = {'added_tokens_file': ADDED_TOKENS_FILE,
																'special_tokens_map_file': SPECIAL_TOKENS_MAP_FILE,
																'tokenizer_config_file': TOKENIZER_CONFIG_FILE,
																}

			# If a path to a file was provided, get the parent directory
			saved_directory = pretrained_model_name_or_path
			if os.path.exists(saved_directory) and not os.path.isdir(saved_directory):
				saved_directory = os.path.dirname(saved_directory)

			for file_id, file_name in additional_files_names.items():
				full_file_name = os.path.join(saved_directory, file_name)
				if not os.path.exists(full_file_name):
					logger.info("Didn't find file {}. We won't load it.".format(full_file_name))
					full_file_name = None
				vocab_files[file_id] = full_file_name

			if all(full_file_name is None for full_file_name in vocab_files.values()):
				raise EnvironmentError(
					"Model name '{}' was not found in tokenizers model name list ({}). "
					"We assumed '{}' was a path or url to a directory containing vocabulary files "
					"named {} but couldn't find such vocabulary files at this path or url.".format(
						pretrained_model_name_or_path, ', '.join(s3_models),
						pretrained_model_name_or_path,
						list(cls.vocab_files_names.values())))

		# Get files from url, cache, or disk depending on the case
		try:
			resolved_vocab_files = {}
			for file_id, file_path in vocab_files.items():
				if file_path is None:
					resolved_vocab_files[file_id] = None
				else:
					resolved_vocab_files[file_id] = cached_path(file_path, cache_dir=cache_dir, force_download=force_download,
																											proxies=proxies)
		except EnvironmentError:
			if pretrained_model_name_or_path in s3_models:
				msg = "Couldn't reach server at '{}' to download vocabulary files."
			else:
				msg = "Model name '{}' was not found in tokenizers model name list ({}). " \
							"We assumed '{}' was a path or url to a directory containing vocabulary files " \
							"named {}, but couldn't find such vocabulary files at this path or url.".format(
					pretrained_model_name_or_path, ', '.join(s3_models),
					pretrained_model_name_or_path,
					list(cls.vocab_files_names.values()))

			raise EnvironmentError(msg)

		for file_id, file_path in vocab_files.items():
			if file_path == resolved_vocab_files[file_id]:
				logger.info("loading file {}".format(file_path))
			else:
				logger.info("loading file {} from cache at {}".format(
					file_path, resolved_vocab_files[file_id]))

		# Prepare tokenizer initialization kwargs
		# Did we saved some inputs and kwargs to reload ?
		tokenizer_config_file = resolved_vocab_files.pop('tokenizer_config_file', None)
		if tokenizer_config_file is not None:
			init_kwargs = json.load(open(tokenizer_config_file, encoding="utf-8"))
			saved_init_inputs = init_kwargs.pop('init_inputs', ())
			if not init_inputs:
				init_inputs = saved_init_inputs
		else:
			init_kwargs = init_configuration

		# Update with newly provided kwargs
		init_kwargs.update(kwargs)

		# Set max length if needed
		if pretrained_model_name_or_path in cls.max_model_input_sizes:
			# if we're using a pretrained model, ensure the tokenizer
			# wont index sequences longer than the number of positional embeddings
			max_len = cls.max_model_input_sizes[pretrained_model_name_or_path]
			if max_len is not None and isinstance(max_len, (int, float)):
				init_kwargs['max_len'] = min(init_kwargs.get('max_len', int(1e12)), max_len)

		# Merge resolved_vocab_files arguments in init_kwargs.
		added_tokens_file = resolved_vocab_files.pop('added_tokens_file', None)
		special_tokens_map_file = resolved_vocab_files.pop('special_tokens_map_file', None)
		for args_name, file_path in resolved_vocab_files.items():
			if args_name not in init_kwargs:
				init_kwargs[args_name] = file_path
		if special_tokens_map_file is not None:
			special_tokens_map = json.load(open(special_tokens_map_file, encoding="utf-8"))
			for key, value in special_tokens_map.items():
				if key not in init_kwargs:
					init_kwargs[key] = value

		# Instantiate tokenizer.
		tokenizer = cls(*init_inputs, **init_kwargs)

		# Save inputs and kwargs for saving and re-loading with ``save_pretrained``
		tokenizer.init_inputs = init_inputs
		tokenizer.init_kwargs = init_kwargs

		# Add supplementary tokens.
		if added_tokens_file is not None:
			added_tok_encoder = json.load(open(added_tokens_file, encoding="utf-8"))
			added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
			tokenizer.added_tokens_encoder.update(added_tok_encoder)
			tokenizer.added_tokens_decoder.update(added_tok_decoder)

		return tokenizer

	def save_pretrained(self, save_directory):
		""" Save the tokenizer vocabulary files together with:
                - added tokens,
                - special-tokens-to-class-attributes-mapping,
                - tokenizer instantiation positional and keywords inputs (e.g. do_lower_case for Bert).

            This won't save modifications other than (added tokens and special token mapping) you may have
            applied to the tokenizer after the instantiation (e.g. modifying tokenizer.do_lower_case after creation).

            This method make sure the full tokenizer can then be re-loaded using the :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
		if not os.path.isdir(save_directory):
			logger.error("Saving directory ({}) should be a directory".format(save_directory))
			return

		special_tokens_map_file = os.path.join(save_directory, SPECIAL_TOKENS_MAP_FILE)
		added_tokens_file = os.path.join(save_directory, ADDED_TOKENS_FILE)
		tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)

		tokenizer_config = copy.deepcopy(self.init_kwargs)
		tokenizer_config['init_inputs'] = copy.deepcopy(self.init_inputs)
		for file_id in self.vocab_files_names.keys():
			tokenizer_config.pop(file_id, None)

		with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
			f.write(json.dumps(tokenizer_config, ensure_ascii=False))

		with open(special_tokens_map_file, 'w', encoding='utf-8') as f:
			f.write(json.dumps(self.special_tokens_map, ensure_ascii=False))

		with open(added_tokens_file, 'w', encoding='utf-8') as f:
			if self.added_tokens_encoder:
				out_str = json.dumps(self.added_tokens_encoder, ensure_ascii=False)
			else:
				out_str = u"{}"
			f.write(out_str)

		vocab_files = self.save_vocabulary(save_directory)

		return vocab_files + (special_tokens_map_file, added_tokens_file)

	def save_vocabulary(self, save_directory):
		""" Save the tokenizer vocabulary to a directory. This method does *NOT* save added tokens
            and special token mappings.

            Please use :func:`~transformers.PreTrainedTokenizer.save_pretrained` `()` to save the full Tokenizer state if you want to reload it using the :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
		raise NotImplementedError

	def vocab_size(self):
		""" Size of the base vocabulary (without the added tokens) """
		raise NotImplementedError

	def __len__(self):
		""" Size of the full vocabulary with the added tokens """
		return self.vocab_size + len(self.added_tokens_encoder)

	def add_tokens(self, new_tokens):
		"""
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the
        vocabulary, they are added to it with indices starting from length of the current vocabulary.

        Args:
            new_tokens: list of string. Each string is a token to add. Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to increase the vocabulary of Bert model and tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        """
		if not new_tokens:
			return 0

		to_add_tokens = []
		for token in new_tokens:
			assert isinstance(token, str) or (six.PY2 and isinstance(token, unicode))
			if token != self.unk_token and \
					self.convert_tokens_to_ids(token) == self.convert_tokens_to_ids(self.unk_token) and \
					token not in to_add_tokens:
				to_add_tokens.append(token)
				logger.info("Adding %s to the vocabulary", token)

		added_tok_encoder = dict((tok, len(self) + i) for i, tok in enumerate(to_add_tokens))
		added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
		self.added_tokens_encoder.update(added_tok_encoder)
		self.added_tokens_decoder.update(added_tok_decoder)

		return len(to_add_tokens)

	def num_added_tokens(self, pair=False):
		"""
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
		token_ids_0 = []
		token_ids_1 = []
		return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

	def add_special_tokens(self, special_tokens_dict):
		"""
        Add a dictionary of special tokens (eos, pad, cls...) to the encoder and link them
        to class attributes. If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Using `add_special_tokens` will ensure your special tokens can be used in several ways:

        - special tokens are carefully handled by the tokenizer (they are never split)
        - you can easily refer to special tokens using tokenizer class attributes like `tokenizer.cls_token`. This makes it easy to develop model-agnostic training and fine-tuning scripts.

        When possible, special tokens are already registered for provided pretrained modelings (ex: BertTokenizer cls_token is already registered to be '[CLS]' and XLM's one is also registered to be '</s>')

        Args:
            special_tokens_dict: dict of string. Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

                Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer assign the index of the ``unk_token`` to them).

        Returns:
            Number of tokens added to the vocabulary.

        Examples::

            # Let's see how to add a new classification token to GPT-2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')

            special_tokens_dict = {'cls_token': '<CLS>'}

            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print('We have added', num_added_toks, 'tokens')
            model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

            assert tokenizer.cls_token == '<CLS>'
        """
		if not special_tokens_dict:
			return 0

		added_tokens = 0
		for key, value in special_tokens_dict.items():
			assert key in self.SPECIAL_TOKENS_ATTRIBUTES
			if key == 'additional_special_tokens':
				assert isinstance(value, (list, tuple)) and all(
					isinstance(t, str) or (six.PY2 and isinstance(t, unicode)) for t in value)
				added_tokens += self.add_tokens(value)
			else:
				assert isinstance(value, str) or (six.PY2 and isinstance(value, unicode))
				added_tokens += self.add_tokens([value])
			logger.info("Assigning %s to the %s key of the tokenizer", value, key)
			setattr(self, key, value)

		return added_tokens

	def tokenize(self, text, **kwargs):
		""" Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """

		def split_on_token(tok, text):
			result = []
			split_text = text.split(tok)
			for i, sub_text in enumerate(split_text):
				sub_text = sub_text.strip()
				if i == 0 and not sub_text:
					result += [tok]
				elif i == len(split_text) - 1:
					if sub_text:
						result += [sub_text]
					else:
						pass
				else:
					if sub_text:
						result += [sub_text]
					result += [tok]
			return result

		def split_on_tokens(tok_list, text):
			if not text:
				return []
			if not tok_list:
				return self._tokenize(text, **kwargs)

			tokenized_text = []
			text_list = [text]
			for tok in tok_list:
				tokenized_text = []
				for sub_text in text_list:
					if sub_text not in self.added_tokens_encoder \
							and sub_text not in self.all_special_tokens:
						tokenized_text += split_on_token(tok, sub_text)
					else:
						tokenized_text += [sub_text]
				text_list = tokenized_text

			return sum((self._tokenize(token, **kwargs) if token not \
																										 in self.added_tokens_encoder and token not in self.all_special_tokens \
										else [token] for token in tokenized_text), [])

		added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
		tokenized_text = split_on_tokens(added_tokens, text)
		return tokenized_text

	def _tokenize(self, text, **kwargs):
		""" Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Do NOT take care of added tokens.
        """
		raise NotImplementedError

	def convert_tokens_to_ids(self, tokens):
		""" Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
		if tokens is None:
			return None

		if isinstance(tokens, str) or (six.PY2 and isinstance(tokens, unicode)):
			return self._convert_token_to_id_with_added_voc(tokens)

		ids = []
		for token in tokens:
			ids.append(self._convert_token_to_id_with_added_voc(token))
		if len(ids) > self.max_len:
			logger.warning("Token indices sequence length is longer than the specified maximum sequence length "
										 "for this model ({} > {}). Running this sequence through the model will result in "
										 "indexing errors".format(len(ids), self.max_len))
		return ids

	def _convert_token_to_id_with_added_voc(self, token):
		if token is None:
			return None

		if token in self.added_tokens_encoder:
			return self.added_tokens_encoder[token]
		return self._convert_token_to_id(token)

	def _convert_token_to_id(self, token):
		raise NotImplementedError

	def encode(self,
						 text,
						 text_pair=None,
						 add_special_tokens=True,
						 max_length=None,
						 stride=0,
						 truncation_strategy='longest_first',
						 return_tensors=None,
						 **kwargs):
		"""
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
		encoded_inputs = self.encode_plus(text,
																			text_pair=text_pair,
																			max_length=max_length,
																			add_special_tokens=add_special_tokens,
																			stride=stride,
																			truncation_strategy=truncation_strategy,
																			return_tensors=return_tensors,
																			**kwargs)

		return encoded_inputs["input_ids"]

	def encode_plus(self,
									text,
									text_pair=None,
									add_special_tokens=True,
									max_length=None,
									stride=0,
									truncation_strategy='longest_first',
									return_tensors=None,
									**kwargs):
		"""
        Returns a dictionary containing the encoded sequence or sequence pair and additional informations:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """

		def get_input_ids(text):
			if isinstance(text, six.string_types):
				return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
			elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], six.string_types):
				return self.convert_tokens_to_ids(text)
			elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
				return text
			else:
				raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

		first_ids = get_input_ids(text)
		second_ids = get_input_ids(text_pair) if text_pair is not None else None

		return self.prepare_for_model(first_ids,
																	pair_ids=second_ids,
																	max_length=max_length,
																	add_special_tokens=add_special_tokens,
																	stride=stride,
																	truncation_strategy=truncation_strategy,
																	return_tensors=return_tensors)

	def prepare_for_model(self, ids, pair_ids=None, max_length=None, add_special_tokens=True, stride=0,
												truncation_strategy='longest_first', return_tensors=None):
		"""
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens and manages a window stride for
        overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful for edge effect removal when using sequential
                list of inputs.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.

        Return:
            A Dictionary of shape::

                {
                    input_ids: list[int],
                    overflowing_tokens: list[int] if a ``max_length`` is specified, else None
                    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True``
                }

            With the fields:
                ``input_ids``: list of tokens to be fed to a model

                ``overflowing_tokens``: list of overflowing tokens if a max length is specified.

                ``special_tokens_mask``: if adding special tokens, this is a list of [0, 1], with 0 specifying special added
                tokens and 1 specifying sequence tokens.
        """
		pair = bool(pair_ids is not None)
		len_ids = len(ids)
		len_pair_ids = len(pair_ids) if pair else 0

		encoded_inputs = {}
		total_len = len_ids + len_pair_ids + (self.num_added_tokens(pair=pair) if add_special_tokens else 0)
		if max_length and total_len > max_length:
			ids, pair_ids, overflowing_tokens = self.truncate_sequences(ids, pair_ids=pair_ids,
																																	num_tokens_to_remove=total_len - max_length,
																																	truncation_strategy=truncation_strategy,
																																	stride=stride)
			encoded_inputs["overflowing_tokens"] = overflowing_tokens
			encoded_inputs["num_truncated_tokens"] = total_len - max_length

		if add_special_tokens:
			sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
			token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
			encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
		else:
			sequence = ids + pair_ids if pair else ids
			token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

		if return_tensors == 'tf' and is_tf_available():
			sequence = tf.constant([sequence])
			token_type_ids = tf.constant([token_type_ids])
		elif return_tensors == 'pt' and is_torch_available():
			sequence = torch.tensor([sequence])
			token_type_ids = torch.tensor([token_type_ids])
		elif return_tensors is not None:
			logger.warning(
				"Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(return_tensors))

		encoded_inputs["input_ids"] = sequence
		encoded_inputs["token_type_ids"] = token_type_ids

		if max_length and len(encoded_inputs["input_ids"]) > max_length:
			encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
			encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
			encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]

		return encoded_inputs

	def truncate_sequences(self, ids, pair_ids=None, num_tokens_to_remove=0, truncation_strategy='longest_first',
												 stride=0):
		"""Truncates a sequence pair in place to the maximum length.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences).
                    Overflowing tokens only contains overflow from the first sequence.
                - 'only_first': Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
        """
		if num_tokens_to_remove <= 0:
			return ids, pair_ids, []

		if truncation_strategy == 'longest_first':
			overflowing_tokens = []
			for _ in range(num_tokens_to_remove):
				if pair_ids is None or len(ids) > len(pair_ids):
					overflowing_tokens = [ids[-1]] + overflowing_tokens
					ids = ids[:-1]
				else:
					pair_ids = pair_ids[:-1]
			window_len = min(len(ids), stride)
			if window_len > 0:
				overflowing_tokens = ids[-window_len:] + overflowing_tokens
		elif truncation_strategy == 'only_first':
			assert len(ids) > num_tokens_to_remove
			window_len = min(len(ids), stride + num_tokens_to_remove)
			overflowing_tokens = ids[-window_len:]
			ids = ids[:-num_tokens_to_remove]
		elif truncation_strategy == 'only_second':
			assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
			window_len = min(len(pair_ids), stride + num_tokens_to_remove)
			overflowing_tokens = pair_ids[-window_len:]
			pair_ids = pair_ids[:-num_tokens_to_remove]
		elif truncation_strategy == 'do_not_truncate':
			raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
		else:
			raise ValueError(
				"Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']")
		return (ids, pair_ids, overflowing_tokens)

	def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
		logger.warning("This tokenizer does not make use of special tokens.")
		if token_ids_1 is None:
			return len(token_ids_0) * [0]
		return [0] * len(token_ids_0) + [1] * len(token_ids_1)

	def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
		"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
		logger.warning("This tokenizer does not make use of special tokens. Input is returned with no modification.")
		if token_ids_1 is None:
			return token_ids_0
		return token_ids_0 + token_ids_1

	def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
		"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """
		return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

	def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
		""" Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.

            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
		if isinstance(ids, int):
			if ids in self.added_tokens_decoder:
				return self.added_tokens_decoder[ids]
			else:
				return self._convert_id_to_token(ids)
		tokens = []
		for index in ids:
			if skip_special_tokens and index in self.all_special_ids:
				continue
			if index in self.added_tokens_decoder:
				tokens.append(self.added_tokens_decoder[index])
			else:
				tokens.append(self._convert_id_to_token(index))
		return tokens

	def _convert_id_to_token(self, index):
		raise NotImplementedError

	def convert_tokens_to_string(self, tokens):
		""" Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
		return ' '.join(self.convert_ids_to_tokens(tokens))

	def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
		"""
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.

        Args:
            token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
            skip_special_tokens: if set to True, will replace special tokens.
            clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
        """
		filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

		# To avoid mixing byte-level and unicode for byte-level BPT
		# we need to build string separatly for added tokens and byte-level tokens
		# cf. https://github.com/huggingface/transformers/issues/1133
		sub_texts = []
		current_sub_text = []
		for token in filtered_tokens:
			if skip_special_tokens and token in self.all_special_ids:
				continue
			if token in self.added_tokens_encoder:
				if current_sub_text:
					sub_texts.append(self.convert_tokens_to_string(current_sub_text))
					current_sub_text = []
				sub_texts.append(" " + token)
			else:
				current_sub_text.append(token)
		if current_sub_text:
			sub_texts.append(self.convert_tokens_to_string(current_sub_text))
		text = ''.join(sub_texts)

		if clean_up_tokenization_spaces:
			clean_text = self.clean_up_tokenization(text)
			return clean_text
		else:
			return text

	@property
	def special_tokens_map(self):
		""" A dictionary mapping special token class attribute (cls_token, unk_token...) to their
            values ('<unk>', '<cls>'...)
        """
		set_attr = {}
		for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
			attr_value = getattr(self, "_" + attr)
			if attr_value:
				set_attr[attr] = attr_value
		return set_attr

	@property
	def all_special_tokens(self):
		""" List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
		all_toks = []
		set_attr = self.special_tokens_map
		for attr_value in set_attr.values():
			all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
		all_toks = list(set(all_toks))
		return all_toks

	@property
	def all_special_ids(self):
		""" List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
		all_toks = self.all_special_tokens
		all_ids = list(self._convert_token_to_id(t) for t in all_toks)
		return all_ids

	@staticmethod
	def clean_up_tokenization(out_string):
		""" Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
		out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
																																														 ).replace(" ' ",
																																																			 "'").replace(
			" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
																									).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
		return out_string


try:
	from functools import lru_cache
except ImportError:
	# Just a dummy decorator to get the checks to run on python2
	# because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
	def lru_cache():
		return lambda func: func

# from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
	'vocab_file': 'vocab.json',
	'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
	'vocab_file':
		{
			'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
			'gpt2-medium': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json",
			'gpt2-large': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json",
			'gpt2-xl': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json",
			'distilgpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-vocab.json",
		},
	'merges_file':
		{
			'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
			'gpt2-medium': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt",
			'gpt2-large': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt",
			'gpt2-xl': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt",
			'distilgpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-merges.txt",
		},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
	'gpt2': 1024,
	'gpt2-medium': 1024,
	'gpt2-large': 1024,
	'gpt2-xl': 1024,
	'distilgpt2': 1024,
}


@lru_cache()
def bytes_to_unicode():
	"""
    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
	_chr = unichr if sys.version_info[0] == 2 else chr
	bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
	cs = bs[:]
	n = 0
	for b in range(2 ** 8):
		if b not in bs:
			bs.append(b)
			cs.append(2 ** 8 + n)
			n += 1
	cs = [_chr(n) for n in cs]
	return dict(zip(bs, cs))


def get_pairs(word):
	"""Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
	pairs = set()
	prev_char = word[0]
	for char in word[1:]:
		pairs.add((prev_char, char))
		prev_char = char
	return pairs


class GPT2Tokenizer(PreTrainedTokenizer):
	"""
    GPT-2 BPE tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """
	vocab_files_names = VOCAB_FILES_NAMES
	pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
	max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

	def __init__(self, vocab_file, merges_file, errors='replace', unk_token="<|endoftext|>",
							 bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs):
		super(GPT2Tokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
		self.max_len_single_sentence = self.max_len  # no default special tokens - you can update this value if you add special tokens
		self.max_len_sentences_pair = self.max_len  # no default special tokens - you can update this value if you add special tokens

		# print('@@@@@@@@@@@@@@@', vocab_file)
		self.encoder = json.load(open(vocab_file, encoding="utf-8"))
		self.decoder = {v: k for k, v in self.encoder.items()}
		self.errors = errors  # how to handle errors in decoding
		self.byte_encoder = bytes_to_unicode()
		self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
		bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
		bpe_merges = [tuple(merge.split()) for merge in bpe_data]
		self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
		self.cache = {}

		# Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
		self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

	@property
	def vocab_size(self):
		return len(self.encoder)

	def bpe(self, token):
		if token in self.cache:
			return self.cache[token]
		word = tuple(token)
		pairs = get_pairs(word)

		if not pairs:
			return token

		while True:
			bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
			if bigram not in self.bpe_ranks:
				break
			first, second = bigram
			new_word = []
			i = 0
			while i < len(word):
				try:
					j = word.index(first, i)
					new_word.extend(word[i:j])
					i = j
				except:
					new_word.extend(word[i:])
					break

				if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
					new_word.append(first + second)
					i += 2
				else:
					new_word.append(word[i])
					i += 1
			new_word = tuple(new_word)
			word = new_word
			if len(word) == 1:
				break
			else:
				pairs = get_pairs(word)
		word = ' '.join(word)
		self.cache[token] = word
		return word

	def _tokenize(self, text, add_prefix_space=False):
		""" Tokenize a string.
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space toto get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """
		if add_prefix_space:
			text = ' ' + text

		bpe_tokens = []
		for token in re.findall(self.pat, text):
			if sys.version_info[0] == 2:
				token = ''.join(self.byte_encoder[ord(b)] for b in
												token)  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
			else:
				token = ''.join(self.byte_encoder[b] for b in token.encode(
					'utf-8'))  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
			bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
		return bpe_tokens

	def _convert_token_to_id(self, token):
		""" Converts a token (str/unicode) in an id using the vocab. """
		return self.encoder.get(token, self.encoder.get(self.unk_token))

	def _convert_id_to_token(self, index):
		"""Converts an index (integer) in a token (string/unicode) using the vocab."""
		return self.decoder.get(index)

	def convert_tokens_to_string(self, tokens):
		""" Converts a sequence of tokens (string) in a single string. """
		text = ''.join(tokens)
		text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
		return text

	def save_vocabulary(self, save_directory):
		"""Save the tokenizer vocabulary and merge files to a directory."""
		if not os.path.isdir(save_directory):
			logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
			return
		vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])
		merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES['merges_file'])

		with open(vocab_file, 'w', encoding='utf-8') as f:
			f.write(json.dumps(self.encoder, ensure_ascii=False))

		index = 0
		with open(merge_file, "w", encoding="utf-8") as writer:
			writer.write(u'#version: 0.2\n')
			for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
				if index != token_index:
					logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
												 " Please check that the tokenizer is not corrupted!".format(merge_file))
					index = token_index
				writer.write(' '.join(bpe_tokens) + u'\n')
				index += 1

		return vocab_file, merge_file


try:
	from functools import lru_cache
except ImportError:
	# Just a dummy decorator to get the checks to run on python2
	# because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
	def lru_cache():
		return lambda func: func

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
	'vocab_file': 'vocab.json',
	'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
	'vocab_file':
		{
			'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
			'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
			'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
			'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
			'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
			'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
		},
	'merges_file':
		{
			'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
			'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
			'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt",
			'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt",
			'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt",
			'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt",
		},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
	'roberta-base': 512,
	'roberta-large': 512,
	'roberta-large-mnli': 512,
	'distilroberta-base': 512,
	'roberta-base-openai-detector': 512,
	'roberta-large-openai-detector': 512,
}


class RobertaTokenizer(GPT2Tokenizer):
	"""
    RoBERTa BPE tokenizer, derived from the GPT-2 tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """
	vocab_files_names = VOCAB_FILES_NAMES
	pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
	max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

	def __init__(self, vocab_file, merges_file, errors='replace', bos_token="<s>", eos_token="</s>", sep_token="</s>",
							 cls_token="<s>", unk_token="<unk>", pad_token='<pad>', mask_token='<mask>', **kwargs):
		super(RobertaTokenizer, self).__init__(vocab_file=vocab_file, merges_file=merges_file, errors=errors,
																					 bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
																					 sep_token=sep_token, cls_token=cls_token, pad_token=pad_token,
																					 mask_token=mask_token, **kwargs)
		self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
		self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens

	def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
		"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            single sequence: <s> X </s>
            pair of sequences: <s> A </s></s> B </s>
        """
		if token_ids_1 is None:
			return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
		cls = [self.cls_token_id]
		sep = [self.sep_token_id]
		return cls + token_ids_0 + sep + sep + token_ids_1 + sep

	def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
		"""
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """
		if already_has_special_tokens:
			if token_ids_1 is not None:
				raise ValueError("You should not supply a second sequence if the provided sequence of "
												 "ids is already formated with special tokens for the model.")
			return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

		if token_ids_1 is None:
			return [1] + ([0] * len(token_ids_0)) + [1]
		return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

	def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
		"""
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A RoBERTa sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
		sep = [self.sep_token_id]
		cls = [self.cls_token_id]

		if token_ids_1 is None:
			return len(cls + token_ids_0 + sep) * [0]
		return len(cls + token_ids_0 + sep + sep) * [0] + len(token_ids_1 + sep) * [1]
