import logging, os, random, torch
import numpy as np
# from transformers.file_utils import is_tf_available
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
# from transformers import glue_processors


logger = logging.getLogger(__name__)


def get_vocab(path):
    # used to get the vocabulary in glove as input
    word_to_idx = {'UNK': 0, 'PAD': 1}
    idx = 2
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')[1:]
            line = ' '.join(line)
            words = line.split()
            for w in words:
                if w not in word_to_idx:
                    word_to_idx[w] = idx
                    idx += 1
    return word_to_idx


def load_glove_vectors(vocab, glove_file):
    with open(glove_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        data = {}
        for line in f:
            tokens = line.strip().split(' ')
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            data[word] = vec
    return data


def get_embedding_table(word_to_idx, glove_file):
    # extract the word embedding from glove cache according to vocabulary
    w2v = load_glove_vectors(word_to_idx, glove_file)
    V = len(word_to_idx)
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, 300))
    for word, vec in w2v.items():
        idx = word_to_idx.get(word, 0)
        if idx == 0 or idx == V:
            print(idx)
        embed[idx] = vec # padding word is positioned at index 0
    return embed


def get_batch_data(args, data_file, vocab_dic, batch_size):
    # generate batch data
    all_data = []
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            label, sentence = line[0], line[1:]
            sentence = ' '.join(sentence)
            words = [vocab_dic.get(w, 0) for w in sentence.split()]
            line_dic = {'text': words[:args.max_seq_length]}
            if args.task == 'yelp-5':
                line_dic['label'] = int(label) -1
            elif args.task == 'sst-5':
                label_dic_sst = {'__label__1': 0, '__label__2': 1, '__label__3': 2, '__label__4': 3, '__label__5': 4}
                line_dic['label'] = label_dic_sst[label]
            elif args.task == 'sst-2':
                line_dic['label'] = int(label)
            elif args.task == 'mr-2':
                line_dic['label'] = int(label)
            else:
                break
            line_dic['length'] = len(words[:args.max_seq_length])
            all_data.append(line_dic)
    random.shuffle(all_data) # , random=random.seed(args.seed)
    dataBuckt = []
    for start, end in zip(range(0, len(all_data), batch_size), range(batch_size, len(all_data), batch_size)):
        batchData = all_data[start: end]
        dataBuckt.append(batchData)
    newDataBuckt = []
    for idx, batch in enumerate(dataBuckt):
        batch_tmp = {"length": [], "text": [], "label": [], "iterations": idx+1}
        for data in batch:
            batch_tmp["length"].append(data['length'])
            batch_tmp["text"].append(data['text'])
            batch_tmp["label"].append(data['label'])
        max_len = args.max_seq_length
        batch_tmp["attn_mask"] = torch.LongTensor([x + [0] * (max_len-len(x)) for x in batch_tmp["text"]])
        batch_tmp["text"] = torch.LongTensor([x + [1] * (max_len-len(x)) for x in batch_tmp["text"]]) # pad = 1
        batch_tmp["length"] = torch.LongTensor(batch_tmp["length"])
        batch_tmp["label"] = torch.LongTensor(batch_tmp["label"])
        newDataBuckt.append(batch_tmp)
    return newDataBuckt


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features


class Sst5Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst_test.txt")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["__label__1", "__label__2", "__label__3", "__label__4", "__label__5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Yelp5Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "Yelp5.sample.train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "Yelp5.sample.dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "Yelp5.sample.test")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ['1', '2', '3', '4', '5']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Mr2Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "MR-2.train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "MR-2.dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "MR-2.test")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst.binary.train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst.binary.dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst.binary.test")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


tasks_num_labels = {
    "sst-5": 5,
    "sst-2": 2,
    "yelp-5": 5,
    "mr-2": 2,
}

processors = {
    "sst-5": Sst5Processor,
    "sst-2": Sst2Processor,
    "mr-2": Mr2Processor,
    "yelp-5": Yelp5Processor,
}

output_modes = {
    "sst-5": "classification",
    "sst-2": "classification",
    "mr-2": "classification",
    "yelp-5": "classification",
}


def load_and_cache_examples(args, task, tokenizer, evaluate):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if evaluate == 'test':
        if args.against:
            cached_features_file = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}".format(
                    "against_test",
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    str(args.max_seq_length),
                    str(task),
                ),
            )
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                label_list = processor.get_labels()
                examples = (processor.get_against_test_examples(args.data_dir))

                features = convert_examples_to_features(
                    examples, tokenizer, max_length=args.max_seq_length, task=args.task_name,
                    label_list=label_list, output_mode=output_mode,
                    pad_token=tokenizer.pad_token, pad_token_segment_id=tokenizer.pad_token_segment_id
                )
                if args.local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
        else:
            cached_features_file = os.path.join(
                args.data_dir,
                "cached_{}_{}_{}_{}".format(
                    "test",
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                    str(args.max_seq_length),
                    str(task),
                ),
            )
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
            else:
                logger.info("Creating features from dataset file at %s", args.data_dir)
                label_list = processor.get_labels()
                examples = (
                    processor.get_test_examples(args.data_dir)
                )
                features = convert_examples_to_features(
                    examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
                )
                if args.local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(features, cached_features_file)
    else:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                "dev" if evaluate == 'dev' else "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            examples = (
                processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            )
            features = convert_examples_to_features(
                examples, tokenizer, max_length=args.max_seq_length, task=args.task_name,
                label_list=label_list, output_mode=output_mode,
                pad_token=tokenizer.pad_token_id, pad_token_segment_id=tokenizer.pad_token_type_id
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` .")

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

