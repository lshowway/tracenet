import csv
import json
import logging, os, torch
import numpy as np
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)


def convert_examples_to_features(examples, max_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        start, end = example.text_b[0], example.text_b[1]
        sentence = example.text_a
        tokens_0_start = tokenizer.tokenize(' '.join(sentence[:start]))
        tokens_start_end = tokenizer.tokenize(' '.join(sentence[start:end]))
        tokens_end_last = tokenizer.tokenize(' '.join(sentence[end:]))
        tokens = [tokenizer.cls_token] + tokens_0_start + tokens_start_end + tokens_end_last + [tokenizer.sep_token]
        tokens = tokens[: max_length]
        start = 1 + len(tokens_0_start)
        end = 1 + len(tokens_0_start) + len(tokens_start_end)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        # pad
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length
        # label
        label_id = example.label
        start_id = np.zeros(max_length)
        if start >= max_length:
            start = 0  # 如果entity被截断了，就使用CLS位代替
        start_id[start: end] = 1

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          start_id=start_id,
                          ))
    return features


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, start_id=None, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_id = start_id

        self.label_id = label_id

class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)


class ABSAProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_test_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_labels(self):
        label_list = ['positive', 'negative', 'neutral']
        return label_list

    def _create_examples(self, lines):
        examples = []
        label_list = self.get_labels()
        label_set = set()
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['token']
            aspects = line['aspects']
            for aspect in aspects:
                text_b = (aspect['from'], aspect['to'])
                label = aspect['polarity']
                label = label_list.index(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



tasks_num_labels = {
    "laptop": 3,
    "restaurants": 3,
}

processors = {
    "laptop": ABSAProcessor,
    "restaurants": ABSAProcessor,
}

output_modes = {
    "laptop": "classification",
    "restaurants": "classification",
}


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            dataset_type,
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
        if dataset_type == 'train':
            examples = processor.get_train_examples(args.data_dir, dataset_type)
        elif dataset_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir, dataset_type)
        else:
            examples = processor.get_test_examples(args.data_dir, dataset_type)
        features = convert_examples_to_features(
            examples, max_length=args.max_seq_length, tokenizer=tokenizer,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_start_ids = torch.tensor([f.start_id for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_ids, all_label_ids)
    return dataset

