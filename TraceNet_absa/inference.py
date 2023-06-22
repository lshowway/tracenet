import os
import glob

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    RobertaConfig,
    AlbertConfig,
    XLNetConfig,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    BertTokenizer,
    RobertaTokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_compute_metrics as compute_metrics
from data_utils import output_modes, processors

from tracenet import XLNetTraceNetForSequenceClassification, RobertTraceNetForSequenceClassification

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in
     (BertConfig, RobertaConfig, AlbertConfig, XLNetConfig)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlnet_tracenet": (XLNetConfig, XLNetTraceNetForSequenceClassification, XLNetTokenizer),
    "roberta_tracenet": (RobertaConfig, RobertTraceNetForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer, prefix="", evaluate='dev', against_file=None):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=evaluate, against_file=against_file)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        if evaluate == 'dev':
            logger.info("***** Running evaluation {} *****".format(prefix))
        else:
            logger.info("***** Running test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert"] else None)  # XLM and DistilBERT don't use segment_ids
                if 'tracenet' in args.model_type:
                    this_batch = inputs["input_ids"].shape[0]
                    inputs["item_weights"] = torch.ones(this_batch,
                                                        args.max_seq_length, 1, dtype=torch.float,
                                                        device=args.device) / args.max_seq_length
                outputs = model(**inputs)
                if 'tracenet' in args.model_type:
                    logits, _, _, all_item_weights = outputs
                else:
                    _, logits = outputs[:2]
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` .")
        result = compute_metrics('sst-2', preds, out_label_ids)
        results.update(result)
    return results['acc']


def load_and_cache_examples(args, task, tokenizer, evaluate, against_file):
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
                examples = (processor.get_against_test_examples(args.data_dir, against_file))
                features = convert_examples_to_features(
                    examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
                )
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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    # hubo net
    parser.add_argument("--output_hidden_states", action="store_true",
                        help="whether output the hidden state of each layer")
    parser.add_argument("--output_item_weights", action="store_true",
                        help="whether output the item weights of each layer")
    parser.add_argument("--num_hubo_layers", type=int, default=3, help="the number of layers of TraceNet")
    parser.add_argument("--method", default='mean', type=str)
    parser.add_argument("--seq_select_prob", default=0.0, type=float,
                        help="the probability to select one sentence to mask its words")
    parser.add_argument("--proactive_masking", action="store_true", help="Whether to use proactive masking.")
    parser.add_argument("--write_item_weights", action="store_true", help="Whether to write item weights.")
    parser.add_argument("--against", action="store_true", help="Whether to use proactive masking.")
    parser.add_argument("--dropout_prob", default=0.1, required=True, type=float)
    parser.add_argument("--output_feature", default=768, required=True, type=int)

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()  # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # tokenizer = tokenizer_class.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     do_lower_case=args.do_lower_case,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.output_hidden_states = args.output_hidden_states
    config.output_item_weights = args.output_item_weights
    config.num_hubo_layers = args.num_hubo_layers
    config.method = args.method
    config.max_seq_length = args.max_seq_length
    config.seq_select_prob = args.seq_select_prob
    config.dropout_prob = args.dropout_prob
    config.output_feature = args.output_feature

    # model = model_class.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )

    # model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_eval:
        logger.info("===========> inference ... ")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        acc_list = []
        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            if args.task_name == 'sst-5':
                file_list = ['sst_test.against.%i' % (i + 1) for i in range(10)]
            elif args.task_name == 'yelp-5':
                file_list = ['Yelp5.sample.test.against.%i' % (i + 1) for i in range(10)]
            elif args.task_name == 'mr-2':
                file_list = ['mr_test.against.%i' % (i + 1) for i in range(10)]
            elif args.task_name == 'sst-2':
                file_list = ['sst2_test.against.%i' % (i + 1) for i in range(10)]
            for against_file in file_list:
                acc = evaluate(args, model, tokenizer, evaluate='test', against_file=against_file)
                acc_list.append(acc)
        print('\t'.join([str(round(x, 4) * 100) for x in acc_list]), flush=True)


if __name__ == "__main__":
    main()
