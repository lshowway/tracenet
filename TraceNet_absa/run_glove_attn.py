import torch, math, logging, argparse
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import AdamW
from transformers import glue_compute_metrics as compute_metrics
from torch import nn

from data_utils import get_vocab, get_embedding_table, get_batch_data
from tracenet import Discriminator
# from attentions import AdditiveAttention
# from attentions import AdditiveAttentionLayer as Attn
from attentions import AdditiveAttentionModel as Attn
# from attentions import ScaledDotProductAttention as Attn
# from tracenet import AdditiveAttention


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"relu": torch.nn.functional.relu, "gelu": _gelu_python}
XLNetLayerNorm = nn.LayerNorm


class GloveForClassification(nn.Module):
    def __init__(self, config):
        super(GloveForClassification, self).__init__()
        self.num_labels = config.num_labels
        self.word_embedding = nn.Embedding(num_embeddings=config.num_words, embedding_dim=300)
        self.proj = nn.Linear(300, config.output_feature)
        self.attn = Attn(config)  # config.output_feature
        self.discriminator = Discriminator(config=args)

        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, input_ids=None, labels=None):
        input_embddings = self.word_embedding(input_ids)  # batch*len*300

        # =====
        input_embddings = self.proj(input_embddings)  # 300d -> output_feature
        # hidden_states, _ = self.attn(query=input_embddings, key=input_embddingsss, value=input_embddings)
        hidden_states = self.attn(query=input_embddings, values=input_embddings)
        # hidden_states = self.attn(hidden_states=input_embddings)  # self-attn
        # hidden_states = input_embddings
        hidden_states = torch.mean(hidden_states, dim=1)
        # =====

        logits = self.discriminator(hidden_states)
        final_outputs = ()
        final_outputs = final_outputs + (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            discriminator_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            final_outputs = final_outputs + (discriminator_loss,)

        return final_outputs  # logits, discriminator_loss


def evaluate(args, model, eval_dataloader):
    # Eval!
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        input_ids = batch['text'].to(args.device)
        labels = batch['label'].to(args.device)
        with torch.no_grad():
            # this_batch = input_ids.shape[0]
            # item_weights = torch.ones(this_batch, args.max_seq_length, 1, dtype=torch.float,
            #                                     device=args.device) / args.max_seq_length
            outputs = model(input_ids=input_ids, labels=labels)
            logits, _ = outputs  # logits, loss, loss, item_weights
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    acc = compute_metrics('sst-2', preds, out_label_ids)['acc']
    return acc


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


def glove_as_input(args):
    if args.task == 'yelp-5':
        all_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.all"
        glove_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.glove.to.tranceNet"
        train_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.train"
        dev_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.dev"
        test_file = "../dataset/Yelp_5/yelp_review_full_csv/Yelp5.sample.test"
    elif args.task == 'sst-5':
        all_file = '../dataset/SST_5/sst_all.txt'
        glove_file = '../dataset/SST_5/SST5.glove.to.TraceNet'
        train_file = "../dataset/SST_5/sst_train.txt"
        dev_file = "../dataset/SST_5/sst_dev.txt"
        test_file = "../dataset/SST_5/sst_test.txt"
    elif args.task == 'mr-2':
        all_file = '../dataset/MR_SentiLARE/MR-2.all'
        glove_file = '../dataset/MR_SentiLARE/MR2.glove.to.TraceNet'
        train_file = "../dataset/MR_SentiLARE/MR-2.train"
        dev_file = "../dataset/MR_SentiLARE/MR-2.dev"
        test_file = "../dataset/MR_SentiLARE/MR-2.test"
    elif args.task == 'sst-2':
        all_file = '../dataset/SST_2/sst.binary.all'
        glove_file = '../dataset/SST_2/SST2.glove.to.TraceNet'
        train_file = "../dataset/SST_2/sst.binary.train"
        dev_file = "../dataset/SST_2/sst.binary.dev"
        test_file = "../dataset/SST_2/sst.binary.test"
    vocab_dic = get_vocab(path=all_file)
    embeddings = get_embedding_table(word_to_idx=vocab_dic, glove_file=glove_file)
    # train_loader = get_batch_data(args, train_file, vocab_dic, args.per_gpu_train_batch_size)
    valid_loader = get_batch_data(args, dev_file, vocab_dic, args.per_gpu_eval_batch_size)
    test_loader = get_batch_data(args, test_file, vocab_dic, args.per_gpu_eval_batch_size)

    args.num_labels = 5
    args.num_words = len(vocab_dic)
    set_seed(args)
    args.hidden_state = 300
    model = GloveForClassification(config=args)
    model.to(args.device)
    model.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
    model.word_embedding.weight.requires_grad = True
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(params=params, lr=args.learning_rate, eps=args.adam_epsilon)

    global_step = 0
    best_dev_acc = 0.0
    preds = None
    out_label_ids = None
    for epoch in range(args.num_train_epochs):
        train_loader = get_batch_data(args, train_file, vocab_dic, args.per_gpu_train_batch_size)
        train_loss = 0.0
        # train
        for batch_iter, train_batch in enumerate(train_loader):
            model.train()
            input_ids = train_batch['text'].to(args.device)
            label = train_batch['label'].to(args.device)
            logits, loss = model(input_ids, label)
            train_loss += loss.item()
            if preds is None:
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                out_label_ids = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, np.argmax(logits.detach().cpu().numpy(), axis=1), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1
        train_acc = compute_metrics('sst-2', preds, out_label_ids)['acc']
        # evaluate
        dev_acc = evaluate(args, model, valid_loader)
        test_acc = evaluate(args, model, test_loader)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            logging.warning("===> epoch: %s, train loss: %s, train acc: %s,  dev acc: %s, test acc: %s ***" % (
            epoch, train_loss, train_acc, dev_acc, test_acc))
        else:
            logging.warning("===> epoch: %s, train loss: %s, train acc: %s,  dev acc: %s, test acc: %s" % (
            epoch, train_loss, train_acc, dev_acc, test_acc))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--max_seq_length', default=49, type=int)
    parser.add_argument('--output_feature', default=32, type=int)
    parser.add_argument('--dropout_prob', default=0.2, type=float)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument("--num_hubo_layers", type=int, default=1, help="the number of layers of TraceNet")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="the number epochs for training")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--task", type=str, default='sst-5', help="random seed for initialization")
    parser.add_argument("--attn_type", type=str, default='additive', help="['additive', 'scaleDP]")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print('===============device： %s' % args.device)
    return args


if __name__ == "__main__":
    args = get_args()
    glove_as_input(args=args)
