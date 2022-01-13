import argparse
import logging
import os

from tensorboardX import SummaryWriter

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from model import SSTModel
from data_embedding_utils import get_batch_data, get_vocab, compute_embed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')


def train(args):
    if args.task == 'sst-5':
        vocab_dic = get_vocab(path="../dataset/SST_5/sst_all.txt")
        embeddings = compute_embed(word_to_idx=vocab_dic, w2v_file='../dataset/SST_5/SST5.glove.to.TraceNet')
        train_loader = get_batch_data(args, "../dataset/SST_5/sst_train.txt", vocab_dic, args.batch_size)
        valid_loader = get_batch_data(args, "../dataset/SST_5/sst_dev.txt", vocab_dic, args.batch_size)
        test_loader = get_batch_data(args, "../dataset/SST_5/sst_test.txt", vocab_dic, args.batch_size)
        num_classes = 5
        num_words = len(vocab_dic)
        logging.info(f'Number of classes: {num_classes}')
    elif args.task == 'yelp-5':
        vocab_dic = get_vocab(path="../dataset/Yelp/yelp_review_full_csv/Yelp5.sample.all")
        embeddings = compute_embed(word_to_idx=vocab_dic, w2v_file='../dataset/Yelp/yelp_review_full_csv/Yelp5.glove.to.tranceNet')
        train_loader = get_batch_data(args, "../../dataset/Yelp/yelp_review_full_csv/Yelp5.sample.train", vocab_dic, args.batch_size)
        valid_loader = get_batch_data(args, "../../dataset/Yelp/yelp_review_full_csv/Yelp5.sample.dev", vocab_dic, args.batch_size)
        test_loader = get_batch_data(args, "../../dataset/Yelp/yelp_review_full_csv/Yelp5.sample.test", vocab_dic, args.batch_size)
        num_classes = 5
        num_words = len(vocab_dic)
        logging.info(f'Number of classes: {num_classes}')

    model = SSTModel(num_classes=num_classes, num_words=num_words,
                     word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                     clf_hidden_dim=args.clf_hidden_dim,
                     clf_num_layers=args.clf_num_layers,
                     use_leaf_rnn=args.leaf_rnn,
                     bidirectional=args.bidirectional,
                     intra_attention=args.intra_attention,
                     use_batchnorm=args.batchnorm,
                     dropout_prob=args.dropout)
    if args.pretrained:
        logging.info('=====> initialize with glove.840B.300d')
        model.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
    logging.info('=====> Will  update word embeddings')
    model.word_embedding.weight.requires_grad = True
    logging.info(f'Using device {args.device}')
    model.to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    elif args.optimizer == 'adagrad':
        optimizer_class = optim.Adagrad
    elif args.optimizer == 'adadelta':
        optimizer_class = optim.Adadelta
    optimizer = optimizer_class(params=params, weight_decay=args.l2reg)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='max', factor=0.5,
        patience=20 * args.halve_lr_every, verbose=True)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'train'))
    valid_summary_writer = SummaryWriter(
        log_dir=os.path.join(args.save_dir, 'log', 'valid'))

    def run_iter(args, batch, is_training):
        model.train(is_training)
        words, length = batch['text'].to(args.device), batch['length'].to(args.device)
        label = batch['label'].to(args.device)
        logits = model(words=words, length=length)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=params, max_norm=5)
            optimizer.step()
        return loss, accuracy

    def add_scalar_summary(summary_writer, name, value, step):
        if torch.is_tensor(value):
            value = value.item()
        summary_writer.add_scalar(tag=name, scalar_value=value,
                                  global_step=step)

    best_vaild_accuacy = 0
    iter_count = 0
    for epoch in range(args.max_epoch):
        ### train
        tr_loss_sum = tr_accuracy_sum = 0
        for batch_iter, train_batch in enumerate(train_loader):
            train_loss, train_accuracy = run_iter(args, batch=train_batch, is_training=True)
            tr_loss_sum += train_loss.item()
            tr_accuracy_sum += train_accuracy.item()
            iter_count += 1
            add_scalar_summary(summary_writer=train_summary_writer,
                name='loss', value=train_loss, step=iter_count)
            add_scalar_summary(summary_writer=train_summary_writer,
                name='accuracy', value=train_accuracy, step=iter_count)
        train_loss = tr_loss_sum / iter_count
        train_accuracy = tr_accuracy_sum / iter_count

        ### dev
        valid_loss_sum = valid_accuracy_sum = 0
        num_valid_batches = len(valid_loader)
        for valid_batch in valid_loader:
            valid_loss, valid_accuracy = run_iter(args, batch=valid_batch, is_training=False)
            valid_loss_sum += valid_loss.item()
            valid_accuracy_sum += valid_accuracy.item()
        valid_loss = valid_loss_sum / num_valid_batches
        valid_accuracy = valid_accuracy_sum / num_valid_batches
        add_scalar_summary(summary_writer=valid_summary_writer,
            name='loss', value=valid_loss, step=iter_count)
        add_scalar_summary(summary_writer=valid_summary_writer,
            name='accuracy', value=valid_accuracy, step=iter_count)
        scheduler.step(valid_accuracy)
        logging.info(f'Epoch {epoch:.2f}:, ' f'train loss = {train_loss:.4f}, ' f'train accuracy = {train_accuracy:.4f} '
                     f'valid loss = {valid_loss:.4f}, ' f'valid accuracy = {valid_accuracy:.4f}')

        ### test
        if not test_loader is None:
            if valid_accuracy > best_vaild_accuacy:
                best_vaild_accuacy = valid_accuracy
                test_loss_sum = test_accuracy_sum = 0
                num_test_batches = len(test_loader)
                for test_batch in test_loader:
                    test_loss, test_accuracy = run_iter(args, batch=test_batch, is_training=False)
                    test_loss_sum += test_loss.item()
                    test_accuracy_sum += test_accuracy.item()
                test_loss = test_loss_sum / num_test_batches
                test_accuracy = test_accuracy_sum / num_test_batches
                logging.info(f' ======> Epoch {epoch:.2f}:, '  f'valid loss = {test_loss:.4f}, ' f'valid accuracy = {test_accuracy:.4f}')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--word-dim', required=True, type=int)
    parser.add_argument('--hidden-dim', required=True, type=int)
    parser.add_argument('--clf-hidden-dim', required=True, type=int)
    parser.add_argument('--clf-num-layers', required=True, type=int)
    parser.add_argument('--leaf-rnn', default=False, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=False, action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--l2reg', default=0.0, type=float)
    parser.add_argument('--pretrained', required=True, default=False)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--max-epoch', required=True, type=int)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--omit-prob', default=0.0, type=float)
    parser.add_argument('--optimizer', default='adadelta')
    parser.add_argument('--task', default='sst-5')
    parser.add_argument('--fine-grained', default=True, action='store_true')
    parser.add_argument('--halve-lr-every', default=2, type=int)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--max_seq_length', default=60, type=int)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

