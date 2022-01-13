from __future__ import division, print_function, unicode_literals
import argparse
import h5py
import numpy as np
import tensorflow as tf
from network import baseline_model_kimcnn, BiLSTM, LSTM
from sklearn.utils import shuffle

tf.reset_default_graph()
np.random.seed(1)
tf.set_random_seed(1)


parser = argparse.ArgumentParser()

parser.add_argument('--embedding_type', type=str, default='rand',
                    help='Options: rand (randomly initialized word embeddings), '
                         'static (pre-trained embeddings from word2vec, static during learning), '
                         'nonstatic (pre-trained embeddings, tuned during learning), '
                         'multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--dataset', type=str, default='../dataset/SST_5/sst5.hdf5',
                    help='Options: sst5.hdf5, yelp5.hdf5')

parser.add_argument('--loss_type', type=str, default='cross_entropy',
                    help='margin_loss, spread_loss, cross_entropy')

parser.add_argument('--model_type', type=str, default='CNN',
                    help='KIMCNN, LSTM, BiLSTM')

parser.add_argument('--n_hidden', type=int, default=100, help='If data has test, we use it. Otherwise, we use CV on folds')
parser.add_argument('--has_test', type=int, default=1, help='If data has test, we use it. Otherwise, we use CV on folds')
parser.add_argument('--has_dev', type=int, default=1, help='If data has dev, we use it, otherwise we split from train')

parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=5, help='Batch size for training')
parser.add_argument('--max_sent', type=int, default=49, help='Batch size for training')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')#CNN 0.0005
parser.add_argument('--margin', type=float, default=0.2, help='the initial value for spread loss')
parser.add_argument('--l2_reg_lambda', type=float, default=0.01, help='L2 regularization lambda (default: 0.0)')

import json
args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_data(dataset):
    f = h5py.File(dataset, 'r')
    print('loading data...', flush=True)
    print(dataset)
    print("Keys: %s" % f.keys(), flush=True)

    w2v = list(f['w2v'])

    train = list(f['train'])
    train_label = list(f['train_label'])

    dev = list(f['dev'])
    dev_label = list(f['dev_label'])

    test = list(f['test'])
    test_label = list(f['test_label'])
    
    for i, v in enumerate(train):
        if np.sum(v) == 0:        
            del(train[i])     
            del(train_label[i])

    for i, v in enumerate(dev):
        if np.sum(v) == 0:
            del(dev[i])
            del(dev_label[i])

    for i, v in enumerate(test):
        if np.sum(v) == 0:
            del(test[i])
            del(test_label[i])
    
    return train, train_label, test, test_label, dev, dev_label, w2v


class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, dataset,label, batch_size,input_size, is_shuffle=True):
      self._dataset = dataset
      self._label = label
      self._batch_size = batch_size    
      self._cursor = 0      
      self._input_size = input_size      
      
      if is_shuffle:
          index = np.arange(len(self._dataset))
          np.random.shuffle(index)
          self._dataset = np.array(self._dataset)[index]
          self._label = np.array(self._label)[index]
      else:
          self._dataset = np.array(self._dataset)
          self._label = np.array(self._label)
    def next(self):
      if self._cursor + self._batch_size > len(self._dataset):
          self._cursor = 0
      """Generate a single batch from the current cursor position in the data."""      
      batch_x = self._dataset[self._cursor : self._cursor + self._batch_size,:]
      batch_y = self._label[self._cursor : self._cursor + self._batch_size]
      self._cursor += self._batch_size
      return batch_x, batch_y


if __name__ == "__main__":
    train, train_label, test, test_label, dev, dev_label, w2v = load_data(args.dataset)
    args.vocab_size = len(w2v)
    args.vec_size = w2v[0].shape[0]
    print('max sent: ', args.max_sent, flush=True)
    print('vocab size: ', args.vocab_size, flush=True)
    print('vec size: ', args.vec_size, flush=True)
    print('num_classes: ', args.num_classes, flush=True)
    train, train_label = shuffle(train, train_label, random_state=1)
    with tf.device('/gpu:0'):
        global_step = tf.train.get_or_create_global_step()
    X = tf.placeholder(tf.int32, [args.batch_size, args.max_sent], name="input_x")
    y = tf.placeholder(tf.int64, [args.batch_size, args.num_classes], name="input_y")
    is_training = tf.placeholder_with_default(False, shape=())
    learning_rate = tf.placeholder(dtype='float32')
    margin = tf.placeholder(shape=(), dtype='float32')

    l2_loss = tf.constant(0.0)

    w2v = np.array(w2v, dtype=np.float32)
    if args.embedding_type == 'rand':
        W1 = tf.Variable(tf.random_uniform([args.vocab_size, args.vec_size], -0.25, 0.25, seed=1), name="Wemb")
        X_embedding = tf.nn.embedding_lookup(W1, X)
        X_embedding = X_embedding[..., tf.newaxis]
    if args.embedding_type == 'static':
        W1 = tf.Variable(w2v, trainable=False)
        X_embedding = tf.nn.embedding_lookup(W1, X)
        X_embedding = X_embedding[..., tf.newaxis]
    if args.embedding_type == 'nonstatic':
        W1 = tf.Variable(w2v, trainable=True)
        X_embedding = tf.nn.embedding_lookup(W1, X)
        X_embedding = X_embedding[..., tf.newaxis]
    if args.embedding_type == 'multichannel':
        W1 = tf.Variable(w2v, trainable=True)
        W2 = tf.Variable(w2v, trainable=True)
        X_1 = tf.nn.embedding_lookup(W1, X)
        X_2 = tf.nn.embedding_lookup(W2, X)
        X_1 = X_1[..., tf.newaxis]
        X_2 = X_2[..., tf.newaxis]
        X_embedding = tf.concat([X_1, X_2], axis=-1)

    tf.logging.info("input dimension:{}".format(X_embedding.get_shape()))
    activations = None
    if args.model_type == 'KIMCNN':
        poses, activations, l2_loss = baseline_model_kimcnn(X_embedding, args.max_sent, args.n_hidden, args.num_classes)
    elif args.model_type == 'BiLSTM':
        poses, activations, l2_loss = BiLSTM(X_embedding, args.max_sent, args.n_hidden, args.num_classes)
    elif args.model_type == 'LSTM':
        poses, activations, l2_loss = LSTM(X_embedding, args.max_sent, args.n_hidden, args.num_classes)

    if args.loss_type == 'cross_entropy':
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=activations, labels=y)
        loss = tf.reduce_mean(loss) + args.l2_reg_lambda * l2_loss

    y_pred = tf.argmax(activations, axis=1, name="y_proba")
    correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name="training_op")
    gradients, variables = zip(*optimizer.compute_gradients(loss))

    grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                  for g in gradients if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]
    with tf.control_dependencies(grad_check):
        training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    sess = tf.InteractiveSession()
    n_iterations_per_epoch = len(train) // args.batch_size
    n_iterations_test = len(test) // args.batch_size
    n_iterations_dev = len(dev) // args.batch_size

    train = BatchGenerator(train, train_label, args.batch_size, 0, is_shuffle=False)
    dev = BatchGenerator(dev, dev_label, args.batch_size, 0, is_shuffle=False)
    test = BatchGenerator(test, test_label, args.batch_size, 0, is_shuffle=False)

    best_model = None
    best_epoch = 0
    best_acc_val = 0.

    init = tf.global_variables_initializer()
    sess.run(init)

    lr = args.learning_rate
    m = args.margin
    best_dev_acc = 0.0
    for epoch in range(args.num_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = train.next()
            y_batch = to_categorical(y_batch, args.num_classes)
            _, loss_train, probs, capsule_pose = sess.run(
                [training_op, loss, activations, poses],
                feed_dict={X: X_batch[:, :args.max_sent],
                           y: y_batch,
                           is_training: True,
                           learning_rate: lr,
                           margin: m})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch,
                loss_train),
                end="", flush=True)
        # evaluate
        loss_vals, acc_vals = [], []
        for iteration in range(1, n_iterations_dev + 1):
            X_batch, y_batch = dev.next()
            y_batch = to_categorical(y_batch, args.num_classes)
            loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch[:, :args.max_sent],
                           y: y_batch,
                           is_training: False,
                           margin: m})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val, acc_val = np.mean(loss_vals), np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.1f}%  Loss: {:.4f}".format(
            epoch + 1, acc_val * 100, loss_val), flush=True)
        if acc_val > best_dev_acc:
            best_dev_acc = acc_val
            loss_tests, acc_tests = [], []
            for iteration in range(1, n_iterations_test + 1):
                X_batch, y_batch = test.next()
                y_batch = to_categorical(y_batch, args.num_classes)
                _, acc_test = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch[:, :args.max_sent],
                               y: y_batch,
                               is_training: False,
                               margin: m})
                acc_tests.append(acc_test)
            acc_test = np.mean(acc_tests)
            print("\r=====> Epoch: {}  test accuracy: {:.1f}%".format(epoch + 1, acc_test * 100), flush=True)

        if args.model_type == 'KIMCNN':
            lr = max(1e-6, lr * 0.8)

