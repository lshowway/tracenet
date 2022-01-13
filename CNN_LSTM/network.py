from __future__ import division, print_function, unicode_literals
import tensorflow as tf


def baseline_model_kimcnn(X, max_sent, filters_num, num_classes):
    pooled_outputs = []
    channel_num = X.shape.as_list()[-1]
    for i, filter_size in enumerate([2, 3, 4, 5]):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, 300, channel_num, filters_num]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[filters_num]), name="b")
            conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, max_sent - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = filters_num * 4
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    h_drop = h_pool_flat
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss = tf.constant(0.0)
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    return tf.zeros([0]), logits, l2_loss


def BiLSTM(X, max_sent, n_hidden, num_classes):
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, activation='relu')
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, activation='relu')

    X = tf.squeeze(X)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=X, dtype=tf.float32)
    outputs = tf.concat(outputs, axis=-1)
    outputs = tf.nn.dropout(outputs, 0.5)
    W = tf.get_variable("w", [max_sent * n_hidden * 2, num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    l2_loss = tf.constant(0.0)
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    outputs = tf.reshape(outputs, [-1, max_sent * n_hidden * 2]) # output[:, -1, :]最后一个unit的输出
    logits = tf.nn.xw_plus_b(outputs, W, b, name="scores")
    return tf.zeros([0]), logits, l2_loss


def LSTM(X, max_sent, n_hidden, num_classes):
    lstmCell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, activation='relu')
    X = tf.squeeze(X)
    outputs, states = tf.nn.dynamic_rnn(cell=lstmCell, inputs=X, dtype=tf.float32)
    outputs = tf.nn.dropout(outputs, 0.5)
    W = tf.get_variable("w", [n_hidden * max_sent, num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    l2_loss = tf.constant(0.0)
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    outputs = tf.reshape(outputs, [-1, n_hidden * max_sent])
    logits = tf.nn.xw_plus_b(outputs, W, b, name="scores")

    return tf.zeros([0]), logits, l2_loss