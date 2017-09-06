import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.framework as framework
import tensorflow.contrib.rnn as rnn

from inception_resnet_v2 import *
from vgg import *
from bnlstm import *

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def common_arg_scope(weight_decay=0.00004,
                     batch_norm_decay=0.9997,
                     batch_norm_epsilon=0.001):
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def get_inception_resnet_feats(inputs, is_training=True):
    # Inception Resnet V2 for feature extraction
    scope = vgg_arg_scope()
    with slim.arg_scope(scope):
        _, end_points = vgg_16(inputs)
        features = end_points['vgg_16/conv5/conv5_1']           # 14 x 14 x 512
        restore_vars = framework.get_variables_to_restore(
                exclude=['global_step', 'vgg_16/fc8'])

    #scope = common_arg_scope()
    #with slim.arg_scope(scope):
        # Reduce Dimensionality
    #    with tf.variable_scope('DimReduce'):
    #        features = slim.conv2d(features, 512, 1, scope='Conv2d_1x1')
    tvars = []

    return features, restore_vars, tvars

def get_temporal_mean_pooled_feats(inputs, is_training=True):
    # Temporal Average pooling
    with tf.variable_scope('temporal_mean_pool'):
        features = slim.conv2d(inputs, 512, 1, scope='Conv2d_1x1')
        pooled_features = slim.avg_pool2d(features, (14, 1), stride=1, padding='VALID', scope='AvgPool_14x1')
        features = slim.flatten(pooled_features)
    tvars = framework.get_variables('temporal_mean_pool')

    return features, tvars

def get_bn_lstm_feats(inputs, is_training=True):
    # 2-layer Batch-Normalized LSTM
    with tf.variable_scope('bnlstm'):
        # Reduce feature dimension
        features = slim.conv2d(inputs, 512, 1, scope='Conv2d_1x1')
        num_hidden = 512; num_layers = 2
        cells = []
        for i in range(num_layers):
            cell = BNLSTMCell(num_hidden, training=tf.cast(is_training, tf.bool))
            cell = rnn.DropoutWrapper(cell, output_keep_prob=0.5)
            cells.append(cell)
        lstmlayer = rnn.MultiRNNCell(cells)
        b = tf.shape(features)[0]
        l = features.get_shape().as_list()[2:]
        l = l[0] * l[1]
        features = tf.reshape(features, shape=(b, 17, l))
        output, _ = tf.nn.dynamic_rnn(lstmlayer, features,
                dtype=tf.float32, sequence_length=length(features))
        last = last_relevant(output, 17)

    tvars = framework.get_variables('bnlstm')
    return last, tvars

def get_classifier_logits(inputs, num_classes, lscope='', is_training=True):
    # Primary Classifier
    scope = common_arg_scope()
    with slim.arg_scope(scope):
        with tf.variable_scope(lscope):
            plogits = slim.fully_connected(inputs, 1024, activation_fn=tf.nn.relu, scope='PreLogits')
            dropout = slim.dropout(plogits, 0.5, is_training=is_training, scope='Logits_dropout')
            logits = slim.fully_connected(dropout, num_classes, activation_fn=None, scope='Final_Logits')

    tvars = framework.get_variables(lscope)
    return logits, tvars
