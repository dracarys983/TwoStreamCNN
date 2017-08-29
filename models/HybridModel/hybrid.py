import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.framework as framework
import tensorflow.contrib.rnn as rnn

from inception_resnet_v2 import *
from bnlstm import *

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def hybrid_model(inputs,
                labels,
                num_classes,
                is_training=True):

    # Inception Resnet V2 for feature extraction
    scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(scope):
        _, end_points = inception_resnet_v2(inputs)
        features = end_points['Conv2d_7b_1x1']          # 8 x 8 x 1536
        restore_vars = framework.get_variables_to_restore(
                exclude=['global_step', 'InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits'])

    # Auxiliary Classifier
    with tf.variable_scope('AuxLogits'):
        pooled_features = slim.max_pool2d(features, (8, 1), stride=1, padding='VALID', scope='MaxPool_8x1')
        pooled_features = slim.flatten(pooled_features)
        fc1_aux = slim.fully_connected(pooled_features, 512, activation_fn=tf.nn.relu, scope='AuxLogits_512')
        dropout_aux = slim.dropout(fc1_aux, 0.5, is_training=is_training, scope='AuxLogits_dropout')
        aux = slim.fully_connected(dropout_aux, num_classes, activation_fn=None, scope='Final_AuxLogits')

    # 2-layer Batch-Normalized LSTM
    with tf.variable_scope('bnlstm'):
        num_hidden = 512; num_layers = 2
        cells = []
        for i in range(num_layers):
            cell = BNLSTMCell(num_hidden, training=tf.cast(is_training, tf.bool))
            cell = rnn.DropoutWrapper(cell, output_keep_prob=0.5)
            cells.append(cell)
        lstmlayer = rnn.MultiRNNCell(cells)
        features = tf.reshape(features, (36, 8, -1))
        output, _ = tf.nn.dynamic_rnn(lstmlayer, features,
                dtype=tf.float32)

    # Primary Classifier
    with tf.variable_scope('Logits'):
        last = last_relevant(output, 8)
        plogits = slim.fully_connected(last, 256, activation_fn=tf.nn.relu, scope='PreLogits')
        dropout = slim.dropout(plogits, 0.5, is_training=is_training, scope='Logits_dropout')
        logits = slim.fully_connected(dropout, num_classes, activation_fn=None, scope='Final_Logits')

    # Losses for classification
    aux_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=aux,
                        name='aux_xentropy'), name='aux_xentropy_mean')
    logit_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                        name='xentropy'), name='xentropy_mean')
    loss = ((0.2 * aux_loss) + (0.8 * logit_loss))

    aux_vars = framework.get_variables('AuxLogits')
    train_vars = framework.get_variables('bnlstm')
    train_vars.extend(framework.get_variables('Logits'))

    predictions = ((0.5 * aux) + (0.5 * logits))

    return predictions, aux_loss, loss, aux_vars, train_vars, restore_vars
