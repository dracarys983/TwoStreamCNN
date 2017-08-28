import os
import math

import tensorflow as tf
import numpy as np
from six.moves import xrange

slim = tf.contrib.slim
framework = tf.contrib.framework
rnn = tf.contrib.rnn
layers = tf.contrib.layers

import bnlstm

from tensorflow.python.ops import variable_scope as vs

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

class SkeletonHRNNet(object):

    def __init__(self):

        self._num_layers_spatial = 3

    def create_model(self, inputs, num_classes, labels, is_training=True, **unused_params):
        outputs = {}

        is_training = tf.constant(is_training, dtype=tf.bool)
        with tf.variable_scope('spatial'):
            cells = []
            num_hidden = [256, 256, 256]
            for i in range(self._num_layers_spatial):
                cell = bnlstm.BNLSTMCell(num_hidden[i], training=is_training)
                cell = rnn.DropoutWrapper(cell, input_keep_prob=0.5, output_keep_prob=0.5)
                cells.append(cell)
            spatial = rnn.MultiRNNCell(cells)
            output, new_states = tf.nn.dynamic_rnn(spatial, inputs,
                dtype=tf.float32, sequence_length=length(inputs))

        last = last_relevant(output, length(output))
        fc4 = layers.fully_connected(last, 128, activation_fn=tf.nn.relu)
        fc5 = layers.fully_connected(fc4, 64, activation_fn=tf.nn.relu)
        logits = layers.fully_connected(fc5, num_classes, activation_fn=None)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        outputs['predictions'] = logits
        outputs['loss'] = loss
        outputs['reg_loss'] = l2_loss
        return outputs
