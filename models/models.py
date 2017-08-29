import tensorflow as tf
slim = tf.contrib.slim
framework = tf.contrib.framework

from NTURGBD import *
from HybridModel import *

class NTURGBD_RNN(object):

    def __init__(self):
        self.name = "NTURGBD_RNN"

    def create_model(self, inputs, num_classes, labels, **unused_params):
        model = nturgbd_rnn.SkeletonHRNNet()
        output = model.create_model(inputs, num_classes, labels)

        return output

class Hybrid(object):

    def __init__(self):
        self.name = "HybridModel"

    def create_model(self, inputs, num_classes, labels, is_training=True, **unused_params):
        predictions, aux_loss, loss,
            aux_vars, train_vars, restore_vars = hybrid.hybrid_model(inputs, labels, num_classes, is_training)
        output = {'predictions': predictions, 'aux_loss': aux_loss, 'loss': loss,
                'aux_vars': aux_vars, 'train_vars': train_vars, 'restore_vars': restore_vars}

        return output
