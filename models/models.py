import tensorflow as tf
slim = tf.contrib.slim
framework = tf.contrib.framework

from NTURGBD import *

class BaseModel(object):

    def create_model(self, unused_inputs, **unused_params):
        raise NotImplementedError()

class NTURGBD_RNN(object):

    def __init__(self):
        self.name = "NTURGBD_RNN"

    def create_model(self, inputs, num_classes, labels, **unused_params):
        model = nturgbd_rnn.SkeletonHRNNet()
        output = model.create_model(inputs, num_classes, labels)

        return output
