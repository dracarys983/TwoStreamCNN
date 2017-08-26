import tensorflow as tf
slim = tf.contrib.slim
framework = tf.contrib.framework

from inception_resnet_v2 import *
from VGG import *
from NTURGBD import *

class BaseModel(object):

    def create_model(self, unused_inputs, **unused_params):
        raise NotImplementedError()

class NTURGBD(object):

    def __init__(self):
        self.name = "NTURGBD"

    def create_model(self, inputs, num_classes, labels, **unused_params):
        output = {}
        arg_scope = nturgbd_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = nturgbd(inputs, num_classes)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        output['predictions'] = logits
        output['loss'] = loss
        return output

class NTURGBD_RNN(object):

    def __init__(self):
        self.name = "NTURGBD_RNN"

    def create_model(self, inputs, num_classes, labels, **unused_params):
        model = nturgbd_rnn.SkeletonHRNNet()
        output = model.create_model(inputs, num_classes, labels)

        return output

class InceptionResnetV2(BaseModel):

    def __init__(self):
        self.name = "InceptionResnetV2"

    def create_model(self, inputs, num_classes, labels, **unused_params):
        output = {}
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = inception_resnet_v2(inputs, num_classes)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        output['predictions'] = logits
        output['loss'] = loss
        return output

class VGG_16(BaseModel):

    def __init__(self):
        self.name = "VGG_16"

    def create_model(self, inputs, num_classes, labels, var_list=None, **unused_params):
        output = {}
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = vgg_16(inputs, num_classes)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        output['predictions'] = logits
        output['loss'] = loss
        return output

class VGG_19(BaseModel):

    def __init__(self):
        self.name = "VGG_19"

    def create_model(self, inputs, num_classes, labels, **unused_params):
        output = {}
        arg_scope = vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, _ = vgg_19(inputs, num_classes)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        output['predictions'] = logits
        output['loss'] = loss
        return output
