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

    def create_feature_model(self, inputs, scope='', is_training=True, **unused_params):
        feature, restore_vars, tvars = hybrid.get_pretrained_model_feats(inputs, scope, is_training)
        return feature, restore_vars, tvars

    def create_aux_model(self, inputs, is_training=True, **unused_params):
        outputs, tvars = hybrid.get_temporal_mean_pooled_feats(inputs, is_training)
        return outputs, tvars

    def create_logits_model(self, inputs, num_classes, is_training=True, scope="logits", reuse=None):
        outputs, tvars = hybrid.get_classifier_logits(inputs, num_classes, is_training=is_training, lscope=scope, reuse=reuse)
        return outputs, tvars
