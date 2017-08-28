import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

_TOP_PREDICTIONS_IN_OUTPUT = 20

class ModelExporter(object):

    def __init__(self, model, reader):
        self.model = model
        self.reader = reader

        with tf.Graph().as_default() as graph:
            self.inputs, self.outputs = self.build_inputs_and_outputs()
            self.graph = graph
            self.saver = tf.train.Saver(tf.trainable_variables(), sharded=True)

    def export_model(self, model_dir, global_step_val, last_checkpoint):
        """ Exports the model so that it can used for batch predictions """

        with self.graph.as_default():
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                self.saver.restore(session, last_checkpoint)

                signature = signature_def_utils.build_signature_def(
                    inputs=self.inputs,
                    outputs=self.outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME)

                signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                signature}

                model_builder = saved_model_builder.SavedModelBuilder(model_dir)
                model_builder.add_meta_graph_and_variables(session,
                    tags=[tag_constants.SERVING],
                    signature_def_map=signature_map,
                    clear_devices=True)
                model_builder.save()

    def build_inputs_and_outputs(self):
        split_num = self.reader.present_split

        top_indices_output, top_predictions_output = (
              self.build_prediction_graph(split_num))

        inputs = {"example_bytes":
                  saved_model_utils.build_tensor_info(tf.constant(split_num))}

        outputs = {
            "class_indexes": saved_model_utils.build_tensor_info(top_indices_output),
            "predictions": saved_model_utils.build_tensor_info(top_predictions_output)}

        return inputs, outputs

    def build_prediction_graph(self, split):
        files, labels = self.reader._read_filelist(split=split)

        files = ops.convert_to_tensor(files, dtypes.string)
        labels = ops.convert_to_tensor(labels, dtypes.int64)

        input_queue = tf.train.slice_input_producer(
                            [files, labels],
                            num_epochs = self.reader.num_epochs,
                            shuffle = True)
        image, label = self.reader._read_samples(input_queue)
        image = tf.image.resize_images(image, (299, 299))

        image_loader, label_loader = tf.train.shuffle_batch(
                [image, label],
                batch_size = self.reader.batch_size,
                capacity = 5 * self.reader.batch_size,
                min_after_dequeue = self.reader.batch_size)

        #with tf.variable_scope("tower"):
        result = self.model.create_model(
          image_loader,
          self.reader.num_classes,
          label_loader,
          is_training=False)

        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

        predictions = result["predictions"]

        top_predictions, top_indices = tf.nn.top_k(predictions,
          _TOP_PREDICTIONS_IN_OUTPUT)
        return top_indices, top_predictions
