import time
import numpy as np

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from utils import *

import data, models

FLAGS = flags.FLAGS

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_evaluation_tensors(reader,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files, labels = reader._read_filelist(split=reader.present_split, train=False)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    files = ops.convert_to_tensor(files, dtypes.string)
    labels = ops.convert_to_tensor(labels, dtypes.int64)

    input_queue = tf.train.slice_input_producer(
                        [files, labels],
                        num_epochs = 1,
                        shuffle = False)
    image, label = reader._read_samples(input_queue)

    test_image_loader, test_label_loader = tf.train.batch(
        [image, label],
        batch_size = batch_size,
        capacity = 5 * batch_size)
    return test_image_loader, test_label_loader

def build_graph(reader,
                model,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """
  global_step = tf.Variable(0, trainable=False, name="global_step")
  images_loader, labels_loader = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
        reader,
        batch_size=batch_size,
        num_readers=num_readers)
  images_batch = tf.placeholder(tf.float32, (None, 224, 224, 3))
  labels_batch = tf.placeholder(tf.int64, (None,))
  # (224, 224, 3) -> (14, 14, 512)
  feature_0, restore_vars_0, train_v0 = model.create_feature_model(
        images_batch, scope="rgb", is_training=False)
  # (224, 224, 3) -> (14, 14, 512)
  feature_1, restore_vars_1, train_v1 = model.create_feature_model(
        images_batch, scope="rgbdiff", is_training=False)
  # (14, 14, 512) -> (7168,)
  aux_feat_batch = tf.placeholder(tf.float32, (None, 14, 14, 512))
  aux_output, train_v2 = model.create_aux_model(
      aux_feat_batch)
  # (21504,) -> (60,)
  aux_fc_batch_0 = tf.placeholder(tf.float32, (None, 21504))
  logits_aux_0, train_v3 = model.create_logits_model(
      aux_fc_batch_0, 60, is_training=False, scope="auxlogs", reuse=None)
  # (21504,) -> (60,)
  aux_fc_batch_1 = tf.placeholder(tf.float32, (None, 21504))
  logits_aux_1, train_v4 = model.create_logits_model(
      aux_fc_batch_1, 60, is_training=False, scope="auxlogs", reuse=True)
  # (21504,) -> (60,)
  aux_fc_batch_2 = tf.placeholder(tf.float32, (None, 21504))
  logits_aux_2, train_v5 = model.create_logits_model(
      aux_fc_batch_2, 60, is_training=False, scope="auxlogs", reuse=True)
  # (21504,) -> (60,)
  aux_fc_batch_3 = tf.placeholder(tf.float32, (None, 21504))
  logits_aux_3, train_v6 = model.create_logits_model(
      aux_fc_batch_3, 60, is_training=False, scope="auxlogs", reuse=True)

  loss_0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_0, labels=labels_batch))
  loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_1, labels=labels_batch))
  loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_2, labels=labels_batch))
  loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_3, labels=labels_batch))

  loss = loss_0 + loss_1 + loss_2 + loss_3
  predictions = ( logits_aux_0 + logits_aux_1 + logits_aux_2 + logits_aux_3 )

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", loss)
  tf.add_to_collection("feature_0", feature_0)
  tf.add_to_collection("feature_1", feature_1)
  tf.add_to_collection("aux_feat_batch", aux_feat_batch)
  tf.add_to_collection("aux_output", aux_output)
  tf.add_to_collection("aux_fc_batch_0", aux_fc_batch_0)
  tf.add_to_collection("logits_aux_0", logits_aux_0)
  tf.add_to_collection("aux_fc_batch_1", aux_fc_batch_1)
  tf.add_to_collection("logits_aux_1", logits_aux_1)
  tf.add_to_collection("aux_fc_batch_2", aux_fc_batch_2)
  tf.add_to_collection("logits_aux_2", logits_aux_2)
  tf.add_to_collection("aux_fc_batch_3", aux_fc_batch_3)
  tf.add_to_collection("logits_aux_3", logits_aux_3)
  tf.add_to_collection("input_batch", images_batch)
  tf.add_to_collection("labels", labels_batch)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("images_loader", images_loader)
  tf.add_to_collection("labels_loader", labels_loader)

  return restore_vars_0.extend(restore_vars_1)

def evaluation_loop(predictions, labels, loss,
              inputs, aux_feat_batch, aux_output, aux_fc_batch_0, logits_aux_0,
              aux_fc_batch_1, logits_aux_1, aux_fc_batch_2, logits_aux_2,
              aux_fc_batch_3, logits_aux_3, inputs_loader, feature_0, feature_1,
              labels_loader, saver, summary_writer, train_dir, evl_metrics, last_global_step_val):

  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session() as sess:
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint:
      logging.info("Loading checkpoint for eval: " + latest_checkpoint)
      # Restores from checkpoint
      saver.restore(sess, latest_checkpoint)

      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/train_dir/model.ckpt-0, extract global_step from it.
      global_step_val = latest_checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [predictions, labels, loss]
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      while not coord.should_stop():
        batch_start_time = time.time()

        input_batch, label_batch = sess.run([inputs_loader, labels_loader])

        # (batch_size, 12, 224, 224, 3)
        input_batch = np.transpose(input_batch, [1, 0, 2, 3, 4])

        # list of (batch_size, 224,  224, 3) of size 6
        tw_inputs = np.split(input_batch, 12)
        tw_inputs = [np.reshape(x, [-1, 224, 224, 3]) for x in tw_inputs]
        s0_inputs = tw_inputs[:6]
        s1_inputs = tw_inputs[6:]

        # [(224, 224, 3), ..] -> [(14, 14, 512), ..]
        features_0 = []
        for inp in s0_inputs:
            feat_vec = sess.run(feature_0, feed_dict={inputs: inp})
            features_0.append(feat_vec)

        # [(224, 224, 3), ..] -> [(14, 14, 512), ..]
        features_1 = []
        for inp in s1_inputs:
            feat_vec = sess.run(feature_1, feed_dict={inputs: inp})
            features_1.append(feat_vec)

        # [(14, 14, 512), ..] -> [(7168,), ..]
        feats_for_aux = []
        for feat in features_0:
            out = sess.run(aux_output, feed_dict={aux_feat_batch: feat})
            feats_for_aux.append(out)

        # [(7168,), ..] -> [(21504,), (21504,)] (RGB stream)
        aux_fcs_0 = [np.concatenate([feats_for_aux[i], feats_for_aux[i+2],
            feats_for_aux[i+4]], axis=1) for i in range(2)]

        # [(14, 14, 512), ..] -> [(7168,), ..]
        feats_for_aux = []
        for feat in features_1:
            out = sess.run(aux_output, feed_dict={aux_feat_batch: feat})
            feats_for_aux.append(out)

        # [(7168,), ..] -> [(21504,), (21504,)] (RGB difference stream)
        aux_fcs_1 = [np.concatenate([feats_for_aux[i], feats_for_aux[i+2],
            feats_for_aux[i+4]], axis=1) for i in range(2)]

        predictions_val, labels_val, loss_val  = sess.run(
                fetches, feed_dict={labels: label_batch,
                    aux_fc_batch_0: aux_fcs_0[0], aux_fc_batch_1: aux_fcs_0[1],
                    aux_fc_batch_2: aux_fcs_1[0], aux_fc_batch_3: aux_fcs_1[1]})
        seconds_per_batch = time.time() - batch_start_time
        example_per_second = labels_val.shape[0] / seconds_per_batch
        examples_processed += labels_val.shape[0]

        iteration_info_dict = evl_metrics.accumulate(predictions_val,
                                                     labels_val, loss_val)
        iteration_info_dict["examples_per_second"] = example_per_second

        iterinfo = utils.AddGlobalStepSummary(
            summary_writer,
            global_step_val,
            iteration_info_dict,
            summary_scope="Eval")
        logging.info("examples_processed: %d | %s", examples_processed,
                     iterinfo)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = evl_metrics.get()
      epoch_info_dict["epoch_id"] = global_step_val

      #summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          global_step_val,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)
      evl_metrics.clear()
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val, epoch_info_dict['avg_hit_at_one']


def evaluate(dataset,
             model,
             train_dir,
             dataset_dir,
             splits_dir,
             num_epochs,
             batch_size,
             split_num,
             label_loss='CrossEntropyLoss',
             run_once=True):
  tf.set_random_seed(0)  # for reproducibility
  with tf.Graph().as_default():
    reader = getattr(data, dataset)(dataset_dir, splits_dir,
                    num_epochs, batch_size, split_num)

    model = find_class_by_name(model,
                            [models])()
    label_loss_fn = find_class_by_name(label_loss, [losses])()

    restore_vars = \
        build_graph(
            reader=reader,
            model=model,
            label_loss_fn=label_loss_fn,
            num_readers=1,
            batch_size=batch_size)
    logging.info("built evaluation graph")
    loss = tf.get_collection("loss")[0]
    predictions = tf.get_collection("predictions")[0]
    labels = tf.get_collection("labels")[0]
    inputs = tf.get_collection("input_batch")[0]
    feature_0 = tf.get_collection("feature_0")[0]
    feature_1 = tf.get_collection("feature_1")[0]
    aux_feat_batch = tf.get_collection("aux_feat_batch")[0]
    aux_output = tf.get_collection("aux_output")[0]
    aux_fc_batch_0 = tf.get_collection("aux_fc_batch_0")[0]
    logits_aux_0 = tf.get_collection("logits_aux_0")[0]
    aux_fc_batch_1 = tf.get_collection("aux_fc_batch_1")[0]
    logits_aux_1 = tf.get_collection("logits_aux_1")[0]
    aux_fc_batch_2 = tf.get_collection("aux_fc_batch_2")[0]
    logits_aux_2 = tf.get_collection("logits_aux_2")[0]
    aux_fc_batch_3 = tf.get_collection("aux_fc_batch_3")[0]
    logits_aux_3 = tf.get_collection("logits_aux_3")[0]
    inputs_loader = tf.get_collection("images_loader")[0]
    labels_loader = tf.get_collection("labels_loader")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, 20)

    last_global_step_val = -1
    while True:
      last_global_step_val, h1 = evaluation_loop(predictions, labels, loss,
              inputs, aux_feat_batch, aux_output, aux_fc_batch_0, logits_aux_0,
              aux_fc_batch_1, logits_aux_1, aux_fc_batch_2, logits_aux_2,
              aux_fc_batch_3, logits_aux_3, inputs_loader, feature_0, feature_1,
              labels_loader, saver, summary_writer, train_dir, evl_metrics, last_global_step_val)

      if run_once:
        break
    return h1
