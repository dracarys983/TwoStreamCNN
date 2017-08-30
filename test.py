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
  images_batch = tf.placeholder(tf.float32, (None, 299, 299, 3))
  labels_batch = tf.placeholder(tf.int64, (None,))
  # 17 x 17 x 512
  feature, restore_vars, train_v0 = model.create_feature_model(
        images_batch, is_training=False)
  # 17 x 17 x (512 * 3)
  aux_feat_batch = tf.placeholder(tf.float32, (None, 17, 17, 1536))
  aux_output, train_v1 = model.create_aux_model(
        aux_feat_batch, is_training=False)
  # 17 x 17 x (512 * 2)
  lstm_feat_batch = tf.placeholder(tf.float32, (None, 17, 17, 1024))
  lstm_output, train_v2 = model.create_lstm_model(
        lstm_feat_batch, is_training=False)
  # (3 * 512)
  lstm_fc_batch = tf.placeholder(tf.float32, (None, 1536))
  logits_lstm, train_v4 = model.create_logits_model(
      lstm_fc_batch, 60, scope="lstmlogs", is_training=False)
  # (17 * 512 * 2) = 17408
  aux_fc_batch = tf.placeholder(tf.float32, (None, 17408))
  logits_aux, train_v3 = model.create_logits_model(
      aux_fc_batch, 60, scope="auxlogs", is_training=False)

  # 60
  loss = (0.4 * tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux, labels=labels_batch))
      + 0.6 * tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_lstm, labels=labels_batch)))

  predictions = tf.placeholder(tf.float32, (None, 60))

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", loss)
  tf.add_to_collection("feature", feature)
  tf.add_to_collection("aux_feat_batch", aux_feat_batch)
  tf.add_to_collection("aux_output", aux_output)
  tf.add_to_collection("lstm_feat_batch", lstm_feat_batch)
  tf.add_to_collection("lstm_output", lstm_output)
  tf.add_to_collection("aux_fc_batch", aux_fc_batch)
  tf.add_to_collection("logits_aux", logits_aux)
  tf.add_to_collection("lstm_fc_batch", lstm_fc_batch)
  tf.add_to_collection("logits_lstm", logits_lstm)
  tf.add_to_collection("input_batch", images_batch)
  tf.add_to_collection("labels", labels_batch)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("images_loader", images_loader)
  tf.add_to_collection("labels_loader", labels_loader)

  return restore_vars

def evaluation_loop(predictions, labels, loss,
              inputs, feature, aux_feat_batch, aux_output, lstm_feat_batch, lstm_output,
              aux_fc_batch, logits_aux, lstm_fc_batch, logits_lstm, inputs_loader,
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

        input_batch = np.transpose(input_batch, [1, 0, 2, 3, 4])
        six_inputs = np.split(input_batch, 6)
        six_inputs = [np.reshape(x, [-1, 299, 299, 3]) for x in six_inputs]

        features = []
        for inp in six_inputs:
            feat_vec = sess.run(feature, feed_dict={inputs: inp})
            features.append(feat_vec)
        feats_for_aux = [np.concatenate([features[i], features[i+2],
            features[i+4]], axis=3) for i in range(2)]
        feats_for_lstm = [np.concatenate([features[i], features[i+1]],
            axis=3) for i in range(0, 5, 2)]

        aux_outputs = []
        for feat in feats_for_aux:
            out = sess.run(aux_output, feed_dict={aux_feat_batch: feat})
            aux_outputs.append(out)
        lstm_outputs = []
        for feat in feats_for_lstm:
            out = sess.run(lstm_output, feed_dict={lstm_feat_batch: feat})
            lstm_outputs.append(out)

        lstm_fc = np.concatenate(lstm_outputs, axis=1)
        aux_fc = np.concatenate(aux_outputs, axis=1)

        aux_fc_output, lstm_fc_output = sess.run(
                [logits_aux, logits_lstm],
                feed_dict={aux_fc_batch: aux_fc, lstm_fc_batch: lstm_fc})

        final_logits = (0.4 * aux_fc_output) + (0.6 * lstm_fc_output)

        predictions_val, labels_val, loss_val  = sess.run(
                fetches, feed_dict={labels: label_batch,
                    predictions: final_logits, aux_fc_batch: aux_fc, lstm_fc_batch: lstm_fc})
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
    feature = tf.get_collection("feature")[0]
    aux_feat_batch = tf.get_collection("aux_feat_batch")[0]
    aux_output = tf.get_collection("aux_output")[0]
    lstm_feat_batch = tf.get_collection("lstm_feat_batch")[0]
    lstm_output = tf.get_collection("lstm_output")[0]
    aux_fc_batch = tf.get_collection("aux_fc_batch")[0]
    logits_aux = tf.get_collection("logits_aux")[0]
    lstm_fc_batch = tf.get_collection("lstm_fc_batch")[0]
    logits_lstm = tf.get_collection("logits_lstm")[0]
    inputs_loader = tf.get_collection("images_loader")[0]
    labels_loader = tf.get_collection("labels_loader")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, 20)

    last_global_step_val = -1
    while True:
      last_global_step_val, h1 = evaluation_loop(predictions, labels, loss,
              inputs, feature, aux_feat_batch, aux_output, lstm_feat_batch, lstm_output,
              aux_fc_batch, logits_aux, lstm_fc_batch, logits_lstm, inputs_loader,
              labels_loader, saver, summary_writer, train_dir, evl_metrics, last_global_step_val)

      if run_once:
        break
    return h1

'''
if __name__=='__main__':
    logging.set_verbosity(tf.logging.INFO)
    dataset = 'HybridModelReader'
    model = 'Hybrid'
    train_dir = '/home/procastinator/nturgbd_hybrid'
    dataset_dir = '/home/procastinator/nturgb+d_images'
    splits_dir = '/home/procastinator/NTU_data'
    checkpoint_file = ''
    num_epochs = 1
    batch_size = 32
    split_num = '1'
    out = evaluate(dataset, model, train_dir, dataset_dir, splits_dir, num_epochs, batch_size, split_num)
    print out
'''
