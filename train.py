import os, argparse, shutil, time, glob

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import logging
from tensorflow import gfile
from tensorflow.python.client import device_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.tools.inspect_checkpoint import *

from tensorflow.contrib.tensorboard.plugins import projector

import data, models
from utils import *
from test import *

slim = tf.contrib.slim
layers = tf.contrib.layers
framework = tf.contrib.framework

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_string("train_dir", "",
                        "Directory to save the model files in")
    flags.DEFINE_string("dataset", "HybridModelReader", "Which dataset to load \
                        for Action Recognition")
    flags.DEFINE_string("dataset_dir", "", \
                        "Path to base directory for video frames (rgb / rgb+flow)")
    flags.DEFINE_string("splits_dir", "", \
                        "Directory where train and test splits are stored")
    flags.DEFINE_string("checkpoint_file", "", \
                        "Checkpoint file to restore variables")
    flags.DEFINE_string("model", "Hybrid", "Which architecture to use for the model")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss", "Which loss function to use \
                            for training the model")
    flags.DEFINE_string("optimizer", "AdamOptimizer", "What optimizer class to use")
    flags.DEFINE_string("split_num", "1", "The train/test split to run the model on")

    flags.DEFINE_integer("batch_size", 32, "Number of examples to process per batch \
                            for training")
    flags.DEFINE_integer("num_epochs", 50, "How many passes to make over the dataset \
                            before halting training")
    flags.DEFINE_integer("export_model_steps", 10000, "The period, in number of steps, \
                            with which the model is exported for batch prediction")
    flags.DEFINE_integer("max_steps", None, "The maximum number of iterations of the \
                            training loop")
    flags.DEFINE_integer("learning_rate_decay_examples", 80182, "Multiply current learning \
                            rate by learning_rate_decay every learning_rate_decay_examples")

    flags.DEFINE_float("base_learning_rate", 0.0001, "Which learning rate to start with")
    flags.DEFINE_float("learning_rate_decay", 0.9, "Learning rate decay factor to be \
                            applied every learning_rate_decay_examples")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to")
    flags.DEFINE_float("regularization_penalty", 0.00005, "How much weight to give to the \
                            regularization loss (the label loss has a weight of 1).")

    flags.DEFINE_bool("start_new_model", False, "If set, this will not resume from a checkpoint \
                            and will instead create a new model instance")
    flags.DEFINE_bool("log_device_placement", False, "Whether to write the device on which every \
                            op will run into the logs on startup.")


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=2,
                           num_epochs=None,
                           num_readers=1):
    logging.info("Using batch size of " + str(batch_size) + " for training.")
    files, labels = reader._read_filelist(split=reader.present_split)

    with tf.name_scope("train_input"):
        logging.info("Number of training files: %s", str(len(files)))

        files = ops.convert_to_tensor(files, dtypes.string)
        labels = ops.convert_to_tensor(labels, dtypes.int64)

        input_queue = tf.train.slice_input_producer(
                            [files, labels],
                            num_epochs = num_epochs,
                            shuffle = True)
        image, label = reader._read_samples(input_queue)

        train_image_loader, train_label_loader = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            capacity = 5 * batch_size,
            min_after_dequeue = batch_size)

    return train_image_loader, train_label_loader

def build_graph(reader,
                model,
                split_num,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):

    global_step = tf.Variable(0, name="global_step", trainable=False)

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = optimizer_class(learning_rate)
    images_loader, labels_loader = (
            get_input_data_tensors(
                    reader,
                    reader.train_split_files[split_num],
                    batch_size=batch_size,
                    num_readers=num_readers,
                    num_epochs=num_epochs))
    images_batch = tf.placeholder(tf.float32, (None, 299, 299, 3))
    labels_batch = tf.placeholder(tf.int64, (None,))
    # 17 x 17 x 512
    feature, restore_vars, train_v0 = model.create_feature_model(
        images_batch)
    # 17 x 17 x (512 * 3)
    aux_feat_batch = tf.placeholder(tf.float32, (None, 17, 17, 1536))
    aux_output, train_v1 = model.create_aux_model(
        aux_feat_batch)
    # 17 x 17 x (512 * 2)
    lstm_feat_batch = tf.placeholder(tf.float32, (None, 17, 17, 1024))
    lstm_output, train_v2 = model.create_lstm_model(
        lstm_feat_batch)
    # (3 * 512)
    lstm_fc_batch = tf.placeholder(tf.float32, (None, 1536))
    logits_lstm, train_v4 = model.create_logits_model(
        lstm_fc_batch, 60, scope="lstmlogs")
    # (17 * 512 * 2) = 17408
    aux_fc_batch = tf.placeholder(tf.float32, (None, 17408))
    logits_aux, train_v3 = model.create_logits_model(
        aux_fc_batch, 60, scope="auxlogs")

    # 60
    loss = (0.4 * tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux, labels=labels_batch))
        + 0.6 * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_lstm, labels=labels_batch)))

    predictions = tf.placeholder(tf.float32, (None, 60))

    train_vars = train_v0
    train_vars.extend(train_v1)
    train_vars.extend(train_v2)
    train_vars.extend(train_v3)
    train_vars.extend(train_v4)
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=train_vars)

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
    tf.add_to_collection("train_op", train_op)
    tf.add_to_collection("images_loader", images_loader)
    tf.add_to_collection("labels_loader", labels_loader)

    return restore_vars

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def task_as_string(task):
    return "/job:%s/task:%s" % (task.type, task.index)

class Trainer(object):
    def __init__(self, cluster, task, train_dir, model, reader,
                log_device_placement=True, max_steps=None,
                export_model_steps=1000):

        self.cluster = cluster
        self.task = task
        self.is_master = (task.type == "master" and task.index == 0)
        self.train_dir = train_dir
        self.config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=log_device_placement)
        self.model = model
        self.reader = reader
        #self.model_exporter = model_exporter
        self.max_steps = max_steps
        self.max_steps_reached = False
        self.export_model_steps = export_model_steps
        self.last_model_export_step = 0

    def remove_training_directory(self, train_dir):
        """Removes the training directory."""
        try:
            logging.info(
              "%s: Removing existing train directory.",
              task_as_string(self.task))
            gfile.DeleteRecursively(train_dir)
        except:
            logging.error(
              "%s: Failed to delete directory " + train_dir +
              " when starting a new model. Please delete it manually and" +
              " try again.", task_as_string(self.task))

    def start_server_if_distributed(self):
        """ Starts a server if the execution is distributed """

        if self.cluster:
            logging.info("%s: Starting trainer within cluster %s.",
                           task_as_string(self.task), self.cluster.as_dict())
            server = start_server(self.cluster, self.task)
            target = server.target
            device_fn = tf.train.replica_device_setter(
                ps_device="/job:ps",
                worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
                cluster=self.cluster)
        else:
            target = ""
            device_fn = ""
        return (target, device_fn)

    def get_meta_filename(self, start_new_model, train_dir):
        if start_new_model:
            logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
            return None

        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if not latest_checkpoint:
            logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
            return None

        meta_filename = latest_checkpoint + ".meta"
        if not gfile.Exists(meta_filename):
            logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
            return None
        else:
            return meta_filename

    def recover_model(self, meta_filename):
        logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
        return tf.train.import_meta_graph(meta_filename)

    def build_model(self, model, reader):
        """ Find the model and build the graph """

        label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
        optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

        restore_vars = \
            build_graph(reader=reader,
                         model=model,
                         optimizer_class=optimizer_class,
                         clip_gradient_norm=FLAGS.clip_gradient_norm,
                         split_num=FLAGS.split_num,
                         label_loss_fn=label_loss_fn,
                         base_learning_rate=FLAGS.base_learning_rate,
                         learning_rate_decay=FLAGS.learning_rate_decay,
                         learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                         regularization_penalty=FLAGS.regularization_penalty,
                         num_readers=1,
                         batch_size=FLAGS.batch_size,
                         num_epochs=FLAGS.num_epochs)

        saver = tf.train.Saver(var_list=restore_vars)

        return saver, tf.train.Saver(max_to_keep=2)

    def export_model(self, global_step_val, saver, save_path, session):

        # If the model has already been exported at this step, return.
        if global_step_val == self.last_model_export_step:
            return

        last_checkpoint = saver.save(session, save_path, global_step_val)

        model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
        logging.info("%s: Exporting the model at step %s to %s.",
                task_as_string(self.task), global_step_val, model_dir)

        self.model_exporter.export_model(
            model_dir=model_dir,
            global_step_val=global_step_val,
            last_checkpoint=last_checkpoint)

    def run(self, start_new_model=0):
        if self.is_master and start_new_model:
            self.remove_training_directory(self.train_dir)

        target, device_fn = self.start_server_if_distributed()

        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

        with tf.Graph().as_default() as graph:
            if meta_filename:
                saver = self.recover_model(meta_filename)

            with tf.device(device_fn):
                if not meta_filename:
                    saver_for_restore, saver = self.build_model(self.model, self.reader)

            global_step = tf.get_collection("global_step")[0]
            loss = tf.get_collection("loss")[0]
            predictions = tf.get_collection("predictions")[0]
            labels = tf.get_collection("labels")[0]
            inputs = tf.get_collection("input_batch")[0]
            train_op = tf.get_collection("train_op")[0]
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
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sv = tf.train.Supervisor(
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=10 * 60,
            save_summaries_secs=120,
            saver=saver)

        with tf.Session(graph=graph) as sess:
            if not meta_filename:
                saver_for_restore.restore(sess, FLAGS.checkpoint_file)

        if os.path.exists('best_checkpoint_path'):
            with open('best_checkpoint_path', 'r') as f:
                lines = f.readlines()
                line = lines[-1].strip().split(',')[1].strip()
                global_h1 = float(line)
        else:
            global_h1 = -1
        validation_steps = 37585
        logging.info("%s: Starting managed session.", task_as_string(self.task))
        with sv.managed_session(config=self.config) as sess:
            try:
                logging.info("%s: Entering training loop.", task_as_string(self.task))
                while (not sv.should_stop()) and (not self.max_steps_reached):
                    #logging.info("%s, Entered", task_as_string(self.task))
                    batch_start_time = time.time()
                    input_batch, label_batch = sess.run([inputs_loader, labels_loader])
                    #logging.info("%s, Doesn't break before", task_as_string(self.task))

                    # (batch_size, 6, 299, 299, 3)
                    input_batch = np.transpose(input_batch, [1, 0, 2, 3, 4])
                    # list of (batch_size, 299,  299, 3) of size 6
                    six_inputs = np.split(input_batch, 6)
                    six_inputs = [np.reshape(x, [-1, 299, 299, 3]) for x in six_inputs]

                    features = []
                    for inp in six_inputs:
                        feat_vec = sess.run(feature, feed_dict={inputs: inp})
                        features.append(feat_vec)
                    # Each entry has size (batch_size, 17, 17, 1536)
                    feats_for_aux = [np.concatenate([features[i], features[i+2],
                        features[i+4]], axis=3) for i in range(2)]
                    # Each entry has size (batch_size, 17, 17, 1024)
                    feats_for_lstm = [np.concatenate([features[i], features[i+1]],
                        axis=3) for i in range(0, 5, 2)]

                    # Each output (2) has size (batch_size, 8704)
                    aux_outputs = []
                    for feat in feats_for_aux:
                        out = sess.run(aux_output, feed_dict={aux_feat_batch: feat})
                        aux_outputs.append(out)
                    # Each output (3) has size (batch_size, 512)
                    lstm_outputs = []
                    for feat in feats_for_lstm:
                        out = sess.run(lstm_output, feed_dict={lstm_feat_batch: feat})
                        lstm_outputs.append(out)

                    # lstm_fc: (batch_size, 1536)
                    # aux_fc: (batch_size, 17408)
                    lstm_fc = np.concatenate(lstm_outputs, axis=1)
                    aux_fc = np.concatenate(aux_outputs, axis=1)

                    #logging.info("%s, %s", lstm_fc.shape, aux_fc.shape)

                    lstm_logits, aux_logits = sess.run(
                            [logits_lstm, logits_aux],
                            feed_dict={aux_fc_batch: aux_fc, lstm_fc_batch: lstm_fc})

                    final_logits = (0.4 * aux_logits) + (0.6 * lstm_logits)

                    _, global_step_val, predictions_val, labels_val, loss_val = sess.run([train_op,
                        global_step, predictions, labels, loss], feed_dict={labels: label_batch,
                            predictions: final_logits, aux_fc_batch: aux_fc, lstm_fc_batch: lstm_fc})

                    seconds_per_batch = time.time() - batch_start_time
                    examples_per_second = labels_val.shape[0] / seconds_per_batch # TODO

                    if self.max_steps and self.max_steps <= global_step_val:
                        self.max_steps_reached = True

                    if self.is_master and global_step_val % 10 == 0 and self.train_dir:
                        eval_start_time = time.time()
                        hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
                        hit_at_five = eval_util.calculate_hit_at_five(predictions_val, labels_val)

                        eval_end_time = time.time()
                        eval_time = eval_end_time - eval_start_time

                        logging.info("training step " + str(global_step_val) + " | Loss: " + ("%.2f" % loss_val) +
                            " Examples/sec: " + ("%.2f" % examples_per_second) + " | Hit@1: " + ("%.2f" % hit_at_one) +
                            " Hit@5: " + ("%.2f" % hit_at_five))

                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                            global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Hit@5", hit_at_five),
                            global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("global_step/Examples/Second",
                                              examples_per_second), global_step_val)
                        sv.summary_writer.flush()

                    else:
                        logging.info("training step " + str(global_step_val) + " | Loss: " + ("%.2f" % loss_val) +
                            " Examples/sec: " + ("%.2f" % examples_per_second))

                    if global_step_val and not (global_step_val % validation_steps):
                        f = open('best_checkpoint_path', 'a+')
                        if not os.path.exists('best_checkpoint'):
                            os.makedirs('best_checkpoint')
                        avg_h1 = evaluate(FLAGS.dataset, FLAGS.model, FLAGS.train_dir,
                                        FLAGS.dataset_dir, FLAGS.splits_dir, 1, FLAGS.batch_size, FLAGS.split_num)
                        if avg_h1 > global_h1:
                            global_h1 = avg_h1
                            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
                            f.write("%s, %f" % (latest_checkpoint, global_h1))
                            del_files = glob.glob('best_checkpoint/*')
                            for d in del_files:
                                os.remove(d)
                            files = glob.glob(latest_checkpoint + '.*')
                            for fn in files:
                                shutil.copy(fn, 'best_checkpoint')
                        f.close()

            except tf.errors.OutOfRangeError:
                logging.info("%s: Done training -- epoch limit reached.",
                            task_as_string(self.task))
        logging.info("%s: Exited training loop.", task_as_string(self.task))
        sv.Stop()

def main(unused_argv):

    cluster = None
    task_data = {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)

    logging.set_verbosity(tf.logging.INFO)
    logging.info("%s: Tensorflow version: %s",
                task_as_string(task), tf.__version__)

    if not cluster or task.type == "master" or task.type == "worker":
        model = find_class_by_name(FLAGS.model,
                    [models])()
        reader = getattr(data, FLAGS.dataset)(FLAGS.dataset_dir, FLAGS.splits_dir,
                FLAGS.num_epochs, FLAGS.batch_size, FLAGS.split_num)

        #model_exporter = export_model.ModelExporter(
        #        model=model,
        #        reader=reader)

        Trainer(cluster, task, FLAGS.train_dir, model, reader,
                FLAGS.log_device_placement, FLAGS.max_steps,
                FLAGS.export_model_steps).run(start_new_model=FLAGS.start_new_model)

    elif task.type == "ps":
        # Distributed server
        raise NotImplementedError()
    else:
        raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))

if __name__ == '__main__':
    app.run()
