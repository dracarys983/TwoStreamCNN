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
    flags.DEFINE_string("optimizer", "GradientDescentOptimizer", "What optimizer class to use")
    flags.DEFINE_string("split_num", "1", "The train/test split to run the model on")

    flags.DEFINE_integer("batch_size", 64, "Number of examples to process per batch \
                            for training")
    flags.DEFINE_integer("num_epochs", 50, "How many passes to make over the dataset \
                            before halting training")
    flags.DEFINE_integer("export_model_steps", 10000, "The period, in number of steps, \
                            with which the model is exported for batch prediction")
    flags.DEFINE_integer("max_steps", None, "The maximum number of iterations of the \
                            training loop")
    flags.DEFINE_integer("learning_rate_decay_examples", 188895, "Multiply current learning \
                            rate by learning_rate_decay every learning_rate_decay_examples")

    flags.DEFINE_float("base_learning_rate", 0.001, "Which learning rate to start with")
    flags.DEFINE_float("learning_rate_decay", 0.95, "Learning rate decay factor to be \
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
    images_batch = tf.placeholder(tf.float32, (None, 224, 224, 3))
    labels_batch = tf.placeholder(tf.int64, (None,))
    # (224, 224, 3) -> (14, 14, 512)
    feature_0, restore_vars_0, train_v0 = model.create_feature_model(
        images_batch)
    feature_1, restore_vars_1, train_v1 = model.create_feature_model(
        images_batch)
    # (14, 14, 512) -> (7168,)
    aux_feat_batch = tf.placeholder(tf.float32, (None, 14, 14, 512))
    aux_output, train_v2 = model.create_aux_model(
        aux_feat_batch)
    # (21504,) -> (60,)
    aux_fc_batch_0 = tf.placeholder(tf.float32, (None, 21504))
    logits_aux_0, train_v3 = model.create_logits_model(
        aux_fc_batch_0, 60, scope="auxlogs", reuse=None)
    # (21504,) -> (60,)
    aux_fc_batch_1 = tf.placeholder(tf.float32, (None, 21504))
    logits_aux_1, train_v4 = model.create_logits_model(
        aux_fc_batch_1, 60, scope="auxlogs", reuse=True)
    # (21504,) -> (60,)
    aux_fc_batch_2 = tf.placeholder(tf.float32, (None, 21504))
    logits_aux_2, train_v5 = model.create_logits_model(
        aux_fc_batch_2, 60, scope="auxlogs", reuse=True)
    # (21504,) -> (60,)
    aux_fc_batch_3 = tf.placeholder(tf.float32, (None, 21504))
    logits_aux_3, train_v6 = model.create_logits_model(
        aux_fc_batch_3, 60, scope="auxlogs", reuse=True)

    loss_0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_0, labels=labels_batch))
    loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_1, labels=labels_batch))
    loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_2, labels=labels_batch))
    loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_aux_3, labels=labels_batch))

    loss = loss_0 + loss_1 + loss_2 + loss_3
    predictions = ( logits_aux_0 + logits_aux_1 + logits_aux_2 + logits_aux_3 )

    train_vars = train_v0
    train_vars.extend(train_v1)
    train_vars.extend(train_v2)
    train_vars.extend(train_v3)
    train_vars.extend(train_v4)
    train_vars.extend(train_v5)
    train_vars.extend(train_v6)
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=train_vars)

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
    tf.add_to_collection("train_op", train_op)
    tf.add_to_collection("images_loader", images_loader)
    tf.add_to_collection("labels_loader", labels_loader)

    return restore_vars_0.extend(restore_vars_1)

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
            if not meta_filename:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            else:
                init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=60 * 60,
            save_summaries_secs=7200,
            saver=saver)

        with tf.Session(graph=graph) as sess:
            if not meta_filename:
                saver_for_restore.restore(sess, FLAGS.checkpoint_file)

        pp = os.path.join(FLAGS.train_dir, 'best_checkpoint_path')
        if os.path.exists(pp):
            with open(pp, 'r') as f:
                lines = f.readlines()
                if len(lines):
                    line = lines[-1].strip().split(',')[1].strip()
                    global_h1 = float(line)
                else:
                    global_h1 = -1
        else:
            global_h1 = -1
        validation_steps = 10001
        logging.info("%s: Starting managed session.", task_as_string(self.task))
        with sv.managed_session(config=self.config) as sess:
            try:
                logging.info("%s: Entering training loop.", task_as_string(self.task))
                while (not sv.should_stop()) and (not self.max_steps_reached):

                    batch_start_time = time.time()
                    input_batch, label_batch = sess.run([inputs_loader, labels_loader])

                    # (batch_size, 12, 224, 224, 3)
                    input_batch = np.transpose(input_batch, [1, 0, 2, 3, 4])

                    # list of (batch_size, 224,  224, 3) of size 12
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

                    _, global_step_val, predictions_val, labels_val, loss_val = sess.run([train_op,
                        global_step, predictions, labels, loss], feed_dict={labels: label_batch,
                            aux_fc_batch_0: aux_fcs_0[0], aux_fc_batch_1: aux_fcs_0[1],
                            aux_fc_batch_2: aux_fcs_1[0], aux_fc_batch_3: aux_fcs_1[1]})

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
                            utils.MakeSummary("model/Training_Loss", loss_val),
                            global_step_val)
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
                        f = open(pp, 'a+')
                        p = os.path.join(FLAGS.train_dir, 'best_checkpoint')
                        if not os.path.exists(p):
                            os.makedirs(p)
                        avg_h1 = evaluate(FLAGS.dataset, FLAGS.model, FLAGS.train_dir,
                                        FLAGS.dataset_dir, FLAGS.splits_dir, 1, FLAGS.batch_size, FLAGS.split_num)
                        if avg_h1 > global_h1:
                            global_h1 = avg_h1
                            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
                            f.write("%s, %f\n" % (latest_checkpoint, global_h1))
                            del_files = glob.glob(p + '/*')
                            for d in del_files:
                                os.remove(d)
                            files = glob.glob(latest_checkpoint + '.*')
                            for fn in files:
                                shutil.copy(fn, p)
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
