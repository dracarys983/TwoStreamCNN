import tensorflow as tf
import numpy as np
import os

from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from nturgbd import Joint, Reader

FLAGS = flags.FLAGS

if __name__ == '__main__':
    flags.DEFINE_string("dataset_dir", "", \
                        "Path to base directory for skeleton files")
    flags.DEFINE_string("splits_dir", "", \
                        "Directory where train and test splits are stored")
    flags.DEFINE_string("tfrecords_dir", "", \
                        "Directory where TFRecord files are to be stored")

    flags.DEFINE_integer("split_num", 1, \
                         "The present train / test split to preprocess")

    flags.DEFINE_bool("is_training", True, \
                      "Whether the present split is for train or test")

def task_as_string(task):
    return "/job:%s/task:%s" % (task.type, task.index)

def _write_to_tfrecords(task,
                        reader,
                        split,
                        outdir='',
                        train=True):
    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    if train:
        fpath = reader.train_splits[split]
        splitname = "train"
    else:
        fpath = reader.test_splits[split]
        splitname = "test"
    logging.info("%s: Converting %s split%d files to TFRecords", task_as_string(task), splitname, split)

    with open(os.path.join(reader.splits, 'faulty_skeletons'), 'r') as f:
        remove = f.readlines()
        remove = [x.strip() for x in remove]

    with open(fpath, 'r') as f:
        lines = f.readlines()
        files = [x.strip().split()[0] for x in lines]
        labels = [int(x.strip().split()[1]) for x in lines]
        n = 0; l = len(files)
        for fname, label in zip(files, labels):
            if not any(fname.split('.')[0] in x for x in remove):
                n += 1
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                    os.makedirs(os.path.join(outdir, splitname + str(split)))
                elif not os.path.exists(os.path.join(outdir, splitname + str(split))):
                    os.makedirs(os.path.join(outdir, splitname + str(split)))

                tfrecord_file = os.path.join(outdir, splitname + str(split), fname + '.tfrecord')
                if not os.path.exists(tfrecord_file):
                    writer = tf.python_io.TFRecordWriter(tfrecord_file)
                else:
                    continue
                video = reader._read_skeleton_file(fname)
                skeletons = video._get_main_actor_skeletons()
                # logging.info("%s: Number of skeletons: %d", task_as_string(task), len(skeletons))

                # Spatial Feature: Relative coordinates for each joint in the frame [300 x 900]
                features = []
                for skeleton in skeletons:
                    feat = []
                    joints = skeleton._get_joint_objects()
                    assert len(joints) == 25
                    for i in range(len(joints)):
                        joint = joints[i]
                        x, y, z = joint._get_cartesian_coordinates()
                        for j in range(i+1, len(joints)):
                            joint_ = joints[j]
                            x_, y_, z_ = joint_._get_cartesian_coordinates()
                            r = x - x_; theta = y - y_; phi = z - z_
                            j = Joint(r, theta, phi)
                            r, theta, phi = j._get_spherical_coordinates()
                            feat.extend([r, theta, phi])
                    features.append(feat)
                feats_spatial = np.array(features, dtype=np.float32)
                feats_spatial = np.pad(feats_spatial, [[0, (300 - feats_spatial.shape[0])], [0, 0]], 'constant', constant_values=0)
                feats_spatial = np.hstack(feats_spatial)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _float32_feature(feats_spatial),
                    'label': _int64_feature(label)}))
                writer.write(example.SerializeToString())
                writer.close()
                logging.info("%s: status: %d of %d done", task_as_string(task), n, l)
    return

def main(unused_argv):
    task_data = {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)

    logging.set_verbosity(tf.logging.INFO)
    logging.info("%s: Tensorflow version: %s",
                task_as_string(task), tf.__version__)

    if FLAGS.dataset_dir == '':
        logging.info("%s: No dataset directory provided. "
                "Please set the --dataset_dir flag when running the script.", task_as_string(task))
        return EXIT_ERROR
    if FLAGS.splits_dir == '':
        logging.info("%s: No split files directory provided. "
                "Please set the --splits_dir flag when running the script.", task_as_string(task))
        return EXIT_ERROR
    if FLAGS.tfrecords_dir == '':
        logging.info("%s: No target directory for TFRecords provided. "
                "Please set the --tfrecords_dir flag when running the script.", task_as_string(task))
        return EXIT_ERROR

    logging.info("%s: Using >\n"
                "           Dataset directory: %s\n"
                "           Split files directory: %s\n"
                "           Directory to store TFRecords: %s\n"
                "           Split number: %d\n"
                "           Train/Test split: %s\n", task_as_string(task),
                        FLAGS.dataset_dir, FLAGS.splits_dir, FLAGS.tfrecords_dir,
                        FLAGS.split_num, "train" if FLAGS.is_training else "test")

    data_dir = FLAGS.dataset_dir
    split_dir = FLAGS.splits_dir
    reader = Reader(dataset_dir=data_dir, splits_dir=split_dir)

    _write_to_tfrecords(task,
                        reader=reader,
                        split=FLAGS.split_num,
                        outdir=FLAGS.tfrecords_dir,
                        train=FLAGS.is_training)

    logging.info("%s: Converting to TFRecords done! Exiting.", task_as_string(task))

if __name__ == '__main__':
    app.run()
