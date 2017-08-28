import tensorflow as tf
import numpy as np
import os
from PIL import Image

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
    flags.DEFINE_string("output_dir", "", \
                        "Directory where TFRecord files are to be stored")

    flags.DEFINE_integer("split_num", 1, \
                         "The present train / test split to preprocess")

    flags.DEFINE_bool("is_training", True, \
                      "Whether the present split is for train or test")
    flags.DEFINE_bool("tfrecords", False, \
                      "Whether to create TFRecords or create images")

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
        n = 0; l = len(files); two_person = 0
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
                skeletons_0, skeletons_1 = video._get_main_actor_skeletons()
                # logging.info("%s: Number of skeletons: %d", task_as_string(task), len(skeletons))

                # Spatial Feature: Relative coordinates for each joint in the frame [300 x 900]
                features = np.zeros((300, 576), dtype=np.float32)
                two_person_action = True
                for i in skeletons_1:
                    if i._is_zero_skeleton:
                        two_person_action = False
                one_person_feat_len = 288
                for nn, skeleton in enumerate(skeletons_0):
                    feat_len = 0
                    joints = skeleton._get_joint_objects()
                    assert len(joints) == 25
                    for i in [5, 8, 12, 16]:
                        joint = joints[i]
                        x, y, z = joint._get_cartesian_coordinates()
                        for j in range(len(joints)):
                            if not j == i:
                                joint_ = joints[j]
                                x_, y_, z_ = joint_._get_cartesian_coordinates()
                                r = x - x_; theta = y - y_; phi = z - z_
                                j = Joint(r, theta, phi)
                                r, theta, phi = j._get_spherical_coordinates()
                                features[nn, feat_len] = r; feat_len += 1
                                features[nn, feat_len] = theta; feat_len += 1
                                features[nn, feat_len] = phi; feat_len += 1
                if two_person_action:
                    two_person += 1
                    logging.info("%s: Two person Action", task_as_string(task))
                    for nn, skeleton in enumerate(skeletons_0):
                        feat_len = 0
                        joints_0 = skeleton._get_joint_objects()
                        joints_1 = skeletons_1[nn]._get_joint_objects()
                        assert len(joints_0) == 25; assert len(joints_1) == 25
                        for i in [5, 8, 12, 16]:
                            joint = joints_0[i]
                            x, y, z = joint._get_cartesian_coordinates()
                            for j in range(len(joints_1)):
                                if not j == i:
                                    joint_ = joints_1[j]
                                    x_, y_, z_ = joint_._get_cartesian_coordinates()
                                    r = x - x_; theta = y - y_; phi = z - z_
                                    j = Joint(r, theta, phi)
                                    r, theta, phi = j._get_spherical_coordinates()
                                    features[nn, one_person_feat_len+feat_len] = r; feat_len += 1
                                    features[nn, one_person_feat_len+feat_len] = theta; feat_len += 1
                                    features[nn, one_person_feat_len+feat_len] = phi; feat_len += 1
                feats_spatial = features
                feats_spatial = np.pad(feats_spatial, [[0, (300 - feats_spatial.shape[0])], [0, 0]], 'constant', constant_values=0)
                feats_spatial = np.hstack(feats_spatial)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'feature': _float32_feature(feats_spatial),
                    'label': _int64_feature(label)}))
                writer.write(example.SerializeToString())
                writer.close()
                logging.info("%s: status: %d of %d done", task_as_string(task), n, l)
    return n, two_person

def _write_to_images(task,
                    reader,
                    split,
                    outdir='',
                    train=True):
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
        n = 0; l = len(files); two_person = 0
        for fname, label in zip(files, labels):
            if not any(fname.split('.')[0] in x for x in remove):
                n += 1
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                    os.makedirs(os.path.join(outdir, splitname + str(split)))
                elif not os.path.exists(os.path.join(outdir, splitname + str(split))):
                    os.makedirs(os.path.join(outdir, splitname + str(split)))

                image_dir = os.path.join(outdir, splitname + str(split), fname)
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                else:
                    if len(os.listdir(image_dir)) == 6:
                        continue
                video = reader._read_skeleton_file(fname)
                skeletons_0, skeletons_1 = video._get_main_actor_skeletons()

                two_person_action = True
                for i in skeletons_1:
                    if i._is_zero_skeleton:
                        two_person_action = False

                if two_person_action:
                    im_size = (2, len(skeletons_0), 48)
                else:
                    im_size = (2, len(skeletons_0), 24)

                im_r = np.zeros(im_size); im_theta = np.zeros(im_size); im_phi = np.zeros(im_size)
                for nn, skeleton in enumerate(skeletons_0):
                    joints = skeleton._get_joint_objects()
                    assert len(joints) == 25
                    im_num = 0
                    for i in [0, 20]:
                        feat_len = 0
                        joint = joints[i]
                        x, y, z = joint._get_cartesian_coordinates()
                        for j in range(len(joints)):
                            if not j == i:
                                joint_ = joints[j]
                                x_, y_, z_ = joint_._get_cartesian_coordinates()
                                r = x - x_; theta = y - y_; phi = z - z_
                                im_r[im_num, nn, feat_len] = r; im_theta[im_num, nn, feat_len] = theta; im_phi[im_num, nn, feat_len] = phi
                                feat_len += 1
                        im_num += 1
                if two_person_action:
                    processed = 24
                    two_person += 1
                    for nn, skeleton in enumerate(skeletons_0):
                        joints_0 = skeleton._get_joint_objects()
                        joints_1 = skeletons_1[nn]._get_joint_objects()
                        assert len(joints_0) == 25; assert len(joints_1) == 25
                        im_num = 0
                        for i in [0, 20]:
                            feat_len = processed
                            joint = joints_0[i]
                            x, y, z = joint._get_cartesian_coordinates()
                            for j in range(len(joints_1)):
                                if not j == i:
                                    joint_ = joints_1[j]
                                    x_, y_, z_ = joint_._get_cartesian_coordinates()
                                    r = x - x_; theta = y - y_; phi = z - z_
                                    im_r[im_num, nn, feat_len] = r; im_theta[im_num, nn, feat_len] = theta; im_phi[im_num, nn, feat_len] = phi
                                    feat_len += 1
                            im_num += 1

                count = 0
                for im in im_r:
                    im += np.amin(im)
                    im *= 255.0 / np.amax(im)
                    im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
                    image = Image.fromarray(im.astype(np.uint8), 'RGB')
                    path = os.path.join(image_dir, 'img_%.4d.jpg' % count)
                    image.save(path)
                    count += 1
                for im in im_theta:
                    im += np.amin(im)
                    im *= 255.0 / np.amax(im)
                    im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
                    image = Image.fromarray(im.astype(np.uint8), 'RGB')
                    path = os.path.join(image_dir, 'img_%.4d.jpg' % count)
                    image.save(path)
                    count += 1
                for im in im_phi:
                    im += np.amin(im)
                    im *= 255.0 / np.amax(im)
                    im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
                    image = Image.fromarray(im.astype(np.uint8), 'RGB')
                    path = os.path.join(image_dir, 'img_%.4d.jpg' % count)
                    image.save(path)
                    count += 1
                logging.info("%s: status: %d of %d done", task_as_string(task), n, l)
    return n, two_person

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
    if FLAGS.output_dir == '':
        logging.info("%s: No target directory for TFRecords provided. "
                "Please set the --tfrecords_dir flag when running the script.", task_as_string(task))
        return EXIT_ERROR

    logging.info("%s: Using >\n"
                "           Dataset directory: %s\n"
                "           Split files directory: %s\n"
                "           Directory to store the output: %s\n"
                "           Split number: %d\n"
                "           Train/Test split: %s\n", task_as_string(task),
                        FLAGS.dataset_dir, FLAGS.splits_dir, FLAGS.output_dir,
                        FLAGS.split_num, "train" if FLAGS.is_training else "test")

    data_dir = FLAGS.dataset_dir
    split_dir = FLAGS.splits_dir
    reader = Reader(dataset_dir=data_dir, splits_dir=split_dir)

    if FLAGS.tfrecords:
        n, two_person = _write_to_tfrecords(task,
                            reader=reader,
                            split=FLAGS.split_num,
                            outdir=FLAGS.output_dir,
                            train=FLAGS.is_training)
    else:
        n, two_person = _write_to_images(task,
                            reader=reader,
                            split=FLAGS.split_num,
                            outdir=FLAGS.output_dir,
                            train=FLAGS.is_training)

    logging.info("%s: Converting to output format done! Total files: %d, Two person actions: %d. Exiting.", task_as_string(task), n, two_person)

if __name__ == '__main__':
    app.run()
