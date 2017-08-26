import os, argparse
import threading
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import nturgbd_reader
from utils import feature_engineer

gfile = tf.gfile
slim = tf.contrib.slim
np.random.seed(267)

class DataLoader(object):

    def __init__(self, **kwargs):
        self.data = kwargs['dataset_dir']
        self.splits = kwargs['split_dir']
        self.num_splits = int(kwargs['num_splits'])
        self.num_classes = int(kwargs['num_classes'])
        self.batch_size = int(kwargs['batch_size'])
        self.feature_len = int(kwargs['feature_len'])
        self.max_seq_length = 300

        self.train_files = {1: 'train_cs.txt', 2: 'train_cv.txt'}
        self.test_files = {1: 'test_cs.txt', 2: 'test_cv.txt'}
        f = open('/home/procastinator/ActRec-skeleton/faulty_skeletons', 'r')
        self.faulty_samples = f.readlines()
        f.close()

        self._idx = 0

    def _init_filelist(self, split_num, train=True):
        if train:
            f = open(os.path.join(self.splits, self.train_files[split_num]), 'r')
        else:
            f = open(os.path.join(self.splits, self.test_files[split_num]), 'r')
        self.files = []; self.labels = []
        for line in f:
            line = line.strip()
            fname, label = line.split()
            if not any(fname.split('.')[0] in x for x in self.faulty_samples):
                self.files.append(fname)
                self.labels.append(label)
        f.close()

    def _get_next_batch(self, shuffle=True):
        sequence_batch = np.zeros((self.batch_size, self.max_seq_length, 900))
        labels_batch = np.zeros((self.batch_size,), dtype=np.int64)
        reader = nturgbd_reader.Reader(self.data, self.splits)
        for i in range(self.batch_size):
            video = reader._read_skeleton_file(self.files[self._idx])
            skeletons = video._get_main_actor_skeletons()
            vsequence = []
            for skeleton in skeletons:
                joints = skeleton._get_joint_objects()
                fsequence = []
                for joint in joints:
                    x, y, z = joint._get_cartesian_coordinates()
                    fsequence.extend([x, y, z])
                vsequence.append(fsequence)
            vsequence = np.array(vsequence, dtype=np.float32)
            vsequence = np.pad(vsequence, [[0, (self.max_seq_length - vsequence.shape[0])], [0, 0]],
                    'constant', constant_values=0)
            vsequence = feature_engineer.get_transformed_features(vsequence)
            sequence_batch[i] = vsequence
            labels_batch[i] = self.labels[self._idx]
            self._idx += 1
            if self._idx == len(self.files):
                self._idx = 0
        sequence_batch = np.array(sequence_batch)
        labels_batch = np.array(labels_batch, dtype=np.int64)

        if shuffle:
            rng = np.random.get_state()
            np.random.shuffle(sequence_batch)
            np.random.set_state(rng)
            np.random.shuffle(labels_batch)

        return sequence_batch, labels_batch

    def _size(self):
        return len(self.files)

    def _reset(self):
        self._idx = 0


class DataGenerator(object):
    def __init__(self, coord, loader, queue_size=5 * 64):
        self.queue = tf.FIFOQueue(capacity=queue_size,
            dtypes=[tf.float32, tf.int64], shapes=[(300, 900), ()])
        self.loader = loader
        self.threads = []
        self.coord = coord
        self.sample_placeholder = tf.placeholder(tf.float32, [300, 900])
        self.label_placeholder = tf.placeholder(tf.int64, [])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.label_placeholder])

    def _size(self):
        return self.queue.size()

    def _dequeue(self, num_elements):
        seq, lab = self.queue.dequeue_many(num_elements)
        return seq, lab

    def _thread_main(self, sess):
        stop = False
        while not stop:
            it_seq, it_lab = self.loader._get_next_batch()
            for seq, lab in zip(it_seq, it_lab):
                if self.coord.should_stop():
                    stop = True
                    break
                sess.run(self.enqueue, feed_dict={self.sample_placeholder: seq, self.label_placeholder: lab})

    def _start_threads(self, sess, n_threads=4):
        for _ in range(n_threads):
            thread = threading.Thread(target=self._thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

class NTURGBD(object):

    def __init__(self, dataset_dir, split_dir, num_epochs, batch_size, split=1):
        self.dataset_dir = dataset_dir
        self.splits_dir = split_dir
        self.num_splits = 2
        self.num_classes = 60
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.present_split = split

        self.train_split_files = {'1': 'train1', '2': 'train2'}
        self.val_split_files = {'1': 'val1', '2': 'val2'}
        self.test_split_files = {'1': 'test1', '2': 'test2'}
        f = open('/home/procastinator/ActRec-skeleton/faulty_skeletons', 'r')
        self.faulty_samples = f.readlines()
        f.close()

    def _read_filelist(self, split='1', train=True, val=False, **kwargs):
        if train:
            files = gfile.ListDirectory(os.path.join(self.dataset_dir, self.train_split_files[split]))
            files = [os.path.join(self.dataset_dir, self.train_split_files[split], x) for x in files]
        elif val:
            files = gfile.ListDirectory(os.path.join(self.dataset_dir, self.val_split_files[split]))
            files = [os.path.join(self.dataset_dir, self.val_split_files[split], x) for x in files]
        else:
            files = gfile.ListDirectory(os.path.join(self.dataset_dir, self.test_split_files[split]))
            files = [os.path.join(self.dataset_dir, self.test_split_files[split], x) for x in files]
        return files

    def _read_samples(self, input_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(input_queue)

        features = tf.parse_single_example(
                serialized_example,
                features={
                    'feature': tf.FixedLenFeature([270000], tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64)})

        label = tf.cast(features['label'], tf.int64)
        feat_vec = tf.reshape(features['feature'], [300, 900])

        return feat_vec, label
