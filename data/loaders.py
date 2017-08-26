import os, argparse
import threading
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

gfile = tf.gfile
slim = tf.contrib.slim

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
