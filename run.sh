#!/bin/bash

python train_old.py --train_dir=/home/procastinator/nturgbd_cs_rnn_2 --dataset_dir=/home/procastinator/nturgb+d_tfrecords \
    --splits_dir=/home/procastinator/NTU_data --split_num 1

python test.py --train_dir=/home/procastinator/nturgbd_cs_rnn_2 --dataset_dir=/home/procastinator/nturgb+d_tfrecords \
    --splits_dir=/home/procastinator/NTU_data --split_num 1
