#!/bin/bash

python train.py --train_dir=/shared/blazar/home/kalpit.t/nturgbd_cs_rnn --dataset_dir=/shared/blazar/home/kalpit.t/nturgb+d_tfrecords \
    --splits_dir=/users/kalpit.t/NTU_data --split_num 1

python test.py --train_dir=/shared/blazar/home/kalpit.t/nturgbd_cs_rnn --dataset_dir=/shared/blazar/home/kalpit.t/nturgb+d_tfrecords \
    --splits_dir=/users/kalpit.t/NTU_data --split_num 1
