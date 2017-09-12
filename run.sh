#!/bin/bash

python train.py --train_dir=/home/procastinator/nturgbd_twostreamcnn --dataset_dir=/home/procastinator/nturgb+d_images_new \
    --splits_dir=/home/procastinator/NTU_data --split_num 2 --checkpoint_file=/home/procastinator/pretrainedCheckpoints/vgg_19.ckpt
