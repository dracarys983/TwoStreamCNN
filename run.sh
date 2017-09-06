#!/bin/bash

python train.py --train_dir=/home/procastinator/nturgbd_hybrid_cnnonly --dataset_dir=/home/procastinator/nturgb+d_images \
    --splits_dir=/home/procastinator/NTU_data --split_num 2 --checkpoint_file=/home/procastinator/ActRec-2D/models/VGG/vgg_16.ckpt
