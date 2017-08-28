#!/bin/bash

python train.py --train_dir=/home/procastinator/nturgbd_hybrid --dataset_dir=/home/procastinator/nturgb+d_images \
    --splits_dir=/home/procastinator/NTU_data --split_num 1 --checkpoint_file=./models/HybridModel/inception_resnet_v2.ckpt
