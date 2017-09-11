#!/bin/bash

python train.py --train_dir=/home/procastinator/nturgbd_irv2_exp1 --dataset_dir=/home/procastinator/nturgb+d_images \
    --splits_dir=/home/procastinator/NTU_data --split_num 2 --checkpoint_file=/home/procastinator/ActRec-2D/models/inception_resnet_v2/inception_resnet_v2.ckpt
