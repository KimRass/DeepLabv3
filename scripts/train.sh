#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train.py\
    --img_dir="/home/jbkim/Documents/datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"\
    --gt_dir="/home/jbkim/Documents/datasets/SegmentationClassAug/SegmentationClassAug"\
    --save_dir="/home/jbkim/Documents/deeplabv3"\
    --batch_size=8\
    --n_cpus=4\
    --n_steps=30_000\
    --init_lr=0.0002\
