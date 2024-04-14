#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../train.py\
    --img_dir="/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"\
    --gt_dir="/Users/jongbeomkim/Documents/datasets/SegmentationClassAug"\
    --save_dir="/Users/jongbeomkim/Documents/deeplabv3"\
    --batch_size=1\
    --n_cpus=1\
    --n_steps=20_000\
