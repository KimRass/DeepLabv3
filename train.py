# References:
    # https://github.com/PengtaoJiang/OAA-PyTorch/blob/master/deeplab-pytorch/libs/utils/lr_scheduler.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path
from time import time

from voc2012 import VOC2012Dataset
from model import DeepLabv3ResNet101
from loss import DeepLabLoss
from evaluate import PixelmIoU
from utils import get_device, get_elapsed_time, save_checkpoint

# "We decouple the DCNN and CRF training stages, assuming the DCNN unary terms are fixed
# when setting the CRF parameters."


def get_lr(init_lr, step, n_steps, power=0.9):
    # "We employ a 'poly' learning rate policy where the initial learning rate is multiplied
    # by $(1 - \frac{iter}{max{\_}iter})^{power}$ with $power = 0.9$."
    lr = init_lr * (1 - (step / n_steps)) ** power
    return lr


def evaluate(val_dl, model, metric):
    with torch.no_grad():
        sum_miou = 0
        for batch, (image, gt) in enumerate(val_dl, start=1):
            image = image.to(DEVICE)
            gt = gt.to(DEVICE)

            pred = model(image)
            miou = metric(pred=pred, gt=gt)

            sum_miou += miou
        avg_miou = sum_miou / batch
    return avg_miou


# "Since large batch size is required to train batch normalization parameters, we employ `output_stride=16`
# and compute the batch normalization statistics with a batch size of 16. The batch normalization parameters
# are trained with $decay = 0.9997$. After training on the 'trainaug' set with 30K iterations
# and $initial learning rate = 0.007$, we then freeze batch normalization parameters,
# employ `output_stride = 8`, and train on the official PASCAL VOC 2012 trainval set
# for another 30K iterations and smaller $base learning rate = 0.001$."
IMG_SIZE = 513
BATCH_SIZE = 16
N_WORKERS = 4
# IMG_DIR = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
# GT_DIR = "/Users/jongbeomkim/Documents/datasets/SegmentationClassAug"
IMG_DIR = "/home/user/cv/voc2012/VOCdevkit/VOC2012/JPEGImages"
GT_DIR = "/home/user/cv/SegmentationClassAug"
INIT_LR = 0.007
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.0005

DEVICE = get_device()
model = DeepLabv3ResNet101(output_stride=16).to(DEVICE)
model = nn.DataParallel(model, output_device=0)
# optim = SGD(params=model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
optim = SGD(params=model.parameters(), lr=INIT_LR)
scaler = GradScaler()

train_ds = VOC2012Dataset(img_dir=IMG_DIR, gt_dir=GT_DIR, split="train")
train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True
)
train_di = iter(train_dl)

val_ds = VOC2012Dataset(img_dir=IMG_DIR, gt_dir=GT_DIR, split="val")
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=N_WORKERS)

crit = DeepLabLoss()
metric = PixelmIoU()

### Train.
# N_STEPS = 300_000 # In the paper
N_STEPS = 600_000 # In my case
N_PRINT_STEPS = 500
N_CKPT_STEPS = 5000
N_EVAL_STEPS = 3000
running_loss = 0
start_time = time()
for step in range(1, N_STEPS + 1):
    model.train()

    try:
        image, gt = next(train_di)
    except StopIteration:
        train_di = iter(train_dl)
        image, gt = next(train_di)
    image = image.to(DEVICE)
    gt = gt.to(DEVICE)

    lr = get_lr(init_lr=INIT_LR, step=step, n_steps=N_STEPS)
    optim.param_groups[0]["lr"] = lr

    optim.zero_grad()

    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
        pred = model(image)
    
    loss = crit(pred=pred, gt=gt)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    running_loss += loss.item()

    if step % N_PRINT_STEPS == 0:
        running_loss /= N_PRINT_STEPS
        print(f"""[ {step:,}/{N_STEPS:,} ][ {lr:4f} ][ {get_elapsed_time(start_time)} ]""", end="")
        print(f"""[ Loss: {running_loss:.4f} ]""")
        running_loss = 0

        start_time = time()

    if step % N_CKPT_STEPS == 0:
        save_checkpoint(
            step=step,
            n_steps=N_STEPS,
            model=model,
            optim=optim,
            save_path=Path(__file__).parent/f"""checkpoints/{step}.pth""",
        )

    ### Evaluate.
    if step % N_EVAL_STEPS == 0:
        start_time = time()

        model.eval()
        avg_miou = evaluate(val_dl=val_dl, model=model, metric=metric)
        print(f"""[ {step:,}/{N_STEPS:,} ][ {lr:4f} ][ {get_elapsed_time(start_time)} ]""", end="")
        print(f"""[ Average mIoU: {avg_miou:.4f} ]""")
