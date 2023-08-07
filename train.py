# References:
    # https://github.com/PengtaoJiang/OAA-PyTorch/blob/master/deeplab-pytorch/libs/utils/lr_scheduler.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path
from time import time

import config
from voc2012 import VOC2012Dataset
from model import DeepLabv3ResNet101
from loss import DeepLabLoss
from evaluate import PixelmIoU
from utils import get_elapsed_time, save_checkpoint

# "We decouple the DCNN and CRF training stages, assuming the DCNN unary terms are fixed
# when setting the CRF parameters."


def get_lr(init_lr, step, n_steps, power=0.9):
    # "We employ a 'poly' learning rate policy where the initial learning rate is multiplied
    # by $(1 - \frac{iter}{max{\_}iter})^{power}$ with $power = 0.9$."
    lr = init_lr * (1 - (step / n_steps)) ** power
    return lr


def validate(val_dl, model, metric):
    with torch.no_grad():
        sum_miou = 0
        for batch, (image, gt) in enumerate(val_dl, start=1):
            image = image.to(config.DEVICE)
            gt = gt.to(config.DEVICE)

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

model = DeepLabv3ResNet101(output_stride=16).to(config.DEVICE)
if config.MULTI_GPU:
    print(f"""Using {torch.cuda.device_count()} GPU(s).""")
    model = nn.DataParallel(model)

optim = SGD(
    params=model.parameters(),
    lr=config.INIT_LR,
    momentum=config.MOMENTUM,
    weight_decay=config.WEIGHT_DECAY,
)

scaler = GradScaler()

train_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="train")
train_dl = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.N_WORKERS,
    pin_memory=True,
    drop_last=True,
)
train_di = iter(train_dl)

val_ds = VOC2012Dataset(img_dir=config.IMG_DIR, gt_dir=config.GT_DIR, split="val")
val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.N_WORKERS)

crit = DeepLabLoss()
metric = PixelmIoU()

# Resume from checkpoint.
if config.CKPT_PATH is not None:
    ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
    init_step = ckpt["step"]
    n_steps = ckpt["number_of_steps"]
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
else:
    init_step = 0
    n_steps = config.N_STEPS

### Train.
running_loss = 0
start_time = time()
for step in range(init_step + 1, n_steps + 1):
    model.train()

    try:
        image, gt = next(train_di)
    except StopIteration:
        train_di = iter(train_dl)
        image, gt = next(train_di)
    image = image.to(config.DEVICE)
    gt = gt.to(config.DEVICE)

    lr = get_lr(init_lr=config.INIT_LR, step=step, n_steps=n_steps)
    optim.param_groups[0]["lr"] = lr

    # with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16):
    pred = model(image)

    optim.zero_grad()
    loss = crit(pred=pred, gt=gt)
    # scaler.scale(loss).backward()
    # scaler.step(optim)
    # scaler.update()
    loss.backward()
    optim.step()

    running_loss += loss.item()

    if step % config.N_PRINT_STEPS == 0:
        running_loss /= config.N_PRINT_STEPS
        print(f"""[ {step:,}/{n_steps:,} ][ {lr:4f} ][ {get_elapsed_time(start_time)} ]""", end="")
        print(f"""[ Loss: {running_loss:.4f} ]""")
        running_loss = 0

        start_time = time()

    if step % config.N_CKPT_STEPS == 0:
        save_checkpoint(
            step=step,
            n_steps=n_steps,
            model=model,
            optim=optim,
            save_path=Path(__file__).parent/f"""checkpoints/{step}.pth""",
        )
        print(f"""Saved checkpoint at step {step:,}/{n_steps:,}.""")

    ### Validate.
    if step % config.N_EVAL_STEPS == 0:
        start_time = time()

        model.eval()
        avg_miou = validate(val_dl=val_dl, model=model, metric=metric)
        print(f"""[ {step:,}/{n_steps:,} ][ {lr:4f} ][ {get_elapsed_time(start_time)} ]""", end="")
        print(f"""[ Average mIoU: {avg_miou:.4f} ]""")
