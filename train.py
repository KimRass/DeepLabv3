import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path
import random

from voc2012 import VOC2012Dataset
from model import DeepLabv3
from loss import DeepLabLoss
from evaluate import PixelmIoU


def get_lr(step, n_steps, power=0.9):
    # "We employ a 'poly' learning rate policy where the initial learning rate is multiplied
    # by $1 - \frac{iter}{max_iter}^{power}$ with $power = 0.9$."
    lr = 1 - (step / n_steps) ** power
    return lr


# "A batch size of 16."
IMG_SIZE = 513
N_EPOCHS = 50
BATCH_SIZE = 16
# N_WORKERS = 4
N_WORKERS = 0
DATA_DIR = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012"
ROOT_DIR = Path(__file__).parent
LR = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

model = DeepLabv3()
ds = VOC2012Dataset(root_dir=DATA_DIR)
val_size = round(len(ds) * 0.05)
train_size = len(ds) - val_size
train_ds, val_ds = random_split(ds, lengths=(train_size, val_size))
train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True
)
train_di = iter(train_dl)

val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True
)
val_di = iter(val_dl)

crit = DeepLabLoss()
# optim = SGD(params=model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
optim = SGD(params=model.parameters(), lr=LR)

# "Since large batch size is required to train batch normalization parameters, we employ `output_stride=16`
# and compute the batch normalization statistics with a batch size of 16. The batch normalization parameters
# are trained with $decay = 0.9997$. After training on the 'trainaug' set with 30K iterations
# and $initial learning rate = 0.007$, we then freeze batch normalization parameters,
# employ `output_stride = 8`, and train on the official PASCAL VOC 2012 trainval set
# for another 30K iterations and smaller $base learning rate = 0.001$."

# "We decouple the DCNN and CRF training stages, assuming the DCNN unary terms are fixed when setting the CRF parameters."

metric = PixelmIoU()

N_STEPS = 30000
for step in range(1, N_STEPS + 1):
    try:
        image, gt = next(train_di)
    except StopIteration:
        train_di = iter(train_dl)
        image, gt = next(train_di)

    lr = get_lr(step=step, n_steps=N_STEPS)
    optim.param_groups[0]["lr"] = lr

    optim.zero_grad()

    pred = model(image)
    
    loss = crit(pred=pred, gt=gt)
    loss.backward()
    optim.step()

    # if step % 100 == 0:
    print(f"""Loss: {loss.item():.6f}""")

    ### Evaluate.
    # if step % 100 == 0:
    # model.eval()
    for _ in range(3):
        try:
            image, gt = next(val_di)
            pred = model(image)
            miou = metric(pred=pred, gt=gt)
            print(f"""mIoU: {miou:.6f}""")
        except StopIteration:
            val_di = iter(val_dl)
            image, gt = next(val_di)

            pred = model(image)
            pred = F.interpolate(pred, size=IMG_SIZE)

    # model.train()
