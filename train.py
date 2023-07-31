import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp.grad_scaler import GradScaler
from pathlib import Path

from voc2012 import VOC2012Dataset
from model import DeepLabv3
from loss import DeepLabLoss


def get_lr(step, n_steps, power=0.9):
    lr = 1 - (step / n_steps) ** power
    return lr


# "A batch size of 16."
IMG_SIZE = 513
N_EPOCHS = 10
BATCH_SIZE = 16
# N_WORKERS = 4
N_WORKERS = 0
DATA_DIR = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012"
ROOT_DIR = Path(__file__).parent
LR = 0.001
WEIGHT_DECAY = 0.9997

model = DeepLabv3()
ds = VOC2012Dataset(root_dir=DATA_DIR)
train_dl = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True
)

crit = DeepLabLoss()
# optim = Adam(params=model.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)
optim = SGD(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# "We employ a 'poly' learning rate policy where the initial learning rate is multiplied
# by $1 - \frac{iter}{max_iter}^{power}$ with $power = 0.9$."
lr = get_lr(step=33, n_steps=700)

# "Since large batch size is required to train batch normalization parameters, we employ `output_stride=16`
# and compute the batch normalization statistics with a batch size of 16. The batch normalization parameters
# are trained with $decay = 0.9997$. After training on the 'trainaug' set with 30K iterations
# and $initial learning rate = 0.007$, we then freeze batch normalization parameters,
# employ `output_stride = 8`, and train on the official PASCAL VOC 2012 trainval set
# for another 30K iterations and smaller $base learning rate = 0.001$."

# "We decouple the DCNN and CRF training stages, assuming the DCNN unary terms are fixed when setting the CRF parameters."

model.train()
for epoch in range(1, N_EPOCHS + 1):
    for batch, (image, label) in enumerate(train_dl, start=1):
        step = len(train_dl) * (epoch - 1) + batch
        lr = get_lr(step=step, n_steps=N_EPOCHS * len(train_dl))
        optim.param_groups[0]["lr"] = lr

        optim.zero_grad()

        pred = model(image)
        pred = F.interpolate(pred, size=IMG_SIZE)
        
        loss = crit(pred=pred, gt=label)
        loss.backward()
        optim.step()

        if batch % 100 == 0:
            print(f"""{loss.item():.6f}""")

optim.param_groups[0].keys()
optim.param_groups[0]["lr"]