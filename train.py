from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp.grad_scaler import GradScaler

from voc2012 import VOC2012Dataset
from model import DeepLabv3
from loss import DeepLabLoss

BATCH_SIZE = 4
# N_WORKERS = 4
N_WORKERS = 0
DATA_DIR = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012"
ROOT_DIR = Path(__file__).parent

model = DeepLabv3()
ds = VOC2012Dataset(root_dir=DATA_DIR)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, pin_memory=True, drop_last=True)
image, label = next(iter(dl))
# label.permute(0, 2, 3, 1).view(-1).unique()

crit = DeepLabLoss()
optim = Adam(params=model.parameters(), lr=LR, betas=(BETA1, BETA2), eps=EPS)