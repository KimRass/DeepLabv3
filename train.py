from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from voc2012 import VOC2012Dataset


root_dir = "/Users/jongbeomkim/Documents/datasets/voc2012/VOCdevkit/VOC2012"
ds = VOC2012Dataset(root_dir=root_dir)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
image, label = next(iter(dl))