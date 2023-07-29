# "We apply data augmentation by randomly scaling the input images (from 0.5 to 2.0) and randomly left-right flipping during training."

# "We employ crop size to be $513$ during both training and test on PASCAL VOC 2012 dataset."

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import xml.etree.ElementTree as et
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import extcolors
import random

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

IMG_SIZE = 448


class VOC2012Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()

        self.labels = list((Path(root_dir)/"SegmentationClass").glob("*.png"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_path = labels[idx]
        label = Image.open(label_path)
        label = TF.pil_to_tensor(label)
        label = torch.where(label == 255, 0, label)
        label.unique()

        img_path = f"""{label_path.parent.parent/"JPEGImages"/label_path.stem}.jpg"""
        image = Image.open(img_path).convert("RGB")
