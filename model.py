# References:
    # https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
    # https://github.com/giovanniguidi/deeplabV3-PyTorch/tree/master/models

import torch
import torch.nn as nn
import torch.nn.functional as F

# "There are three $3 \times 3$ convolutions in those blocks, and the last convolution contains
# stride $2$ except the one in last block, similar to original ResNet."

# "we apply atrous convolution with rates determined by the desired output stride value."

# "We employ crop size to be $513$ during both training and test on PASCAL VOC 2012 dataset."
from torchvision.models import resnet50

model = resnet50()
model

# x = torch.randn(2, 3, 256, 224)
# # 'Conv1 + Pool1'
# x = model.conv1(x)
# x = model.bn1(x)
# x = model.relu(x)
# x = model.maxpool(x) # output_stride = 4
# x = model.layer1(x) # output_stride = 4

# x = model.layer2(x) # output_stride = 8 # 'Block1'
# x = model.layer3(x) # output_stride = 16 # 'Block2'
# x = model.layer4(x) # output_stride = 32 # 'Block3'

x = torch.randn(2, 3, 256, 224)
# 'Conv1 + Pool1'
x = model.conv1(x)
x = model.bn1(x)
x = model.relu(x)
x = model.maxpool(x)
x = model.layer1(x) # output_stride = 4
x = model.layer2(x) # output_stride = 8 # 'Block1'
x = model.layer3(x) # output_stride = 16 # 'Block2'



class ASPPConv(nn.Module):
    def __init__(self, in_ch, kernel_size, stride=1, dilation=1, out_ch=256):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding="same", bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPool(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(
            64, out_ch, kernel_size=1, stride=1, padding="same", bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        _, _, w, h = x.shape

        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=False)
        return x


class ASPP(nn.Module):
    def __init__(self, atrous_rates):
        super().__init__()

        self.aspp_conv1 = ASPPConv(in_ch=64, kernel_size=1)
        self.aspp_conv2 = ASPPConv(in_ch=64, kernel_size=3, dilation=atrous_rates[0])
        self.aspp_conv3 = ASPPConv(in_ch=64, kernel_size=3, dilation=atrous_rates[1])
        self.aspp_conv4 = ASPPConv(in_ch=64, kernel_size=3, dilation=atrous_rates[2])
        self.aspp_pool = ASPPPool()
    
    def forward(self, x):
        # x = torch.randn(8, 64, 720, 480)

        x1 = self.aspp_conv1(x)
        x2 = self.aspp_conv2(x)
        x3 = self.aspp_conv3(x)
        x4 = self.aspp_conv4(x)
        x5 = self.aspp_pool(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return x


class DeepLabv3(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        n_classes = 1_000
        output_stride = 16
        if output_stride == 16:
            atrous_rates = (6, 12, 18)
        
        aspp = ASPP(atrous_rates=atrous_rates)
        aspp_conv = ASPPConv(in_ch=1280, kernel_size=1)
        conv = nn.Conv2d(256, n_classes, kernel_size=1)
    
    def forward(self, x):
        x = torch.randn(8, 64, 720, 480)
        x = aspp(x)
        x = aspp_conv(x)
        x = conv(x)
        x.shape


# class Block(nn.Module):
#     def __init__(self, in_ch, out_ch, rate, last_stride, multi_grid=(1, 2, 4)):
#         super().__init__()
#         rate = 8
#         multi_grid=(1, 2, 4)

#         self.last_stride = last_stride
#         self.multi_grid = multi_grid

#         nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=rate * multi_grid[0])
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=rate * multi_grid[1])
#         nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=last_stride, dilation=rate * multi_grid[2])
    

    # def forward(self, x):


