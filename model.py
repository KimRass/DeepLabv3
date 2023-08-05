# References:
    # https://github.com/PengtaoJiang/OAA-PyTorch/blob/master/deeplab-pytorch/libs/models/deeplabv3.py
    # https://gaussian37.github.io/vision-segmentation-deeplabv3/

import torch
import torch.nn as nn
import torch.nn.functional as F
import ssl
from torchvision.models import resnet101, ResNet101_Weights

ssl._create_default_https_context = ssl._create_unverified_context

# RESNET101 = resnet101(weights=ResNet101_Weights.DEFAULT)
RESNET101 = resnet101()


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.downsample = downsample

    def forward(self, x):
        skip = x.clone()
        if self.downsample is not None:
            skip = self.downsample(skip)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = x + skip
        x = F.relu(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, n_blocks):
        super().__init__()

        self.layers = list()
        self.layers.append(
            Bottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation,
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * 4),
                ),
            )
        )
        for _ in range(n_blocks - 1):
            self.layers.append(
                Bottleneck(in_channels=out_channels * 4, out_channels=out_channels, stride=1, dilation=1),
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


# "We define as `multi_grid = (r1, r2, r3)` the unit rates for the three convolutional layers within block4.
# The final atrous rate for the convolutional layer is equal to the multiplication of the unit rate
# and the corresponding rate."
class MultiGridResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, rate, multi_grid):
        super().__init__()

        self.layers = list()
        self.layers.append(
            Bottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=rate * multi_grid[0],
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * 4),
                ),
            )
        )
        for grid in multi_grid[1:]:
            self.layers.append(
                Bottleneck(
                    in_channels=out_channels * 4, out_channels=out_channels, stride=1, dilation=rate * grid
                ),
            )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResNet101Backbone(nn.Module):
    def __init__(self, output_stride, multi_grid=(1, 2, 4)):
        super().__init__()

        self.conv1 = RESNET101.conv1
        self.bn1 = RESNET101.bn1
        self.maxpool = RESNET101.maxpool
        self.block1 = RESNET101.layer1
        self.block2 = RESNET101.layer2

        if output_stride == 16:
            self.block3 = RESNET101.layer3
            self.block4 = MultiGridResNetBlock(
                in_channels=1024, out_channels=512, stride=1, rate=2, multi_grid=multi_grid,
            )
        elif output_stride == 8:
            # "When output `output_stride = 8`, the last two blocks ('block3' and 'block4')
            # in the original ResNet contains atrous convolution with `rate = 2` and `rate = 4` respectively."
            self.block3 = ResNetBlock(in_channels=512, out_channels=256, stride=1, dilation=2, n_blocks=23)
            self.block4 = MultiGridResNetBlock(
                in_channels=1024, out_channels=512, stride=1, rate=4, multi_grid=multi_grid,
            )

    def forward(self, x):
        x = self.conv1(x) # "'Conv1 + Pool1'"
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        x = self.block1(x) # "'Block1'"
        x = self.block2(x) # "'Block2'"
        x = self.block3(x) # "'Block3'"
        x = self.block4(x) # "'Block4'"
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # "All with 256 filters and batch normalization"
        self.conv = nn.Conv2d(
            in_channels,
            256,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            bias=False
        )
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ImagePooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv2d(2048, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x): # `(b, 64, h, w)`
        _, _, w, h = x.shape

        # "We apply global average pooling on the last feature map of the model, feed
        # the resulting image-level features to a 1×1 convolution with 256 filters
        # (and batch normalization), and then bilinearly upsample the feature to the desired spatial dimension.
        x = self.global_avg_pool(x) # `(b, 64, 1, 1)`
        x = self.conv(x) # `(b, 256, 1, 1)`
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=False) # `(b, 256, h, w)`
        return x


class ASPP(nn.Module):
    def __init__(self, atrous_rates):
        super().__init__()

        # "ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions
        # with `rates = (6, 12, 18)` when `output_stride = 16`, and (b) the image-level features.
        # "Four parallel atrous convolutions with different atrous rates are applied on top of the feature map."
        # "We include batch normalization within ASPP."
        self.conv_block1 = ConvBlock(in_channels=2048, kernel_size=1, dilation=1)
        self.conv_block2 = ConvBlock(in_channels=2048, kernel_size=3, dilation=atrous_rates[0])
        self.conv_block3 = ConvBlock(in_channels=2048, kernel_size=3, dilation=atrous_rates[1])
        self.conv_block4 = ConvBlock(in_channels=2048, kernel_size=3, dilation=atrous_rates[2])
        self.image_pooling = ImagePooling()
    
    def forward(self, x): # `(b, 64, h, w)`
        x1 = self.conv_block1(x) # `(b, 256, h, w)`
        x2 = self.conv_block2(x) # `(b, 256, h, w)`
        x3 = self.conv_block3(x) # `(b, 256, h, w)`
        x4 = self.conv_block4(x) # `(b, 256, h, w)`
        x5 = self.image_pooling(x) # `(b, 256, h, w)`
        # The resulting features from all the branches are then concatenated."
        x = torch.cat([x1, x2, x3, x4, x5], dim=1) # `(b, 256 * 5, h, w)`
        return x


class DeepLabv3ResNet101(nn.Module):
    def __init__(self, output_stride=16, n_classes=21):
        super().__init__()

        # "we apply atrous convolution with rates determined by the desired output stride value."
        # Note that the rates are doubled when `output_stride = 8`.
        if output_stride == 16:
            self.atrous_rates = (6, 12, 18)
        elif output_stride == 8:
            self.atrous_rates = (12, 24, 36)

        self.backbone = ResNet101Backbone(output_stride=output_stride)
        self.aspp = ASPP(atrous_rates=self.atrous_rates)
        # "Pass through another 1×1 convolution (also with 256 filters and batch normalization)
        # before the final 1×1 convolution which generates the final logits."
        self.conv_block = ConvBlock(in_channels=1280, kernel_size=1, dilation=1)
        self.fin_conv = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape

        x = self.backbone(x)
        x = self.aspp(x)
        x = self.conv_block(x)
        x = self.fin_conv(x)

        x = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 513, 513)
    model = DeepLabv3ResNet101(output_stride=16)
    # model = DeepLabv3ResNet101(output_stride=8)
    pred = model(x)
    print(pred.shape)
