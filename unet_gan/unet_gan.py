""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
import torch
from .unet_gan_parts import *


class UNetGenerator(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNetGenerator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.sigmoid(logits)
        return out


class UNetDiscriminator(nn.Module):
    def __init__(self, n_channels=6, n_classes=1, bilinear=True):
        super(UNetDiscriminator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.outc = OutConv(1024 // factor, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.pool(x)
        x = self.outc(x)
        out = self.sigmoid(x)
        return out


class UNetGAN(nn.Module):
    def __init__(self, pretrained_unet_G=None):
        super(UNetGAN, self).__init__()
        self.netG = pretrained_unet_G
        if self.netG == None:
            self.netG = UNetGenerator()
        self.netD = UNetDiscriminator()

    def forward(self, x, train=True, judge_img=None):
        img_G = self.netG(x)
        if train:
            if judge_img is None:
                pair = torch.cat((img_G, x), dim=1)
            else:
                pair = torch.cat((judge_img, x), dim=1)
            fake_or_true = self.netD(pair)
            return img_G, fake_or_true
        return img_G



if __name__ == '__main__':
    u = UNetGenerator().cuda()
    t = torch.rand((1, 3, 1024, 768)).cuda()
    i = u(t)
    print(i.shape)
    pass