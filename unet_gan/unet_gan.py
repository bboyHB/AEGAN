""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
import torch
from unet_gan_parts import *


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
        self.tanh = nn.Tanh()

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
        out = self.tanh(logits)
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
    def __init__(self, pretrained_unet_G=None, bilinear=True):
        super(UNetGAN, self).__init__()
        self.netG = pretrained_unet_G
        if self.netG is None:
            self.netG = UNetGenerator(bilinear=bilinear)
        self.netD = UNetDiscriminator(bilinear=bilinear)

    def forward(self, x, gt_img=None):
        img_G = self.netG(x)
        if self.training:
            assert gt_img is not None, 'no gt img'
            pair_fake = torch.cat((img_G, x), dim=1)
            pair_real = torch.cat((gt_img, x), dim=1)
            judge_fake = self.netD(pair_fake)
            judge_real = self.netD(pair_real)
            return img_G, judge_fake, judge_real
        else:
            assert gt_img is None, 'no need for gt img'
            return img_G



if __name__ == '__main__':
    ug = UNetGAN(bilinear=False)
    t = torch.rand((1, 3, 512, 386))
    ug.eval()
    i = ug(t, t)
    # u = UNetGenerator().cuda()
    # t = torch.rand((1, 3, 1024, 768)).cuda()
    # i = u(t)
    # print(i.shape)
    pass