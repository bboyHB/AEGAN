import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleconvO(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        return self.double_conv(x)


class Doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.deconv_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            Doubleconv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.deconv_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            Doubleconv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleAEGAN(nn.Module):
    def __init__(self, input_size=160, ae_level=3):
        super(SimpleAEGAN, self).__init__()
        # self.kernel_size = 3
        # self.padding = self.kernel_size // 2
        self.ae_level = ae_level
        self.init_channel_in = 3
        self.init_channel_out = 64

        self.ae_layers = [Doubleconv(self.init_channel_in, self.init_channel_out)]
        for l in range(self.ae_level):
            self.ae_layers.append(Down(self.init_channel_out * (2**l), self.init_channel_out * (2**l) * 2))
        for l in range(self.ae_level-1, -1, -1):
            self.ae_layers.append(Up(self.init_channel_out * (2**l) * 2, self.init_channel_out * (2**l)))
        self.ae_layers.extend([nn.Conv2d(self.init_channel_out, self.init_channel_in, kernel_size=1),
                               nn.Tanh()])
        self.ae = nn.Sequential(*self.ae_layers)

        self.discriminator_layers = [Doubleconv(self.init_channel_in, self.init_channel_out)]
        for l in range(self.ae_level):
            self.discriminator_layers.append(Down(self.init_channel_out * (2**l), self.init_channel_out * (2**l) * 2))
        self.discriminator_layers = self.discriminator_layers + [nn.Conv2d(self.init_channel_out * (2**self.ae_level), self.init_channel_out * (2**self.ae_level), kernel_size=1),
                                        nn.Flatten(),
                                        nn.Linear(self.init_channel_out * (2**self.ae_level) * ((input_size // (2**self.ae_level))**2), 1),
                                        nn.Sigmoid()]
        self.discriminator = nn.Sequential(*self.discriminator_layers)

    def forward(self, x, judge_only=False):
        if judge_only:
            prob = self.discriminator(x)
            return prob
        new_img = self.ae(x)
        prob = self.discriminator(new_img)
        return new_img, prob


if __name__ == '__main__':
    s = SimpleAEGAN()
    c = s.ae.__class__
    n = c.__name__
    # a = torch.randn((3, 3, 160, 160))
    # b, c = s(a)
    pass