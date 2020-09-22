import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image
from torch.optim import SGD
from config import config
import os
from torch.utils.tensorboard import SummaryWriter

from unet_gan.unet_gan import UNetGAN
from model import AutoEncoder

ae = UNetGAN()
criterion = nn.BCELoss()
optimizer = SGD(ae.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)

img_path = '/home/xmu_stu/chb/weiya_detectron2/11603/FALSE'
writer = SummaryWriter()

for e in range(10):
    for index, img_name in enumerate(os.listdir(img_path)):
        print(e, index)
        img = Image.open(os.path.join(img_path, img_name))
        img = img.resize((1024, 768))
        img_tensor = F.to_tensor(img).unsqueeze(0)

        output = ae(img_tensor)
        loss = criterion(output, img_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            writer.add_image('output', output.squeeze())
