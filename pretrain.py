import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from config import config
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from unet_gan.unet_gan import UNetGenerator
import cv2
from model import AutoEncoder
import numpy as np
from skimage.feature import local_binary_pattern
from datetime import datetime
from FocalLoss2d import FocalLossG

def pretrain(path=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if path is not None:
        if path == '':
            return UNetGenerator(bilinear=False).to(device)
        return torch.load(path, map_location=device)
    ae = UNetGenerator(bilinear=False).to(device)
    criterion = nn.L1Loss()
    optimizer = SGD(ae.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)
    step_lr = StepLR(optimizer, config.lr_step, config.lr_decay_gamma)

    img_path = 'C:/Users/DeepLearning/Desktop/11603/FALSE'
    stamp = 'UNET_pretrain_e' + str(config.epoch) + '_lr' + str(config.lr)
    time_stamp = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='aelog/' + stamp + '_' + time_stamp)

    glbstep = 0
    for e in range(config.epoch):
        iter_per_epoch = len(os.listdir(img_path))
        for index, img_name in enumerate(os.listdir(img_path)):
            print(e, index)
            img = Image.open(os.path.join(img_path, img_name))
            img = img.resize((512, 384))
            img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

            output = ae(img_tensor)
            loss = criterion(output, img_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = iter_per_epoch * e + index
            writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=iters)
            if iters % 100 == 0:
                writer.add_image('img/input', img_tensor.squeeze(), glbstep)
                writer.add_image('img/output', output.squeeze(), glbstep)
                input_numpy = img_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
                input_gray = cv2.cvtColor(input_numpy, cv2.COLOR_RGB2GRAY)
                input_LBP = local_binary_pattern(input_gray, P=8, R=1)
                output_numpy = output.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                output_gray = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2GRAY)
                output_LBP = local_binary_pattern(output_gray, P=8, R=1)
                # writer.add_image('diff/LBP', np.abs(input_LBP - output_LBP), glbstep, dataformats='HW')
                bin_img = np.array(np.abs(input_LBP - output_LBP) > 230, dtype=np.uint8)
                writer.add_image('diff/LBP_bin', bin_img * 255, glbstep, dataformats='HW')
                filtered_img = extract_flaws(bin_img)
                writer.add_image('diff/LBP_filter_bin', filtered_img * 255, glbstep, dataformats='HW')
                glbstep += 1
        step_lr.step()
    torch.save(ae, stamp + time_stamp + '.pth')
    return ae


def extract_flaws(bin_img):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=4)
    filtered_img = np.zeros_like(bin_img)
    for i in range(retval):
        if i == 0:
            continue
        if stats[i][4] > 100:
            filtered_img[labels == i] = 1
    return filtered_img


def fine_tune(model_path=None):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ae = pretrain(model_path)
    criterion = nn.L1Loss()

    optimizer = SGD(ae.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)
    step_lr = StepLR(optimizer, config.lr_step, config.lr_decay_gamma)

    data_path = 'paste_good'
    stamp = 'UNET_funetune_e' + str(config.epoch) + '_lr' + str(config.lr)
    time_stamp = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='aelog/' + stamp + '_' + time_stamp)

    glbstep = 0
    for e in range(config.epoch):
        if e > 0:
            criterion = FocalLossG(2)
        classes = os.listdir(data_path)
        for cls in classes:
            origin_img_path = os.path.join(data_path, cls, 'origin')
            pasted_img_path = os.path.join(data_path, cls, 'pasted')
            for index, img_name in enumerate(os.listdir(origin_img_path)):
                print(e, cls, index)
                origin_img = Image.open(os.path.join(origin_img_path, img_name))
                pasted_img = Image.open(os.path.join(pasted_img_path, img_name))
                origin_img = origin_img.resize((512, 384))
                pasted_img = pasted_img.resize((512, 384))  # (768, 576)
                origin_img_tensor = F.to_tensor(origin_img).unsqueeze(0).to(device)
                pasted_img_tensor = F.to_tensor(pasted_img).unsqueeze(0).to(device)

                output = ae(origin_img_tensor)
                loss = criterion(output, pasted_img_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                glbstep += 1
                writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=glbstep)
                if glbstep % 100 == 0:
                    writer.add_image('img/input', origin_img_tensor.squeeze(), glbstep)
                    writer.add_image('img/output', output.squeeze(), glbstep)
                    writer.add_image('img/pasted', pasted_img_tensor.squeeze(), glbstep)
                    input_numpy = origin_img_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
                    input_gray = cv2.cvtColor(input_numpy, cv2.COLOR_RGB2GRAY)
                    input_LBP = local_binary_pattern(input_gray, P=8, R=1)
                    output_numpy = output.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                    output_gray = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2GRAY)
                    output_LBP = local_binary_pattern(output_gray, P=8, R=1)
                    # writer.add_image('diff/LBP', np.abs(input_LBP - output_LBP), glbstep, dataformats='HW')
                    bin_img = np.array(np.abs(input_LBP - output_LBP) > 230, dtype=np.uint8)
                    writer.add_image('diff/LBP_bin', bin_img * 255, glbstep, dataformats='HW')
                    filtered_img = extract_flaws(bin_img)
                    writer.add_image('diff/LBP_filter_bin', filtered_img * 255, glbstep, dataformats='HW')
        step_lr.step()
    torch.save(ae, stamp + time_stamp + '.pth')
    return ae

if __name__ == '__main__':
    fine_tune('')
    # tensorboard --logdir=aelog