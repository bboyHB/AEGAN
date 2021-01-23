import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import functional as F

from config import config
from FocalLoss2d import FocalLossG
from unet_gan.unet_gan import UNetGenerator, UNetDiscriminator
from defect_generate_gan.dg_gan import add_defect


trans2tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
iters_per_epoch = 1000
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def Unet_repair_autodata():
    G = UNetGenerator().to(device)
    D = UNetDiscriminator().to(device)
    criterion_G = nn.MSELoss()
    criterion_D = nn.BCELoss()

    optimizer_G = SGD(G.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)
    optimizer_D = SGD(D.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)
    step_lr_G = StepLR(optimizer_G, config.lr_step, config.lr_decay_gamma)
    step_lr_D = StepLR(optimizer_D, config.lr_step, config.lr_decay_gamma)

    data_path = 'paste_good'
    stamp = 'UNET_repair_e' + str(config.epoch) + '_lr' + str(config.lr)
    time_stamp = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='AEGAN_log/' + stamp + '_' + time_stamp)

    for e in range(config.epoch):
        # if e > 0:
        #     criterion = FocalLossG(2)
        for i in tqdm(range(iters_per_epoch)):
            bg_imgs = os.listdir('normal_sample')
            img_origin = Image.open(os.path.join('normal_sample', bg_imgs[random.randint(0, len(bg_imgs) - 1)])).convert('RGB')
            img_numpy = np.array(img_origin)
            img_origin_tensor = trans2tensor(img_origin).unsqueeze(0).to(device)
            img_defect = Image.fromarray(add_defect(img_numpy)).convert('RGB')
            img_defect_tensor = trans2tensor(img_defect).unsqueeze(0).to(device)
            # 添加高斯噪声
            noise_img_defect_tensor = add_noise(img_defect_tensor)

            # hard labels
            gt_recon_hard = torch.ones((img_origin_tensor.shape[0], 1)).to(device)
            gt_real_hard = torch.zeros((img_origin_tensor.shape[0], 1)).to(device)
            target_recon_hard = torch.zeros((img_origin_tensor.shape[0], 1)).to(device)

            # soft labels
            gt_recon_soft = torch.tensor(np.random.uniform(0.7, 1.0, (img_origin_tensor.shape[0], 1)), dtype=torch.float).to(device)
            gt_real_soft = torch.tensor(np.random.uniform(0.0, 0.3, (img_origin_tensor.shape[0], 1)), dtype=torch.float).to(device)
            target_recon_soft = torch.tensor(np.random.uniform(0.0, 0.3, (img_origin_tensor.shape[0], 1)), dtype=torch.float).to(device)

            img_G = G(noise_img_defect_tensor)
            recon_loss = criterion_G(img_G, img_origin_tensor)
            optimizer_G.zero_grad()
            recon_loss.backward()
            optimizer_G.step()

            pair_fake = torch.cat((img_G, noise_img_defect_tensor), dim=1)
            judge_fake = D(pair_fake)
            lie_loss = criterion_D(judge_fake.squeeze(0).squeeze(0), target_recon_soft)
            optimizer_G.zero_grad()
            lie_loss.backward()
            optimizer_G.step()

            pair_fake = torch.cat((img_G.detach(), noise_img_defect_tensor), dim=1)
            judge_fake = D(pair_fake)
            detect_loss = criterion_D(judge_fake, gt_recon_soft)
            optimizer_D.zero_grad()
            detect_loss.backward()
            optimizer_D.step()

            pair_real = torch.cat((img_origin_tensor, noise_img_defect_tensor), dim=1)
            judge_real = D(pair_real)
            real_loss = criterion_D(judge_real, gt_real_soft)
            optimizer_D.zero_grad()
            real_loss.backward()
            optimizer_D.step()

            glbstep = e * iters_per_epoch + i

            if glbstep % 100 == 0:
                writer.add_image('img/input', noise_img_defect_tensor.squeeze(), glbstep)
                writer.add_image('img/output', img_G.squeeze(), glbstep)
                writer.add_image('img/origin', img_origin_tensor.squeeze(), glbstep)
                # input_numpy = origin_img_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
                # input_gray = cv2.cvtColor(input_numpy, cv2.COLOR_RGB2GRAY)
                # input_LBP = local_binary_pattern(input_gray, P=8, R=1)
                # output_numpy = output.squeeze().cpu().detach().numpy().transpose((1, 2, 0))
                # output_gray = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2GRAY)
                # output_LBP = local_binary_pattern(output_gray, P=8, R=1)
                # writer.add_image('diff/LBP', np.abs(input_LBP - output_LBP), glbstep, dataformats='HW')
                # bin_img = np.array(np.abs(input_LBP - output_LBP) > 230, dtype=np.uint8)
                # writer.add_image('diff/LBP_bin', bin_img * 255, glbstep, dataformats='HW')
                # filtered_img = extract_flaws(bin_img)
                # writer.add_image('diff/LBP_filter_bin', filtered_img * 255, glbstep, dataformats='HW')
        step_lr_G.step()
        step_lr_D.step()
    torch.save(ug, stamp + time_stamp + '.pth')


def Unet_repair_data():
    ug = UNetGAN(bilinear=True).to(device)
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()

    optimizer1 = SGD(ug.netG.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)
    optimizer2 = SGD(ug.netD.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-8)
    step_lr1 = StepLR(optimizer1, config.lr_step, config.lr_decay_gamma)
    step_lr2 = StepLR(optimizer2, config.lr_step, config.lr_decay_gamma)

    data_path = 'paste_good'
    stamp = 'UNET_repair_e' + str(config.epoch) + '_lr' + str(config.lr)
    time_stamp = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(log_dir='AEGAN_log/' + stamp + '_' + time_stamp)

    glbstep = 0
    for e in range(config.epoch):
        # if e > 0:
        #     criterion = FocalLossG(2)
        classes = os.listdir(data_path)
        for cls in classes:
            origin_img_path = os.path.join(data_path, cls, 'origin')
            pasted_img_path = os.path.join(data_path, cls, 'pasted')
            for index, img_name in enumerate(os.listdir(origin_img_path)):
                print(e, cls, index)

                origin_img = Image.open(os.path.join(origin_img_path, img_name))
                pasted_img = Image.open(os.path.join(pasted_img_path, img_name))
                origin_img = origin_img.resize((768, 576))
                pasted_img = pasted_img.resize((768, 576))  # (768, 576)
                origin_img_tensor = F.to_tensor(origin_img).unsqueeze(0).to(device)
                pasted_img_tensor = F.to_tensor(pasted_img).unsqueeze(0).to(device)

                output = ug(origin_img_tensor)
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
    torch.save(ug, stamp + time_stamp + '.pth')
    return ae


def add_noise(img, alpha=0.01):
    noise_img = img + (alpha ** 0.5) * torch.randn(img.shape).to(device)
    return noise_img


if __name__ == '__main__':
    Unet_repair_autodata()