import torch
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import random
from torchvision import transforms
import numpy as np

from simple_aegan import SimpleAEGAN



input_size = 160
ae_level = 2
trans2tensor = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
lr = 0.001
epoch = 7
show_interval = 100
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train():
    data_path = 'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/commom1'
    flaw_path = [
        'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/flaw01',
        'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/flaw02',
        'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/flaw05',
        'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/flaw07',
        'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/other',
        'C:/Users/DeepLearning/Desktop/flaw_detect/img/test/watermark'
    ]
    time_stamp = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    model = SimpleAEGAN(input_size=input_size, ae_level=ae_level).to(device)
    optim_G = torch.optim.Adam(model.ae.parameters(), lr=lr)
    optim_D = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, 3, 0.1)
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, 3, 0.1)
    criterion_G = torch.nn.MSELoss()
    criterion_D = torch.nn.BCELoss()
    writer = SummaryWriter(log_dir='runs/aegan'+time_stamp)

    for e in range(epoch):
        img_paths = os.listdir(data_path)

        # Warm up Train 暖身训练 让lr从一个很小的值线性增长到初始设定的lr
        lr_warmup_G = None
        lr_warmup_D = None
        if e == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(img_paths) - 1)
            lr_warmup_G = warmup_lr_scheduler(optim_G, warmup_iters, warmup_factor)
            lr_warmup_D = warmup_lr_scheduler(optim_D, warmup_iters, warmup_factor)
        losses = {'recon_loss': [],
                  'lie_loss': [],
                  'detect_loss': [],
                  'real_loss': []}
        for index, img_path in tqdm(enumerate(img_paths)):
            img = Image.open(os.path.join(data_path, img_path)).convert("RGB")
            img_tensor = trans2tensor(img).unsqueeze(0).to(device)
            # 添加高斯噪声
            noise_img_tensor = add_noise(img_tensor)

            # hard labels
            gt_recon_hard = torch.ones((img_tensor.shape[0], 1)).to(device)
            gt_real_hard = torch.zeros((img_tensor.shape[0], 1)).to(device)
            target_recon_hard = torch.zeros((img_tensor.shape[0], 1)).to(device)

            # soft labels
            gt_recon_soft = torch.tensor(np.random.uniform(0.7, 1.0, (img_tensor.shape[0], 1)), dtype=torch.float).to(device)
            gt_real_soft = torch.tensor(np.random.uniform(0.0, 0.3, (img_tensor.shape[0], 1)), dtype=torch.float).to(device)
            target_recon_soft = torch.tensor(np.random.uniform(0.0, 0.3, (img_tensor.shape[0], 1)), dtype=torch.float).to(device)

            global_step = e * len(img_paths) + index

            # 训练判别器
            # for repeat in range(2):
            real_prob = model(img_tensor, True)
            real_loss = criterion_D(real_prob, gt_real_hard)
            loss_D = real_loss
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            _, recon_prob2 = model(noise_img_tensor)
            detect_loss = criterion_D(recon_prob2, gt_recon_soft)
            loss_D = detect_loss
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            # 训练AE生成器
            if global_step % 50 == 0:
                recon_img, _ = model(noise_img_tensor)
                recon_loss = criterion_G(recon_img, img_tensor)
                loss_G = recon_loss
                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()

                _, recon_prob = model(noise_img_tensor)
                lie_loss = criterion_D(recon_prob, target_recon_soft)
                loss_G = lie_loss
                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()


            if lr_warmup_G is not None:
                lr_warmup_G.step()
            if lr_warmup_D is not None:
                lr_warmup_D.step()

            # 输出内容到tensorboard观察训练情况
            losses['recon_loss'].append(recon_loss)
            losses['lie_loss'].append(lie_loss)
            losses['detect_loss'].append(detect_loss)
            losses['real_loss'].append(real_loss)

            if global_step % show_interval == 0 and global_step != 0:
                recon_img_nonoise, recon_prob_nonoise = model(img_tensor.detach())
                writer.add_scalars('loss', {'recon_loss': sum(losses['recon_loss']) / show_interval,
                                                'lie_loss': sum(losses['lie_loss']) / show_interval,
                                                'detect_loss': sum(losses['detect_loss']) / show_interval,
                                                'real_loss': sum(losses['real_loss']) / show_interval}, global_step // show_interval)
                # writer.add_scalar('loss_lie_recon', lie_loss / recon_loss, global_step // show_interval)
                # writer.add_scalar('loss_real_detect', real_loss / detect_loss, global_step // show_interval)
                writer.add_image('common/origin', img_tensor.squeeze(0)/2+0.5, global_step // show_interval)
                writer.add_image('common/noise', noise_img_tensor.squeeze(0) / 2 + 0.5, global_step // show_interval)
                writer.add_image('common/recon', recon_img.squeeze(0)/2+0.5, global_step // show_interval)
                writer.add_image('common/recon_nonoise', recon_img_nonoise.squeeze(0) / 2 + 0.5, global_step // show_interval)
                writer.add_text('judge_common',
                                f'recon:{"{:.2%}".format(float(recon_prob))}_'
                                f'orgin:{"{:.2%}".format(float(real_prob))}_'
                                f'recon_nonoise:{"{:.2%}".format(float(recon_prob_nonoise))}',
                                global_step // show_interval)
                # 重新累计
                losses = {'recon_loss': [],
                          'lie_loss': [],
                          'detect_loss': [],
                          'real_loss': []}
                # 输出网络权重和梯度
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step // show_interval)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step // show_interval)
                # 对瑕疵进行测试
                for p in flaw_path:
                    get_img_paths = os.listdir(p)
                    get_one = random.choice(get_img_paths)
                    get_img = Image.open(os.path.join(p, get_one)).convert("RGB")
                    get_img_tensor = trans2tensor(get_img).unsqueeze(0).to(device)
                    # noise_get_img_tensor = add_noise(get_img_tensor)
                    get_recon_flaw, get_flaw_prob1 = model(get_img_tensor)
                    get_flaw_prob2 = model(get_img_tensor.detach(), True)
                    writer.add_image(f'recon_flaw/{p.split("/")[-1]}_o', get_img_tensor.squeeze(0)/2+0.5, global_step // show_interval)
                    writer.add_image(f'recon_flaw/{p.split("/")[-1]}', get_recon_flaw.squeeze(0)/2+0.5, global_step // show_interval)
                    writer.add_text(f'judge_flaw/{p.split("/")[-1]}',
                    f'recon:{"{:.2%}".format(float(get_flaw_prob1))}_orgin:{"{:.2%}".format(float(get_flaw_prob2))}',
                                    global_step // show_interval)
                    if global_step % (show_interval * 10) == 0 and global_step != 0:
                        random.shuffle(get_img_paths)
                        test_imgs = get_img_paths[:200]
                        right = 0
                        probs = []
                        for test_img in test_imgs:
                            get_test_img = Image.open(os.path.join(p, test_img)).convert("RGB")
                            get_test_img_tensor = trans2tensor(get_test_img).unsqueeze(0).to(device)
                            out = model(get_test_img_tensor, True)
                            if float(out) > 0.5:
                                right += 1
                                probs.append(float(out))
                        writer.add_scalars(f'flaw_acc/{p.split("/")[-1]}', {'acc': right/200,
                                                                            'summup': sum(probs)/200}, global_step // (show_interval * 10))
        lr_scheduler_G.step()
        lr_scheduler_D.step()



def add_noise(img, alpha=0.01):
    noise_img = img + (alpha ** 0.5) * torch.randn(img.shape).to(device)
    return noise_img


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


if __name__ == '__main__':
    train()

