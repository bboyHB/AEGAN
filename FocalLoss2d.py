import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLossG(nn.Module):
    def __init__(self, gamma=0):
        super(FocalLossG, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        mseloss = F.mse_loss(input, target)
        # l1loss = F.l1_loss(input, target)
        diff = torch.abs(input - target)
        mse = diff ** 2
        ratios = (mse - mse.min()) * (1 / (mse - mse.min()).max())
        loss = (ratios * diff).mean()
        print(diff.min(), diff.max(), loss)
        # alpha = F.tanh(diff.max())
        # beta = torch.exp(diff.max())
        #
        # loss = (alpha ** self.gamma) * (beta ** 4) * mseloss
        # print(float(alpha), float(beta.float()), float((alpha ** self.gamma)), float((beta ** 4)))
        return loss

