import torch
import torch.nn as nn
import torch.nn.functional as F

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, target):
        target = F.adaptive_max_pool2d(target, output.shape[2:])
        mask = F.adaptive_max_pool2d(mask, output.shape[2:])
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss
