import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    return nn.BCELoss()(output, target)
