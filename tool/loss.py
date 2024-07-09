import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        # input_flat = input.view(N, -1)
        # target_flat = target.view(N, -1)

        input_flat = input.view(input.size(0), input.size(1), -1)
        input_flat = torch.transpose(input_flat, 1, 2).contiguous()
        input_flat = input_flat.view(-1, input_flat.size(2))
        target_flat = target.long()
        target_flat = target_flat.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


class Mixup(nn.Module):
    def __init__(self, use_mixloss=True):
        super(Mixup, self).__init__()
        self.use_mixloss = use_mixloss

    def criterion(self, lam, outputs, targets_a, targets_b, criterion):
        return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

    def forward(self, inputs, targets, criterion, model):
        lam = 0.5
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size).cuda()
        mix_inputs = lam*inputs + (1-lam)*inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        outputs = model(mix_inputs)

        losses = 0
        if isinstance(outputs, (list, tuple)):
                if self.use_mixloss:
                    for i in range(len(outputs)):
                        loss = self.criterion(lam, outputs[i], targets_a, targets_b, criterion[0])
                        losses += loss
                    edge_targets_b = edge_targets_a[index]
                    loss2 = self.criterion(lam, outputs[-1], edge_targets_a, edge_targets_b, criterion)
                    losses += loss2
                else:
                    for i in range(len(outputs)):
                        loss = self.criterion(lam, outputs[i], targets_a, targets_b, criterion[0])
                        losses += loss
        else:
            losses = self.criterion(lam, outputs, targets_a, targets_b, criterion[0])
        return losses