#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def inter_ocular_normalized_mse(input, target, left_eye_index, right_eye_index):
    n = F.mse_loss(input, target)
    d = F.mse_loss(target[:, left_eye_index, :], target[:, right_eye_index, :])
    return n / d

def focal_loss(input, target, weight, focus, reduction='mean', logits=False):
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(input, target, reduce=False)

    pt = torch.exp(-BCE_loss)
    F_loss = weight * torch.pow(1 - pt, focus) * BCE_loss

    if reduction == 'mean':
        return torch.mean(F_loss)

    elif reduction == 'sum':
        return torch.sum(F_loss)

    elif reduction == 'none':
        return F_loss

def iou_loss(input, target):
    area_p = None
    area_t = None
    inter_height = None
    inter_width = None
    area_i = inter_width * inter_height
    area_u = area_p + area_t - area_i
    iou = area_i / area_u
    IOU_loss = -torch.log(iou)#Only compute where target[i,j] inside a bbox
    return iou_loss

class InterOcularNormalizedMSE(nn.Module):
    def __init__(self, left_eye_index, right_eye_index):
        super(IONormMSE, self).__init__()
        self.left_eye_index  = left_eye_index
        self.right_eye_index = right_eye_index

    def forward(self, input, target):
        return inter_ocular_normalized_mse(input, taget, self.left_eye_index, self.right_eye_index)

class FocalLoss(nn.Module):
    def __init__(self, weight, focus, reduction='mean', logits=False):
        super(FocalLoss, self).__init__()
        self.weight    = weight
        self.focus     = focus
        self.reduction = reduction
        self.logits    = logits

    def forward(self, input, target):
        return focal_loss(input, target, self.weight, self.focus, self.reduction, self.logits)

class IOULoss(nn.Module):
    pass
