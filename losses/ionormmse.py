#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class IONormMSE(nn.Module):
    def __init__(self, l_eye, r_eye):
        super(IONormMSE, self).__init__()
        self.l_eye = l_eye
        self.r_eye = r_eye

    def forward(self, input, target):
        n = F.mse_loss(input, target)
        d = F.mse_loss(target[:, self.l_eye, :], target[:, self.r_eye, :])
        return n / d
