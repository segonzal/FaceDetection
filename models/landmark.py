#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        size = x.size()
        return torch.reshape(F.max_pool2d(x, kernel_size=size[2:]), (size[0], -1))

def abstanh(x):
    return torch.abs(torch.tanh(x))

class LandmarkModel(nn.Module):
    """
    | Type     | num |  k  | p | s |
    |----------|-----|-----|---|---|
    | Conv2d   |  16 | 5x5 | 2 | 1 |
    | Mpool2d  |     | 2x2 |   | 2 |
    | Conv2d   |  48 | 3x3 | 1 | 1 |
    | Mpool2d  |     | 2x2 |   | 2 |
    | Conv2d   |  64 | 3x3 | 0 | 1 |
    | Mpool2d  |     | 2x2 |   | 2 |
    | Conv2d   |  64 | 2x2 | 0 | 1 |
    | Gmpool2d |     |     |   |   |
    | Linear   | 100 |     |   |   |
    | Linear   |  2L |     |   |   |

    @article{wu2017facial,
      title={Facial landmark detection with tweaked convolutional neural networks},
      author={Wu, Yue and Hassner, Tal and Kim, KangGeon and Medioni, Gerard and Natarajan, Prem},
      journal={IEEE transactions on pattern analysis and machine intelligence},
      volume={40},
      number={12},
      pages={3067--3074},
      year={2017},
      publisher={IEEE}
    }
    @inproceedings{zhang2014facial,
      title={Facial landmark detection by deep multi-task learning},
      author={Zhang, Zhanpeng and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
      booktitle={European conference on computer vision},
      pages={94--108},
      year={2014},
      organization={Springer}
    }
    """
    def __init__(self, heads, shared_feature_size=100):
        super(LandmarkModel, self).__init__()
        self.conv1 = nn.Conv2d( 3, 16, 5, 1, 2, bias=True)
        self.conv2 = nn.Conv2d(16, 48, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(48, 64, 3, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(64, 64, 2, 1, 0, bias=True)
        self.fc    = nn.Linear(64, shared_feature_size, bias=True)

        self.heads = nn.ModuleDict({
            head_name: self.create_head(shared_feature_size, head_out_channels)
            for head_name, head_out_channels in heads.items()})

        self.mpool = nn.MaxPool2d(2, 2)
        self.gpool = GlobalMaxPool2d()

    def create_head(self, shared_feature_size, out_channels):
        return nn.Linear(shared_feature_size, out_channels, bias=True)

    def init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.02)

        for m in [self.fc, *self.heads.children()]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def forward(self, x):
        bz = x.size(0)
        x = self.mpool(abstanh(self.conv1(x)))
        x = self.mpool(abstanh(self.conv2(x)))
        x = self.mpool(abstanh(self.conv3(x)))
        x = self.gpool(abstanh(self.conv4(x)))
        x = torch.abs(torch.tanh(self.fc(x)))
        return {
            head_name: head_module(x)
            for head_name, head_module in self.heads.items()}
