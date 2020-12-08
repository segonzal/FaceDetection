import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.models as models
import numpy as np
from collections import OrderedDict
from torchvision.models._utils import IntermediateLayerGetter


class FCOS(nn.Module):
    prior_prob = 0.01

    def __init__(self, backbone, intermediate_layers, strides, fpn_out_channels=128):
        super(FCOS, self).__init__()
        self.strides = strides

        return_layers = {k: k for k in intermediate_layers.keys()}
        in_channels_list = list(intermediate_layers.values())

        self.backbone = IntermediateLayerGetter(backbone, return_layers)
        self.fpn = ops.FeaturePyramidNetwork(
            in_channels_list,
            fpn_out_channels,
            ops.feature_pyramid_network.LastLevelP6P7(fpn_out_channels, fpn_out_channels))

        self.cls_tower = self.make_tower(fpn_out_channels, 4)
        self.reg_tower = self.make_tower(fpn_out_channels, 4)

        self.cls_head = self.make_head(fpn_out_channels, 1)
        self.box_head = self.make_head(fpn_out_channels, 4)
        self.ctr_head = self.make_head(fpn_out_channels, 1)

        self.scale = self.make_scales(len(strides))

        self.init_weights()

    @staticmethod
    def make_tower(channels, conv_num):
        tower = []
        for _ in range(conv_num):
            tower.extend([
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(32, channels),
                nn.ReLU()])
        return nn.Sequential(*tower)

    @staticmethod
    def make_head(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 3)

    @staticmethod
    def make_scales(num):
        return nn.ParameterList((nn.Parameter(torch.tensor(1.0)) for _ in range(num)))

    def init_weights(self):
        self.backbone.requires_grad = False

        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)

    def forward(self, inputs):
        features = self.fpn(self.backbone(inputs))

        out= dict()
        for i, p in enumerate(features.values()):
            cla_tower = self.cls_tower(p)
            reg_tower = self.reg_tower(p)

            cls_prediction = self.cls_head(cla_tower)
            box_prediction = self.box_head(reg_tower)
            ctr_prediction = self.ctr_head(reg_tower)

            box_prediction = F.relu(self.scale[i] * box_prediction)

            out[f'cls_{i}'] = cls_prediction
            out[f'ctr_{i}'] = ctr_prediction
            out[f'box_{i}'] = box_prediction
        return out


def fcos_resnet50():
    backbone = models.resnet50(pretrained=True)
    intermediate_layers = OrderedDict([
        ('layer2', 512),
        ('layer3', 1024),
        ('layer4', 2048)])
    return FCOS(backbone, intermediate_layers, [8, 16, 32, 64, 128])
