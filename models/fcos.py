#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.ops as tvo
import torchvision.models as tvm
import torchvision.models._utils as tvu

class FCOS(nn.Module):
    use_p5 = True
    centerness_on_regression_tower = True
    normalize_regression_targets = False
    fpn_strides = [128, 64, 32, 16, 8]
    prior_prob = 0.01

    """
    @inproceedings{tian2019fcos,
      title={Fcos: Fully convolutional one-stage object detection},
      author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
      booktitle={Proceedings of the IEEE international conference on computer vision},
      pages={9627--9636},
      year={2019}
    }
    """
    def __init__(self, backbone, backbone_layers, num_classes, tower_channels=256):
        super(FCOS, self).__init__()

        backbone_names, backbone_sizes = zip(*backbone_layers.items())

        self.backbone = self.make_backbone(backbone, backbone_names)
        self.fpyramid = self.make_pyramid(backbone_sizes, tower_channels, self.use_p5)

        self.cls_tower = self.make_tower(tower_channels, 4)
        self.reg_tower = self.make_tower(tower_channels, 4)

        self.cls_head = self.make_head(tower_channels, num_classes)
        self.ctr_head = self.make_head(tower_channels, 1)
        self.box_head = self.make_head(tower_channels, 4)

        self.box_scales = nn.ParameterList([
                            nn.Parameter(torch.tensor(1.0))
                            for _ in range(len(backbone_names) + 2)])

        self.init_weights()

    def init_weights(self):
        self.backbone.requires_grad = False

        for m in [
                self.cls_tower, self.reg_tower,
                self.cls_head, self.ctr_head, self.box_head]:
            for l in m.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_head.bias, bias_value)

    @staticmethod
    def make_backbone(backbone, backbone_layers):
        return_layers = {l:f'p{i}'
                for i, l in enumerate(backbone_layers, 3)}
        back = tvu.IntermediateLayerGetter(backbone, return_layers)
        return back

    @staticmethod
    def make_pyramid(in_channel_list, out_channels, use_p5=True):
        extra_in_channels = out_channels if use_p5 else in_channel_list[-1]
        fpn_extra = tvo.feature_pyramid_network.LastLevelP6P7(extra_in_channels, out_channels)
        fpn = tvo.FeaturePyramidNetwork(in_channel_list, out_channels, fpn_extra)
        return fpn

    @staticmethod
    def make_tower(in_channels, num_convs):
        tower = []
        for i in range(num_convs):
            tower.extend([
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True),
                nn.GroupNorm(32, in_channels),
                nn.ReLU()])
        return nn.Sequential(*tower)

    @staticmethod
    def make_head(in_channels, out_channels):
        return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpyramid(x)

        results = dict(yp_cls=[], yp_ctr=[], yp_box=[])

        for i, (name, feature) in enumerate(x.items()):
            cls_tower = self.cls_tower(feature)
            reg_tower = self.reg_tower(feature)

            yp_cls = self.cls_head(cls_tower)

            if self.centerness_on_regression_tower:
                yp_ctr = self.ctr_head(reg_tower)
            else:
                yp_ctr = self.ctr_head(cls_tower)

            yp_box = self.box_scales[i] * self.box_head(reg_tower)
            if self.normalize_regression_targets:
                yp_box = F.relu(yp_box)
                #if self.training: multiply by fpn_strides
            else:
                yp_box = torch.exp(yp_box)

            results['yp_cls'].append(yp_cls)
            results['yp_ctr'].append(yp_ctr)
            results['yp_box'].append(yp_box)
        return results

BACKBONES = {
    'resnet18':  {'layer2': 128, 'layer3':  256, 'layer4':  512},
    'resnet34':  {'layer2': 128, 'layer3':  256, 'layer4':  512},
    'resnet50':  {'layer2': 512, 'layer3': 1024, 'layer4': 2048},
    'resnet101': {'layer2': 512, 'layer3': 1024, 'layer4': 2048},
    'resnet152': {'layer2': 512, 'layer3': 1024, 'layer4': 2048},
}

def _fcos(model_name, num_classes):
    backbone = tvm.__dict__[model_name](pretrained=True)
    backbone.requires_grad = False
    backbone_layers = BACKBONES[model_name]
    return FCOS(backbone, backbone_layers, num_classes)

def fcos_resnet18(num_classes):
    return _fcos('resnet18', num_classes)

def fcos_resnet34(num_classes):
    return _fcos('resnet34', num_classes)

def fcos_resnet50(num_classes):
    return _fcos('resnet50', num_classes)

def fcos_resnet101(num_classes):
    return _fcos('resnet101', num_classes)

def fcos_resnet152(num_classes):
    return _fcos('resnet152', num_classes)
