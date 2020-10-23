import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as tvm
from .feature_extractor import FeatureExtractor
from .feature_pyramid_network import FeaturePyramidNetwork


class FCOSHeads(nn.Module):
    prior_prob = 0.01
    fpn_strides = [128, 64, 32, 16, 8]

    def __init__(self, stage_num, class_num, fpn_out_channels):
        super(FCOSHeads, self).__init__()
        self.stage_num = stage_num
        self.class_num = class_num
        self.fpn_out_channels = fpn_out_channels

        cls_tower = []
        reg_tower = []
        for _ in range(4):
            cls_tower.extend([
                nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
                nn.GroupNorm(32, self.fpn_out_channels),
                nn.ReLU()])
            reg_tower.extend([
                nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 3, padding=1),
                nn.GroupNorm(32, self.fpn_out_channels),
                nn.ReLU()])
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_head = nn.Conv2d(self.fpn_out_channels, self.class_num, 3)
        self.box_head = nn.Conv2d(self.fpn_out_channels, 4, 3)
        self.ctr_head = nn.Conv2d(self.fpn_out_channels, 1, 3)

        self.scales = nn.ParameterList()

        for _ in range(self.stage_num):
            self.scales.append(nn.Parameter(torch.tensor(1.0)))

        self.init_weights()

    def forward(self, inputs):
        outputs = []
        for i, feature in enumerate(inputs):
            ctower = self.cls_tower(feature)
            rtower = self.reg_tower(feature)

            cls_pred = self.cls_head(ctower)
            box_pred = self.box_head(rtower)
            ctr_pred = self.ctr_head(rtower)

            box_pred = F.relu(self.scales[i] * box_pred)

            outputs.extend([cls_pred, box_pred, ctr_pred])

        return outputs

    def init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

        bias_value = -np.log((1-self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)


def _fcos(model, return_layers, class_num):
    layer_names, layer_sizes = zip(*return_layers.items())
    return nn.Sequential(
        FeatureExtractor(model, layer_names),
        FeaturePyramidNetwork(layer_sizes, 256, 2),
        FCOSHeads(len(layer_names) + 2, class_num, 256))


def fcos_resnet50(class_num):
    model = tvm.resnet50(pretrained=True)
    model.requires_grad = False
    return_layers = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}
    return _fcos(model, return_layers, class_num)
