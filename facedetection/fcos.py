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
                nn.ReLU(inplace=True)])
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
        batch_size = inputs.shape[0]
        features = self.fpn(self.backbone(inputs))

        predicted_cls = []
        predicted_ctr = []
        predicted_box = []
        for i, p in enumerate(features.values()):
            cla_tower = self.cls_tower(p)
            reg_tower = self.reg_tower(p)

            cls_prediction = self.cls_head(cla_tower)
            box_prediction = self.box_head(reg_tower)
            # By default uses regression tower for the centerness head
            ctr_prediction = self.ctr_head(reg_tower)

            box_prediction = self.scale[i] * box_prediction
            # By default normalizes the regression targets
            box_prediction = F.relu(box_prediction)

            if not self.training:
                box_prediction = box_prediction * self.strides[i]
                # TODO: Check if needs to add locations

            cls_prediction = cls_prediction.reshape(batch_size, -1, 1)
            ctr_prediction = ctr_prediction.reshape(batch_size, -1, 1)
            box_prediction = box_prediction.reshape(batch_size, -1, 4)

            predicted_cls.append(cls_prediction)
            predicted_ctr.append(ctr_prediction)
            predicted_box.append(box_prediction)

        predicted_cls = torch.cat(predicted_cls, 1)
        predicted_ctr = torch.cat(predicted_ctr, 1)
        predicted_box = torch.cat(predicted_box, 1)

        if self.training:
            return predicted_cls, predicted_ctr, predicted_box
        else:
            return ops.nms(predicted_box, predicted_cls, 0.05)


def fcos_resnet50():
    backbone = models.resnet50(pretrained=True)
    intermediate_layers = OrderedDict([
        ('layer2', 512),
        ('layer3', 1024),
        ('layer4', 2048)])
    return FCOS(backbone, intermediate_layers, [8, 16, 32, 64, 128])


def focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='none'):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def iou_loss(inputs, targets, loss_type='iou', reduction='none'):
    p_left = inputs[:, :, 0]
    p_top = inputs[:, :, 1]
    p_right = inputs[:, :, 2]
    p_bottom = inputs[:, :, 3]

    t_left = targets[:, :, 0]
    t_top = targets[:, :, 1]
    t_right = targets[:, :, 2]
    t_bottom = targets[:, :, 3]

    p_area = (p_left + p_right) * (p_top + p_bottom)
    t_area = (t_left + t_right) * (t_top + t_bottom)

    h_intersect = torch.min(p_bottom, t_bottom) + torch.min(p_top, t_top)
    w_intersect = torch.min(p_left, t_left) + torch.min(p_right, t_right)

    area_intersect = w_intersect * h_intersect
    area_union = t_area + p_area - area_intersect

    iou = (area_intersect + 1.0) / (area_union + 1.0)

    if loss_type == 'iou':
        loss = -torch.log(iou)
    elif loss_type == 'liou':
        loss = 1 - iou
    elif loss_type == 'giou':
        enclosing_h_intersect = torch.max(p_bottom, t_bottom) + torch.max(p_top, t_top)
        enclosing_w_intersect = torch.max(p_left, t_left) + torch.max(p_right, t_right)

        enclosing_area = enclosing_w_intersect * enclosing_h_intersect + 1e-7

        giou = iou - (enclosing_area - area_union) / enclosing_area

        loss = 1 - giou
    else:
        raise NotImplementedError

    loss = loss.unsqueeze(-1)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def fcos_loss(prediction, target, mask=None, reduction='none'):
    p_cls, p_ctr, p_box = prediction
    t_cls, t_ctr, t_box = target

    indication = t_cls.sum(dim=-1, keepdim=True).gt(0)

    # class_loss is focal loss
    cls_loss = focal_loss(p_cls, t_cls, reduction='none')
    # regression_loss is the iou loss as in unitbox
    reg_loss = iou_loss(p_box, t_box, reduction='none') * indication
    # centerness_loss is the binary cross-entropy loss for sigmoid outputs
    ctr_loss = F.binary_cross_entropy_with_logits(p_ctr, t_ctr, reduction='none')

    loss = cls_loss + reg_loss + ctr_loss

    if mask is not None:
        loss = (loss * mask.float()).sum(dim=1, keepdim=True) / torch.sum(mask)
    else:
        loss = loss.mean(dim=1, keepdim=True)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss
