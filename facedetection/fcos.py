import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetEncoder(nn.Module):
    def __init__(self, strides, sizes):
        super(TargetEncoder, self).__init__()
        self.strides = strides
        self.sizes = sizes

    def forward(self, images, target, image_shapes, target_shapes):
        batch_size = images.shape[0]

        boxes = []
        classes = []
        centerness = []
        for s in self.strides:
            map_size = images.shape[2:] // s
            box = torch.zeros((batch_size, 4, *map_size))
            cls = torch.zeros((batch_size, 1, *map_size))
            ctr = torch.zeros((batch_size, 1, *map_size))
            boxes.append(box)
            classes.append(cls)
            centerness.append(ctr)
        return boxes, classes, centerness

    def compute_boxes(self, boxes, stride):
        return 0

    def compute_centerness(self):
        return 0

    def compute_classes(self):
        return 0


class FCOSLoss(nn.Module):
    def __init__(self):
        super(FCOSLoss, self).__init__()

    def forward(self, prediction, target):
        return 0


class FCOS(nn.Module):
    def __init__(self, strides, sizes):
        super(FCOS, self).__init__()
        self.strides = strides
        self.sizes = sizes

    def forward(self, inputs):
        if self.training:
            target_encoder = TargetEncoder(self.strides, self.sizes)
            loss_fn = FCOSLoss()
        return inputs
