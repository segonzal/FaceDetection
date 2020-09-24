#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import imgaug.augmenters as iaa

from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as T

from models.fcos import fcos_resnet50
from datasets.wider import WIDERDataset
from losses import focal_loss, iou_loss
from trainer import Trainer


class FCOSLoss(nn.Module):
    def __init__(self):
        super(FCOSLoss, self).__init__()

    def forward(self, input, target):
        #focal_loss(input, target, weight, focus, reduction='mean', logits=False)
        #iou_loss(input, target)
        return 0

class TrainFCOSTransform(object):
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.PadToSquare(),
            iaa.Resize({'width': 128, 'height': 128}),
            #iaa.PadToFixedSize(width=800, height=800),
        ])

    def imgaug_wrap(self, image, bbox, keypoints):
        kps_shape = keypoints.shape
        box_shape = bbox.shape

        kpsoi = KeypointsOnImage.from_xy_array(keypoints, image.shape)
        bbsoi = BoundingBoxesOnImage.from_xyxy_array(bbox, image.shape)

        image, kpsoi, bbsoi = self.seq(
                                image=image,
                                keypoints=kpsoi,
                                bounding_boxes=bbsoi)

        keypoints = kpsoi.to_xy_array().reshape(kps_shape)
        bbox = bbsoi.to_xyxy_array().reshape(box_shape)

        return image, bbox, keypoints

    def bbox_list_to_matrix(self, class_list, bbox_list, image_shape, num_classes=1):
        # order boxes by area
        class_list, bbox_list = zip(
            *sorted(zip(class_list, bbox_list),
                    key=lambda p: (p[1][2]-p[1][0])*(p[1][1]-p[1][3])))

        # clamp bbox to image
        bbox_list = np.int32(bbox_list).reshape(-1,2,2)
        bbox_list = np.clip(bbox_list, 0, image_shape).reshape(-1, 4)

        # compute class and ltrb matrix
        cls = np.zeros((*image_shape, num_classes))
        box = np.zeros((*image_shape, 4))

        for i, (x0,y0,x1,y1) in enumerate(bbox_list):
            cls[y0:y1, x0:x1, class_list[i]-1] = 1

            x, y = np.meshgrid(range(x0,x1), range(y0,y1))

            box[y0:y1, x0:x1, 0] = x - x0 # l
            box[y0:y1, x0:x1, 1] = y - y0 # t
            box[y0:y1, x0:x1, 2] = x1 - x # r
            box[y0:y1, x0:x1, 3] = y1 - y # b

        # compute centerness
        min_lr = np.min(np.stack([box[:,:,0], box[:,:,2]]), axis=0)
        min_tb = np.min(np.stack([box[:,:,1], box[:,:,3]]), axis=0)
        max_lr = np.max(np.stack([box[:,:,0], box[:,:,2]]), axis=0)
        max_tb = np.max(np.stack([box[:,:,1], box[:,:,3]]), axis=0)

        ctr = np.sqrt(
            np.multiply(
                np.divide(min_lr, max_lr, out=np.zeros_like(min_lr), where=max_lr!=0),
                np.divide(min_tb, max_tb, out=np.zeros_like(min_tb), where=max_tb!=0)))

        return cls, box, ctr)

    def __call__(self, data):
        img = data['image']
        box = data['bbox']
        kps = data['keypoints']

        # From xywh to xyxy
        box[:,:,0] = box[:,:,0] + box[:,:,1]
        box = box.reshape(-1, 4)
        kps = kps.reshape(-1, 2)

        img, box_list, kps_list = self.imgaug_wrap(img, box, kps)

        bbox_list = np.int32().reshape(-1, 4).tolist()
        class_list = np.ones(len(bbox_list), dtype=np.int32).tolist()
        image_shape = img.shape[:2]

        cls, box, ctr = self.bbox_list_to_matrix(
                class_list, bbox_list, image_shape, num_classes=1)

        # TODO: Normalize image here

        data = dict(
            img=T.to_tensor(img),
            cls=T.to_tensor(cls),
            ctr=T.to_tensor(ctr),
            box=T.to_tensor(box))

        return data

class FCOSTrainer(Trainer):
    def __init__(self, wider_path, model_path, milestones, num_workers=1, use_cuda=False):
        transform = TrainFCOSTransform()

        model         = fcos_resnet50(2)
        train_dataset = WIDERDataset(wider_path, 'train', transform=transform)
        criterion     = FCOSLoss()
        optimizer     = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
        scheduler     = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        use_cuda = use_cuda and torch.cuda.is_available()
        device   = torch.device("cuda:0" if use_cuda else "cpu")

        super(FCOSTrainer, self).__init__(model, train_dataset, num_workers=num_workers, device=device)

        # TODO: load weights from pretrained model
        # self.model.load_state_dict(weights)
        # self.epoch_start = 0

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.model_path = model_path
        self.save_epoch = -1
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def filename(self, model_name, epoch=None):
        if epoch is None:
            pass
        elif self.save_epoch > 0 and epoch % self.save_epoch == (self.save_epoch - 1):
            model_name = f"{model_name}.Epoch-{epoch:03d}"
        else:
            return None
        return os.path.join(self.model_path, model_name + '.pth')

    def train(self, model, data):
            self.optimizer.zero_grad()

            yp = model(data['img'])
            loss = self.criterion(yp, data)

            loss.backward()
            self.scheduler.step()

            return loss.item()

if __name__ == '__main__':
    wider_path = r"C:\Users\seba-\Desktop\WIDER"
    model_path = r"C:\Users\seba-\Desktop"
    num_workers = 4
    use_cuda = True

    # Wider.train has 12880 items
    # num_epochs = 90000
    # milestones = [60000, 80000]

    num_epochs = 110
    milestones = [60000, 80000]
    batch_size = 16

    # TODO: Log modifications to model

    trainer = FCOSTrainer(wider_path, model_path, milestones, num_workers, use_cuda)
    trainer(num_epochs, batch_size)
