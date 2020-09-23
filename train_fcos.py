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
from losses import FocalLoss, IOULoss
from trainer import Trainer


class FCOSLoss(nn.Module):
    pass

class TrainFCOSTransform(object):
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.PadToSquare(),
            iaa.Resize({'width': 800, 'height': 800}),
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

    def __call__(self, data):
        img = data['image']
        box = data['bbox']
        kps = data['keypoints']

        # From xywh to xyxy
        box[:,:,0] = box[:,:,0] + box[:,:,1]
        box = box.reshape(-1, 4)
        kps = kps.reshape(-1, 2)

        img, box_list, kps_list = self.imgaug_wrap(img, box, kps)
        box_list = box_list.astype(np.int32)

        # Compute centerness and ltrb
        cls = np.zeros((*img.shape[:2], 1))
        box = np.zeros((*img.shape[:2], 4))

        for x0,y0,x1,y1 in box_list:
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, img.shape[1])
            y1 = min(y1, img.shape[0])
            cls[y0:y1, x0:x1, 0] = 1

            x, y = np.meshgrid(range(x0,x1), range(y0,y1))

            box[y0:y1, x0:x1, 0] = x - x0
            box[y0:y1, x0:x1, 1] = y - y0
            box[y0:y1, x0:x1, 2] = x1 - x
            box[y0:y1, x0:x1, 3] = y1 - y

        min_lr = np.min(np.stack([box[:,:,0], box[:,:,2]]), axis=0)
        min_tb = np.min(np.stack([box[:,:,1], box[:,:,3]]), axis=0)
        max_lr = np.max(np.stack([box[:,:,0], box[:,:,2]]), axis=0)
        max_tb = np.max(np.stack([box[:,:,1], box[:,:,3]]), axis=0)
        ctr = np.sqrt(
            np.multiply(
                np.divide(min_lr, max_lr, out=np.zeros_like(min_lr), where=max_lr!=0),
                np.divide(min_tb, max_tb, out=np.zeros_like(min_tb), where=max_tb!=0)))

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
    batch_size = 8

    # TODO: Log modifications to model

    trainer = FCOSTrainer(wider_path, model_path, milestones, num_workers, use_cuda)
    trainer(num_epochs, batch_size)
