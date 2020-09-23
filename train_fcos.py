#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.optim as optim

from models.fcos import fcos_resnet50
from datasets.wider import WIDERDataset
from losses import FocalLoss, IOULoss
from trainer import Trainer

class FCOSLoss(nn.Module):
    pass

def train_fcos_trtansform(data):
    # TODO: Format data
    return data

class FCOSTrainer(Trainer):
    def __init__(self, wider_path, model_path, num_workers=1, use_cuda=False):
        model         = fcos_resnet50(2)
        train_dataset = WIDERDataset(wider_path, 'train', transform=train_fcos_trtansform)
        criterion     = FCOSLoss()
        optimizer     = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
        scheduler     = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60000, 80000], gamma=0.1)

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

            x, yt = data

            yp = model(x)
            loss = self.criterion(yp, yt)

            loss.backward()
            self.scheduler.step()

            return loss.item()

if __name__ == '__main__':
    wider_path = r"C:\Users\seba-\Desktop\WIDER"
    model_path = r"C:\Users\seba-\Desktop"
    num_workers = 4
    use_cuda = True

    # TODO: I think this is how many iterations, not epochs
    num_epochs = 90000
    batch_size = 16

    # TODO: Log modifications to model

    trainer = FCOSTrainer(wider_path, model_path,num_workers, use_cuda)
    trainer(num_epochs, batch_size)
