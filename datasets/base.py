#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch.utils.data as data

from PIL import Image

class BaseImageDataset(data.Dataset):
    def __init__(self, images, targets, image_path, transform=None):
        self.images = images
        self.targets = targets
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        item = dict()
        item.update(self.get_image(key))
        item.update(self.get_target(key))

        if self.transform:
            item = self.transform(item)

        return item

    def get_image(self, key):
        image = os.path.join(self.image_path, self.images[key])
        image = Image.open(image).convert('RGB')
        image = np.float32(image) / 255.0
        return dict(image=image)

    @property
    def color_mean(self):
        if not hasattr(self, 'COLOR_MEAN'):
            agg_num = 0.0
            agg_avg = 0.0

            for image, _ in self:
                img_num = image.shape[0] * image.shape[1]
                img_avg = np.mean(image, axis=(0, 1))

                num = agg_num + img_num
                avg = (agg_num * agg_avg + img_num * img_avg) / num

                agg_num = num
                agg_avg = avg

            self.COLOR_MEAN = agg_avg
        return self.COLOR_MEAN

    def get_target(self, key):
        return self.targets[key]
