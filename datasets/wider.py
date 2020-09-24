#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import numpy as np

from datasets.base import BaseImageDataset

class WIDERDataset(BaseImageDataset):
    COLOR_MEAN = np.float32([0.47651113, 0.43808137, 0.41200404])

    def __init__(self, path, subset, transform=None):
        assert subset in ['test', 'train', 'val']

        image_path = os.path.join(path, subset, 'images')
        images  = []
        targets = []

        with open(os.path.join(path, f"{subset}/label.txt"), 'r') as fp:
            for line in fp:
                if line.startswith('#'):
                    images.append(line[2:-1])
                    targets.append([])
                else:
                    numbers = line[:-1].split()
                    box = np.int32(list(map(int, numbers[:4])))
                    kps = np.float32(list(map(float, numbers[4:-1]))).reshape(5, 3)
                    kps = kps[:,:2].flatten()
                    targets[-1].append((box, kps))

        super(WIDERDataset, self).__init__(
            images,
            targets,
            image_path,
            transform)

    def get_target(self, key):
        # The image with max faces contains 1968
        box, kps = zip(*self.targets[key])
        box = np.stack(box).reshape(-1,4)
        kps = np.stack(kps).reshape(-1,5,2)
        return dict(bbox=box,  keypoints=kps)
