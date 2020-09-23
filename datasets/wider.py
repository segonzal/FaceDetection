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
                    targets[-1].append(line[:-1].split())

        super(WIDERDataset, self).__init__(
            images,
            targets,
            image_path,
            transform)

    def preprocess_target(self, target):
        bbox = [list(map(int, l[:4])) for l in target]
        #xywh_to_xyxy
        bbox = [[x,y,x+w,y+h] for x,y,w,h in bbox]

        keypoints = [list(map(float, l[4:-1])) for l in target]
        for l in keypoints:
            del l[2::3]

        return dict(bbox=bbox, keypoints=keypoints)
