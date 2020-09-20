#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import numpy as np

from base import BaseImageDataset

class IBug300WDataset(BaseImageDataset):
    def __init__(self, path, transform=None):
        images  = []
        targets = []
        for folder, name in [("01_Indoor", "indoor"),
                             ("02_Outdoor", "outdoor")]:
            for i in range(1, 301):
                fname = f"{folder}/{name}_{i:03d}"
                kps = []
                with open(os.path.join(path, fname + '.pts'), 'r') as fp:
                    for line in fp:
                        if line[0].isdigit():
                            kps.append(list(map(float, line[:-1].split())))

                images.append(fname + ".png")
                targets.append(kps)

        super(IBug300WDataset, self).__init__(
            images,
            targets,
            path,
            transform)
