#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class IBug300WDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.images = []
        self.keypoints = []

        for folder, name in [("01_Indoor", "indoor"),
                             ("02_Outdoor", "outdoor")]:
            for i in range(1, 301):
                fname = f"{folder}/{name}_{i:03d}"
                kps = []
                with open(os.path.join(path, fname + '.pts'), 'r') as fp:
                    for line in fp:
                        if line[0].isdigit():
                            kps.append(list(map(float, line[:-1].split())))

                self.images.append(fname + ".png")
                self.keypoints.append(np.float32(kps).flatten())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        file_path = os.path.join(self.path, self.images[key])
        keypoints = self.keypoints[key]

        image = Image.open(file_path).convert('RGB')
        image = np.float32(image) / 255.0

        out = dict(image=image, keypoints=keypoints)
        if self.transform:
            out = self.transform(out)
        return out
