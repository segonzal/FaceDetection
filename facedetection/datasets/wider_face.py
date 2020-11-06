import os
import numpy as np
import torch.utils.data as data
from PIL import Image


class WiderFace(data.Dataset):
    def __init__(self, path, subset, transform=None):
        assert subset in ['test', 'train', 'val'], 'subset should be one of (test, train, val), got {}' % (subset,)
        self.path = path
        self.subset = subset
        self.transform = transform

        self.image_path = os.path.join(self.path, self.subset, 'images')
        self.images = []
        self.targets = []

        with open(os.path.join(self.path, subset, 'label.txt'), 'r') as fp:
            for line in fp:
                if line.startswith('#'):
                    self.images.append(line[2:-1])
                    self.targets.append([])
                else:
                    numbers = line[:-1].split()
                    box = np.int32(list(map(int, numbers[:4])))
                    kps = np.float32(list(map(float, numbers[4:-1]))).reshape(5, 3)
                    kps = kps[:, :2].flatten()
                    self.targets[-1].append((box, kps))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.open(os.path.join(self.image_path, img)).convert('RGB')
        img = np.float32(img) / 255.0

        box, kps = zip(*self.targets[idx])
        box = np.stack(box).reshape(-1, 4)
        kps = np.stack(kps).reshape(-1, 5, 2)

        cls = np.ones((kps.shape[0], 1))

        return dict(img=img, box=box, kps=kps, cls=cls)
