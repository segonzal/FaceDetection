import os
import numpy as np
import torch.utils.data as data
from PIL import Image


class WIDERFace(data.Dataset):
    def __init__(self, path, subset='train', transforms=None):
        self.path = path
        self.subset = subset
        self.transforms = transforms

        self.images = []
        self.boxes = []

        with open(os.path.join(self.path, self.subset, 'label.txt'), 'r') as fp:
            for line in fp:
                line = line[:-1].split()

                if line[0].startswith('#'):
                    self.images.append(line[1])
                    self.boxes.append([])
                else:
                    self.boxes[-1].append(line[:4])

        def sort_key(x):
            return int(x[2]) * int(x[3])

        for box in self.boxes:
            box.sort(key=sort_key, reverse=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.get_image_path(idx)).convert('RGB')
        image = np.float32(image) / 255.0

        target = np.float32(self.boxes[idx]).reshape(-1, 2, 2)
        target[:, 1, :] = target[:, 0, :] + target[:, 1, :]

        item = dict(image=image, target=target)

        if self.transforms:
            item = self.transforms(item)

        return item

    def get_image_path(self, idx):
        image_path = os.path.join(self.path, self.subset, 'images', self.images[idx])
        return image_path

    def get_height_and_width(self, idx):
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(self.get_image_path(idx)).size
        return height, width
