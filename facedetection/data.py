import os
import math
import copy
import bisect
import numpy as np
import torch.utils.data as data

from PIL import Image
from collections import defaultdict
from itertools import repeat, chain
from torch.utils.data.sampler import BatchSampler, Sampler


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
                    self.boxes[-1].extend(line[:4])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.get_image_path(idx)).convert('RGB')

        target = np.float32(self.boxes[idx]).reshape(-1, 4)
        target[:, 2:] = target[:, 2:] + target[:, :2]

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def get_image_path(self, idx):
        image_path = os.path.join(self.path, self.subset, 'images', self.images[idx])
        return image_path

    def get_height_and_width(self, idx):
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(self.get_image_path(idx)).size
        return height, width


class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                'Sampler should be an instance of '
                'torch.utils.data.Sampler, but got sampler={}'.format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            for group_id, _ in sorted(buffer_per_group.items(), key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                repeat_times = math.ceil(remaining / len(samples_per_group[group_id]))
                samples_from_group_id = list(chain.from_iterable(repeat(samples_per_group[group_id], repeat_times)))
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


def compute_aspect_ratios(dataset):
    indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def group_aspect_ratio(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = quantize(aspect_ratios, bins)
    return groups


def get_aspect_ratios(images):
    aspect_ratios = []
    for img in images:
        width, height = img.size
        aspect_ratios.append(float(width) / float(height))
    return aspect_ratios


def resize(image, target, width, height):
    image = image.resize((width, height), Image.ANTIALIAS)

    return image, target


def collate_fn(batch):
    image, target = tuple(zip(*batch))

    # TODO: add image_size parameter elsewhere
    image_size = 640
    aspect_ratio = np.median(get_aspect_ratios(image))

    if aspect_ratio < 1:
        width, height = int(image_size * aspect_ratio), image_size
    else:
        width, height = image_size, int(image_size / aspect_ratio)

    out_image, out_target = [], []
    for img, tgt in zip(image, target):
        img, tgt = resize(img, tgt, width, height)
        img = np.float32(img)

        # TODO: Resize target accordingly

        out_image.append(img)
        out_target.append(tgt)

    # TODO: transform target as FCOS expects it

    out_image = np.float32(out_image)
    return out_image, out_target
