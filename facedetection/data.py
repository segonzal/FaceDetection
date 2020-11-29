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


def resize(image, target, new_size):
    new_width, new_height = new_size
    old_width, old_height = image.size

    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    target[:, 0::2] = target[:, 0::2] * (new_width / old_width)
    target[:, 1::2] = target[:, 1::2] * (new_height / old_height)

    return image, target


def list_of_tuples_to_tuple_of_lists(batch):
    return tuple(zip(*batch))


def encode_target(image, target, strides, sizes):
    img_width, img_height = image.size
    # Order targets by area (bigger first)
    target = target.tolist()
    target.sort(key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)
    out_target = []
    for stride, min_size, max_size in zip(strides, [0, *sizes], [*sizes, np.inf]):
        map_height, map_width = img_height // stride, img_width // stride
        tgt = np.zeros((map_height, map_width, 6))

        y = np.linspace(np.floor(stride / 2), img_height, map_height)
        x = np.linspace(np.floor(stride / 2), img_width, map_width)
        xx, yy = np.meshgrid(x, y)

        for x0, y0, x1, y1 in target:
            left = xx - x0
            top = yy - y0
            right = x1 - xx
            bottom = y1 - yy
            box = np.stack([left, top, right, bottom])
            box_non_zero = (box > 0).all(0)
            box_max = box.max(0)

            k = np.where(box_non_zero & (min_size < box_max) & (box_max < max_size))

            # Compute LTRB box coordinates
            tgt[:, :, 1][k] = left[k]
            tgt[:, :, 2][k] = top[k]
            tgt[:, :, 3][k] = right[k]
            tgt[:, :, 4][k] = bottom[k]

            # Compute Class target
            tgt[:, :, 0][k] = 1

            # Compute Centerness target from LTRB
            lr = np.stack([left, right])
            tb = np.stack([top, bottom])
            ctr = lr.min(0)/lr.max(0) * tb.min(0)/tb.max(0)
            tgt[:, :, 5][k] = np.sqrt(ctr[k])

        out_target.append(tgt)
    return image, out_target


class CollateFn(object):
    def __init__(self, image_size, strides, sizes):
        self.image_size = image_size
        self.strides = strides
        self.sizes = sizes

    def compute_median_aspect_ratio(self, images):
        aspect_ratio = np.median(get_aspect_ratios(images))

        if aspect_ratio < 1:
            width, height = int(self.image_size * aspect_ratio), self.image_size
        else:
            width, height = self.image_size, int(self.image_size / aspect_ratio)

        return width, height

    def __call__(self, batch):
        new_size = self.compute_median_aspect_ratio([i for i, t in batch])
        batch = [
            encode_target(*resize(image, target, new_size), self.strides, self.sizes)
            for image, target in batch]
        return list_of_tuples_to_tuple_of_lists(batch)
