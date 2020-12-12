import torch
import numpy as np


def pad_array(array, stack_arrays=True):
    """Pads a numpy array with zeros. Returns both the padded array and an array wit the original shapes."""
    shapes = np.stack([a.shape for a in array])
    max_shape = shapes.max(0)
    array = [np.pad(a, [(0, max_shape[i] - a.shape[i]) for i in range(len(max_shape))]) for a in array]
    if stack_arrays:
        array = np.stack(array)
    return array


def tuple_collate_fn(batch):
    return tuple(zip(*batch))


def dict_collate_fn(batch):
    return {k: [dic[k] for dic in batch] for k in batch[0].keys()}


def default_collate_fn(batch):
    if isinstance(batch, tuple):
        return tuple_collate_fn(batch)
    elif isinstance(batch, dict):
        return dict_collate_fn(batch)
    else:
        raise NotImplementedError


def fcos_collate_fn(batch):
    batch = (list((pad_array(c, stack_arrays=False) for c in zip(*b))) for b in zip(*batch))
    batch = ((list(zip(*b)) for b in batch))
    batch = (((np.concatenate(c) for c in b) for b in batch))
    batch = ((np.stack(list(b)) for b in batch))
    batch = (map(torch.tensor, batch))
    return tuple(batch)
