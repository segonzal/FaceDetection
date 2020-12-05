import torch
import numpy as np


def pad_array(array):
    """Pads a numpy array with zeros. Returns both the padded array and an array wit the original shapes."""
    shapes = np.stack([a.shape for a in array])
    max_shape = shapes.max(0)
    array = np.stack([
        np.pad(a, [(0, max_shape[i] - a.shape[i]) for i in range(len(max_shape))])
        for a in array
    ])
    return array, shapes


def default_collate_fn(batch):
    return tuple(zip(*batch))


def dict_collate_fn(batch):
    return {k: [dic[k] for dic in batch] for k in batch[0].keys()}


def padded_dict_collate_fn(batch):
    batch = dict_collate_fn(batch)

    # Wrap dict_keys with a list so we do not have troubles updating the dict
    for key in list(batch.keys()):
        array, shapes = pad_array(batch[key])
        batch[key] = torch.tensor(array)
        batch[key + '_shape'] = shapes
    return batch
