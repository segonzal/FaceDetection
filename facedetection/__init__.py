import torch
import random
import numpy as np
import imgaug as ia
from facedetection import data, sampler, utils


def train_one_epoch(model, data_loader, optimizer, epoch, device):
    for image, target, mask, length in data_loader:
        print(type(image), type(target), type(mask), type(length))
        print(image.shape, target.shape, mask.shape, length.shape)


def collate_fn(batch):
    image, target = tuple(zip(*batch))
    image_shape = np.stack([i.shape[:2] for i in image])
    target_shape = np.stack([t.shape[0] for t in target])
    mask = (np.ones(shape) for shape in image_shape)
    length = target_shape

    # image, target and mask require padding
    max_image_shape = image_shape.max(0)
    max_target_shape = target_shape.max(0)
    image = (
        np.pad(im, [
            [0, max_image_shape[0] - shape[0]],
            [0, max_image_shape[1] - shape[1]],
            [0, 0]])
        for im, shape in zip(image, image_shape))
    mask = (
        np.pad(m, [
            [0, max_image_shape[0] - shape[0]],
            [0, max_image_shape[1] - shape[1]],
            [0, 0]])
        for m, shape in zip(mask, image_shape))
    target = (
        np.pad(tg, [
            [0, max_target_shape - l],
            [0, 0],
            [0, 0]])
        for tg, l in zip(target, length)
    )

    image = np.rollaxis(np.stack(image), 1, 3)
    target = np.stack(target)
    mask = np.stack(mask)

    return image, target, mask, length


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ia.seed(args.seed)
    # If running on cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    train_transform = None
    train_dataset = data.WIDERFace(args.path, subset='train', transforms=train_transform)
    train_sampler = sampler.get_sampler(train_dataset, args.batch_size, k=5)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn)

    model = None
    optimizer = None
    # lr weight_decay momentum
    # log_frequency

    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, epoch, device)
