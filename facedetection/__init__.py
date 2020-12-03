import torch
import random
import numpy as np
import imgaug as ia
from facedetection import data, sampler, utils


def train_one_epoch(model, data_loader, optimizer, epoch, device):
    for image, target in data_loader:
        #print(type(target), type(target[0]), type(target[0][0]))
        print(epoch, image.shape, *[t.shape for t in target])


def collate_fn(batch):
    image, target = tuple(zip(*batch))
    target = list(zip(*target))
    print(type(image), len(image), type(image[0]))
    print(type(target), len(target), type(target[0]))
    return image, target


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

    train_transform = utils.Compose([
        # TODO augmenters
        # utils.ImgaugWrapper(lambda i, t: (i, t)),
        utils.ResizeAndPad(640),
        utils.EncodeTarget([8, 16, 32, 64, 128], [32, 64, 128, 256, 512]),
        utils.ToTensor(),
        utils.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # TODO: Pad to homogeneous size and return the original size?
    ])
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