import torch
import random
import numpy as np
import imgaug as ia
from facedetection import data, sampler, transforms, collate_functions, fcos


def train_one_epoch(model, data_loader, optimizer, epoch, device):
    loss_fn = fcos.fcos_loss

    model.train()
    for item in data_loader:
        image = item['image'].to(device)
        target = item['target'].to(device)
        image_shape = item['image_shape']
        target_shape = item['target_shape']

        # prediction = model(image, image_shape, target, target_shape)
        #
        # optimizer.zero_grad()
        # loss = loss_fn(prediction, target)
        # loss.backward()
        # optimizer.step()
        #
        # print(epoch, loss.item())


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
    # transforms.EncodeTarget([8, 16, 32, 64, 128], [32, 64, 128, 256, 512])
    train_dataset = data.WIDERFace(args.path, subset='train', transforms=train_transform)
    train_sampler = sampler.get_sampler(train_dataset, args.batch_size, k=5)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_functions.padded_dict_collate_fn)

    model = None
    optimizer = None
    # lr weight_decay momentum
    # log_frequency

    model = model.to(device)
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, epoch, device)
