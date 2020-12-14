import torch
import random
import numpy as np
import wandb
import imgaug as ia
import imgaug.augmenters as iaa
from facedetection import data, sampler, transforms, collate_functions, fcos


def train_one_epoch(model, data_loader, optimizer, epoch, device):
    loss_fn = fcos.fcos_loss

    model.train()
    base_step = epoch * len(data_loader)
    for b, batch in enumerate(data_loader):
        img, t_cls, t_ctr, t_box, mask = tuple((t.to(device) for t in batch))

        prediction = model(img)
        target = (t_cls, t_ctr, t_box)
        loss = loss_fn(prediction, target, mask, reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'epoch': epoch, 'loss': loss.item()}, step=base_step + b)


def train(args):
    wandb.init(project="=fcos-face-detection", config=args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ia.seed(args.seed)
    # If running on cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    network_strides = [8, 16, 32, 64, 128]
    network_sizes = [32, 64, 128, 256, 512]

    train_transform = transforms.Compose([
        transforms.ImgaugWrapper([
            iaa.Resize({"shorter-side": 320, "longer-side": "keep-aspect-ratio"}),
        ]),
        transforms.EncodeTarget(network_strides, network_sizes)])
    train_dataset = data.WIDERFace(args.path, subset='train', transforms=train_transform)
    train_sampler = sampler.get_sampler(train_dataset, args.batch_size, k=5)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_functions.fcos_collate_fn)

    model = fcos.fcos_resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = model.to(device)
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, epoch, device)
    torch.save(model.state_dict(), 'fcos.pth')
