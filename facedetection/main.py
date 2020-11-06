import os
import wandb
import argparse
import numpy as np

import torch
import torch.onnx

from .datasets.wider_face import WiderFace
from .models.fcos import fcos_resnet50


def dir_path(arg):
    if os.path.isdir(arg):
        return arg
    raise NotADirectoryError(arg)


def fcos_loss(output, target):
    pass


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # TODO: Log here


def test(args, model, device, test_loader):
    model.eval()
    # TODO: testing


def main(use_wandb=False):
    if use_wandb:
        wandb.init()

    parser = argparse.ArgumentParser('Train FCOS')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='WD',
                        help='SGD weight decay (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('path', type=dir_path,
                        help='Path to WIDER Face dataset')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_wandb:
        wandb.config.update(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # If running on cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if use_cuda else 'cpu')

    # TODO: data transforms
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        WiderFace(args.path, 'train', transform=None),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        WiderFace(args.path, 'val', transform=None),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = fcos_resnet50(1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    if use_wandb:
        wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, fcos_loss, device, train_loader, optimizer, epoch)
        # TODO: test every X epochs
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
