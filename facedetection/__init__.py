import torch
import random
import numpy as np
import imgaug as ia
from facedetection import data, sampler, transforms, collate_functions, fcos


def train_one_epoch(model, data_loader, optimizer, epoch, device):
    # loss_fn = fcos.fcos_loss

    model.train()
    for batch in data_loader:
        batch = {k: v.to(device) if not k.startswith('_') else v for k, v in batch.items()}

        prediction = model(batch['image'])

        print('target', 'prediction')
        for k in prediction:
            print(k, batch[k].shape, prediction[k].shape)

        # optimizer.zero_grad()
        # loss = loss_fn(prediction, target)
        # loss.backward()
        # optimizer.step()
        #
        # print(epoch, loss.item())
        return


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

    network_strides = [8, 16, 32, 64, 128]
    network_sizes = [32, 64, 128, 256, 512]

    train_transform = transforms.EncodeTarget(network_strides, network_sizes)
    train_dataset = data.WIDERFace(args.path, subset='train', transforms=train_transform)
    train_sampler = sampler.get_sampler(train_dataset, args.batch_size, k=5)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_functions.padded_dict_collate_fn)

    model = fcos.fcos_resnet50()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # log_frequency

    model = model.to(device)
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, epoch, device)
        return
