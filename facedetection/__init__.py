import torch
import numpy as np
import facedetection.data as data


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # If running on cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    train_dataset = data.WIDERFace(args.path, subset='train', transforms=None)
    group_ids = data.group_aspect_ratio(train_dataset, k=5)
    train_sampler = data.GroupedBatchSampler(torch.utils.data.RandomSampler(train_dataset), group_ids, args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=data.collate_fn)

    mi = (np.inf, np.inf)
    ma = (-np.inf, -np.inf)
    for image, target in train_loader:
        mi = min(mi, image.shape)
        ma = max(ma, image.shape)
    print(mi, ma)
