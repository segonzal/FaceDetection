import torch
import numpy as np
import facedetection.data as data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def draw_images(batch, n=1):
    images, targets = batch
    x = int(np.sqrt(n))
    y = x + (1 if n - x * x > 0 else 0)
    fig, ax = plt.subplots(x, y)
    ax = [ax] if n == 1 else ax.flat
    for i in range(n):
        ax[i].imshow(images[i])
        for x1, y1, x2, y2 in targets[i]:
            ax[i].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'))
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # If running on cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    collate_fn = data.CollateFn(640, [8, 16, 32, 64, 128], [32, 64, 128, 256, 512])

    train_dataset = data.WIDERFace(args.path, subset='train', transforms=None)
    group_ids = data.group_aspect_ratio(train_dataset, k=5)
    train_sampler = data.GroupedBatchSampler(torch.utils.data.RandomSampler(train_dataset), group_ids, args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn)

    print('test')
    for image, target in train_loader:
        for tt in target:
            for i, t in enumerate(tt):
                for j in range(6):
                    im = Image.fromarray(np.uint8(t[:, :, j] * 255) , 'L')
                    im.save(f"target_lvl-{i}_ch-{j}.jpeg")
        break