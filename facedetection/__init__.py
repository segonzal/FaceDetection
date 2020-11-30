import torch
import numpy as np
from facedetection import data, sampler
from PIL import Image
from torchvision.transforms.functional import to_tensor


def compute_size(width, height, side_size):
    """Computes the target size so that the largest side is equal to the given value and keeping the aspect ratio."""
    if height < width:
        height = int(side_size * (height / width))
        width = side_size
    else:
        width = int(side_size * (width / height))
        height = side_size
    return width, height


def square_pad(mat, side=None):
    """Pads a matrix to be squared. By default chooses the largest side."""
    h, w = mat.shape[:2]
    if side is None:
        side = max(h, w)
    h = max(side - h, 0)
    w = max(side - w, 0)
    h2 = h // 2
    w2 = w // 2

    mat = np.pad(mat, [[h2, h - h2], [w2, 2 - w2], [0, 0]], mode='constant', constant_values=0)

    return mat


def resize(image, target, new_size):
    new_width, new_height = new_size
    old_width, old_height = image.size

    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    target[:, 0::2] = target[:, 0::2] * (new_width / old_width)
    target[:, 1::2] = target[:, 1::2] * (new_height / old_height)

    return image, target


def encode_target(image, target, strides, sizes):
    img_width, img_height = image.size
    # Order targets by area (bigger first)
    target = target.tolist()
    target.sort(key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)
    out_target = []
    for stride, min_size, max_size in zip(strides, [0, *sizes], [*sizes, np.inf]):
        map_height, map_width = img_height // stride, img_width // stride
        tgt = np.zeros((map_height, map_width, 6))

        y = np.linspace(np.floor(stride / 2), img_height, map_height)
        x = np.linspace(np.floor(stride / 2), img_width, map_width)
        xx, yy = np.meshgrid(x, y)

        for x0, y0, x1, y1 in target:
            left = xx - x0
            top = yy - y0
            right = x1 - xx
            bottom = y1 - yy
            box = np.stack([left, top, right, bottom])
            box_non_zero = (box > 0).all(0)
            box_max = box.max(0)

            k = np.where(box_non_zero & (min_size < box_max) & (box_max < max_size))

            # Compute LTRB box coordinates
            tgt[:, :, 1][k] = left[k]
            tgt[:, :, 2][k] = top[k]
            tgt[:, :, 3][k] = right[k]
            tgt[:, :, 4][k] = bottom[k]

            # Compute Class target
            tgt[:, :, 0][k] = 1

            # Compute Centerness target from LTRB
            lr = np.stack([left, right])
            tb = np.stack([top, bottom])
            ctr = lr.min(0)/lr.max(0) * tb.min(0)/tb.max(0)
            tgt[:, :, 5][k] = np.sqrt(ctr[k])

        out_target.append(tgt)
    return image, out_target


class TrainTransform:
    def __init__(self, strides, sizes, image_size):
        self.strides = strides
        self.sizes = sizes
        self.image_size = image_size

    def __call__(self, image, target):
        width, height = image.size
        width, height = compute_size(width, height, self.image_size)
        image, target = resize(image, target, (width, height))
        image, target = encode_target(image, target, self.strides, self.sizes)

        image = to_tensor(square_pad(np.float32(image)))
        target = list(map(torch.from_numpy, map(square_pad, target)))

        return image, target


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # If running on cudnn backend
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # collate_fn = data.CollateFn(640, [8, 16, 32, 64, 128], [32, 64, 128, 256, 512])

    train_transform = TrainTransform([8, 16, 32, 64, 128], [32, 64, 128, 256, 512], 640)

    train_dataset = data.WIDERFace(args.path, subset='train', transforms=train_transform)
    train_sampler = sampler.get_sampler(train_dataset, args.batch_size, k=5)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers)

    for image, target in train_loader:
        print(type(image), image.shape)
        for t in target:
            print(type(t),  t.shape)
        break
