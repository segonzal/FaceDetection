import torch
import numpy as np
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import torchvision.transforms.functional as F
from skimage.transform import resize


class ImgaugWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, image, target):
        image = np.float32(image)
        bbs = BoundingBoxesOnImage.fill_from_xyxy_array_(target)
        image, bbs = self.seq(image=image, bounding_boxes=bbs)
        target = bbs.to_xyxy_array()
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torch.from_numpy(np.rollaxis(image, 2, 0))
        target = list(map(torch.from_numpy, target))
        return image, target


class Normalize(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std, inplace=True)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for fn in self.transforms:
            image, target = fn(image, target)
        return image, target


class EncodeTarget(object):
    def __init__(self, strides, sizes):
        self.strides = strides
        self.sizes = sizes

    def __call__(self, batch):
        image = batch['image']
        target = batch['target']

        img_height, img_width = image.shape[:2]
        # Order targets by area (bigger first)
        target = target.tolist()
        target.sort(key=lambda t: (t[1][0] - t[0][0]) * (t[1][1] - t[0][1]), reverse=True)

        out = dict(image=image)
        for i, (stride, min_size, max_size) in enumerate(zip(self.strides, [0, *self.sizes], [*self.sizes, np.inf])):
            map_height, map_width = img_height // stride, img_width // stride
            cls = np.zeros((map_height, map_width, 1), dtype=np.float32)
            ctr = np.zeros((map_height, map_width, 1), dtype=np.float32)
            box = np.zeros((map_height, map_width, 4), dtype=np.float32)

            y = np.linspace(np.floor(stride / 2), img_height, map_height)
            x = np.linspace(np.floor(stride / 2), img_width, map_width)
            xx, yy = np.meshgrid(x, y)

            for (x0, y0), (x1, y1) in target:
                left = xx - x0
                top = yy - y0
                right = x1 - xx
                bottom = y1 - yy

                ltrb = np.stack([left, top, right, bottom])
                ltrb_non_zero = (ltrb > 0).all(0)
                ltrb_max = ltrb.max(0)

                k = np.where(ltrb_non_zero & (min_size < ltrb_max) & (ltrb_max < max_size))

                # Compute LTRB box coordinates
                box[:, :, 0][k] = left[k]
                box[:, :, 1][k] = top[k]
                box[:, :, 2][k] = right[k]
                box[:, :, 3][k] = bottom[k]

                # Compute Class target
                cls[:, :, 0][k] = 1

                # Compute Centerness target from LTRB
                lr = np.stack([left, right])
                tb = np.stack([top, bottom])
                centerness = lr.min(0) / lr.max(0) * tb.min(0) / tb.max(0)
                ctr[:, :, 0][k] = np.sqrt(centerness[k])

            out[f'cls_{i}'] = cls
            out[f'ctr_{i}'] = ctr
            out[f'box_{i}'] = box
        return out


class ResizeAndPad(object):
    def __init__(self, side):
        self.side = side

    def resize(self, image, target):
        old_height, old_width = image.shape[:2]

        if old_height < old_width:
            new_height = int(self.side * (old_height / old_width))
            new_width = self.side
        else:
            new_width = int(self.side * (old_width / old_height))
            new_height = self.side

        image = resize(image, (new_height, new_width))

        target[:, 0::2] = target[:, 0::2] * (new_width / old_width)
        target[:, 1::2] = target[:, 1::2] * (new_height / old_height)

        return image, target

    def pad(self, image, target):
        h, w = image.shape[:2]

        h = max(self.side - h, 0)
        w = max(self.side - w, 0)
        h2 = h // 2
        w2 = w // 2

        image = np.pad(image, [[h2, h - h2], [w2, w - w2], [0, 0]], mode='constant', constant_values=0)

        target[:, 0::2] += w2
        target[:, 1::2] += h2

        h, w = image.shape[:2]
        assert h == w == self.side

        return image, target

    def __call__(self, image, target):
        image, target = self.resize(image, target)
        image, target = self.pad(image, target)
        return image, target
