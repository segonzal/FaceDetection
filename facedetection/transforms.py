import torch
import numpy as np
import math
from imgaug.augmentables.bbs import BoundingBoxesOnImage


class ImgaugWrapper(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, image, target):
        image = np.float32(image)
        bbs = BoundingBoxesOnImage.fill_from_xyxy_array_(target)
        image, bbs = self.seq(image=image, bounding_boxes=bbs)
        target = bbs.to_xyxy_array()
        return image, target


class EncodeTarget(object):
    def __init__(self, strides, sizes):
        self.strides = strides
        self.sizes = sizes

    def encode_boxes(self, boxes, img_width, img_height):
        # TODO: Check encode target for index 8 with default seeds,
        #  there is a size mismatch between target and predicted outputs.
        # Order targets by area (bigger first)
        boxes = boxes.tolist()
        boxes.sort(key=lambda t: (t[1][0] - t[0][0]) * (t[1][1] - t[0][1]), reverse=True)

        target_cls, target_ctr, target_box, target_mask = [], [], [], []
        for i, (stride, min_size, max_size) in enumerate(zip(self.strides, [0, *self.sizes], [*self.sizes, np.inf])):
            map_height = math.ceil(img_height / stride) - 2
            map_width = math.ceil(img_width / stride) - 2

            cls = np.zeros((map_height, map_width, 1), dtype=np.float32)
            ctr = np.zeros((map_height, map_width, 1), dtype=np.float32)
            box = np.zeros((map_height, map_width, 4), dtype=np.float32)

            y = np.linspace(np.floor(stride / 2), img_height, map_height)
            x = np.linspace(np.floor(stride / 2), img_width, map_width)
            xx, yy = np.meshgrid(x, y)

            for (x0, y0), (x1, y1) in boxes:
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

            mask = np.ones_like(ctr, dtype=bool)

            target_cls.append(cls.reshape(-1, 1))
            target_ctr.append(ctr.reshape(-1, 1))
            target_box.append(box.reshape(-1, 4))
            target_mask.append(mask.reshape(-1, 1))
        return target_cls, target_ctr, target_box, target_mask

    def __call__(self, batch):
        img, box = batch
        img_height, img_width = img.shape[:2]

        img = [np.rollaxis(img, 2, 0)]
        cls, ctr, box, mask = self.encode_boxes(box, img_width, img_height)

        return img, cls, ctr, box, mask


class ToTorch(object):
    def __call__(self, image, target):
        image = torch.from_numpy(np.rollaxis(image, 2, 0))
        target = list(map(torch.from_numpy, target))
        return image, target