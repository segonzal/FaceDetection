import torch
import numpy as np
import math
import imgaug as ia
import imgaug.augmenters as iaa


class ImgaugWrapper(object):
    def __init__(self, seq):
        self.seq = iaa.Sequential(seq)

    def __call__(self, batch):
        img, box = batch

        bbs = ia.BoundingBoxesOnImage.from_xyxy_array(box.reshape(-1, 4), shape=img.shape)
        img, bbs = self.seq(image=img, bounding_boxes=bbs)
        box = bbs.to_xyxy_array().reshape(-1, 2, 2)

        return img, box


class EncodeTarget(object):
    def __init__(self, strides, sizes):
        self.strides = strides
        self.sizes = sizes

    def encode_boxes(self, boxes, img_width, img_height):
        # Order targets by area (bigger first)
        boxes = boxes.tolist()
        boxes.sort(key=lambda t: (t[1][0] - t[0][0]) * (t[1][1] - t[0][1]), reverse=True)

        target_cls, target_ctr, target_box, target_mask = [], [], [], []
        for i, (stride, min_size, max_size) in enumerate(zip(self.strides, [0, *self.sizes], [*self.sizes, np.inf])):
            map_height = math.ceil(img_height / stride) - 2
            map_width = math.ceil(img_width / stride) - 2

            assert map_height > 0 and map_width > 0, f"The input image (W:{img_width}, H:{img_height}) is too small " \
                                                     f"to reach level {i}:{stride}."

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


class Compose(object):
    def __init__(self, callables=[]):
        self.callables = callables

    def __call__(self, batch):
        for fn in self.callables:
            batch = fn(batch)
        return batch
