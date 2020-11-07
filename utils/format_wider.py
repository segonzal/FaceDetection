import os
import numpy as np
import argparse
from PIL import Image, ImageDraw
from itertools import chain, zip_longest as zipl


def read_labels(path, subset):
    outputs = {'img': [], 'box': [], 'kps': []}
    with open(os.path.join(path, subset, 'label.txt'), 'r') as fp:
        for line in fp:
            if line.startswith('#'):
                outputs['img'].append(line[2:-1])
                outputs['box'].append([])
                outputs['kps'].append([])
            else:
                numbers = line[:-1].split()
                if len(numbers) > 0:
                    box = np.int32(list(map(int, numbers[:4])))
                    if subset == 'train':
                        box[2] += box[0]
                        box[3] += box[1]
                    outputs['box'][-1].append(box)
                if len(numbers) > 4:
                    kps = np.float32(list(map(float, numbers[4:-1]))).reshape(5, 3)
                    kps = kps[:, :2].flatten()
                    outputs['kps'][-1].append(kps)
    return outputs


def intersection(abox, bbox):
    x_left   = max(abox[0], bbox[0])
    y_top    = max(abox[1], bbox[1])
    x_right  = min(abox[2], bbox[2])
    y_bottom = min(abox[3], bbox[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    return intersection_area


def split_image(image, boxes, keypoints, x_splits, y_splits):
    width, height = image.size
    size_x = width // x_splits
    size_y = height // y_splits
    for i in range(x_splits):
        for j in range(y_splits):
            img_box = (i * size_x, j * size_y, (i + 1) * size_x, (j + 1) * size_y)
            new_image = image.crop(img_box)

            new_boxes = []
            new_kps = []
            for box, kps in zipl(boxes, keypoints, fillvalue=None):

                if intersection(box, img_box) > 0:
                    box = np.float32(box).reshape(-1, 2)
                    box -= img_box[:2]
                    new_boxes.append(box.flatten().tolist())

                    if kps is None:
                        kps = -np.ones(10)
                    else:
                        kps = np.float32(kps).reshape(-1, 2)
                        kps -= img_box[:2]
                    new_kps.append(kps.flatten().tolist())

            yield new_image, new_boxes, new_kps


def resize_image(image, boxes, keypoints, new_size, enlarge=True):
    new_width, new_height = new_size
    old_width, old_height = image.size

    new_width  = new_width if enlarge else min(new_width, old_width)
    new_height = new_height if enlarge else min(new_height, old_height)

    rnd = min(new_width / old_width, new_height / old_height)
    new_width = int(rnd * old_width)
    new_height = int(rnd * old_height)

    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    boxes = (rnd * np.float32(boxes)).tolist()
    keypoints = (rnd * np.float32(keypoints))
    keypoints[keypoints < 0] = -1
    keypoints = keypoints.tolist()

    return image, boxes, keypoints


def prepare_dataset(path, out_path, out_size, show_boxes=False):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for subset in ['val', 'train']:
        os.makedirs(os.path.join(out_path, subset, 'images'), exist_ok=True)
        out_file = open(os.path.join(out_path, subset, 'labels.txt'), 'w')
        i = 1

        data = read_labels(path, subset)
        for image_path, box, kps in zip(data['img'], data['box'], data['kps']):

            with Image.open(os.path.join(path, subset, 'images', image_path)) as img:
                width, height = img.size
                x_splits = int((width / height) + 0.5)
                y_splits = int((height / width) + 0.5)

                x_splits = 1 if x_splits <= 2 else x_splits
                y_splits = 1 if y_splits <= 2 else y_splits

                for out_img, out_box, out_kps in split_image(img, box, kps, x_splits, y_splits):
                    # drop slices with no boxes
                    if len(out_box) == 0:
                        continue

                    out_img, out_box, out_kps = resize_image(out_img, out_box, out_kps, (out_size, out_size))

                    fname = f"{i:06d}.jpg"
                    i += 1
                    print(subset, fname)
                    width, height = out_img.size
                    out_file.write(f'# {fname} {width} {height} {len(out_box)} {subset}\n')

                    rect = ImageDraw.Draw(out_img)
                    for b, k in zip(out_box, out_kps):

                        out_file.write(' '.join(chain(
                            map(lambda s: str(int(s)), b),
                            map(str, k))) + '\n')

                        if show_boxes:
                            rect.rectangle(b)
                            for p in range(5):
                                rect.text((k[2*p], k[2*p+1]), f'{p:d}')

                    out_img.save(os.path.join(out_path, subset, 'images', fname))


def main():
    parser = argparse.ArgumentParser('Format WIDER')
    parser.add_argument('in_path', type=str, help='Path to were the train and val folders of WIDER is stored')
    parser.add_argument('out_path', type=str, help='Where to store the formated data')
    parser.add_argument('--show-boxes', type=bool, default=False, help='Draw bounding boxes and keypoints')
    parser.add_argument('--out-size', type=int, default=1024, help='Size of the big side of the image (default 1024)')

    args = parser.parse_args()

    prepare_dataset(args.in_path, args.out_path, args.out_size, show_boxes=args.show_boxes)


if __name__ == '__main__':
    main()
