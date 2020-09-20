#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import os
import numpy as np

from base import BaseImageDataset

class CelebADataset(BaseImageDataset):
    def __init__(self, path, transform=None):
        raise NotImplementedError
        super(CelebADataset, self).__init__(
            images,
            targets,
            image_path,
            transform)


# #!/usr/bin/env/python
# # -*- coding: utf-8 -*-
#
# import os
# import numpy as np
# import pandas as pd
#
# from PIL import Image
# from torch.utils.data import Dataset
#
# class CelebADataset(Dataset):
#     list_eval_partition         = 'Eval/list_eval_partition.txt'
#     identity_CelebA             = 'Anno/identity_CelebA.txt'
#     list_attr_celeba            = 'Anno/list_attr_celeba.txt'
#     list_bbox_celeba            = 'Anno/list_bbox_celeba.txt'
#     list_landmarks_celeba       = 'Anno/list_landmarks_celeba.txt'
#     list_landmarks_align_celeba = 'Anno/list_landmarks_align_celeba.txt'
#     img_celeba                  = 'Img/img_celeba'
#     img_align_celeba            = 'Img/img_align_celeba'
#
#     partition = ['partition']
#     identity  = ['identity']
#     attributes = [
#         '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
#         'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
#         'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
#         'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
#         'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
#         'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
#         'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
#         'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
#         'Wearing_Necklace', 'Wearing_Necktie', 'Young']
#     bbox = ['x_1', 'y_1', 'width', 'height']
#     keypoints = [
#         'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x',
#         'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
#
#     def __init__(self, path, transform=None, use_aligned=False, subset=None):
#         self.path = path
#         self.transform = transform
#         self.data =  self._read_annotations(path, use_aligned)
#         self.aligned = use_aligned
#         imfolder = 'img_align_celeba' if use_aligned else 'img_celeba'
#         self.impath = os.path.join(path, 'Img', imfolder)
#
#         if subset in [0,1,2]:
#             self.data = self.data[self.data['partition']==subset]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, key):
#         data = self.data.iloc[key]
#         file_path = os.path.join(self.impath, data.name)
#
#         image = Image.open(file_path).convert('RGB')
#         image = np.float32(image) / 255.0
#
#         keypoints = data[self.keypoints].to_numpy(np.float32).reshape(-1, 2)
#
#         identity = data[self.identity][0]
#         attributes = data[self.attributes].to_numpy(np.float32)
#
#         out = dict(
#             image=image,
#             keypoints=keypoints,
#             identity=identity,
#             attributes=attributes)
#
#         if not self.aligned:
#             out['bbox'] = data[self.bbox].to_numpy(np.float32)
#             # TODO: to x1y1x2y2 (?)
#
#         if self.transform:
#             out = self.transform(out)
#         return out
#
#     def _read_annotations(self, path, use_aligned):
#         file_list = [
#             (self.list_eval_partition, dict(names=self.partition, header=None)),
#             (self.identity_CelebA,     dict(names=self.identity, header=None)),
#             (self.list_attr_celeba,    dict(header=1))]
#         if use_aligned:
#             file_list.append(
#                 (self.list_landmarks_align_celeba, dict(header=1)))
#         else:
#             file_list.extend([
#                 (self.list_bbox_celeba, dict(header=1)),
#                 (self.list_landmarks_celeba, dict(header=1))])
#
#         dataframes = []
#         for filename, kwargs in file_list:
#             dataframes.append(pd.read_csv(
#                     os.path.join(path, filename),
#                     index_col=0,
#                     sep='\s+',
#                     **kwargs))
#             dataframes[-1].index.name = 'image_id'
#         return pd.concat(dataframes, axis=1, join='inner')
