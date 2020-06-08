# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from misc_utils import smpl_to_openpose
from projection_utils import Projection


Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(dataset='openpose', data_folder='data', **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                 keyp_folder='keypoints',
                 calib_dir='',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 depth_folder='Depth',
                 mask_folder='BodyIndex',
                 mask_color_folder='BodyIndexColor',
                 read_depth=False,
                 read_mask=False,
                 mask_on_color=False,
                 depth_scale=1e-3,
                 depth_max=10.0,
                 flip=False,
                 start=0,
                 step=1,
                 scale_factor=1,
                 frame_ids=None,
                 init_mode='sk',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)
        self.img_folder = osp.join(data_folder, img_folder)
        if not os.path.exists(self.img_folder):
            self.img_folder = osp.join(data_folder, "color")
        self.keyp_folder = osp.join(keyp_folder)
        self.depth_folder = os.path.join(data_folder, depth_folder)
        if not os.path.exists(self.depth_folder):
            self.depth_folder = osp.join(data_folder, "depth")
        self.mask_folder = os.path.join(data_folder, mask_folder)
        self.mask_color_folder = os.path.join(data_folder, mask_color_folder)

        self.img_paths = [osp.join(self.img_folder, img_fn)
                          for img_fn in os.listdir(self.img_folder)
                          if img_fn.endswith('.png') or
                          img_fn.endswith('.jpg') and
                          not img_fn.startswith('.')]
        self.img_paths = sorted(self.img_paths)
        if frame_ids is None:
            self.img_paths = self.img_paths[start::step]
        else:
            self.img_paths = [self.img_paths[id -1] for id in frame_ids]

        self.cnt = 0
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.flip = flip
        self.read_depth = read_depth
        self.read_mask = read_mask
        self.scale_factor = scale_factor
        self.init_mode = init_mode
        self.mask_on_color = mask_on_color
        self.projection = Projection(calib_dir)

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        if self.flip:
            img = cv2.flip(img, 1)
        color_im = img.copy()
        img_fn = osp.split(img_path)[1]
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        keypoint_fn = osp.join(self.keyp_folder,
                               img_fn + '_keypoints.json')
        keyp_tuple = read_keypoints(keypoint_fn, use_hands=self.use_hands,
                                    use_face=self.use_face,
                                    use_face_contour=self.use_face_contour)

        if len(keyp_tuple.keypoints) < 1:
            return {}
        keypoints = np.stack(keyp_tuple.keypoints)

        depth_im = None
        if self.read_depth:
            depth_im = cv2.imread(os.path.join(self.depth_folder, img_fn + '.png'), flags=-1).astype(float)
            depth_im = depth_im / self.depth_scale
            depth_im[depth_im > self.depth_max] = 0
            if self.flip:
                depth_im = cv2.flip(depth_im, 1)

        mask = None
        if self.read_mask:
            if self.mask_on_color:
                mask = cv2.imread(os.path.join(self.mask_color_folder, img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
            elif osp.exists(os.path.join(self.mask_folder, img_fn + '.png')):
                mask = cv2.imread(os.path.join(self.mask_folder, img_fn + '.png'), cv2.IMREAD_GRAYSCALE)
                mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
            else:
                mask = np.ones(depth_im.shape, dtype=np.uint8)
                mask[depth_im > 0.3]= 0
                mask[depth_im < 2.0] = 0
            if self.flip:
                mask = cv2.flip(mask, 1)
        elif depth_im is not None:
            mask = np.ones(depth_im.shape, dtype=np.uint8)
            mask[depth_im > 0.3]= 0
            mask[depth_im < 2.0] = 0

        scan_dict = None
        init_trans = None
        if depth_im is not None and mask is not None:
            scan_dict = self.projection.create_scan(mask, depth_im, color_im=color_im, mask_on_color=self.mask_on_color, keypoints=keypoints)
            init_trans = np.mean(scan_dict.get('points'), axis=0)

        output_dict = {'fn': img_fn,
                       'img_path': img_path,
                       'keypoints': keypoints,
                       'img': img,
                       'init_trans': init_trans,
                       'depth_im': depth_im,
                       'mask': mask,
                       'scan_dict':scan_dict}
        if keyp_tuple.gender_gt is not None:
            if len(keyp_tuple.gender_gt) > 0:
                output_dict['gender_gt'] = keyp_tuple.gender_gt
        if keyp_tuple.gender_pd is not None:
            if len(keyp_tuple.gender_pd) > 0:
                output_dict['gender_pd'] = keyp_tuple.gender_pd
                
        depth_output = os.path.join(self.depth_folder, "pointcloud")
        if not os.path.exists(depth_output):
            os.makedirs(depth_output)
        import trimesh
        
        if scan_dict is not None:
            if 'points' in scan_dict.keys():
                m = trimesh.Trimesh(scan_dict['points'], None, process=False)
                m.export(os.path.join(depth_output, img_fn+".ply"))

            if 'keypoints3d' in scan_dict.keys():
                m = trimesh.Trimesh(scan_dict['keypoints3d'], None, process=False)
                m.export(os.path.join(depth_output, img_fn+"_keypoints.ply"))
        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)
