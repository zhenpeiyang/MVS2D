from __future__ import absolute_import, division, print_function
import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from time import time
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2

cv2.setNumThreads(0)
import glob
import utils
import torch.nn.functional as F
from utils import npy
import os.path as osp
import re
from utils import *


class DTU(data.Dataset):
    def __init__(self, data_split, opt):
        super(DTU, self).__init__()
        self.data_split = data_split
        self.opt = opt
        self.output_scale = self.opt.output_scale
        self.output_width = self.opt.width // (2**self.output_scale)
        self.output_height = self.opt.height // (2**self.output_scale)
        self.data_path = './data/DTU_hr/'

        self.is_train = data_split == 'train'
        if self.data_split == 'train':
            lighting_set = [0, 1, 2, 3, 4, 5, 6]
            data_set = [
                2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68,
                69, 70, 71, 72, 74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93,
                94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108,
                109, 111, 112, 113, 115, 116, 119, 120, 121, 122, 123, 124,
                125, 126, 127, 128
            ]
        elif self.data_split == 'val':
            data_set = [
                3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86,
                106, 117
            ]
            lighting_set = [3]
            if self.opt.debug:
                lighting_set = [0, 1, 2, 3, 4, 5, 6]
                data_set = [
                    2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42,
                    44, 45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64,
                    65, 68, 69, 70, 71, 72, 74, 76, 83, 84, 85, 87, 88, 89, 90,
                    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                    104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128
                ]
        elif self.data_split == 'test':
            data_set = [
                1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49,
                62, 75, 77, 110, 114, 118
            ]
            lighting_set = [3]
        self.cluster_file_path = osp.join(self.data_path, "Cameras_1/pair.txt")
        self.cluster_list = open(self.cluster_file_path).read().split()
        self.path_list = self._load_dataset(data_set, lighting_set)
        self.num_frame = self.opt.num_frame if self.data_split == 'train' else self.opt.num_frame_test

    def _load_dataset(self, dataset, lighting_set):
        path_list = []
        for ind in dataset:
            if self.data_split in ['train', 'val']:
                image_folder = osp.join(self.data_path,
                                        "Rectified/scan{}".format(ind))
                cam_folder = osp.join(self.data_path, "Cameras_1")
                depth_folder = osp.join(self.data_path,
                                        "Depths_raw/scan{}".format(ind))
            else:
                image_folder = osp.join(self.data_path,
                                        "Eval/scan{}".format(ind))
                cam_folder = osp.join(self.data_path, "Cameras_1")
                depth_folder = ''

            for lighting_ind in lighting_set:
                # for each reference image
                for p in range(0, int(self.cluster_list[0])):
                    #p = 41
                    paths = {}
                    # pts_paths = []
                    view_image_paths = []
                    view_cam_paths = []
                    view_depth_paths = []

                    # ref image
                    ref_index = int(self.cluster_list[22 * p + 1])
                    ref_image_path = osp.join(
                        image_folder, "rect_{:03d}_{}_r5000.png".format(
                            ref_index + 1, lighting_ind))
                    ref_cam_path = osp.join(cam_folder,
                                            "{:08d}_cam.txt".format(ref_index))
                    ref_depth_path = osp.join(
                        depth_folder, "depth_map_{:04d}.pfm".format(ref_index))

                    view_image_paths.append(ref_image_path)
                    view_cam_paths.append(ref_cam_path)
                    view_depth_paths.append(ref_depth_path)

                    # view images
                    for view in range(9):
                        view_index = int(self.cluster_list[22 * p + 2 * view +
                                                           3])
                        view_image_path = osp.join(
                            image_folder, "rect_{:03d}_{}_r5000.png".format(
                                view_index + 1, lighting_ind))
                        view_cam_path = osp.join(
                            cam_folder, "{:08d}_cam.txt".format(view_index))
                        view_depth_path = osp.join(
                            depth_folder,
                            "depth_map_{:04d}.pfm".format(view_index))
                        view_image_paths.append(view_image_path)
                        view_cam_paths.append(view_cam_path)
                        view_depth_paths.append(view_depth_path)
                    paths["view_image_paths"] = view_image_paths
                    paths["view_cam_paths"] = view_cam_paths
                    paths["view_depth_paths"] = view_depth_paths

                    path_list.append(paths)

        return path_list

    def get_color_aug(self):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter.get_params(brightness, contrast,
                                                      saturation, hue)
        return color_aug

    def __len__(self):
        return len(self.path_list)

    def parse(self, index):
        sample = self.path_list[index]
        scene = sample['view_image_paths'][0].split('/')[-2]
        frame_ids = [
            x.split('/')[-1].split('.')[0] for x in sample['view_image_paths']
        ]
        return scene, frame_ids

    def __getitem__(self, idx):
        while True:
            try:
                data = self.__getitem__helper(idx)
                break
            except Exception as e:
                print(e)
                idx = np.random.choice(self.__len__(), 1)[0]
        return data

    def __getitem__helper(self, index):
        if self.opt.debug:
            index = 0

        scale = 1
        inputs = {}
        index = index % self.__len__()
        inputs['index'] = index

        paths = self.path_list[index].copy()
        if self.data_split == 'train':
            idx = [0] + list(
                1 + np.random.choice(9, self.num_frame - 1, replace=False))
        else:
            idx = list(np.arange(self.num_frame))
        paths["view_image_paths"] = [paths["view_image_paths"][x] for x in idx]
        if self.opt.random_lighting and self.data_split == 'train':
            for _, x in enumerate(paths['view_image_paths']):
                ind = np.random.randint(7)
                paths['view_image_paths'][_] = x[:-11] + f"{ind}_r5000.png"

        paths["view_cam_paths"] = [paths["view_cam_paths"][x] for x in idx]
        paths["view_depth_paths"] = [paths["view_depth_paths"][x] for x in idx]

        images = []
        cams = []
        for view in range(self.num_frame):
            image = cv2.imread(paths["view_image_paths"][view])
            cam = self.load_cam_dtu(open(paths["view_cam_paths"][view]),
                                    num_depth=128,
                                    interval_scale=1.6)

            images.append(image)
            cams.append(cam)

        depth = np.ones([1200, 1600]) * 500
        if self.data_split != 'test':
            depth = self.load_pfm(paths["view_depth_paths"][0])[0]

            # mask out-of-range depth pixels (in a relaxed range)
            depth[depth > self.opt.max_depth] = 0
            depth[depth < self.opt.min_depth] = 0

            inputs[("depth_gt", 0, 0)] = cv2.resize(
                depth, (self.opt.width, self.opt.height),
                interpolation=cv2.INTER_NEAREST)[None, :, :].astype('float32')
        else:
            inputs[("depth_gt", 0, 0)] = cv2.resize(
                depth, (self.opt.width, self.opt.height),
                interpolation=cv2.INTER_NEAREST)[None, :, :].astype('float32')

        img_list = np.stack(images, axis=0)
        cam_params_list = np.stack(cams, axis=0)
        cam_params_list_old = cam_params_list.copy()
        new_img_list = []
        if not self.opt.disable_color_aug:
            do_color_aug = self.is_train and random.random() > 0.5
        else:
            do_color_aug = False
        for i in range(self.num_frame):
            color = cv2.resize(img_list[i], (self.opt.width, self.opt.height))
            if do_color_aug:
                color_aug = self.get_color_aug()
                color = torch.from_numpy(color).permute(2, 0, 1) / 255.
                anchors = transforms.ToPILImage()(color)
                color = np.array(color_aug(anchors)).astype('float32')
            new_img_list.append(color)
        depth = cv2.resize(depth, (self.opt.width, self.opt.height),
                           interpolation=cv2.INTER_NEAREST)
        img_list = np.stack(new_img_list, axis=0)

        cam_params_list[:, 1, 0, :] *= (self.opt.width) / 1600.
        cam_params_list[:, 1, 1, :] *= (self.opt.height) / 1200.

        scene = paths['view_image_paths'][0].split('/')[-2]
        frame_ids = [
            x.split('/')[-1].split('.')[0] for x in paths['view_image_paths']
        ]
        inputs['filenames'] = scene + '-' + '-'.join('%s' % x
                                                     for x in frame_ids)

        ho, wo = self.opt.height // scale, self.opt.width // scale
        cam_params_list[:, 1, :2, :] /= scale

        for i in range(len(frame_ids)):
            K = cam_params_list[i, 1, :3, :3]

            inv_K = np.linalg.inv(K)
            K_pool = {}
            inv_K_pool = {}
            K_pool[(1200, 1600)] = cam_params_list_old[i, 1, :3, :3]
            for s in range(4):
                K_pool[(ho // 2**s, wo // 2**s)] = K.copy().astype('float32')
                K_pool[(ho // 2**s, wo // 2**s)][:2, :] /= 2**s

            for k, v in K_pool.items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inv_K_pool[k] = np.linalg.inv(K44).astype('float32')
            inputs[('K_pool', i)] = K_pool
            inputs[('inv_K_pool', i)] = inv_K_pool
            inputs[("K", i, 2)] = K.astype('float32')
            inputs[("inv_K", i, 2)] = inv_K.astype('float32')

            inputs[(
                "color", i,
                0)] = img_list[i].transpose(2, 0, 1).astype('float32') / 255.0
            inputs[("pose", i)] = cam_params_list[i][0].astype('float32')
            inputs[("pose_inv",
                    i)] = np.linalg.inv(inputs[("pose", i)]).astype('float32')
            if i == 0:
                inputs[("depth_gt", i, 2)] = cv2.resize(
                    depth, (self.opt.width // scale, self.opt.height // scale),
                    interpolation=cv2.INTER_NEAREST)[None, :, :].astype(
                        'float32')

        cam_params_list[:, 1, :2, :] *= scale
        inputs[('inv_K_pool',
                0)][(self.opt.height,
                     self.opt.width)] = np.linalg.inv(cam_params_list[0, 1])

        inputs = self.compute_projection_matrix(inputs)
        inputs['num_frame'] = self.num_frame

        return inputs

    def load_cam_dtu(self, file, num_depth=0, interval_scale=1.0):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = num_depth
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        return cam

    def load_pfm(self, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                             file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

    def compute_projection_matrix(self, inputs):
        for i in range(self.num_frame):
            inputs[("proj", i)] = {}
            for k, v in inputs[('K_pool', i)].items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inputs[("proj",
                        i)][k] = np.matmul(K44, inputs[("pose",
                                                        i)]).astype('float32')
        return inputs
