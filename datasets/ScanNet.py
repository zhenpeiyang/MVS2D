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


class ScanNet(data.Dataset):
    def __init__(self, data_split, opt):
        super(ScanNet, self).__init__()
        self.data_split = data_split
        self.opt = opt
        self.output_scale = self.opt.output_scale
        self.output_width = self.opt.width // (2**self.output_scale)
        self.output_height = self.opt.height // (2**self.output_scale)

        split_file = "./splits/ScanNet_%d_frame_%s.npy" % (
            self.opt.num_frame, data_split)
            self.samples = np.load(split_file, allow_pickle=True).tolist()

        self.is_train = data_split == 'train'
        if self.opt.perturb_pose and data_split == 'train':
            self.pose_cache = np.load('data/ScanNet_%d_frame_jitter_pose.npy' %
                                      self.opt.num_frame,
                                      allow_pickle=True).item()

    def __len__(self):
        return len(self.samples)

    def parse(self, index):
        sample = self.samples[index]
        scene = sample['scene']
        frame_ids = [sample['target']]
        assert (len(sample['refs']) == (self.opt.num_frame - 1))
        if not self.opt.mono:
            frame_ids.extend(sample['refs'])
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
        inputs = {}
        index = index % self.__len__()
        inputs['index'] = index
        scene, frame_ids = self.parse(index)
        inputs['filenames'] = scene + '-' + '_'.join('%04d' % x
                                                     for x in frame_ids)

        inputs = self.get_K(scene, inputs)
        for i, frame_id in enumerate(frame_ids):

            inputs[("color", i, 0)] = self.get_color(
                scene,
                frame_id,
            )
            inputs[("pose", i)] = self.get_pose(scene,
                                                frame_id).astype('float32')
            inputs[("pose_inv",
                    i)] = np.linalg.inv(inputs[("pose", i)]).astype('float32')

            if i == 0:
                inputs[("depth_gt", i, self.output_scale)] = self.get_depth(
                    scene,
                    frame_id,
                    size=(self.output_width,
                          self.output_height)).astype('float32')[None, :, :]
                inputs[("depth_gt", i, 0)] = self.get_depth(
                    scene, frame_id,
                    size=(640, 480)).astype('float32')[None, :, :]

        if self.opt.perturb_pose and self.is_train:
            inputs = self.get_perturb_pose(inputs)

        inputs = self.compute_projection_matrix(inputs)
        inputs['num_frame'] = self.opt.num_frame

        return inputs

    def get_perturb_pose(self, inputs):
        fID = inputs['filenames']
        if fID in self.pose_cache:
            for i in range(self.opt.num_frame):
                inputs[('pose', i)] = self.pose_cache[fID][('pose', i)]
                inputs[('pose_inv', i)] = np.linalg.inv(inputs[('pose', i)])
        return inputs

    def get_pose(self, scene, frame_id):
        path = os.path.join(self.opt.data_path, scene, 'pose',
                            "%06d.txt" % frame_id)
        pose = np.loadtxt(path).astype('float32')
        return pose

    def get_color(self, folder, frame_id):
        path = os.path.join(self.opt.data_path, folder, 'rgb',
                            "%06d.jpg" % frame_id)
        color = cv2.imread(path)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = torch.from_numpy(color).permute(2, 0, 1) / 255.
        if not self.opt.disable_color_aug:
            do_color_aug = self.is_train and random.random() > 0.5
        else:
            do_color_aug = False
        if do_color_aug:
            color_aug = self.get_color_aug()
            anchors = transforms.ToPILImage()(color)
            color = torch.from_numpy(
                np.array(color_aug(anchors)).astype('float32').transpose(
                    2, 0, 1) / 255.0)
        return color

    def get_color_aug(self):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter.get_params(brightness, contrast,
                                                      saturation, hue)
        return color_aug

    def get_depth(self, folder, frame_id, size=None):
        path = os.path.join(self.opt.data_path, folder, 'depth',
                            "%06d.png" % frame_id)
        depth_gt = cv2.imread(path, 2) / 1000.0
        if size is not None:
            depth_gt = cv2.resize(depth_gt,
                                  size,
                                  interpolation=cv2.INTER_NEAREST)
        return depth_gt

    def get_K(self, scene, inputs):
        path = os.path.join(self.opt.data_path, scene, "intrinsics.txt")
        K = np.loadtxt(path).astype('float32')
        inv_K = np.linalg.inv(K)
        gt_K = K.copy()
        gt_K[:2, :] /= 2**self.output_scale
        gt_inv_K = np.linalg.inv(gt_K)
        K_pool = {}
        ho, wo = 480, 640
        for i in range(6):
            K_pool[(ho // 2**i, wo // 2**i)] = K.copy().astype('float32')
            K_pool[(ho // 2**i, wo // 2**i)][:2, :] /= 2**i

        inputs[("inv_K_pool", 0)] = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3, :3] = v
            inputs[("inv_K_pool", 0)][k] = np.linalg.inv(K44).astype('float32')

        inputs[("inv_K", 0)] = torch.from_numpy(inv_K.astype('float32'))
        inputs[("inv_K", self.output_scale)] = torch.from_numpy(
            gt_inv_K.astype('float32'))
        inputs[("K", 0)] = torch.from_numpy(K.astype('float32'))
        inputs[("K",
                self.output_scale)] = torch.from_numpy(gt_K.astype('float32'))
        inputs['K_pool'] = K_pool
        return inputs

    def compute_projection_matrix(self, inputs):
        for i in range(self.opt.num_frame):
            inputs[("proj", i)] = {}
            for k, v in inputs['K_pool'].items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inputs[("proj",
                        i)][k] = np.matmul(K44, inputs[("pose",
                                                        i)]).astype('float32')
        return inputs
