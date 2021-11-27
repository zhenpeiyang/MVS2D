from __future__ import absolute_import, division, print_function
import pickle
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


class DeMoN(data.Dataset):
    def __init__(self, data_split, opt):
        super(DeMoN, self).__init__()
        self.data_split = data_split
        self.opt = opt
        self.output_scale = self.opt.output_scale
        self.output_width = self.opt.width // (2**self.output_scale)
        self.output_height = self.opt.height // (2**self.output_scale)
        if data_split == 'test':
            self.data_path = f"{self.opt.data_path}/test"
        else:
            self.data_path = f"{self.opt.data_path}/train"
        split_file = "./splits/DeMoN_samples_%s_2_frame.npy" % data_split
        samples = np.load(split_file, allow_pickle=True).tolist()
        self.samples = self.filter(samples)
        self.is_train = data_split == 'train'
        self.pose_cache = {}

    def get_color_aug(self):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter.get_params(
            brightness, contrast, saturation, hue)
        return color_aug

    def __len__(self):
        return len(self.samples)

    def filter(self, samples):
        if self.opt.filter is not None:
            samples = list(
                filter(lambda x: x['scene'].split('_')[0] in self.opt.filter,
                       samples))
        return samples

    def parse(self, index):
        sample = self.samples[index]
        scene = sample['scene']
        frame_ids = [sample['target']]
        assert len(sample['refs']) == (self.opt.num_frame - 1), "wrong num_frame"
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
            inputs[("color", i, 0)] = self.get_color(scene, frame_id)
            inputs[("pose", i)] = self.get_pose(scene, frame_id).astype('float32')
            inputs[("pose_inv", i)] = np.linalg.inv(inputs[("pose", i)]).astype('float32')

            if i == 0:
                inputs[("depth_gt", i, 0)] = self.get_depth(
                    scene,
                    frame_id,
                    ).astype('float32')[None,:,:]
                inputs[('depth_gt', i, self.output_scale)] = cv2.resize(
                    inputs[("depth_gt", i, 0)][0],
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_NEAREST)[None, :, :]

        inputs = self.compute_projection_matrix(inputs)
        inputs['num_frame'] = self.opt.num_frame
        return inputs

    def compute_projection_matrix(self, inputs):
        for i in range(self.opt.num_frame):
            inputs[("proj", i)] = {}
            for k, v in inputs['K_pool'].items():
                K44 = np.eye(4)
                K44[:3, :3] = v
                inputs[("proj", i)][k] = np.matmul(K44, inputs[("pose", i)]).astype('float32')
        return inputs

    def get_pose(self, scene, frame_id):
        if scene not in self.pose_cache:
            path = os.path.join(self.data_path, scene, "poses.txt")
            pose = np.loadtxt(path).astype('float32')
            self.pose_cache[scene] = pose
        pose = self.pose_cache[scene][frame_id].reshape(3, 4)
        pose = np.concatenate((pose, np.array([0, 0, 0, 1]).reshape(1, -1)))
        return pose

    def get_color(self, folder, frame_id):
        path = os.path.join(self.data_path, folder, "%04d.jpg" % frame_id)
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
            color = transforms.ToTensor()(
                color_aug(anchors))
        return color

    def get_depth(self, folder, frame_id, size=None):

        path = os.path.join(self.data_path, folder, "%04d.npy" % frame_id)
        depth_gt = np.load(path)
        if size is not None:
            depth_gt = cv2.resize(depth_gt,
                                  size,
                                  interpolation=cv2.INTER_NEAREST)
        depth_gt[~np.isfinite(depth_gt)] = 0
        return depth_gt

    def get_K(self, scene, inputs):
        path = os.path.join(self.data_path, scene, "cam.txt")
        K = np.loadtxt(path).astype('float32')
        K_pool = {}
        ho, wo = self.opt.height // 2**self.opt.input_scale, self.opt.width // 2**self.opt.input_scale
        for i in range(6):
            K_pool[(ho // 2**i, wo // 2**i)] = K.copy().astype('float32')
            K_pool[(ho // 2**i, wo // 2**i)][:2, :] /= 2**i
        inputs[("inv_K_pool", 0)] = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3, :3] = v
            inputs[("inv_K_pool", 0)][k] = np.linalg.inv(K44).astype('float32')
        inputs['K_pool'] = K_pool
        return inputs
