#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import torch
import torch.nn as nn
from options import MVS2DOptions, EvalCfg
import networks


def load_data(args):
    h, w = args.height, args.width
    scene = 'scene0708_00'
    frames = [2, 3, 4]
    images = []
    poses = []
    depths = []
    for id in frames:
        images.append(
            torch.from_numpy(
                cv2.imread(f"./demo/{scene}/rgb/{id:06d}.jpg") / 255.).permute(
                    2, 0, 1).unsqueeze(0).float().cuda())
        poses.append(np.loadtxt(f"./demo/{scene}/pose/{id:06d}.txt"))
    depths.append(
        cv2.imread(f"./demo/{scene}/depth/{frames[0]:06d}.png", 2) / 1000.0)
    K = np.loadtxt(f"./demo/{scene}/intrinsics.txt")
    K_pool = {}
    ho, wo = h, w
    for i in range(6):
        K_pool[(ho // 2**i, wo // 2**i)] = K.copy().astype('float32')
        K_pool[(ho // 2**i, wo // 2**i)][:2, :] /= 2**i

    inv_K_pool = {}
    for k, v in K_pool.items():
        K44 = np.eye(4)
        K44[:3, :3] = v
        inv_K_pool[k] = np.linalg.inv(K44).astype('float32')
        inv_K_pool[k] = torch.from_numpy(inv_K_pool[k]).float().cuda()

    proj_mats = []
    for i in range(args.num_frame):
        proj_temp = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3, :3] = v
            proj_temp[k] = np.matmul(K44,
                                     poses[i]).astype('float32')[None, :, :]
            proj_temp[k] = torch.from_numpy(proj_temp[k]).float().cuda()
        proj_mats.append(proj_temp)

    return images, proj_mats, inv_K_pool, depths


options = MVS2DOptions()
opts = options.parse()
opts.num_frame = 3
opts.width = int(640)
opts.height = int(480)
model = networks.MVS2D(opt=opts, pretrained=False).cuda()
pretrained_dict = torch.load("./pretrained_model/scannet/MVS2D/model.pth")
model.load_state_dict(pretrained_dict)
model.eval()

with torch.no_grad():
    imgs, proj_mats, inv_K_pool, depths = load_data(opts)
    assert (len(imgs) == opts.num_frame)
    outputs = model(imgs[0], imgs[1:], proj_mats[0], proj_mats[1:], inv_K_pool)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.imshow(imgs[0][0].data.cpu().numpy().transpose(1, 2, 0))
    ax1.set_title("View0")
    ax1.axis('off')
    ax1 = fig.add_subplot(1, 5, 2)
    ax1.imshow(imgs[1][0].data.cpu().numpy().transpose(1, 2, 0))
    ax1.set_title("View1")
    ax1.axis('off')
    ax1 = fig.add_subplot(1, 5, 3)
    ax1.imshow(imgs[2][0].data.cpu().numpy().transpose(1, 2, 0))
    ax1.set_title("View2")
    ax1.axis('off')
    depth_pred = outputs[('depth_pred', 0)][0, 0].data.cpu().numpy()
    ax1 = fig.add_subplot(1, 5, 4)
    ax1.imshow(depth_pred, cmap='hot', vmin=0.3, vmax=3.0)
    ax1.set_title("Depth Pred")
    ax1.axis('off')

    depth_gt = depths[0]
    ax1 = fig.add_subplot(1, 5, 5)
    ax1.imshow(depth_gt, cmap='hot', vmin=0.3, vmax=3.0)
    ax1.set_title("Depth GT")
    ax1.axis('off')

    fig.savefig('demo.png', bbox_inches='tight', dpi=500)
    print(f"demo results sved in demo.png")
