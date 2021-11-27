import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool
import time
from utils import *
import sys
import cv2
from PIL import Image
from dtu_pyeval import reducePts

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--n_views', type=int, default=5, help='num of view')
parser.add_argument('--n_proc', type=int, default=0, help='num of view')
parser.add_argument(
    '--dense_folder',
    default='experiments/release/DTU/exp_resol/recon/eval_000/prediction/',
    help='output dir')
parser.add_argument('--outdir',
                    default='./tmp/dtu_pyfusion',
                    help='output dir')
parser.add_argument('--mask_background',
                    action='store_true',
                    help='display depth images and masks')
parser.add_argument('--geo_pixel_thres',
                    type=float,
                    default=1,
                    help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres',
                    type=float,
                    default=0.01,
                    help='depth threshold for geometric consistency filtering')
parser.add_argument('--conf_thres',
                    type=float,
                    default=3.5,
                    help='threshold for photometric consistency filtering')
parser.add_argument('--att_thres',
                    type=float,
                    default=0.25,
                    help='threshold for photometric consistency filtering')
parser.add_argument('--use_conf_thres', type=int, default=0)
parser.add_argument('--num_consistent', type=int, default=2)

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])


def read_pfm(file):
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

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
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
    return data


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32,
                               sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]),
                               dtype=np.float32,
                               sep=' ').reshape((3, 3))

    return intrinsics, extrinsics


# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth.astype(np.float32) * 255
    Image.fromarray(depth).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src,
                         intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack(
            (x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src,
                                  x_src,
                                  y_src,
                                  interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack(
            (xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
        np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height,
                                                    width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height,
                                               width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height,
                                               width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref,
                                depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres, geo_depth_thres):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src,
        extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref)**2 + (y2d_reprojected - y_ref)**2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres,
                          relative_depth_diff < geo_depth_thres)
    #(dist < geo_pixel_thres).mean()
    #(relative_depth_diff < geo_depth_thres).mean()
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scanID, out_folder, plyfilename, geo_pixel_thres,
                 geo_depth_thres, conf_thres, att_thres, use_conf_thres,
                 num_consistent):
    # the pair file
    pair_file = './data/DTU_hr/Cameras_1/pair.txt'
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    base_dir = args.dense_folder
    all_data = {}
    for view in range(49):
        all_data[view] = np.load(glob.glob(
            f"{base_dir}/scan{scanID}/scan{scanID}-rect_{view+1:03d}*npy")[0],
                                 allow_pickle=True).item()
        h, w = 1152, 1536
        K = np.linalg.inv(all_data[view]['inv_K'])
        K[0, :] *= float(w) / all_data[view]['depth_pred'].shape[1]
        K[1, :] *= float(h) / all_data[view]['depth_pred'].shape[0]
        all_data[view]['inv_K'] = np.linalg.inv(K)
        import cv2
        all_data[view]['depth_pred'] = cv2.resize(
            all_data[view]['depth_pred'], (w, h),
            interpolation=cv2.INTER_NEAREST)
        if 'conf' in all_data[view]:
            all_data[view]['conf'] = cv2.resize(all_data[view]['conf'], (w, h))
        if 'att_score' in all_data[view]:
            all_data[view]['att_score'] = cv2.resize(
                all_data[view]['att_score'], (w, h))
        all_data[view]['color'] = cv2.resize(all_data[view]['color'], (w, h))

    m1, m2, m3 = [], [], []
    for ref_view, src_views in pair_data:
        # load the camera parameters
        data = all_data[ref_view]
        ref_intrinsics = np.linalg.inv(data['inv_K'])
        ref_extrinsics = data['pose']
        ref_depth_est = data['depth_pred']
        if use_conf_thres:
            confidence = data['conf']
            photo_mask = confidence < conf_thres
        else:
            confidence = data['att_score']
            photo_mask = confidence > att_thres
        # load the reference image
        ref_img = data['color']

        all_srcview_depth_ests = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            data_src = all_data[src_view]
            src_intrinsics = np.linalg.inv(data_src['inv_K'])
            src_extrinsics = data_src['pose']
            src_depth_est = data_src['depth_pred']

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est,
                src_intrinsics, src_extrinsics, geo_pixel_thres,
                geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (sum(all_srcview_depth_ests) +
                              ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= num_consistent
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)),
            photo_mask)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)),
            geo_mask)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)),
            final_mask)
        os.makedirs(os.path.join(out_folder, "depth_img"), exist_ok=True)

        print(
            "processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}"
            .format(scanID, ref_view, geo_mask.mean(), photo_mask.mean(),
                    final_mask.mean()))

        m1.append(geo_mask.mean())
        m2.append(photo_mask.mean())
        m3.append(final_mask.mean())

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[
            valid_points]

        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        xyz_world = xyz_world.transpose((1, 0))
        if args.mask_background:
            mask = np.linalg.norm(color, axis=1) > 30
            xyz_world = xyz_world[mask]
            color = color[mask]
        vertexs.append(xyz_world)
        vertex_colors.append((color).astype(np.uint8))

    print('avg mask: %f %f %f' % (np.mean(m1), np.mean(m2), np.mean(m3)))
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    print('Voxel-downsampling...')
    vertexs, index = reducePts(vertexs, 0.2, verbose=True)
    vertex_colors = vertex_colors[index]
    print('Write ply file %s' % plyfilename)
    write_ply(plyfilename, vertexs, color=vertex_colors[:, ::-1] / 255.0)
    try:
        size_in_mb = os.path.getsize(plyfilename) >> 20
        print(f"saving the final model to {plyfilename} {size_in_mb} Mb")
    except:
        pass


def process(scan_id):
    out_folder = os.path.join(args.outdir, 'scan%d' % scan_id)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    filter_depth(
        scan_id, out_folder,
        os.path.join(args.outdir, 'mvs2d{:0>3}_l3.ply'.format(scan_id)),
        args.geo_pixel_thres, args.geo_depth_thres, args.conf_thres,
        args.att_thres, args.use_conf_thres, args.num_consistent)


if __name__ == '__main__':
    scans = [
        1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75,
        77, 110, 114, 118
    ]

    os.makedirs(args.outdir, exist_ok=True)
    if args.n_proc == 0:
        for scan_id in scans:
            process(scan_id)
    else:
        with Pool(args.n_proc) as p:
            p.map(process, scans)
