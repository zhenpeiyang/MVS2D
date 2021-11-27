import numpy as np
import os
import scipy.io as sio
import glob
import open3d as o3d
from pykdtree.kdtree import KDTree
from scipy import spatial


def MaxDistCP(Qto, Qfrom, BB, MaxDist):
    dst = np.ones([len(Qfrom)]) * MaxDist
    mask = (Qfrom[:,0] >= BB[0,0]) & (Qfrom[:,0] <BB[1,0])\
            &(Qfrom[:,1] >= BB[0,1]) & (Qfrom[:,1] <BB[1,1])\
            &(Qfrom[:,2] >= BB[0,2]) & (Qfrom[:,2] <BB[1,2])
    tree = KDTree(Qto)
    dst, _ = tree.query(Qfrom, 1)
    dst[~mask] = MaxDist
    dst = np.clip(dst, None, MaxDist)
    return dst


def reducePts(pts, dst, verbose=False):
    n = len(pts)
    indexSet = np.ones([n]).astype('bool')
    RandOrd = np.random.permutation(n)
    tree = spatial.cKDTree(pts)
    Chunks = np.arange(0, n, min(1e7, n - 1))
    Chunks[-1] = n - 1
    Chunks = Chunks.astype('int')
    idx = np.arange(n)
    for i in range(0, len(Chunks) - 1):
        if verbose:
            print(i, len(Chunks) - 1)
        Range = idx[Chunks[i]:Chunks[i + 1]]
        res = tree.query_ball_point(pts[RandOrd[Range]], dst, workers=8)
        for j in range(len(res)):
            id = RandOrd[j + Chunks[i]]
            if indexSet[id]:
                indexSet[res[j]] = False
                indexSet[id] = True
    print('down sampling factor ', (indexSet).mean())
    index = np.where(indexSet)
    return pts[index], index


def dtu_pyeval_single(index,
                      pred_fn,
                      gt_dir,
                      voxel_down_sample,
                      fn=None,
                      down_sample=False):

    margin = 10
    OutlierDist = 20
    MaxDist = 60
    pred = o3d.io.read_point_cloud(pred_fn)

    if down_sample:
        if voxel_down_sample:
            pred = o3d.geometry.PointCloud.voxel_down_sample(pred,
                                                             voxel_size=0.2)
            pred = np.array(pred.points)
        else:
            pred = np.array(pred.points)
            pred, _ = reducePts(pred, 0.2)
    else:
        pred = np.array(pred.points)
    gt = o3d.io.read_point_cloud(
        f"{gt_dir}/Points/stl/stl{index:03d}_total.ply")
    gt = np.array(gt.points)
    MaskName = f"{gt_dir}/ObsMask/ObsMask{index}_{margin}.mat"
    anno = sio.loadmat(MaskName)
    has_plane = False
    plane_fn = f"{gt_dir}/ObsMask/Plane{index}.mat"
    if os.path.exists(plane_fn):
        has_plane = True
        P = sio.loadmat(plane_fn)['P']

    Ddata = MaxDistCP(gt, pred, anno['BB'], MaxDist)
    Dstl = MaxDistCP(pred, gt, anno['BB'], MaxDist)
    Qv = (pred - anno['BB'][0][None, :]) / (float(anno['Res'][0, 0]))
    ObsMask = anno['ObsMask']
    Qv = np.round(Qv).astype('int')
    Midx1=(Qv[:,0]>=0) & (Qv[:,0]<ObsMask.shape[0]) \
            &(Qv[:,1]>=0) & (Qv[:,1]<ObsMask.shape[1]) \
            &(Qv[:,2]>=0) & (Qv[:,2]<ObsMask.shape[2])

    DataInMask = np.zeros([len(pred)])
    DataInMask[Midx1] = ObsMask[Qv[Midx1, 0], Qv[Midx1, 1], Qv[Midx1, 2]]
    DataInMask = DataInMask.astype('bool')
    if has_plane:
        StlAbovePlane = (np.concatenate(
            (gt, np.ones([len(gt), 1])), 1) @ P)[:, 0] > 0

        Dstl = Dstl[StlAbovePlane]
    Dstl = Dstl[Dstl < OutlierDist]

    Ddata = Ddata[DataInMask]
    Ddata = Ddata[Ddata < OutlierDist]

    Comp = np.mean(Dstl)
    Acc = np.mean(Ddata)
    Avg = (Acc + Comp) / 2.0
    return Acc, Comp, Avg


def dtu_pyeval(prediction_dir,
               gt_dir,
               voxel_down_sample,
               fn=None,
               down_sample=False):

    test_indices = [
        1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75,
        77, 110, 114, 118
    ]
    MeanStl = {}
    MeanData = {}
    MeanAvg = {}

    margin = 10
    OutlierDist = 20
    MaxDist = 60

    for index in test_indices:
        pred_fn = glob.glob(f"{prediction_dir}/*{index:03d}_l3.ply")
        if len(pred_fn) == 0:
            print(pred_fn, ' Not found ')
            continue
        pred = o3d.io.read_point_cloud(pred_fn[0])

        if down_sample:
            if voxel_down_sample:
                pred = o3d.geometry.PointCloud.voxel_down_sample(
                    pred, voxel_size=0.2)
                pred = np.array(pred.points)
            else:
                pred = np.array(pred.points)
                pred, _ = reducePts(pred, 0.2)
        else:
            pred = np.array(pred.points)
        gt = o3d.io.read_point_cloud(
            f"{gt_dir}/Points/stl/stl{index:03d}_total.ply")
        gt = np.array(gt.points)
        MaskName = f"{gt_dir}/ObsMask/ObsMask{index}_{margin}.mat"
        anno = sio.loadmat(MaskName)
        has_plane = False
        plane_fn = f"{gt_dir}/ObsMask/Plane{index}.mat"
        if os.path.exists(plane_fn):
            has_plane = True
            P = sio.loadmat(plane_fn)['P']

        Ddata = MaxDistCP(gt, pred, anno['BB'], MaxDist)
        Dstl = MaxDistCP(pred, gt, anno['BB'], MaxDist)
        Qv = (pred - anno['BB'][0][None, :]) / (float(anno['Res'][0, 0]))
        ObsMask = anno['ObsMask']
        Qv = np.round(Qv).astype('int')
        Midx1=(Qv[:,0]>=0) & (Qv[:,0]<ObsMask.shape[0]) \
                &(Qv[:,1]>=0) & (Qv[:,1]<ObsMask.shape[1]) \
                &(Qv[:,2]>=0) & (Qv[:,2]<ObsMask.shape[2])

        DataInMask = np.zeros([len(pred)])
        DataInMask[Midx1] = ObsMask[Qv[Midx1, 0], Qv[Midx1, 1], Qv[Midx1, 2]]
        DataInMask = DataInMask.astype('bool')

        if has_plane:
            StlAbovePlane = (np.concatenate(
                (gt, np.ones([len(gt), 1])), 1) @ P)[:, 0] > 0

            Dstl = Dstl[StlAbovePlane]
        Dstl = Dstl[Dstl < OutlierDist]

        Ddata = Ddata[DataInMask]
        Ddata = Ddata[Ddata < OutlierDist]

        MeanStl[index] = np.mean(Dstl)
        MeanData[index] = np.mean(Ddata)
        MeanAvg[index] = (MeanStl[index] + MeanData[index]) / 2.0

    indices = sorted(MeanData.keys())
    Acc = np.mean([MeanData[x] for x in indices])
    Comp = np.mean([MeanStl[x] for x in indices])
    Avg = np.mean([MeanAvg[x] for x in indices])

    print('Acc ', Acc)
    print('Comp ', Comp)
    print('F-score ', Avg)

    if fn is not None:
        with open(fn, "w") as f:
            f.write(f"index Acc Comp Mean\n")
            for index in test_indices:
                if index in indices:
                    f.write(
                        f"{index} {MeanData[index]:.4f} {MeanStl[index]:.4f} {MeanAvg[index]:.4f}\n"
                    )
                else:
                    f.write(f"{index} nan nan nan\n")
            f.write(f"Mean {Acc:.4f} {Comp:.4f} {Avg:.4f}")

    return MeanData, MeanStl, MeanAvg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="MVS2D options")
    parser.add_argument("--voxel_down_sample", help="", action="store_true")
    parser.add_argument("--gt_dir",
                        help="",
                        type=str,
                        default='/home/yzp/Documents/SampleSet/MVS Data/')
    parser.add_argument(
        "--prediction_dir",
        help="",
        type=str,
        default='/home/yzp/Documents/SampleSet/dtu_pyfusion_config6_v2_latest')
    args = parser.parse_args()
    dtu_pyeval(args.prediction_dir,
               args.gt_dir,
               args.voxel_down_sample,
               fn=f"pyeval_mvs2d.txt")
