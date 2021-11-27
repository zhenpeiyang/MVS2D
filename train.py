from __future__ import absolute_import, division, print_function
from open3d import *
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import json
from utils import *
import networks
import os
import glob
import random
import torch.optim as optim
from options import MVS2DOptions, EvalCfg
from trainer_base import BaseTrainer
from hybrid_evaluate_depth import evaluate_depth_maps
from dtu_pyeval import dtu_pyeval
import pprint


class Trainer(BaseTrainer):
    def __init__(self, options):
        super(Trainer, self).__init__(options)

    def build_model(self):
        self.parameters_to_train = []
        self.model = networks.MVS2D(opt=self.opt).cuda()
        self.parameters_to_train += list(self.model.parameters())
        parameters_count(self.model, 'MVS2D')

    def build_optimizer(self):
        if self.opt.optimizer.lower() == 'adam':
            self.model_optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.opt.LR,
                weight_decay=self.opt.WEIGHT_DECAY)
        elif self.opt.optimizer.lower() == 'sgd':
            self.model_optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.opt.LR,
                weight_decay=self.opt.WEIGHT_DECAY)

    def val_epoch(self):
        print("Validation")
        writer = self.writers['val']
        self.set_eval()
        results_depth = []
        val_loss = []
        config = EvalCfg(
            eigen_crop=False,
            garg_crop=False,
            min_depth=self.opt.EVAL_MIN_DEPTH,
            max_depth=self.opt.EVAL_MAX_DEPTH,
            vis=self.epoch % 10 == 0 and self.opt.eval_vis,
            disable_median_scaling=self.opt.disable_median_scaling,
            print_per_dataset_stats=self.opt.dataset == 'DeMoN',
            save_dir=os.path.join(self.log_path, 'eval_%03d' % self.epoch))
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        print('evaluation results save to folder %s' % config.save_dir)
        times = []
        val_stats = defaultdict(list)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                if self.opt.val_epoch_size != -1 and batch_idx >= self.opt.val_epoch_size:
                    break
                if batch_idx % 100 == 0:
                    print(batch_idx, len(self.val_loader))
                filenames = inputs["filenames"]
                losses, outputs = self.process_batch(inputs, 'val')
                b = len(inputs["filenames"])

                s = 0
                pred_depth = npy(outputs[('depth_pred', s)])
                depth_gt = npy(inputs[('depth_gt', 0, s)])
                if self.opt.pred_conf:
                    log_conf_pred = outputs[('log_conf_pred', s)]
                    conf_pred = torch.exp(log_conf_pred)
                    val_stats['conf_mean'].append(conf_pred.mean().item())
                    val_stats['conf_median'].append(conf_pred.median().item())
                inv_K = npy(inputs[('inv_K_pool', 0)][(depth_gt.shape[2],
                                                       depth_gt.shape[3])])
                if self.opt.mode == 'test' and self.opt.save_prediction:
                    pred_depth_ = pred_depth[0, 0].copy()
                    color = cv2.resize(
                        (npy(inputs[('color', 0, 0)][0]).transpose(1, 2, 0) *
                         255).astype('uint8'),
                        (pred_depth_.shape[1], pred_depth_.shape[0]))
                    odir = os.path.join(config.save_dir, 'prediction')
                    if not os.path.exists(odir):
                        os.mkdir(odir)
                    odir_ = os.path.join(odir, filenames[0].split('-')[0])
                    if not os.path.exists(odir_):
                        os.mkdir(odir_)
                    data_ = {
                        'depth_pred': pred_depth_,
                        'inv_K': inv_K[0, :3, :3],
                        'pose': npy(inputs[('pose', 0)][0]),
                        'color': color,
                        'depth_gt': depth_gt[0, 0],
                        'fID': filenames[0]
                    }
                    if self.opt.pred_conf:
                        data_['conf'] = cv2.resize(
                            npy(conf_pred[0,
                                          0]), (data_['depth_pred'].shape[1],
                                                data_['depth_pred'].shape[0]),
                            interpolation=cv2.INTER_NEAREST)
                    np.save(f"{odir_}/{filenames[0]}.npy", data_)
                for i in range(len(filenames)):
                    fID = filenames[i]
                    results_depth.append((
                        pred_depth[i, 0],
                        depth_gt[i, 0],
                        fID,
                        inv_K[i],
                    ))
                val_loss.append(losses['loss'].item())

        metrics = {}
        for k, v in val_stats.items():
            metrics[k] = np.mean(v)

        errors = evaluate_depth_maps(results_depth, config)
        self.log_string("\n depth")
        self.log_string("\n  " + ("{:>9} | " *
                                  13).format(*errors['depth']['error_names']))
        self.log_string(("&{: 9.3f}  " * 13).format(
            *errors['depth']['errors'].tolist()) + "\\\\")

        for error, name in zip(errors['depth']['errors'],
                               errors['depth']['error_names']):
            metrics[name] = error
            writer.add_scalar(name, error, self.step)

        self.update_monitor_key(metrics, self.opt.monitor_key,
                                self.opt.monitor_goal)

        print('Eval Metrics\n' + pprint.pformat(metrics))

        writer.add_scalar('val_loss', np.mean(val_loss), self.step)
        self.set_train()

    def process_batch(self, inputs, mode):
        self.to_gpu(inputs)

        imgs, proj_mats, pose_mats = [], [], []
        for i in range(inputs['num_frame'][0].item()):
            imgs.append(inputs[('color', i, self.opt.input_scale)])
            proj_mats.append(inputs[('proj', i)])
            pose_mats.append(inputs[('pose', i)])

        outputs = self.model(imgs[0], imgs[1:], proj_mats[0], proj_mats[1:],
                             inputs[('inv_K_pool', 0)])
        losses = self.compute_losses(inputs, outputs)
        return losses, outputs

    def compute_losses(self, inputs, outputs):
        losses, loss, s = {}, 0, 0
        depth_pred = outputs[('depth_pred', s)]
        depth_gt = inputs[('depth_gt', 0, s)]

        if self.opt.dataset == 'ScanNet':
            valid_depth = (depth_gt > 0)
        elif self.opt.dataset == 'DeMoN':
            valid_depth = ((depth_gt != 0) & (depth_gt >= self.opt.min_depth) &
                           (depth_gt <= self.opt.max_depth))
        elif self.opt.dataset == 'DTU':
            valid_depth = (depth_gt > 0)

        if self.opt.pred_conf:
            log_conf_pred = outputs[('log_conf_pred', s)]
            conf_pred = torch.exp(log_conf_pred)
            min_conf = self.opt.min_conf
            max_conf = self.opt.max_conf if self.opt.max_conf != -1 else None
            conf_pred = conf_pred.clamp(min_conf, max_conf)
            loss_depth = ((depth_pred - depth_gt).abs() / conf_pred +
                          log_conf_pred)[valid_depth].mean()
        else:
            loss_depth = (depth_pred[valid_depth] -
                          depth_gt[valid_depth]).abs().mean()

        losses["depth"] = loss_depth
        loss += loss_depth
        losses["loss"] = loss

        return losses


def run_fusion(dense_folder, out_folder, opts):
    cmd = f"python patchmatch_fusion.py \
                --dense_folder {dense_folder} \
                --outdir {out_folder} \
                --n_proc 4 \
                --conf_thres {opts.conf_thres} \
                --att_thres {opts.att_thres} \
                --use_conf_thres {opts.pred_conf} \
                --geo_depth_thres {opts.geo_depth_thres} \
                --geo_pixel_thres {opts.geo_pixel_thres} \
                --num_consistent {opts.num_consistent} \
                "

    os.system(cmd)


if __name__ == "__main__":
    options = MVS2DOptions()
    opts = options.parse()

    set_random_seed(666)

    if torch.cuda.device_count() > 1 and not opts.multiprocessing_distributed:
        raise Exception(
            "Detected more than 1 GPU. Please set multiprocessing_distributed=1 or set CUDA_VISIBLE_DEVICES"
        )

    opts.distributed = opts.world_size > 1 or opts.multiprocessing_distributed
    if opts.multiprocessing_distributed:
        total_gpus, opts.rank = init_dist_pytorch(opts.tcp_port,
                                                  opts.local_rank,
                                                  backend='nccl')
        opts.ngpus_per_node = total_gpus
        opts.gpu = opts.rank
        print("Use GPU: {}/{} for training".format(opts.gpu,
                                                   opts.ngpus_per_node))
    else:
        opts.gpu = 0

    if opts.mode == 'train':
        trainer = Trainer(opts)
        trainer.train()

    elif opts.mode == 'test':
        trainer = Trainer(opts)
        trainer.val()

    elif opts.mode == 'full_test':
        ##  save depth prediction
        opts.mode = 'test'
        trainer = Trainer(opts)
        trainer.val()

        ## fuse dense prediction into final point cloud
        dense_folder = f"{opts.log_dir}/{opts.model_name}/eval_000/prediction"
        out_folder = f"{opts.log_dir}/{opts.model_name}/recon"
        run_fusion(dense_folder, out_folder, opts)

        ## eval point cloud
        MeanData, MeanStl, MeanAvg = dtu_pyeval(
            f"{out_folder}",
            gt_dir='./data/SampleSet/MVS Data/',
            voxel_down_sample=False,
            fn=f"{out_folder}/result.txt")
