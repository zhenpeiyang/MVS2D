from __future__ import absolute_import, division, print_function
from open3d import *
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import json
from utils import *
import os
import glob
import shutil
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_sched
from options import MVS2DOptions
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

g = torch.Generator()
g.manual_seed(0)


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    import random
    random.seed(seed)


class BaseTrainer(object):
    def __init__(self, options):

        self.is_best = {}
        self.epoch = 0
        self.step = 0
        self.opt = options
        self.opt.is_master = not self.opt.multiprocessing_distributed or (
            self.opt.multiprocessing_distributed
            and self.opt.rank % self.opt.ngpus_per_node == 0)

        if self.opt.is_master:
            base_dir = '.'
            self.opt.log_dir = os.path.join(base_dir, self.opt.log_dir)
            self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
            if os.path.exists(self.log_path) and self.opt.overwrite:
                try:
                    shutil.rmtree(self.log_path)
                except:
                    print('overwrite folder failed')
            self.log_file = os.path.join(self.log_path, "log.txt")
            self.writers = {}
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(
                    os.path.join(self.log_path, mode))
            with open(self.log_file, 'w') as f:
                f.write(self.opt.note + '\n')

            self.save_opts()

        self.build_dataset()

        self.build_model()

        self.build_optimizer()

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.distributed:
            if self.opt.gpu is not None:
                print(
                    f"batch size on GPU: {self.opt.gpu}: {self.opt.batch_size}"
                )

                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.opt.gpu],
                    find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    self.model, find_unused_parameters=True)

        self.build_scheduler()

        self.total_data_time = 0
        self.total_op_time = 0
        if self.opt.epoch_size == -1:
            self.opt.epoch_size = len(self.train_loader)

        if self.opt.is_master:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ",
                  self.opt.log_dir)

            self.num_total_steps = len(self.train_loader) * self.opt.num_epochs
            print("There are {:d} training items and {:d} validation items\n".
                  format(
                      len(self.train_loader) * self.opt.batch_size,
                      len(self.val_loader) * 1))

    def build_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.opt.LR,
                               weight_decay=self.opt.WEIGHT_DECAY)

        self.model_optimizer = optimizer

    def build_scheduler(self):
        total_iters_each_epoch = len(self.train_loader)
        decay_steps = [
            x * total_iters_each_epoch for x in self.opt.DECAY_STEP_LIST
        ]
        total_steps = total_iters_each_epoch * self.opt.num_epochs

        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * self.opt.LR_DECAY
            return max(cur_decay, self.opt.LR_CLIP / self.opt.LR)

        self.model_lr_scheduler = lr_sched.LambdaLR(self.model_optimizer,
                                                    lr_lbmd,
                                                    last_epoch=-1)

    def to_gpu(self, inputs, keys=None):
        if keys == None:
            keys = inputs.keys()
        for key in keys:
            if key not in inputs:
                continue
            ipt = inputs[key]
            if type(ipt) == torch.Tensor:
                inputs[key] = ipt.cuda(self.opt.gpu, non_blocking=True)
            elif type(ipt) == list and type(ipt[0]) == torch.Tensor:
                inputs[key] = [
                    x.cuda(self.opt.gpu, non_blocking=True) for x in ipt
                ]
            elif type(ipt) == dict:
                for k in ipt.keys():
                    if type(ipt[k]) == torch.Tensor:
                        ipt[k] = ipt[k].cuda(self.opt.gpu, non_blocking=True)

    def build_dataset(self):
        if self.opt.dataset == 'ScanNet':
            from datasets.ScanNet import ScanNet as Dataset
        elif self.opt.dataset == 'DeMoN':
            from datasets.DeMoN import DeMoN as Dataset
        elif self.opt.dataset == 'DTU':
            from datasets.DTU import DTU as Dataset
        else:
            raise Exception("Unknown Dataset")

        train_dataset = Dataset('train', self.opt)
        if self.opt.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset)
        else:
            self.train_sampler = None
        self.train_loader = DataLoader(train_dataset,
                                       self.opt.batch_size,
                                       shuffle=(self.train_sampler is None),
                                       num_workers=self.opt.num_workers,
                                       pin_memory=True,
                                       collate_fn=default_collatev1_1,
                                       worker_init_fn=worker_init_fn,
                                       drop_last=True,
                                       sampler=self.train_sampler)
        if self.opt.is_master:
            val_dataset = Dataset(
                'test' if self.opt.use_test else 'val',
                self.opt,
            )
            self.val_sampler = None
            self.val_loader = DataLoader(val_dataset,
                                         1,
                                         shuffle=False,
                                         num_workers=self.opt.num_workers,
                                         pin_memory=True,
                                         collate_fn=custom_collate,
                                         drop_last=False,
                                         sampler=self.val_sampler)

    def log_time(self, batch_idx, op_time, step_time, loss):
        """Print a logging statement to the terminal
        """
        if self.opt.distributed:
            ops_per_sec = self.opt.ngpus_per_node * self.opt.batch_size / op_time
            steps_per_sec = self.opt.ngpus_per_node * self.opt.batch_size / step_time
        else:
            ops_per_sec = self.opt.batch_size / op_time
            steps_per_sec = self.opt.batch_size / step_time
        time_sofar = time.time() - self.start_time

        training_time_left = (self.num_total_steps / self.step -
                              1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6}/{:>6} | ops/s: {:5.1f} | steps/s: {:5.1f} | t_data/t_op: {:5.1f} " + \
                " | loss: {:.5f} | time elapsed: {} | time left: {} | lr: {:.7f}"
        self.log_string(
            print_string.format(self.epoch, batch_idx, len(self.train_loader),
                                ops_per_sec, steps_per_sec,
                                self.total_data_time / self.total_op_time,
                                loss, sec_to_hm_str(time_sofar),
                                sec_to_hm_str(training_time_left),
                                self.model_optimizer.param_groups[0]['lr']))

    def train_epoch(self):
        if self.opt.is_master:
            print("Training")
            self.writers['train'].add_scalar(
                "lr", self.model_optimizer.param_groups[0]['lr'], self.step)
        self.set_train()
        before_data_loader_time = time.time()
        time_last_step = time.time()

        if self.opt.epoch_size == 0:
            return

        for batch_idx, inputs in enumerate(self.train_loader):
            if batch_idx >= self.opt.epoch_size:
                break
            after_data_loader_time = time.time()
            duration_data = after_data_loader_time - before_data_loader_time
            self.total_data_time += duration_data
            before_op_time = time.time()

            self.model_lr_scheduler.step(self.step)

            if self.opt.is_master:
                try:
                    cur_lr = float(self.model_optimizer.lr)
                except:
                    cur_lr = self.model_optimizer.param_groups[0]['lr']

                self.writers['train'].add_scalar('meta_data/learning_rate',
                                                 cur_lr, self.step)

            self.model_optimizer.zero_grad()
            losses, outputs = self.process_batch(inputs, 'train')
            losses['loss'].backward()

            torch.nn.utils.clip_grad_norm_(self.parameters_to_train,
                                           self.opt.GRAD_NORM_CLIP)

            contain_nan = False
            for weight in self.parameters_to_train:
                if weight.grad is not None:
                    if torch.any(torch.isnan(weight.grad)):
                        print('skip parameters update because of nan in grad')
                        contain_nan = True
            if not contain_nan:
                self.model_optimizer.step()

            duration = time.time() - before_op_time
            self.total_op_time += duration

            if self.opt.is_master and batch_idx % self.opt.log_frequency == 0:
                duration_step = time.time() - time_last_step
                self.log_time(batch_idx, duration, duration_step,
                              losses["loss"].cpu().data)
                self.log("train", inputs, losses)
            self.step += 1
            before_data_loader_time = time.time()
            time_last_step = time.time()

    def update_monitor_key(self, metrics, keys, goals):
        if len(keys):
            if type(keys) != list:
                keys = [keys]
            for key, goal in zip(keys, goals):
                val = metrics[key]
                if not hasattr(self, key):
                    setattr(self, key, val)
                    self.is_best[key] = True
                else:
                    if goal == 'minimize':
                        if val < getattr(self, key):
                            self.is_best[key] = True
                            setattr(self, key, val)
                        else:
                            self.is_best[key] = False
                    elif goal == 'maximize':
                        if val > getattr(self, key):
                            self.is_best[key] = True
                            setattr(self, key, val)
                        else:
                            self.is_best[key] = False

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def train(self):
        self.start_time = time.time()
        if self.opt.is_master:
            print("Total epoch: %d " % self.opt.num_epochs)
            print("train loader size: %d " % len(self.train_loader))
            print("val loader size: %d " % len(self.val_loader))
            print("log_frequency: %d " % self.opt.log_frequency)
        for self.epoch in range(self.opt.num_epochs):
            if self.opt.distributed:
                self.train_sampler.set_epoch(self.epoch)
            self.train_epoch()
            if self.opt.is_master:
                self.val_epoch()
            if self.opt.is_master and (self.epoch +
                                       1) % self.opt.save_frequency == 0:
                self.save_model(monitor_key=self.opt.monitor_key)

    def val(self):
        self.val_epoch()

    def process_batch(self, inputs, mode):
        raise Exception("Need to implement process_batch")

    def compute_losses(self, inputs, outputs):
        raise Exception("Need to implement compute_losses")

    def log_string(self, content):
        with open(self.log_file, 'a') as f:
            f.write(content + '\n')
        print(content, flush=True)

    def log(self, mode, inputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if type(losses[l]) == dict:
                writer.add_scalars("{}".format(l), v, self.step)
            else:
                writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2, sort_keys=True)

    def clean_models(self, keep_ids):
        models = glob.glob(os.path.join(self.log_path, "models", "weights_*"))
        models = sorted(models,
                        key=lambda x: int(x.split('/')[-1].split('_')[-1]))
        for i in range(len(models) - 1):
            epoch = int(models[i].split('/')[-1].split('_')[-1])
            if epoch not in keep_ids:
                shutil.rmtree(models[i])

    def save_model(self, monitor_key=""):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_latest")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print("save model to folder %s" % save_folder)

        save_path = os.path.join(save_folder, "model.pth")
        if self.opt.distributed:
            to_save = self.model.module.state_dict()
        else:
            to_save = self.model.state_dict()

        torch.save(to_save, save_path)
        save_path_opt = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path_opt)
        if len(monitor_key):
            if type(monitor_key) != list:
                monitor_key = [monitor_key]
            for key in monitor_key:
                if not self.is_best[key]:
                    continue
                save_folder = os.path.join(self.log_path, "models",
                                           f"weights_best_{key}")
                os.makedirs(save_folder, exist_ok=True)
                cmd = f"cp {save_path} {save_folder}/model.pth"
                os.system(cmd)
                cmd = f"cp {save_path_opt} {save_folder}/adam.pth"
                os.system(cmd)
                with open(f"{save_folder}/key.txt", "w") as f:
                    val = getattr(self, key)
                    f.write(f"{key} {val}\n")

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(
            self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(
            self.opt.load_weights_folder))

        try:
            self.epoch = int(
                self.opt.load_weights_folder.split('/')[-2].split('_')[1])
        except:
            self.epoch = 0

        try:
            path = os.path.join(self.opt.load_weights_folder,
                                "{}.pth".format("model"))
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(path)
            for k, v in pretrained_dict.items():
                if k not in model_dict:
                    print('model dict missing ', k, v.shape)
            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    print('pretrained_dict missing ', k, v.shape)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        except Exception as e:
            print(e)
            print("Fail loading {}".format("model"))

        # loading optimizer state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder,
                                           "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading optimizer weights")
            optimizer_dict = torch.load(optimizer_load_path)
            try:
                self.model_optimizer.load_state_dict(optimizer_dict)
            except Exception as e:
                print(e)
                print("Fail loading optimizer weights")
        else:
            print("Cannot find optimizer weights so optimizer is randomly initialized")
