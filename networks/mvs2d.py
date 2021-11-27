import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
import cv2
import time
import numpy as np
import torchvision.models as models
from utils import *
from .module import UNet


class MVS2D(nn.Module):
    def __init__(self, opt, pretrained=True):
        super(MVS2D, self).__init__()
        self.opt = opt
        self.output_layer = 0
        if self.opt.multi_view_agg:
            if self.opt.robust:
                self.attn_layers = ['layer2', 'layer3', 'layer4']
            else:
                self.attn_layers = ['layer2']
        else:
            self.attn_layers = []
        self.base_model = models.resnet18(pretrained=pretrained)
        self.base_model2 = models.resnet18(pretrained=pretrained)
        delattr(self.base_model2, 'avgpool')
        delattr(self.base_model2, 'fc')
        cnt = 0
        to_delete = []
        for k, v in self.base_model._modules.items():
            if cnt == len(self.attn_layers):
                to_delete.append(k)
            if k in self.attn_layers:
                cnt += 1
        for k in to_delete:
            delattr(self.base_model, k)

        ## ----------- ResNet-18 Specification --------
        self.feat_names = [
            'relu',
            'layer1',  # 1/4 resol
            'layer2',  # 1/8 resol
            'layer3',  # 1/16 resol
            'layer4',  # 1/32 resol
        ]
        self.feat_name2ch = {
            'relu': 64,
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512
        }
        self.feat_channels = [self.feat_name2ch[x] for x in self.feat_names]

        ## ----------- Epipolar Attention Layer --------
        self.layers = {}
        for k in self.attn_layers:
            inp_dim = self.feat_name2ch[k]
            att_dim = inp_dim // self.opt.att_rate
            self.layers[('query', k)] = nn.Conv2d(inp_dim,
                                                  att_dim,
                                                  kernel_size=1)
            self.layers[('m_embed', k)] = nn.Embedding(2, att_dim)
            self.layers[('key', k)] = nn.Conv2d(inp_dim,
                                                att_dim,
                                                kernel_size=1,
                                                bias=False)
            if self.opt.depth_embedding == 'learned':
                self.layers[('pos_embed',
                             k)] = nn.Embedding(self.opt.nlabel, att_dim)
            elif self.opt.depth_embedding == 'cosine':
                self.layers[('pos_embed',
                             k)] = CosineEmbedding(self.opt.nlabel, att_dim)
            elif self.opt.depth_embedding == 'linear':
                self.layers[('pos_embed',
                             k)] = LinearEmbedding(self.opt.nlabel, att_dim)
            elif self.opt.depth_embedding == 'uniform':
                self.layers[('pos_embed',
                             k)] = UniformEmbedding(self.opt.nlabel, att_dim)
            else:
                raise NotImplementedError

            self.layers[('linear_out', k)] = nn.Conv2d(att_dim,
                                                       inp_dim,
                                                       kernel_size=1)
            self.layers[('linear_out2', k)] = nn.Conv2d(inp_dim,
                                                        inp_dim,
                                                        kernel_size=1)
            self.layers[('linear_out3', k)] = nn.Conv2d(inp_dim,
                                                        inp_dim,
                                                        kernel_size=1)

        ## ----------- Decoder ----------
        self.num_ch_dec = [64, 64, 64, 128, 256]
        ch_cur = self.feat_channels[-1]
        for i in range(4, self.output_layer, -1):
            k = 1 if i == 4 else 3
            self.layers[("upconv", i, 0)] = ConvBlock(ch_cur,
                                                      self.num_ch_dec[i],
                                                      kernel_size=k)
            ch_mid = self.num_ch_dec[i]
            if self.opt.use_skip:
                ch_mid += self.feat_channels[i - 1]
            self.layers[("upconv", i, 1)] = ConvBlock(ch_mid,
                                                      self.num_ch_dec[i],
                                                      kernel_size=k)
            ch_cur = self.num_ch_dec[i]

        ## ----------- Depth Regressor ----------
        ch_cur = self.num_ch_dec[self.opt.output_scale - self.opt.input_scale -
                                 1]
        odim = 256
        output_chal = odim if not self.opt.pred_conf else odim + 1
        if self.opt.use_unet:
            self.conv_out = UNet(inp_ch=ch_cur,
                                 output_chal=output_chal,
                                 down_sample_times=3,
                                 channel_mode=self.opt.unet_channel_mode)
        else:
            self.conv_out = nn.Conv2d(ch_cur, output_chal, kernel_size=1)
        self.depth_regressor = nn.Sequential(
            nn.Conv2d(odim,
                      self.opt.num_depth_regressor_anchor,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(self.opt.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.opt.num_depth_regressor_anchor,
                      self.opt.num_depth_regressor_anchor,
                      kernel_size=1),
        )

        ## ----------- Depth hypothesis ----------
        self.register_buffer(
            'depth_expectation_anchor',
            torch.from_numpy(
                1.0 /
                np.linspace(1.0 / self.opt.max_depth, 1.0 / self.opt.min_depth,
                            self.opt.num_depth_regressor_anchor)).float())
        if self.opt.inv_depth:
            depth_values = torch.from_numpy(
                1.0 /
                np.linspace(1.0 / self.opt.max_depth, 1.0 / self.opt.min_depth,
                            self.opt.nlabel)).float()
        else:
            depth_values = torch.linspace(self.opt.min_depth,
                                          self.opt.max_depth, self.opt.nlabel)
        self.register_buffer('depth_values', depth_values)

        ## TODO: remove this line
        self.temp = nn.ModuleList(list(self.layers.values()))

        ## ----------- Up sample to input resolution ----------
        h, w = self.opt.height // 4, self.opt.width // 4
        idv, idu = np.meshgrid(np.linspace(0, 1, h),
                               np.linspace(0, 1, w),
                               indexing='ij')
        self.meshgrid = torch.from_numpy(np.stack((idu, idv))).float()
        self.conv_up = nn.Sequential(
            nn.Conv2d(1 + 2 + odim, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, padding=0),
        )

        ## ----------- Count Parameters ----------
        parameters_count(self.base_model, 'feature extractor')
        parameters_count(self.base_model2, 'depth encoder')
        param = 0
        for key in self.layers.keys():
            if key[0] in [
                    'key', 'pos_embed', 'query', 'm_embed', 'linear_out',
                    'linear_out2', 'linear_out3'
            ]:
                param += parameters_count(self.layers[key],
                                          key,
                                          do_print=False)
        print('#params %s: %.3f M' % ('epipolar attention', param / 1e6))
        param = 0
        for key in self.layers.keys():
            if key[0] == 'upconv':
                param += parameters_count(self.layers[key],
                                          key,
                                          do_print=False)
        param += parameters_count(self.conv_out, 'linear_out', do_print=False)
        param += parameters_count(self.depth_regressor,
                                  'depth_regressor',
                                  do_print=False)
        print('#params %s: %.3f M' % ('depth decoder', param / 1e6))

    def upsample(self, x, scale_factor=2):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=scale_factor, mode="nearest")

    def epipolar_fusion(
        self,
        ref_feature,
        src_features,
        ref_proj,
        src_projs,
        depth_values,
        layer,
        ref_img,
        src_imgs,
    ):
        query = self.layers[('query', layer)]
        m_embed = self.layers[('m_embed', layer)]
        pos_embed = self.layers[('pos_embed', layer)]
        key = self.layers[('key', layer)]
        linear_out = self.layers[('linear_out', layer)]
        linear_out2 = self.layers[('linear_out2', layer)]

        num_depth = depth_values.shape[1]
        num_views = len(src_features) + 1
        b, _, h, w = ref_feature.shape
        nhead = self.opt.nhead
        device = ref_feature.device
        agg = 0

        q = query(ref_feature)

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            k = key(src_fea)
            k, proj_mask, grid = homo_warping(k, src_proj, ref_proj,
                                              depth_values)
            m = m_embed(proj_mask.long()).permute(0, 4, 1, 2, 3)
            pos = pos_embed.weight[None, :, None, None, :].repeat(
                b, 1, h, w, 1).permute(0, 4, 1, 2, 3) * m
            att_dim = k.shape[1]
            attn = ((q.unsqueeze(2) * k).view(b, -1, nhead, num_depth, h,
                                              w).sum(1, keepdim=True) /
                    np.sqrt(att_dim // nhead)).softmax(dim=3)

            v = pos.view(b, -1, nhead, num_depth, h, w)
            agg = agg + (attn * v).sum(3).view(b, -1, h, w)

        # test frame different from train frame
        if len(src_features) + 1 != self.opt.num_frame:
            agg = agg / float(len(src_features)) * (self.opt.num_frame - 1)

        agg = linear_out(agg) + linear_out2(ref_feature)
        return agg

    def decoder(self, ref_feature):
        x = ref_feature[-1]
        for i in range(4, self.output_layer, -1):
            x = self.layers[("upconv", i, 0)](x)
            if i >= 2 - self.opt.input_scale:
                x = self.upsample(x)
                if self.opt.use_skip:
                    x = torch.cat((x, ref_feature[i - 1]), 1)
                x = self.layers[("upconv", i, 1)](x)
            else:
                break
        return x

    def regress_depth(self, feature_map_d):
        x = self.depth_regressor(feature_map_d).softmax(dim=1)
        d = compute_depth_expectation(
            x,
            self.depth_expectation_anchor.unsqueeze(0).repeat(x.shape[0],
                                                              1)).unsqueeze(1)
        return d

    def forward(
        self,
        ref_img,
        src_imgs,
        ref_proj,
        src_projs,
        inv_K,
    ):
        outputs = {}
        ref_feature = ref_img
        src_features = [x for x in src_imgs]
        V = len(src_imgs) + 1
        ref_feature2 = ref_img
        ref_skip_feat2 = []
        cnt = 0
        ## pass through encoder, and integrate multi-view attention
        for k, v in self.base_model2._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            if cnt < len(self.attn_layers):
                ref_feature = getattr(self.base_model, k)(ref_feature)
                for i in range(V - 1):
                    src_features[i] = getattr(self.base_model,
                                              k)(src_features[i])
            ref_feature2 = v(ref_feature2)
            if k in self.attn_layers:
                b = ref_img.shape[0]
                depth_values = self.depth_values[None, :, None, None].repeat(
                    b, 1, ref_feature.shape[2], ref_feature.shape[3])

                sz_ref = (ref_feature.shape[2], ref_feature.shape[3])
                sz_src = (src_features[0].shape[2], src_features[0].shape[3])
                linear_out3 = self.layers[('linear_out3', k)]
                att_f = self.epipolar_fusion(
                    ref_feature,
                    src_features,
                    ref_proj[sz_ref],
                    [proj[sz_src] for proj in src_projs],
                    depth_values,
                    k,
                    ref_img,
                    src_imgs,
                )
                ref_feature2 = ref_feature2 + linear_out3(ref_feature2) + att_f
                cnt += 1

            if any(x in k for x in self.feat_names):
                ref_skip_feat2.append(ref_feature2)

        ## decode into depth map of 1/4 input resolution
        feature_map = self.decoder(ref_skip_feat2)
        if self.opt.pred_conf:
            feature_map = self.conv_out(feature_map)
            outputs[('log_conf_pred',
                     self.opt.output_scale)] = feature_map[:, -1:, :, :]
            feature_map_d = feature_map[:, :-1, :, :]
        else:
            feature_map_d = self.conv_out(feature_map)

        depth_pred = self.regress_depth(feature_map_d)

        ## upsample depth map into input resolution
        depth_pred = self.upsample(
            depth_pred, scale_factor=4) + 1e-1 * self.conv_up(
                torch.cat((depth_pred, self.meshgrid.unsqueeze(0).repeat(
                    depth_pred.shape[0], 1, 1,
                    1).to(depth_pred), feature_map_d), 1))
        if self.opt.pred_conf:
            outputs[('log_conf_pred',
                     0)] = F.interpolate(outputs[('log_conf_pred',
                                                  self.opt.output_scale)],
                                         scale_factor=4)
            outputs[('log_conf_pred', 0)] = F.interpolate(
                outputs[('log_conf_pred', self.opt.output_scale)],
                size=(self.opt.height, self.opt.width))
        outputs[('depth_pred', 0)] = depth_pred

        return outputs
