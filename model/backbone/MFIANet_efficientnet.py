# ------------------------------------------
# EfficientTransformer
# Copyright (c) East China University of Technology.
# written By Fuyang Zhou
# ------------------------------------------
from functools import partial
from typing import Callable, Optional, List, Any
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import model.config.efficientnetv2_config as config
from model.backbone.head import seg,fcn, mlp, fcfpn, uper,unet
from tool.checkpoint import load_checkpoint
from tool.flops_params_fps_count import flops_params_fps_1
from model.module.Attention import SE, CBAM, CA
from tool.init_func import init_weight
from model.module.MFIM_MFAM_MLFA import MS_FIM, MS_FFM_MLP, MS_FFM_FPN, Cross_Attention_MLP
from tool.logger import get_logger

logger = get_logger()

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.GroupNorm
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result

class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SE(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)

        self.out_channels = out_c

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        # result = self.ca(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)
        else:
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)

        self.out_channels = out_c

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x
        return result

class EfficientNetV2(nn.Module):
    def __init__(self, in_chans,
                 model_cnf: list,
                 num_classes: int = 1000,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)
        # norm_layer = partial(nn.GroupNorm, eps=1e-5)

        stem_filter_num = model_cnf[0][4]

        self.RGB_stem = ConvBNAct(in_chans, stem_filter_num, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.X_stem = ConvBNAct(in_chans, stem_filter_num, kernel_size=3, stride=2, norm_layer=norm_layer)

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks1, blocks2, blocks3, blocks4, blocks5, blocks6 = [], [], [], [], [], []

        repeats1 = model_cnf[0][0]
        for i in range(repeats1):
            blocks1.append(FusedMBConv(kernel_size=model_cnf[0][1],
                                      input_c=model_cnf[0][4] if i ==0 else model_cnf[0][5],
                                      out_c=model_cnf[0][5],
                                      expand_ratio=model_cnf[0][3],
                                      stride=model_cnf[0][2] if i == 0 else 1,
                                      se_ratio=model_cnf[0][-1],
                                      drop_rate=drop_connect_rate * i / total_blocks,
                                      norm_layer=norm_layer))
        self.RGB_stage1 = nn.Sequential(*blocks1)
        self.X_stage1 = nn.Sequential(*blocks1)

        block_id = block_id + repeats1
        repeats2 = model_cnf[1][0]
        for i in range(repeats2):
            blocks2.append(FusedMBConv(kernel_size=model_cnf[1][1],
                                      input_c=model_cnf[1][4] if i ==0 else model_cnf[1][5],
                                      out_c=model_cnf[1][5],
                                      expand_ratio=model_cnf[1][3],
                                      stride=model_cnf[1][2] if i == 0 else 1,
                                      se_ratio=model_cnf[1][-1],
                                      drop_rate=drop_connect_rate * (block_id + i) / total_blocks,
                                      norm_layer=norm_layer))
        self.RGB_stage2 = nn.Sequential(*blocks2)
        self.X_stage2 = nn.Sequential(*blocks2)

        block_id = block_id + repeats2
        repeats3 = model_cnf[2][0]
        for i in range(repeats3):
            blocks3.append(FusedMBConv(kernel_size=model_cnf[2][1],
                                       input_c=model_cnf[2][4] if i == 0 else model_cnf[2][5],
                                       out_c=model_cnf[2][5],
                                       expand_ratio=model_cnf[2][3],
                                       stride=model_cnf[2][2] if i == 0 else 1,
                                       se_ratio=model_cnf[2][-1],
                                       drop_rate=drop_connect_rate * (block_id + i) / total_blocks,
                                       norm_layer=norm_layer))
        self.RGB_stage3 = nn.Sequential(*blocks3)
        self.X_stage3 = nn.Sequential(*blocks3)

        block_id = block_id + repeats3
        repeats4 = model_cnf[3][0]
        for i in range(repeats4):
            blocks4.append(MBConv(kernel_size=model_cnf[3][1],
                                       input_c=model_cnf[3][4] if i == 0 else model_cnf[3][5],
                                       out_c=model_cnf[3][5],
                                       expand_ratio=model_cnf[3][3],
                                       stride=model_cnf[3][2] if i == 0 else 1,
                                       se_ratio=model_cnf[3][-1],
                                       drop_rate=drop_connect_rate * (block_id + i) / total_blocks,
                                       norm_layer=norm_layer))
        self.RGB_stage4 = nn.Sequential(*blocks4)
        self.X_stage4 = nn.Sequential(*blocks4)

        block_id = block_id + repeats4
        repeats5 = model_cnf[4][0]
        for i in range(repeats5):
            blocks5.append(MBConv(kernel_size=model_cnf[4][1],
                                       input_c=model_cnf[4][4] if i == 0 else model_cnf[4][5],
                                       out_c=model_cnf[4][5],
                                       expand_ratio=model_cnf[4][3],
                                       stride=model_cnf[4][2] if i == 0 else 1,
                                       se_ratio=model_cnf[4][-1],
                                       drop_rate=drop_connect_rate * (block_id +i) / total_blocks,
                                       norm_layer=norm_layer))
        self.RGB_stage5 = nn.Sequential(*blocks5)
        self.X_stage5 = nn.Sequential(*blocks5)

        self.MS_FIM_1 = MS_FIM(model_cnf[1][5], reduction=1)
        self.MS_FIM_2 = MS_FIM(model_cnf[2][5], reduction=1)
        self.MS_FIM_3 = MS_FIM(model_cnf[3][5], reduction=1)
        self.MS_FIM_4 = MS_FIM(model_cnf[4][5], reduction=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        features = []
        #stage
        rgb_stem = self.RGB_stem(x1)
        x_stem = self.X_stem(x2)

        rgb_1 = self.RGB_stage1(rgb_stem)
        x_1 = self.X_stage1(x_stem)

        rgb_2 = self.RGB_stage2(rgb_1)
        x_2 = self.X_stage2(x_1)
        rgb_fim_1, x_fim_1, merge_fea_1 = self.MS_FIM_1(rgb_2, x_2)

        rgb_3 = self.RGB_stage3(rgb_2 + rgb_fim_1)
        x_3 = self.X_stage3(x_2 + x_fim_1)
        rgb_fim_2, x_fim_2, merge_fea_2 = self.MS_FIM_2(rgb_3, x_3)

        rgb_4 = self.RGB_stage4(rgb_3 + rgb_fim_2)
        x_4 = self.X_stage4(x_3 + x_fim_2)
        rgb_fim_3, x_fim_3, merge_fea_3 = self.MS_FIM_3(rgb_4, x_4)

        rgb_5 = self.RGB_stage5(rgb_4 + rgb_fim_3)
        x_5 = self.X_stage5(x_4 + x_fim_3)
        rgb_fim_4, x_fim_4, merge_fea_4 = self.MS_FIM_4(rgb_5, x_5)

        features.append(merge_fea_1)
        features.append(merge_fea_2)
        features.append(merge_fea_3)
        features.append(merge_fea_4)

        return features


class MFIANet(nn.Module):
    def __init__(self, in_chans,aux, model_cnf, num_class, output_channel,
                 pretrained_root=None, head=False, head_type='mlphead', norm_layer=nn.LayerNorm, drop_connect_rate=0.2):
        super(MFIANet, self).__init__()
        self.head = head
        self.aux = aux
        self.head_name = head_type
        self.output_channel = output_channel
        self.pretrained_root = pretrained_root
        self.norm_layer = norm_layer

        self.backbone = EfficientNetV2(in_chans=in_chans, model_cnf=model_cnf, num_classes=num_class,
                                       drop_connect_rate=drop_connect_rate)

        if head:
            if self.head_name == 'seghead':
                self.decoder = seg.SegHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3])

            if self.head_name == 'mlphead':
                self.decoder = mlp.MLPHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3],
                                               channels=256)

            if self.head_name == 'fcfpnhead':
                self.decoder = fcfpn.FCFPNHead(in_channels=output_channel, num_classes=num_class,
                                                   in_index=[0, 1, 2, 3], channels=256)

            if self.head_name == 'unethead':
                self.decoder = unet.UNetHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3])

            if self.head_name == 'uperhead':
                self.decoder = uper.UPerHead(in_channels=output_channel, num_classes=num_class)

            if self.head_name == 'mlphead':
                self.decoder = mlp.MLPHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3],
                                           channels=256)
        else:
            self.decoder = Cross_Attention_MLP(in_channels=output_channel, channels=256, num_classes=num_class,
                                               norm_layer=norm_layer)

        if self.aux:
            self.auxiliary_head = mlp.MLPHead(in_channels=output_channel, num_classes=num_class,
                                              in_index=[0, 1, 2, 3], channels=256)

        self.init_weights(pretrained=self.pretrained_root)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        else:
            logger.info('Initing encoder weights ...')
        init_weight(self.decoder, nn.init.kaiming_normal_,
                    self.norm_layer, 1e-3, 0.01,
                    mode='fan_in', nonlinearity='relu')
        logger.info('Initing decoder weights ...')
        if self.aux:
            init_weight(self.aux, nn.init.kaiming_normal_,
                        self.norm_layer, 1e-3, 0.01,
                        mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2):
        size = x1.size()[2:]
        outputs = []
        backbone_features = self.backbone(x1, x2)

        if self.head:
            out = self.decoder(backbone_features)
        else:
            out = self.decoder(backbone_features[0], backbone_features[1], backbone_features[2], backbone_features[3])

        x0 = out
        if isinstance(x0, (list, tuple)):
            for out in x0:
                out = F.interpolate(out, size, **up_kwargs)
                outputs.append(out)
        else:
            x0 = F.interpolate(x0, size, **up_kwargs)
            outputs.append(x0)

        if self.aux:
            x1 = self.auxiliary_head(backbone_features)
            x1 = F.interpolate(x1, size, **up_kwargs)
            outputs.append(x1)
        return outputs

def MFIANet_s(num_classes, in_chans, aux, head=False, head_type='mlphead'):
    model_config = config.get_st_config()

    model = MFIANet(in_chans=in_chans,
                    aux=aux,
                    model_cnf=model_config,
                    output_channel=[48, 64, 128, 256],
                    num_class=num_classes,
                    head = head,
                    head_type=head_type)
    return model

def MFIANet_m(num_classes, in_chans, aux, head=False, head_type='mlphead'):
    model_config = config.get_mt_config()
    model = MFIANet(in_chans=in_chans,
                    aux=aux,
                    model_cnf=model_config,
                    output_channel=[48, 80, 176, 512],
                    num_class=num_classes,
                    head=head,
                    head_type=head_type)
    return model

def MFIANet_l(num_classes, in_chans, aux, head=False, head_type='mlphead'):
    model_config = config.get_lt_config()

    model = MFIANet(in_chans=in_chans,
                    aux=aux,
                    model_cnf=model_config,
                    output_channel=[64, 96, 224, 640],
                    num_class=num_classes,
                    head=head,
                    head_type=head_type)
    return model

if __name__ == '__main__':
    model = MFIANet_M(num_classes=45, in_chans=3, aux=False, head=False, head_type='mlphead')
    RGB_inputs = torch.randn(size=(8, 3, 256, 256))
    X_inputs = torch.randn(size=(8, 1, 256, 256))
    outputs = model(RGB_inputs, X_inputs)

    flops_params_fps_1(model, RGB_shape=(8, 3, 256, 256), X_shape=(8, 1, 256, 256))