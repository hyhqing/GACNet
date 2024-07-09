from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import model.config.efficientnetv2_config as config
# from tool.checkpoint import load_checkpoint
from model.backbone.head import seg,fcn, mlp, fcfpn
from tool.heatmap_fun import draw_features
from tool.flops_params_fps_count import flops_params_fps

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
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
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.GroupNorm
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(8,out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result

class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


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

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

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

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
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
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

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

        # norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        norm_layer = partial(nn.GroupNorm, eps=1e-5)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(in_chans,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

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
        self.stage1 = nn.Sequential(*blocks1)

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
        self.stage2 = nn.Sequential(*blocks2)

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
        self.stage3 = nn.Sequential(*blocks3)

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
        self.stage4 = nn.Sequential(*blocks4)

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
        self.stage5 = nn.Sequential(*blocks5)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x: Tensor) -> Tensor:
        conv_features = []
        feature = []
        x = self.stem(x)

        x = self.stage1(x)

        x = self.stage2(x)
        feature.append(x)

        x = self.stage3(x)
        feature.append(x)

        x = self.stage4(x)
        feature.append(x)


        x = self.stage5(x)
        feature.append(x)

        # x = self.stage6(x)
        # feature.append(x)

        return feature

class EffinientNetv2(nn.Module):
    def __init__(self, in_chans, aux, model_cnf,num_class,output_channel, pretrained_root=None,
                 head='mlphead', drop_connect_rate=0.2):
        super(EffinientNetv2, self).__init__()
        self.aux = aux
        self.head_name = head
        self.output_channel = output_channel
        self.backbone = EfficientNetV2(in_chans=in_chans, model_cnf=model_cnf, num_classes=num_class,drop_connect_rate=drop_connect_rate)

        if self.head_name == 'seghead':
            self.decode_head = seg.SegHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3])

        if self.head_name == 'mlphead':
            self.decode_head = mlp.MLPHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3], channels=256)

        if self.head_name == 'fcfpnhead':
            self.decode_head = fcfpn.FCFPNHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3], channels=256)

        # if self.aux:
        #     self.auxiliary_head = fcn.FCNHead(num_convs=1, in_channels=output_channel[2], num_classes=num_class, in_index=2,
        #                                       channels=256)
        if self.aux:
            self.auxiliary_head = mlp.MLPHead(in_channels=output_channel, num_classes=num_class, in_index=[0, 1, 2, 3], channels=256)

        if pretrained_root is None:
            self.backbone.init_weights()
        else:
            self.backbone.init_weights(pretrained=pretrained_root, strict=False)

    def forward(self, x):
        size = x.size()[2:]
        outputs = []

        out_backbone = self.backbone(x)

        x0 = self.decode_head(out_backbone)
        if isinstance(x0, (list, tuple)):
            for out in x0:
                out = F.interpolate(out, size, **up_kwargs)
                outputs.append(out)
        else:
            x0 = F.interpolate(x0, size, **up_kwargs)
            outputs.append(x0)

        if self.aux:
            x1 = self.auxiliary_head(out_backbone)
            x1 = F.interpolate(x1, size, **up_kwargs)
            outputs.append(x1)

        return outputs


def efficientnetv2_s(num_classes, in_chans, aux, head='mlphead'):
    model_config = config.get_st_config()

    model = EffinientNetv2(in_chans=in_chans,
                           aux=aux,
                           model_cnf=model_config,
                           output_channel=[48, 64, 128, 256],
                           num_class=num_classes,
                           head=head)
    return model


def efficientnetv2_m(num_classes, in_chans, aux, head='mlphead'):
    model_config = config.get_mt_config()
    model = EffinientNetv2(in_chans=in_chans,
                           aux=aux,
                           model_cnf=model_config,
                           output_channel=[48, 80, 176, 512],
                           num_class=num_classes,
                           head=head)
    return model


def efficientnetv2_l(num_classes, in_chans, aux, head='mlphead'):
    model_config = config.get_lt_config()

    model = EffinientNetv2(in_chans=in_chans,
                           aux=aux,
                           model_cnf=model_config,
                           output_channel=[64, 96, 224, 640],
                           num_class=num_classes,
                           head=head)
    return model

if __name__ == '__main__':
    model = efficientnetv2_s(num_classes=2, in_chans=3, aux=False, head='fcfpnhead')
    inputs = torch.randn(size=(8,3,256,256))
    outputs = model(inputs)

    flops_params_fps(model, input_shape=(8, 3, 256, 256))

