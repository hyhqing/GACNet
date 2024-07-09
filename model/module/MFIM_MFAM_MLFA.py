import torch
import torch.nn as nn
import  torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
import numpy as np
from model.module.Attention import SE, CBAM, CA
from model.backbone.head.mlp import MLP
from model.backbone.head.base_decoder import resize
# from model.module.gn_module import GNConvModule
# from model.module.AdaptiveDilatedConv import AdaptiveDilatedConv

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

class PositionWeights(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(PositionWeights, self).__init__()
        self.in_channel = in_channel
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv = nn.Conv2d(self.in_channel * 2, self.in_channel * 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.mlp = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2 // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channel * 2),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # (8, 6, 256, 256)
        x_avg = self.pool_w(x).permute(0, 1, 3, 2)  # (8, 6, 256, 1)
        y_avg = self.pool_h(x)  # (8, 6, 256, 1)
        y = torch.cat((x_avg, y_avg), dim=2) # (8, 6, 512, 1)

        y = self.mlp(y)  # (8, 6, 512, 1)
        x_H, x_W = torch.split(y, [H, W], dim=2)   # (8, 6, 1, 256), (8, 6, 256, 1)
        x_W = x_W.permute(0, 1, 3, 2)
        xH_weight = self.sigmoid(self.conv(x_H)) # (8, 6, 1, 256)
        xW_weight = self.sigmoid(self.conv(x_W)) # (8, 6, 256, 1)

        H_weights = xH_weight.reshape(B, 2, self.in_channel, H, 1).permute(1, 0, 2, 3, 4)
        W_weights = xW_weight.reshape(B, 2, self.in_channel, 1, W).permute(1, 0, 2, 3, 4)
        return H_weights, W_weights


class ChannelWeights(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(ChannelWeights, self).__init__()
        self.in_channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channel * 4, self.in_channel * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channel * 4 // reduction, self.in_channel * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.in_channel * 2)
        max = self.max_pool(x).view(B, self.in_channel * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.in_channel * 2, 1)
        channel_weights = y.reshape(B, 2, self.in_channel, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(SpatialWeights, self).__init__()
        self.in_channel = in_channel
        self.mlp = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights

class FeatureInteractionModule(nn.Module):
    def __init__(self, in_channel, reduction=1, lambda_p=0.5, lambda_c=0.2, lambda_s=0.3):
        super(FeatureInteractionModule, self).__init__()
        self.in_channel = in_channel
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.position_weights = PositionWeights(in_channel=in_channel, reduction=reduction)
        self.channel_weights = ChannelWeights(in_channel=in_channel, reduction=reduction)
        self.spatial_weights = SpatialWeights(in_channel=in_channel, reduction=reduction)

    def forward(self, x1, x2):
        position_weights_H, position_weights_W = self.position_weights(x1, x2)
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)

        out_1 = x1 + self.lambda_p * position_weights_H[1] * position_weights_W[1] * x2 \
                    + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_2 = x2 + self.lambda_p * position_weights_H[0] * position_weights_W[0] * x1 \
                    + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1

        return out_1, out_2

class FeatureAggregationModule(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(FeatureAggregationModule, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.conv_rgb = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_x = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        self.dw_conv_r = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, bias=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=in_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channel)
        )

        self.dw_conv_x = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, bias=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=in_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_channel)
        )

        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x_rgb = self.conv_rgb(x1)
        x_x = self.conv_x(x2)

        merge_rgb_x = torch.cat((x1, x2), dim=1)
        y_rgb = self.dw_conv_r(merge_rgb_x)
        y_x = self.dw_conv_x(merge_rgb_x)
        y_rgb_x = torch.cat((y_rgb, y_x), dim=1)
        weights = self.softmax(y_rgb_x)
        weights_rgb, weights_x = torch.split(weights, [C, C], dim=1)
        merge_feature = x_rgb * weights_rgb + x_x * weights_x
        out = self.norm(merge_feature)

        return out


class FeatureInteractionModule_without(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(FeatureInteractionModule_without, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.conv_rgb = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_x = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        rgb = self.conv_rgb(x1)
        x = self.conv_x(x2)
        merge_rgb_x = torch.cat((x1, x2), dim=1)
        weights = self.softmax(merge_rgb_x)
        weights_rgb, weights_x = torch.split(weights, [C, C], dim=1)
        out_rgb = rgb * weights_rgb
        out_x = x * weights_x

        return out_rgb, out_x

class FeatureAggregationModule_without(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(FeatureAggregationModule_without, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.conv_rgb = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_x = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        rgb = self.conv_rgb(x1)
        x = self.conv_x(x2)
        merge_rgb_x = torch.cat((x1, x2), dim=1)
        weights = self.softmax(merge_rgb_x)
        weights_rgb, weights_x = torch.split(weights, [C, C], dim=1)
        out = rgb * weights_rgb + x * weights_x
        out = self.norm(out)

        return out

class MS_FIM(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(MS_FIM, self).__init__()
        self.in_channel = in_channel
        self.feature_interaction = FeatureInteractionModule(in_channel=in_channel, reduction=reduction)
        self.feature_aggregation = FeatureAggregationModule(in_channel=in_channel, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        interaction_feature_1, interaction_feature_2 = self.feature_interaction(x1, x2)
        aggregation_feature = self.feature_aggregation(interaction_feature_1, interaction_feature_2)

        return interaction_feature_1, interaction_feature_2, aggregation_feature


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,stride=stride, padding=(kernel_size - 1) // 2)
        )

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=True):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,stride=stride, padding=(kernel_size - 1) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        H, W = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CA_residual(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA_residual, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out1 = identity * a_w * a_h
        # out1 = a_h.expand_as(x) * a_w.expand_as(x) * identity
        out = identity + out1
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class Cross_Path(nn.Module):
    def __init__(self, dim=(96, 192, 384, 768), num_heads=[1, 2, 5, 8], norm_layer=nn.LayerNorm):
        super().__init__()

        self.pre_conv_1 = nn.Sequential(ConvBNAct(in_channels=dim[3], out_channels=dim[2]),
                                      nn.UpsamplingBilinear2d(scale_factor=2),
                                      ConvBNAct(in_channels=dim[2], out_channels=dim[2]))
        self.pre_conv_2 = nn.Sequential(ConvBNAct(in_channels=dim[2], out_channels=dim[1]),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(in_channels=dim[1], out_channels=dim[1]))
        self.pre_conv_3 = nn.Sequential(ConvBNAct(in_channels=dim[1], out_channels=dim[0]),
                                        nn.UpsamplingBilinear2d(scale_factor=2),
                                        ConvBNAct(in_channels=dim[0], out_channels=dim[0]))

        self.channel_proj1 = nn.Linear(dim[3], dim[3])
        self.channel_proj2 = nn.Linear(dim[2], dim[2])
        self.channel_proj3 = nn.Linear(dim[1], dim[1])
        self.channel_proj4 = nn.Linear(dim[0], dim[0])
        self.channel_proj5 = nn.Linear(dim[2], dim[2])
        self.channel_proj6 = nn.Linear(dim[1], dim[1])
        self.channel_proj7 = nn.Linear(dim[0], dim[0])

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.act3 = nn.ReLU(inplace=True)
        self.act4 = nn.ReLU(inplace=True)
        self.act5 = nn.ReLU(inplace=True)
        self.act6 = nn.ReLU(inplace=True)
        self.act7 = nn.ReLU(inplace=True)

        self.cross_attn1 = CrossAttention(dim[2], num_heads=num_heads[2])
        self.cross_attn2 = CrossAttention(dim[1], num_heads=num_heads[1])
        self.cross_attn3 = CrossAttention(dim[0], num_heads=num_heads[0])
        self.FFN_1 = nn.Sequential(
            nn.Linear(dim[2], dim[2]),
            DWConv(dim[2]),
            nn.Linear(dim[2], dim[2])
        )
        self.FFN_2 = nn.Sequential(
            nn.Linear(dim[2], dim[2]),
            DWConv(dim[2]),
            nn.Linear(dim[2], dim[2])
        )
        self.FFN_3 = nn.Sequential(
            nn.Linear(dim[1], dim[1]),
            DWConv(dim[1]),
            nn.Linear(dim[1], dim[1])
        )
        self.FFN_4 = nn.Sequential(
            nn.Linear(dim[1], dim[1]),
            DWConv(dim[1]),
            nn.Linear(dim[1], dim[1])
        )
        self.FFN_5 = nn.Sequential(
            nn.Linear(dim[0], dim[0]),
            DWConv(dim[0]),
            nn.Linear(dim[0], dim[0])
        )
        self.FFN_6 = nn.Sequential(
            nn.Linear(dim[0], dim[0]),
            DWConv(dim[0]),
            nn.Linear(dim[0], dim[0])
        )
        self.end_proj1 = nn.Linear(dim[2], dim[2])
        self.end_proj2 = nn.Linear(dim[2], dim[2])
        self.end_proj3 = nn.Linear(dim[1], dim[1])
        self.end_proj4 = nn.Linear(dim[1], dim[1])
        self.end_proj5 = nn.Linear(dim[0], dim[0])
        self.end_proj6 = nn.Linear(dim[0], dim[0])

        self.norm1 = norm_layer(dim[2])
        self.norm2 = norm_layer(dim[2])
        self.norm3 = norm_layer(dim[1])
        self.norm4 = norm_layer(dim[1])
        self.norm5 = norm_layer(dim[0])
        self.norm6 = norm_layer(dim[0])

        self.final_proj1 = nn.Linear(dim[2] * 2, dim[2])
        self.final_dw_1 = DWConv(dim[2])
        self.final_proj_1 = nn.Linear(dim[2], dim[2])

        self.final_proj2 = nn.Linear(dim[1] * 2, dim[1])
        self.final_dw_2 = DWConv(dim[1])
        self.final_proj_2 = nn.Linear(dim[1], dim[1])

        self.final_proj3 = nn.Linear(dim[0] * 2, dim[0])
        self.final_dw_3 = DWConv(dim[0])
        self.final_proj_3 = nn.Linear(dim[0], dim[0])

    def forward(self, x1, x2, x3, x4):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        B3, C3, H3, W3 = x3.shape
        B4, C4, H4, W4 = x4.shape

        x1_up = self.pre_conv_1(x1)

        x1_fla = x1.flatten(2).transpose(1, 2)
        x2_fla = x2.flatten(2).transpose(1, 2)
        x3_fla = x3.flatten(2).transpose(1, 2)
        x4_fla = x4.flatten(2).transpose(1, 2)
        x1_up = x1_up.flatten(2).transpose(1, 2)

        u1 = self.act1(self.channel_proj1(x1_fla))
        u2 = self.act2(self.channel_proj2(x2_fla))
        u3 = self.act3(self.channel_proj3(x3_fla))
        u4 = self.act4(self.channel_proj4(x4_fla))
        u11 = self.act5(self.channel_proj5(x1_up))

        # cross attention 1
        v1, v2 = self.cross_attn1(u11, u2)
        out_1_1 = self.FFN_1(v1)
        out_1_2 = self.FFN_2(v2)
        out_1 = self.norm1(v1 + out_1_1)
        out_2 = self.norm2(v2 + out_1_2)
        out1 = torch.cat((out_1, out_2), dim=-1)
        out1 = self.final_proj1(out1)
        out1 = self.final_dw_1(out1)
        out1 = self.final_proj_1(out1)

        B_out1, N_out1, _C_out1 = out1.shape
        output_1 = out1.permute(0, 2, 1).reshape(B_out1, _C_out1, H2, W2).contiguous()
        output_1_residual = self.pre_conv_2(output_1)
        output_1_residual = output_1_residual.flatten(2).transpose(1, 2)
        output_1_residual = self.act6(self.channel_proj6(output_1_residual))

        # cross attention 2
        v3, v4 = self.cross_attn2(output_1_residual, u3)
        out_2_1 = self.FFN_3(v3)
        out_2_2 = self.FFN_4(v4)
        out_3 = self.norm3(v3 + out_2_1)
        out_4 = self.norm4(v4 + out_2_2)
        out2 = torch.cat((out_3, out_4), dim=-1)
        out2 = self.final_proj2(out2)
        out2 = self.final_dw_2(out2)
        out2 = self.final_proj_2(out2)

        B_out2, N_out2, _C_out2 = out2.shape
        output_2 = out2.permute(0, 2, 1).reshape(B_out2, _C_out2, H3, W3).contiguous()
        output_2_residual = self.pre_conv_3(output_2)
        output_2_residual = output_2_residual.flatten(2).transpose(1, 2)
        output_2_residual = self.act7(self.channel_proj7(output_2_residual))

        # cross attention 3
        v5, v6 = self.cross_attn3(output_2_residual, u4)
        out_3_1 = self.FFN_5(v5)
        out_3_2 = self.FFN_6(v6)
        out_5 = self.norm5(v5 + out_3_1)
        out_6 = self.norm6(v6 + out_3_2)
        out3 = torch.cat((out_5, out_6), dim=-1)
        out3 = self.final_proj3(out3)
        out3 = self.final_dw_3(out3)
        out3 = self.final_proj_3(out3)

        B_out3, N_out3, _C_out3 = out3.shape
        output_3 = out3.permute(0, 2, 1).reshape(B_out3, _C_out3, H4, W4).contiguous()

        return x1, output_1, output_2, output_3


class Cross_Attention_MLP(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), channels=512, num_classes=6, num_heads=[1, 2, 4, 8], norm_layer=nn.LayerNorm):
        super(Cross_Attention_MLP, self).__init__()
        self.in_channels = in_channels
        self.cross = Cross_Path(dim=in_channels, num_heads=num_heads, norm_layer=norm_layer)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels[0], self.in_channels[1], \
                                                                         self.in_channels[2], self.in_channels[3]
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=channels)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=channels)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=channels)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=channels)

        self.linear_c3_out = MLP(input_dim=channels, embed_dim=channels)
        self.linear_c2_out = MLP(input_dim=channels, embed_dim=channels)
        self.linear_c1_out = MLP(input_dim=channels, embed_dim=channels)

        self.linear_fuse = MLP(input_dim=channels * 4, embed_dim=channels)

        self.head = nn.Sequential(ConvBNAct(channels, channels // 2),
                                  nn.Dropout(0.1),
                                  nn.UpsamplingBilinear2d(scale_factor=2),
                                  ConvBNAct(channels // 2, channels // 2),
                                  Conv(channels // 2, num_classes, kernel_size=1))

    def forward(self, x1, x2, x3, x4):
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x1, x2, x3, x4
        out = []
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        c4, c3, c2, c1 = self.cross(c4, c3, c2, c1)

        _c4 = self.linear_c4(c4).permute(0, 2, 1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c3.size()[2:], **up_kwargs)

        out.append(resize(_c4, size=c1.size()[2:], **up_kwargs))

        _c3 = self.linear_c3(c3).permute(0, 2, 1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = _c4 + _c3

        _c3_out = self.linear_c3_out(_c3).permute(0, 2, 1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        out.append(resize(_c3_out, size=c1.size()[2:], **up_kwargs))

        _c2 = self.linear_c2(c2).permute(0, 2, 1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c3 = resize(_c3, size=c2.size()[2:], **up_kwargs)
        _c2 = _c3 + _c2

        _c2_out = self.linear_c2_out(_c2).permute(0, 2, 1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        out.append(resize(_c2_out, size=c1.size()[2:], **up_kwargs))

        _c1 = self.linear_c1(c1).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], **up_kwargs)
        _c1 = _c2 + _c1

        _c1_out = self.linear_c1_out(_c1).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        out.append(_c1_out)

        _c = self.linear_fuse(torch.cat(out, dim=1)).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        # _c = self.dropout(_c)
        # x = self.linear_pred(_c).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])
        out = self.head(_c)

        return out

if __name__ == '__main__':
    from tool.flops_params_fps_count import flops_params_fps
    model = FeatureInteractionModule(in_channel=3)
    rgb = torch.randn(size=(8, 3, 256, 256))
    modal_x = torch.randn(size=(8, 3, 256, 256))
    outputs = model(rgb, modal_x)

