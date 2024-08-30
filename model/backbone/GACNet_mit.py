import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from model.module.MFIM_MFAM_MLFA import MS_FIM, Cross_Attention_MLP
from tool.flops_params_fps_count import flops_params_fps_dual
from tool.init_func import init_weight
from tool.logger import get_logger
from model.backbone.head import mlp, seg, fcfpn, uper, unet

logger = get_logger()

up_kwargs = {'mode': 'bilinear', 'align_corners': False}

class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

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

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)

        return x, H, W


class MixTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                                    in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                                    in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                                    in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

        self.MS_FIM_1 = MS_FIM(embed_dims[0], reduction=1)
        self.MS_FIM_2 = MS_FIM(embed_dims[1], reduction=1)
        self.MS_FIM_3 = MS_FIM(embed_dims[2], reduction=1)
        self.MS_FIM_4 = MS_FIM(embed_dims[3], reduction=1)

        self.apply(self._init_weights)

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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x1, x2):
        features = []
        B = x1.shape[0]

        # stage 1
        rgb_1, H, W = self.patch_embed1(x1)
        # B H*W/16 C
        x_1, _, _ = self.extra_patch_embed1(x2)
        for i, blk in enumerate(self.block1):
            rgb_1 = blk(rgb_1, H, W)
        for i, blk in enumerate(self.extra_block1):
            x_1 = blk(x_1, H, W)
        rgb_1 = self.norm1(rgb_1)
        x_1 = self.extra_norm1(x_1)

        rgb_1 = rgb_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_1 = x_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_fim_1, x_fim_1, merge_fea_1 = self.MS_FIM_1(rgb_1, x_1)
        features.append(merge_fea_1)

        # stage 2
        rgb_2, H, W = self.patch_embed2(rgb_1 + rgb_fim_1)
        x_2, _, _ = self.extra_patch_embed2(x_1 + x_fim_1)
        for i, blk in enumerate(self.block2):
            rgb_2 = blk(rgb_2, H, W)
        for i, blk in enumerate(self.extra_block2):
            x_2 = blk(x_2, H, W)
        rgb_2 = self.norm2(rgb_2)
        x_2 = self.extra_norm2(x_2)

        rgb_2 = rgb_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_2 = x_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_fim_2, x_fim_2, merge_fea_2 = self.MS_FIM_2(rgb_2, x_2)
        features.append(merge_fea_2)

        # stage 3
        rgb_3, H, W = self.patch_embed3(rgb_2 + rgb_fim_2)
        x_3, _, _ = self.extra_patch_embed3(x_2 + x_fim_2)
        for i, blk in enumerate(self.block3):
            rgb_3 = blk(rgb_3, H, W)
        for i, blk in enumerate(self.extra_block3):
            x_3 = blk(x_3, H, W)
        rgb_3 = self.norm3(rgb_3)
        x_3 = self.extra_norm3(x_3)

        rgb_3 = rgb_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_3 = x_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_fim_3, x_fim_3, merge_fea_3 = self.MS_FIM_3(rgb_3, x_3)
        features.append(merge_fea_3)

        # stage 4
        rgb_4, H, W = self.patch_embed4(rgb_3 + rgb_fim_3)
        x_4, _, _ = self.extra_patch_embed4(x_3 + x_fim_3)
        for i, blk in enumerate(self.block4):
            rgb_4 = blk(rgb_4, H, W)
        for i, blk in enumerate(self.extra_block4):
            x_4 = blk(x_4, H, W)
        rgb_4 = self.norm4(rgb_4)
        x_4 = self.extra_norm4(x_4)

        rgb_4 = rgb_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_4 = x_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_fim_4, x_fim_4, merge_fea_4 = self.MS_FIM_4(rgb_4, x_4)
        features.append(merge_fea_4)

        return features


class GACNet(nn.Module):
    def __init__(self, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios,
                 qkv_bias, drop_rate, drop_path_rate, depths, sr_ratios,
                 aux, norm_layer=nn.LayerNorm, pretrained_root=None, head=False,head_type='mlphead'):
        super(GACNet, self).__init__()
        self.head = head
        self.aux = aux
        self.head_name = head_type
        self.pretrained_root = pretrained_root
        self.norm_layer = norm_layer

        self.backbone = MixTransformer(in_chans=in_chans, patch_size=patch_size,
                                       num_classes=num_classes,embed_dims=embed_dims, num_heads=num_heads,
                                       mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,drop_rate=drop_rate,
                                       drop_path_rate=drop_path_rate, depths=depths, sr_ratios=sr_ratios)

        if head:
            if self.head_name == 'seghead':
                self.decoder = seg.SegHead(in_channels=embed_dims, num_classes=num_classes, in_index=[0, 1, 2, 3])

            if self.head_name == 'mlphead':
                self.decoder = mlp.MLPHead(in_channels=embed_dims, num_classes=num_classes, in_index=[0, 1, 2, 3],
                                               channels=256)

            if self.head_name == 'fcfpnhead':
                self.decoder = fcfpn.FCFPNHead(in_channels=embed_dims, num_classes=num_classes,
                                                   in_index=[0, 1, 2, 3], channels=256)

            if self.head_name == 'unethead':
                self.decoder = unet.UNetHead(in_channels=embed_dims, num_classes=num_classes, in_index=[0, 1, 2, 3])

            if self.head_name == 'uperhead':
                self.decoder = uper.UPerHead(in_channels=embed_dims, num_classes=num_classes)

            if self.head_name == 'mlphead':
                self.decoder = mlp.MLPHead(in_channels=embed_dims, num_classes=num_classes, in_index=[0, 1, 2, 3],
                                           channels=256)
        else:
            self.decoder = Cross_Attention_MLP(in_channels=embed_dims, num_heads=num_heads, channels=256, num_classes=num_classes, norm_layer=norm_layer)

        if self.aux:
            self.auxiliary_head = mlp.MLPHead(in_channels=embed_dims, num_classes=num_classes, in_index=[0, 1, 2, 3], channels=256)

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


def load_dualpath_model(model, model_file):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        # raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))


def GACNet_b0(num_classes, in_chans, aux, head=False, head_type='mlphead'):

    model = GACNet(in_chans=in_chans, aux=aux, num_classes=num_classes, head = head,
                    head_type=head_type, patch_size=4, embed_dims=[32, 64, 160, 256],
                    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, depths=[2, 2, 2, 2],
                    sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
    return model

def GACNet_b1(num_classes, in_chans, aux, head=False, head_type='mlphead'):

    model = GACNet(in_chans=in_chans, aux=aux, num_classes=num_classes, head = head,
                    head_type=head_type, patch_size=4, embed_dims=[64, 128, 320, 512],
                    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                    depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
    return model

def GACNet_b2(num_classes, in_chans, aux, head=False, head_type='mlphead'):

    model = GACNet(in_chans=in_chans, aux=aux, num_classes=num_classes, head = head,
                    head_type=head_type, patch_size=4, embed_dims=[64, 128, 320, 512],
                    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                    depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
    return model

def GACNet_b3(num_classes, in_chans, aux, head=False, head_type='mlphead'):

    model = GACNet(in_chans=in_chans, aux=aux, num_classes=num_classes, head = head,
                    head_type=head_type, patch_size=4, embed_dims=[64, 128, 320, 512],
                    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                    depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
    return model

def GACNet_b4(num_classes, in_chans, aux, head=False, head_type='mlphead'):

    model = GACNet(in_chans=in_chans, aux=aux, num_classes=num_classes, head = head,
                    head_type=head_type, patch_size=4, embed_dims=[64, 128, 320, 512],
                    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                    depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
    return model

def GACNet_b5(num_classes, in_chans, aux, head=False, head_type='mlphead'):

    model = GACNet(in_chans=in_chans, aux=aux,num_classes=num_classes, head = head, head_type=head_type,
                    patch_size=4, embed_dims=[64, 128, 320, 512],
                    num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                    depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
    return model

if __name__ == '__main__':
    model = GACNet_b4(num_classes=45, in_chans=3, aux=False, head=False, head_type='seghead')
    RGB_inputs = torch.randn(size=(8, 3, 256, 256))
    X_inputs = torch.randn(size=(8, 1, 256, 256))
    outputs = model(RGB_inputs, X_inputs)

    flops_params_fps_dual(model, RGB_shape=(8, 3, 256, 256), X_shape=(8, 1, 256, 256))
