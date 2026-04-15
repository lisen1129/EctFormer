import math
from typing import Tuple
import torch.utils.checkpoint as checkpoint
from torch import Tensor
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath,trunc_normal_
from models.repvit import RepViTBlock



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC
        return x


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

class BiLevelRoutingAttention(nn.Module):
    """
    BiLevelRoutingAttention类实现了一个双层路由注意力机制。
    n_win: 单边窗口的数量（实际窗口数为n_win*n_win）。
    kv_per_win: 仅当kv_downsample_mode='ada_xxxpool'时有效，每个窗口的键/值对数量。实际数量为kv_per_win*kv_per_win。
    topk: 窗口过滤的topk值。
    param_attention: 'qkvo'-线性变换用于q,k,v和o, 'none': 无参数的注意力。
    param_routing: 额外的路由线性变换。
    diff_routing: 路由是否可微分。
    soft_routing: 是否乘以软路由权重。
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # 局部注意力设置
        self.dim = dim
        self.n_win = n_win  # 窗口高宽
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        ################ side_dwconv，例如ShuntedTransformer中的LCE ###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        ################  全局路由设置 #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # 路由器
        # 不能同时设置param_routing=True和diff_routing=False
        assert not (self.param_routing and not self.diff_routing)
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # 软路由，始终可微分（如果没有detach）
            mul_weight = 'soft'
        elif self.diff_routing:  # 硬可微分路由
            mul_weight = 'hard'
        else:
            mul_weight = 'none'  # 硬不可微分路由
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv映射（全局路由和局部注意力共享）
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        # 根据kv_downsample_mode选择不同的键值对下采样策略
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # 不进行键值对下采样
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # 局部注意力的softmax激活
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC格式的张量

        返回:
            NHWC格式的张量
        """
        # 如果启用auto_pad，对于语义分割任务使用填充
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # 填充后的尺寸
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  # 确保H和W能被n_win整除
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        # 将x重排为适合处理的形状
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        ################# qkv投影 ###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        # 对q和kv进行像素级别的重排
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ################## 侧边深度可分离卷积（如果配置） ##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################
        # 根据路由权重选择性地聚集键和值
        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### 执行正常的注意力计算 ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
                                  q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # 如果启用了auto_pad并且进行了填充，则裁剪掉填充的部分
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out

class BiformerBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                       num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                       kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=4, kv_downsample_mode='ada_avgpool',
                       topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, mlp_ratio=4, mlp_dwconv=False,
                       side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        """
        dim: 输入和输出特征的维度。
        drop_path: DropPath的比率，用于正则化和防止过拟合。
        layer_scale_init_value: 层缩放初始化值，用于控制层缩放技巧的初始比例。
        num_heads: 注意力头的数量。
        n_win: 窗口大小，用于局部注意力计算。
        qk_dim: 查询（Q）和键（K）的维度。
        qk_scale: 查询和键的缩放因子。
        kv_per_win: 每个窗口的键值对（K-V）数量。
        kv_downsample_ratio: 键值下采样的比例。
        kv_downsample_kernel: 下采样的卷积核大小。
        kv_downsample_mode: 下采样的模式，如ada_avgpool。
        topk: 顶部K个值，用于注意力计算。
        param_attention: 注意力参数化方式。
        param_routing: 路由参数化。
        diff_routing: 差异化路由。
        soft_routing: 软路由。
        mlp_ratio: MLP的隐藏层扩展比例。
        mlp_dwconv: 是否在MLP中使用深度可分离卷积。
        side_dwconv: 侧边深度可分离卷积的大小。
        before_attn_dwconv: 注意力计算前的深度可分离卷积大小。
        pre_norm: 是否使用预归一化。
        auto_pad: 是否自动填充。
        """
        super().__init__()
        # 如果没有指定qk_dim，则使用dim作为其值
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:# 如果指定了在注意力机制前的深度卷积尺寸
            # 位置嵌入使用深度卷积
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            # 否则，位置嵌入不做任何操作
            self.pos_embed = lambda x: 0
        # 重要的归一化操作，避免注意力机制崩溃
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # 双层路由注意力机制
        self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                        qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                        kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                        topk=topk, param_attention=param_attention, param_routing=param_routing,
                                        diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                        auto_pad=auto_pad)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),# MLP的第一层
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),# 可选的深度卷积
                                 nn.GELU(),# 激活函数
                                 nn.Linear(int(mlp_ratio*dim), dim)# MLP的第二层
                                )
        # 可选的DropPath正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 技巧：层缩放 & 预归一化/后归一化
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            # 第一层缩放参数
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            # 第二层缩放参数
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        # 是否使用预归一化
        self.pre_norm = pre_norm


    def forward(self, x):
        """
        前向传播函数
        x: NCHW格式的张量
        """
        # 卷积位置嵌入
        x = x + self.pos_embed(x)
        # 重排到NHWC格式，以便于注意力和MLP计算
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # 注意力和MLP计算
        if self.pre_norm:
            if self.use_layer_scale:
                # 使用层缩放的注意力计算
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                # 使用层缩放的MLP计算
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else:
                # 普通的注意力计算
                x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
                # 普通的MLP计算
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                # 使用层缩放，先计算后归一化
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                # 使用层缩放，先计算后归一化
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                # 先计算后归一化
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)
        # 重排回NCHW格式
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x


class Biformer_Block(nn.Module):
    def __init__(self, dim, depth, num_head):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            BiformerBlock(dim=dim, num_heads=num_head)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class Rep_Biformer_Block(nn.Module):
    def __init__(self, dim, depth, num_head):
        super().__init__()
        self.cnn = RepViTBlock(inp=dim, hidden_dim=dim*2, oup=dim)
        self.transformer = Biformer_Block(dim, depth, num_head)

    def forward(self, x):
        x = self.cnn(x)
        x = self.transformer(x)
        return x

# SimCSPSPPF池化
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvBNReLU, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class CSPSPPFModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, block=ConvBNReLU):
        super(CSPSPPFModule, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1, 0)
        self.cv2 = block(in_channels, c_, 1, 1, 0)
        self.cv3 = block(c_, c_, 3, 1, 1)
        self.cv4 = block(c_, c_, 1, 1, 0)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.cv5 = block(4 * c_, c_, 1, 1, 0)
        self.cv6 = block(c_, c_, 3, 1, 1)
        self.cv7 = block(2 * c_, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


class SimCSPSPPF(nn.Module):
    '''CSPSPPF with ReLU activation'''

    def __init__(self, in_channels, out_channels, depth, kernel_size=5, e=0.5, block=ConvBNReLU):
        super(SimCSPSPPF, self).__init__()
        self.cspsppf = nn.ModuleList([
            CSPSPPFModule(in_channels, out_channels, kernel_size, e, block)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.cspsppf:
            x = blk(x)
        return x

# 下采样
class DownSampler(nn.Module):
    """Down sample the feature map.
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

# 上采样
class UpSampler(nn.Module):
    """Up sample the feature map."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x


class Ectformer(nn.Module):
    def __init__(self, in_channel, out_channel, channels=[64, 128, 256, 512, 512], depths=[1, 1, 2, 2, 2], num_heads=[2, 4, 8, 16, 16],):
        super().__init__()

        self.pre= nn.Sequential(
                nn.Conv2d(in_channel, channels[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.GELU()
            )
        self.Down1 = DownSampler(channels[0], channels[0])
        self.encoder1 = Rep_Biformer_Block(dim=channels[0], depth=depths[0], num_head=num_heads[0])

        self.Down2 = DownSampler(channels[0], channels[1])
        self.encoder2 = Rep_Biformer_Block(dim=channels[1], depth=depths[1], num_head=num_heads[1])

        self.Down3 = DownSampler(channels[1], channels[2])
        self.encoder3 = Rep_Biformer_Block(dim=channels[2], depth=depths[2], num_head=num_heads[2])

        self.Down4 = DownSampler(channels[2], channels[3])
        self.encoder4 = Rep_Biformer_Block(dim=channels[3], depth=depths[3], num_head=num_heads[3])

        self.Down5 = DownSampler(channels[3], channels[4])
        self.encoder5 = Rep_Biformer_Block(dim=channels[4], depth=depths[4], num_head=num_heads[4])


        self.pool = SimCSPSPPF(in_channels=channels[4], out_channels=channels[4], depth=1)


        self.decoder5 = Rep_Biformer_Block(dim=channels[4]*2, depth=depths[4], num_head=num_heads[4])
        self.up5 = UpSampler(in_channel=channels[4]*2, out_channel=channels[3])

        self.decoder4 = Rep_Biformer_Block(dim=channels[3]*2, depth=depths[3], num_head=num_heads[3])
        self.up4 = UpSampler(in_channel=channels[3] * 2, out_channel=channels[2])

        self.decoder3 = Rep_Biformer_Block(dim=channels[2]*2, depth=depths[2], num_head=num_heads[2])
        self.up3 = UpSampler(in_channel=channels[2] * 2, out_channel=channels[1])

        self.decoder2 = Rep_Biformer_Block(dim=channels[1]*2, depth=depths[1], num_head=num_heads[1])
        self.up2 = UpSampler(in_channel=channels[1] * 2, out_channel=channels[0])

        self.decoder1 = Rep_Biformer_Block(dim=channels[0]*2, depth=depths[0], num_head=num_heads[0])
        self.up1 = UpSampler(in_channel=channels[0] * 2, out_channel=channels[0])

        self.out = nn.Conv2d(channels[0], out_channel, 3, 1, 1)

    def forward(self, x):

        x = self.pre(x)

        x1 = self.Down1(x)
        # print(x1.shape)
        x1 = self.encoder1(x1)

        x2 = self.Down2(x1)
        # print(x2.shape)
        x2 = self.encoder2(x2)

        x3 = self.Down3(x2)
        # print(x3.shape)
        x3 = self.encoder3(x3)

        x4 = self.Down4(x3)
        # print(x4.shape)
        x4 = self.encoder4(x4)

        x5 = self.Down5(x4)
        # print(x5.shape)
        x5 = self.encoder5(x5)

        p = self.pool(x5)

        c5 = torch.cat([p, x5], dim=1)
        c5 = self.decoder5(c5)
        c5 = self.up5(c5)

        c4 = torch.cat([c5, x4], dim=1)
        c4 = self.decoder4(c4)
        c4 = self.up4(c4)

        c3 = torch.cat([c4, x3], dim=1)
        c3 = self.decoder3(c3)
        c3 = self.up3(c3)

        c2 = torch.cat([c3, x2], dim=1)
        c2 = self.decoder2(c2)
        c2 = self.up2(c2)

        c1 = torch.cat([c2, x1], dim=1)
        c1 = self.decoder1(c1)
        c1 = self.up1(c1)

        c = self.out(c1)

        return c

if __name__ == '__main__':
    img = torch.randn(1, 6, 256, 256)

    net = Ectformer(in_channel=6, out_channel=3)

    out = net(img)
    print(out.shape)