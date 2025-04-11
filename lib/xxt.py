import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import os
import gc

from torch.nn import init
import math
import numpy as np
from collections import OrderedDict
from torch.nn import Dropout
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states


class d34_Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(d34_Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class d2_Conv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return d2_Conv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return d2_Conv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = residual + y

        return y


class triple_haed_feature_receptor(nn.Module):
    def __init__(self, channel):
        super(triple_haed_feature_receptor, self).__init__()
        self.relu = nn.ReLU(True)
        self.channel = channel

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.downsample = d34_Conv2d(3 * channel, channel, 3, padding=1)

        self.PreActBTN1_1 = PreActBottleneck(cin=channel, cout=channel * 2, cmid=channel)
        self.PreActBTN1_2 = PreActBottleneck(cin=channel * 2, cout=channel * 2, cmid=channel)

    def forward(self, x1, x2, x3):
        x_1 = x1
        x_2 = x2
        x_3 = x3

        x1_1_1 = self.upsample4(x_1)
        x1_1_2 = x1_1_1 + x_3

        x1_1_3 = self.upsample2(x_1)
        x1_1_4 = (x1_1_3 + x_2) * x_2

        x1_1_5 = self.upsample2(x1_1_4)

        x2_1_1 = self.upsample2(x_2)
        x_3 = x1_1_1 + x1_1_2 + x1_1_5 + x2_1_1 + x_3

        x1_1 = self.PreActBTN1_1(x_1)
        x1_2 = self.PreActBTN1_2(x1_1)
        x1_3 = self.upsample2(x1_2)
        x1_3 = torch.cat((x1_3, x1_1_4), dim=1)
        x1_4 = self.upsample4(x1_2)
        x1_4 = torch.cat((x1_4, x1_1_5), dim=1)
        x1_3 = self.upsample2(x1_3)
        x1_4 = x1_4 + x1_3
        x1_4 = self.downsample(x1_4)

        return (x3 + x1_4) * x3

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0,
                             0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6,
                             3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0,
                             1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4,
                             3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4,
                             6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2,
                             2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, in_planes, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(
            dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((self.dct_h, self.dct_w))

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = self.avgpool(x)
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        y = self.sigmoid(out)
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)

        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.weight = self.get_dct_filter(
            height, width, mapper_x, mapper_y, channel).cuda()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + \
                                  str(len(x.shape))
        # n, c, h, w = x.shape

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)

        x = x * self.weight
        result = torch.sum(torch.sum(x, dim=2), dim=2)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros((channel, tile_size_x, tile_size_y))

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(
                        t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter

class four_haed_feature_receptor(nn.Module):
    def __init__(self, channel):
        super(four_haed_feature_receptor, self).__init__()
        self.relu = nn.ReLU(True)
        self.channel = channel

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.downsample = d34_Conv2d(4 * channel, channel, 3, padding=1)

        self.PreActBTN1_1 = PreActBottleneck(cin=channel, cout=channel * 2, cmid=channel)
        self.PreActBTN1_2 = PreActBottleneck(cin=channel * 2, cout=channel * 2, cmid=channel)

        self.PreActBTN2_1 = PreActBottleneck(cin=channel, cout=channel * 4, cmid=channel * 2)
        self.PreActBTN2_2 = PreActBottleneck(cin=channel * 4, cout=channel * 4, cmid=channel * 2)

    def forward(self, x1, x2, x3, x4):
        x_1 = x1
        x_2 = x2
        x_3 = x3
        x_4 = x4

        x1_1_1 = self.upsample4(x_1)
        x1_1_2 = x1_1_1 + x_3

        x1_1_3 = self.upsample2(x_1)
        x1_1_4 = (x1_1_3 + x_2) * x_2

        x1_1_5 = self.upsample2(x1_1_4)

        x2_1_1 = self.upsample2(x_2)
        x_3 = x1_1_1 + x1_1_2 + x1_1_5 + x2_1_1 + x_3

        x3_1 = self.upsample8(x_1)
        x3_2 = self.upsample2(x_3)
        x3_3 = x3_1 + x3_2
        x4 = (x3_3 + x4) * x4

        x1_1 = self.PreActBTN1_1(x_1)
        x1_2 = self.PreActBTN1_2(x1_1)
        x1_3 = self.upsample2(x1_2)
        x1_3 = torch.cat((x1_3, x1_1_4), dim=1)
        x1_4 = self.upsample4(x1_2)
        x1_4 = torch.cat((x1_4, x1_1_5), dim=1)
        x1_5 = self.upsample2(x1_4)
        x1_6 = torch.cat((x1_5, x4), dim=1)
        x1_7 = self.upsample2(x1_4)
        x1_8 = torch.cat((x1_7, x4), dim=1)
        x1_9 = x1_6 + x1_8
        x1_9 = self.downsample(x1_9)

        return (x_4 + x1_9) * x_4

class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, ):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim=1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)


def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        axial = axial.reshape(-1, t, d)

        axial = self.fn(axial, **kwargs)

        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


class AxialImageTransformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding=1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(
            axial_pos_emb_shape) else nn.Identity()

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList(
                [PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in
                 permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_functions)
            layers.append(conv_functions)

        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)


class xxt(nn.Module):
    def __init__(self, channel=32):
        super(xxt, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer64_32 = d34_Conv2d(64, channel, 1)
        self.Translayer128_32 = d34_Conv2d(128, channel, 1)

        self.Translayer32_1 = d34_Conv2d(128, channel, 1)
        self.Translayer32_2 = d34_Conv2d(320, channel, 1)
        self.Translayer32_3 = d34_Conv2d(512, channel, 1)

        self.Translayer64_1 = d34_Conv2d(64, channel * 2, 1)
        self.Translayer64_2 = d34_Conv2d(128, channel * 2, 1)
        self.Translayer64_3 = d34_Conv2d(320, channel * 2, 1)
        self.Translayer64_4 = d34_Conv2d(512, channel * 2, 1)

        self.Translayer128_1 = d34_Conv2d(64, channel * 4, 1)
        self.Translayer128_2 = d34_Conv2d(128, channel * 4, 1)
        self.Translayer128_3 = d34_Conv2d(320, channel * 4, 1)
        self.Translayer128_4 = d34_Conv2d(512, channel * 4, 1)

        # self.se = SEAttention(channel=32, reduction=4)

        # self.sk = SKAttention(channel=64, reduction=4)

        self.triple_haed_feature_receptor_32 = triple_haed_feature_receptor(channel)
        self.triple_haed_feature_receptor_64 = triple_haed_feature_receptor(channel * 2)
        self.four_haed_feature_receptor = four_haed_feature_receptor(channel * 4)

        self.axial32 = AxialImageTransformer(
            dim=32,
            depth=12,
            reversible=True
        ).cuda()

        self.axial64 = AxialImageTransformer(
            dim=64,
            depth=12,
            reversible=True
        ).cuda()

        self.axial128 = AxialImageTransformer(
            dim=128,
            depth=12,
            reversible=True
        ).cuda()

        self.axial320 = AxialImageTransformer(
            dim=320,
            depth=12,
            reversible=True
        ).cuda()

        self.axial512 = AxialImageTransformer(
            dim=512,
            depth=12,
            reversible=True
        ).cuda()

        # self.ca = ChannelAttention(64)
        # #self.sa = SpatialAttention()
        # self.SAM = SAM()
        self.fcanet1 = MultiSpectralAttentionLayer(64, 16, 16, 64).cuda()
        self.fcanet2 = MultiSpectralAttentionLayer(128, 16, 128, 128).cuda()
        self.fcanet3 = MultiSpectralAttentionLayer(320, 16, 320, 320).cuda()
        self.fcanet4 = MultiSpectralAttentionLayer(512, 16, 512, 512).cuda()

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_32 = nn.Conv2d(channel, 1, 1)
        self.out_64 = nn.Conv2d(channel * 2, 1, 1)
        self.out_320 = nn.Conv2d(channel * 10, 1, 1)
        self.out_512 = nn.Conv2d(channel * 16, 1, 1)

        # self.dropout = Dropout(0.5)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        # fcanet
        x_1 = self.fcanet1(x1)
        x_2 = self.fcanet2(x2)
        x_3 = self.fcanet3(x3)
        x_4 = self.fcanet4(x4)

        # x_1 = self.axial64(x_1)
        # x_2 = self.axial128(x_2)
        # x_3 = self.axial320(x_3)
        # x_4 = self.axial512(x_4)

        x_2a = self.Translayer32_1(x_2)
        x_3a = self.Translayer32_2(x_3)
        x_4a = self.Translayer32_3(x_4)

        x_1b = self.Translayer64_1(x_1)
        x_2b = self.Translayer64_2(x_2)
        x_3b = self.Translayer64_3(x_3)

        # x1axial = self.Translayer64_1(x1)
        # x2axial = self.Translayer64_2(x2)
        # x3axial = self.Translayer64_3(x3)
        # x4axial = self.Translayer64_4(x4)

        x_1c = self.Translayer128_1(x_1)
        x_2c = self.Translayer128_2(x_2)
        x_3c = self.Translayer128_3(x_3)
        x_4c = self.Translayer128_4(x_4)

        # x1axial = self.axial(x_1c)
        # x2axial = self.axial(x_2c)
        # x3axial = self.axial(x_3c)
        # x4axial = self.axial(x_4c)

        # module2_feature2 = self.module2(x4axial, x3axial, x2axial, x1axial)

        module1_feature1 = self.triple_haed_feature_receptor_32(x_4a, x_3a, x_2a)
        module1_feature2 = self.triple_haed_feature_receptor_64(x_3b, x_2b, x_1b)

        module2_feature = self.four_haed_feature_receptor(x_4c, x_3c, x_2c, x_1c)

        module1_feature2 = self.Translayer64_32(module1_feature2)
        module1_feature2 = self.down05(module1_feature2)

        module2_feature = self.Translayer128_32(module2_feature)
        module2_feature = self.down05(module2_feature)

        # module2_feature2 = self.Translayer128_32(module2_feature2)
        # module2_feature2 = self.down05(module2_feature2)
        #
        # #module1_feature = self.se(module1_feature)
        # # T2 = self.se(T2)
        #
        # sam_feature = self.SAM(module1_feature, T2)

        module1_feature1 = self.axial32(module1_feature1)
        module1_feature2 = self.axial32(module1_feature2)
        module2_feature = self.axial32(module2_feature)

        prediction1 = self.out_32(module1_feature1)
        prediction2 = self.out_32(module1_feature2)
        prediction3 = self.out_32(module2_feature)
        # prediction4 = self.out_32(module2_feature2)

        # prediction5 = self.out_512(x_4)
        # prediction5 = self.down05(prediction5)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3_8 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')
        prediction4_8 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')
        # prediction7_8 = F.interpolate(prediction4, scale_factor=8, mode='bilinear')

        return prediction1_8, prediction2_8, prediction3_8, prediction4_8


if __name__ == '__main__':
    model = xxt().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2, prediction3, prediction4 = model(input_tensor)
    print(prediction1.size(), prediction2.size(), prediction3.size())