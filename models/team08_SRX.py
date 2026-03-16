# =============================================================================
#  00_EnsembleDenoiser.py
#  NTIRE 2026 Image Denoising Challenge (sigma=50)
#  Ensemble of SCUNet + XFormer + Restormer
#  -- All three architectures are self-contained in this single file --
# =============================================================================

import math
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath
from timm.models.layers import to_2tuple


# =============================================================================
#  SECTION 1 — SCUNet Architecture
#  Original: https://github.com/cszn/SCUNet
# =============================================================================

class SCU_WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        self.linear = nn.Linear(input_dim, output_dim)
        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = nn.Parameter(
            self.relative_position_params
            .view(2 * window_size - 1, 2 * window_size - 1, self.n_heads)
            .transpose(1, 2).transpose(0, 1))

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool,
                                device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def relative_embedding(self):
        cord = torch.tensor(
            np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

    def forward(self, x):
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c',
                      p1=self.window_size, p2=self.window_size)
        h_windows, w_windows = x.size(1), x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c',
                      p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c',
                             c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size,
                                           shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
        probs = F.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c',
                           w1=h_windows, p1=self.window_size)
        if self.type != 'W':
            output = torch.roll(output,
                                shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return output


class SCU_Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path,
                 type='W', input_resolution=None):
        super().__init__()
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = SCU_WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SCU_ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path,
                 type='W', input_resolution=None):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'
        self.trans_block = SCU_Block(trans_dim, trans_dim, head_dim, window_size,
                                     drop_path, self.type, input_resolution)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        x = x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x


class SCUNet(nn.Module):
    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64,
                 drop_path_rate=0.0, input_resolution=256):
        super().__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))

        begin = 0
        self.m_down1 = nn.Sequential(
            *[SCU_ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution)
              for i in range(config[0])],
            nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False))
        begin += config[0]
        self.m_down2 = nn.Sequential(
            *[SCU_ConvTransBlock(dim, dim, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2)
              for i in range(config[1])],
            nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False))
        begin += config[1]
        self.m_down3 = nn.Sequential(
            *[SCU_ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4)
              for i in range(config[2])],
            nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False))
        begin += config[2]
        self.m_body = nn.Sequential(
            *[SCU_ConvTransBlock(4 * dim, 4 * dim, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 8)
              for i in range(config[3])])
        begin += config[3]
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False),
            *[SCU_ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4)
              for i in range(config[4])])
        begin += config[4]
        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False),
            *[SCU_ConvTransBlock(dim, dim, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2)
              for i in range(config[5])])
        begin += config[5]
        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False),
            *[SCU_ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size,
                                 dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution)
              for i in range(config[6])])
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]


# =============================================================================
#  SECTION 2 — Restormer Architecture
#  Original: https://github.com/swz30/Restormer
# =============================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class R_BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(torch.Size(normalized_shape)))
        self.normalized_shape = torch.Size(normalized_shape)

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class R_WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class R_LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = R_BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' \
            else R_WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class R_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class R_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class R_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = R_LayerNorm(dim, LayerNorm_type)
        self.attn = R_Attention(dim, num_heads, bias)
        self.norm2 = R_LayerNorm(dim, LayerNorm_type)
        self.ffn = R_FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class R_OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class R_Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class R_Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias', dual_pixel_task=False):
        super().__init__()
        self.patch_embed = R_OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[R_TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])
        self.down1_2 = R_Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[R_TransformerBlock(int(dim*2), heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])])
        self.down2_3 = R_Downsample(int(dim*2))
        self.encoder_level3 = nn.Sequential(*[R_TransformerBlock(int(dim*4), heads[2], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[2])])
        self.down3_4 = R_Downsample(int(dim*4))
        self.latent = nn.Sequential(*[R_TransformerBlock(int(dim*8), heads[3], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[3])])
        self.up4_3 = R_Upsample(int(dim*8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*8), int(dim*4), 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[R_TransformerBlock(int(dim*4), heads[2], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[2])])
        self.up3_2 = R_Upsample(int(dim*4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*4), int(dim*2), 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[R_TransformerBlock(int(dim*2), heads[1], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[1])])
        self.up2_1 = R_Upsample(int(dim*2))
        self.decoder_level1 = nn.Sequential(*[R_TransformerBlock(int(dim*2), heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[R_TransformerBlock(int(dim*2), heads[0], ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2), 1, bias=bias)
        self.output = nn.Conv2d(int(dim*2), out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        inp_dec_level3 = self.reduce_chan_level3(torch.cat([self.up4_3(latent), out_enc_level3], 1))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        inp_dec_level2 = self.reduce_chan_level2(torch.cat([self.up3_2(out_dec_level3), out_enc_level2], 1))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = torch.cat([self.up2_1(out_dec_level2), out_enc_level1], 1)
        out_dec_level1 = self.refinement(self.decoder_level1(inp_dec_level1))
        if self.dual_pixel_task:
            return self.output(out_dec_level1 + self.skip_conv(inp_enc_level1))
        return self.output(out_dec_level1) + inp_img


# =============================================================================
#  SECTION 3 — XFormer Architecture
#  Original: https://github.com/gladzhang/Xformer
# =============================================================================

def drop_path_func(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()

class X_DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_func(x, self.drop_prob, self.training)

class X_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

def x_window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)

def x_window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)

class X_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = self.softmax(attn.view(-1, self.num_heads, n, n))
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))

class X_SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = X_WindowAttention(dim, to_2tuple(self.window_size), num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = X_DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = X_Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        attn_mask = self._calculate_mask(self.input_resolution) if self.shift_size > 0 else None
        self.register_buffer('attn_mask', attn_mask)

    def _calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = x_window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = to_3d(x)
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        size_par = self.window_size
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hd, Wd, _ = x.shape
        x_size = (Hd, Wd)
        if min(x_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(x_size)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_windows = x_window_partition(x, self.window_size).view(-1, self.window_size ** 2, c)
        mask = self.attn_mask if self.input_resolution == x_size else self._calculate_mask(x_size).to(x.device)
        attn_windows = self.attn(x_windows, mask=mask).view(-1, self.window_size, self.window_size, c)
        shifted_x = x_window_reverse(attn_windows, self.window_size, Hd, Wd)
        if self.shift_size > 0:
            shifted_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_r > 0 or pad_b > 0:
            shifted_x = shifted_x[:, :h, :w, :].contiguous()
        x = shortcut + self.drop_path(shifted_x.view(b, h * w, c))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return to_4d(x, h, w)

class X_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class X_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = rearrange(attn.softmax(dim=-1) @ v, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class X_ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = R_LayerNorm(dim, LayerNorm_type)   # reuse R_LayerNorm (identical)
        self.attn = X_Attention(dim, num_heads, bias)
        self.norm2 = R_LayerNorm(dim, LayerNorm_type)
        self.ffn = X_FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class X_OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class X_Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class X_Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class XFormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, img_size=128, dim=48,
                 num_blocks=[2, 4, 4], spatial_num_blocks=[2, 4, 4, 6],
                 num_refinement_blocks=4, heads=[1, 2, 4, 8],
                 window_size=[16, 16, 16, 16], drop_path_rate=0.1,
                 ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', dual_pixel_task=False):
        super().__init__()
        self.alpha = self.beta = 1
        self.Convs = nn.ModuleList([
            nn.Conv2d(dim*2, dim*2, 3, 1, 1),
            nn.Conv2d(dim*4, dim*4, 3, 1, 1),
            nn.Conv2d(dim*2, dim*2, 3, 1, 1),
            nn.Conv2d(dim,   dim,   3, 1, 1),
        ])
        self.DWconvs = nn.ModuleList([
            nn.Conv2d(dim*2, dim*2, 3, 1, 1, groups=dim*2),
            nn.Conv2d(dim*4, dim*4, 3, 1, 1, groups=dim*4),
            nn.Conv2d(dim*2, dim*2, 3, 1, 1, groups=dim*2),
            nn.Conv2d(dim,   dim,   3, 1, 1, groups=dim),
        ])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(spatial_num_blocks))]

        def _stb(d, res, nh, ws, snb_idx, n):
            return nn.Sequential(*[
                X_SpatialTransformerBlock(
                    dim=d, input_resolution=(res, res), num_heads=nh,
                    window_size=ws, shift_size=0 if i % 2 == 0 else ws // 2,
                    mlp_ratio=ffn_expansion_factor,
                    drop_path=dpr[sum(spatial_num_blocks[:snb_idx]):sum(spatial_num_blocks[:snb_idx+1])][i])
                for i in range(n)])

        def _ctb(d, nh, n):
            return nn.Sequential(*[X_ChannelTransformerBlock(d, nh, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(n)])

        self.patch_embed = X_OverlapPatchEmbed(inp_channels, dim)
        # channel branch
        self.encoder_level1 = _ctb(dim,    heads[0], num_blocks[0])
        self.down1_2         = X_Downsample(dim)
        self.encoder_level2 = _ctb(dim*2,  heads[1], num_blocks[1])
        self.down2_3         = X_Downsample(dim*2)
        self.encoder_level3 = _ctb(dim*4,  heads[2], num_blocks[2])
        self.down3_4         = X_Downsample(dim*4)
        self.up4_3           = X_Upsample(dim*8)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, 1, bias=bias)
        self.decoder_level3 = _ctb(dim*4,  heads[2], num_blocks[2])
        self.up3_2           = X_Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.decoder_level2 = _ctb(dim*2,  heads[1], num_blocks[1])
        self.up2_1           = X_Upsample(dim*2)
        self.reduce_chan_level1 = nn.Conv2d(dim*2, dim, 1, bias=bias)
        self.decoder_level1 = _ctb(dim,    heads[0], num_blocks[0])
        # spatial branch
        self.encoder1  = _stb(dim,    img_size,    heads[0], window_size[0], 0, spatial_num_blocks[0])
        self.d1_2      = X_Downsample(dim)
        self.encoder2  = _stb(dim*2,  img_size//2, heads[1], window_size[1], 1, spatial_num_blocks[1])
        self.d2_3      = X_Downsample(dim*2)
        self.encoder3  = _stb(dim*4,  img_size//4, heads[2], window_size[2], 2, spatial_num_blocks[2])
        self.d3_4      = X_Downsample(dim*4)
        self.s_latent  = _stb(dim*8,  img_size//8, heads[3], window_size[3], 3, spatial_num_blocks[3])
        self.u4_3      = X_Upsample(dim*8)
        self.reduce3   = nn.Conv2d(dim*8, dim*4, 1, bias=bias)
        self.decoder3  = _stb(dim*4,  img_size//4, heads[2], window_size[2], 2, spatial_num_blocks[2])
        self.u3_2      = X_Upsample(dim*4)
        self.reduce2   = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.decoder2  = _stb(dim*2,  img_size//2, heads[1], window_size[1], 1, spatial_num_blocks[1])
        self.u2_1      = X_Upsample(dim*2)
        self.reduce1   = nn.Conv2d(dim*2, dim, 1, bias=bias)
        self.decoder1  = _stb(dim,    img_size,    heads[0], window_size[0], 0, spatial_num_blocks[0])
        # refinement + output
        self.refinement = _ctb(dim*2, heads[0], num_refinement_blocks)
        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim*2, 1, bias=bias)
        self.output = nn.Conv2d(dim*2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        inp = self.patch_embed(inp_img)
        # encoder
        e1c = self.encoder_level1(inp);   e1s = self.encoder1(inp)
        i2c = self.down1_2(e1c);          i2s = self.d1_2(e1s)
        sc = i2c; i2c = i2c + self.alpha*self.DWconvs[0](i2s); i2s = i2s + self.beta*self.Convs[0](sc)
        e2c = self.encoder_level2(i2c);   e2s = self.encoder2(i2s)
        i3c = self.down2_3(e2c);          i3s = self.d2_3(e2s)
        sc = i3c; i3c = i3c + self.alpha*self.DWconvs[1](i3s); i3s = i3s + self.beta*self.Convs[1](sc)
        e3c = self.encoder_level3(i3c);   e3s = self.encoder3(i3s)
        i4c = self.down3_4(e3c);          i4s = self.d3_4(e3s)
        lc = self.up4_3.__class__   # just reuse s_latent for both (as original code does)
        lc = self.s_latent(i4c);          ls = self.s_latent(i4s)
        # decoder
        d3c = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(lc), e3c], 1)))
        d3s = self.decoder3(self.reduce3(torch.cat([self.u4_3(ls), e3s], 1)))
        d2c_in = self.reduce_chan_level2(torch.cat([self.up3_2(d3c), e2c], 1))
        d2s_in = self.reduce2(torch.cat([self.u3_2(d3s), e2s], 1))
        sc = d2c_in; d2c_in = d2c_in + self.alpha*self.DWconvs[2](d2s_in); d2s_in = d2s_in + self.beta*self.Convs[2](sc)
        d2c = self.decoder_level2(d2c_in); d2s = self.decoder2(d2s_in)
        d1c_in = self.reduce_chan_level1(torch.cat([self.up2_1(d2c), e1c], 1))
        d1s_in = self.reduce1(torch.cat([self.u2_1(d2s), e1s], 1))
        sc = d1c_in; d1c_in = d1c_in + self.alpha*self.DWconvs[3](d1s_in); d1s_in = d1s_in + self.beta*self.Convs[3](sc)
        d1c = self.decoder_level1(d1c_in); d1s = self.decoder1(d1s_in)
        res = self.refinement(torch.cat([d1c, d1s], 1))
        if self.dual_pixel_task:
            return self.output(res + self.skip_conv(inp))
        return self.output(res) + inp_img


# =============================================================================
#  SECTION 4 — Ensemble Wrapper  (THIS IS THE CLASS test_demo.py imports)
# =============================================================================

def _tile_inference(model, img_tensor, tile_size=128, overlap=32):
    b, c, h, w = img_tensor.size()
    stride = tile_size - overlap
    output = torch.zeros_like(img_tensor)
    weight = torch.zeros_like(img_tensor)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y + tile_size, h);  x1 = min(x + tile_size, w)
            y0 = max(y1 - tile_size, 0); x0 = max(x1 - tile_size, 0)
            patch = img_tensor[:, :, y0:y1, x0:x1]
            with torch.cuda.amp.autocast():
                out = model(patch)
                out = out[0] if isinstance(out, (list, tuple)) else out
            output[:, :, y0:y1, x0:x1] += out
            weight[:, :, y0:y1, x0:x1] += 1
    return output / weight


def _tta(model, img_tensor, tile_size=128, overlap=32):
    def _t(x, m):
        if m == 0: return x
        if m == 1: return torch.flip(x, [3])
        if m == 2: return torch.flip(x, [2])
        if m == 3: return torch.flip(x, [2, 3])
        if m == 4: return x.transpose(2, 3)
        if m == 5: return torch.flip(x.transpose(2, 3), [3])
        if m == 6: return torch.flip(x.transpose(2, 3), [2])
        if m == 7: return torch.flip(x.transpose(2, 3), [2, 3])

    def _inv(x, m):
        if m == 0: return x
        if m == 1: return torch.flip(x, [3])
        if m == 2: return torch.flip(x, [2])
        if m == 3: return torch.flip(x, [2, 3])
        if m == 4: return x.transpose(2, 3)
        if m == 5: return torch.flip(x, [3]).transpose(2, 3)
        if m == 6: return torch.flip(x, [2]).transpose(2, 3)
        if m == 7: return torch.flip(x, [2, 3]).transpose(2, 3)

    return torch.stack([_inv(_tile_inference(model, _t(img_tensor, m), tile_size, overlap), m)
                        for m in range(8)]).mean(0)


class EnsembleDenoiser(nn.Module):
    """
    Weighted ensemble of SCUNet + XFormer + Restormer for sigma=50 denoising.
    Loads three separate checkpoint files passed at construction time.

    data_range : set to 1.0  (test_demo.py normalises to [0,1])
    """
    data_range = 1.0

    def __init__(self,
                 scunet_ckpt:   str = "model_zoo/00_EnsembleDenoiser_scunet.pth",
                 xformer_ckpt:  str = "model_zoo/00_EnsembleDenoiser_xformer.pth",
                 restormer_ckpt:str = "model_zoo/00_EnsembleDenoiser_restormer.pth",
                 w_scunet:   float = 0.3,
                 w_xformer:  float = 0.5,
                 w_restormer:float = 0.2,
                 tile_size: int = 128,
                 overlap:   int = 32,
                 use_tta:   bool = True):
        super().__init__()

        self.w_s = w_scunet
        self.w_x = w_xformer
        self.w_r = w_restormer
        self.tile_size = tile_size
        self.overlap   = overlap
        self.use_tta   = use_tta

        # ── SCUNet ──────────────────────────────────────────────────────────
        self.scunet = SCUNet(in_nc=3, config=[4,4,4,4,4,4,4], dim=64)
        ckpt = torch.load(scunet_ckpt, map_location='cpu')
        self.scunet.load_state_dict(ckpt)

        # ── XFormer ─────────────────────────────────────────────────────────
        self.xformer = XFormer(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[2,4,4], spatial_num_blocks=[2,4,4,6],
            num_refinement_blocks=4, heads=[1,2,4,8],
            window_size=[16,16,16,16], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
        ckpt = torch.load(xformer_ckpt, map_location='cpu')
        sd  = ckpt.get('params') or ckpt.get('state_dict') or ckpt
        sd  = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        self.xformer.load_state_dict(sd, strict=True)

        # ── Restormer ───────────────────────────────────────────────────────
        self.restormer = Restormer(
            inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4,6,6,8], num_refinement_blocks=4,
            heads=[1,2,4,8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type='BiasFree', dual_pixel_task=False)
        ckpt = torch.load(restormer_ckpt, map_location='cpu')
        self.restormer.load_state_dict(ckpt['params'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        infer = _tta if self.use_tta else _tile_inference
        out_s = infer(self.scunet,    x, self.tile_size, self.overlap)
        out_x = infer(self.xformer,   x, self.tile_size, self.overlap)
        out_r = infer(self.restormer, x, self.tile_size, self.overlap)
        return self.w_s * out_s + self.w_x * out_x + self.w_r * out_r