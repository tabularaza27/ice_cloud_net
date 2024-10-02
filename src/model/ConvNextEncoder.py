import math
import torch
import torch.nn as nn
from torchvision.ops import stochastic_depth
from inspect import isfunction
from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from model.Embedding import MetadataEmbedding

class SinusoidalPosEmb2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, xy):
        device = xy.device
        x = xy[...,0]
        y = xy[...,1]
        quarter_dim = self.dim // 4
        emb = math.log(10000) / (quarter_dim - 1)
        emb = torch.exp(torch.arange(quarter_dim, device=device) * -emb)
        xemb = x[:, None] * emb[None, :]
        yemb = y[:, None] * emb[None, :]
        emb = torch.cat((xemb.sin(), xemb.cos(), yemb.sin(), yemb.cos()), dim=-1)
        return emb
    
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    """
        Based on transformer-like embedding from 'Attention is all you need'
        Note: 10,000 corresponds to the maximum sequence length
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
# building block modules
class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)
        ) if exists(emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # add stochastic depth - is turned on by lightning model in callback during training
        self.stochastic_depth = False

    def forward(self, x, emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(emb), 'time emb must be passed in'
            condition = self.mlp(emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)

        if self.stochastic_depth:
            x = stochastic_depth(x,0.25,mode="batch")

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# Main Model
class UnetConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_pos_emb = True,
        output_mean_scale = False,
        residual = False
    ):
        super().__init__()
        self.channels = channels
        self.residual = residual
        print("Is Time embed used ? ", with_pos_emb)
        self.output_mean_scale = output_mean_scale

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_pos_emb:
            pos_dim = dim
            self.pos_mlp = nn.Sequential(
                SinusoidalPosEmb2d(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            pos_dim = None
            self.pos_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, emb_dim = pos_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim = pos_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim = pos_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim = pos_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, emb_dim = pos_dim),
                ConvNextBlock(dim_in, dim_in, emb_dim = pos_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1),
            #nn.Tanh() # ADDED
        )

    def forward(self, x, time=None):
        orig_x = x
        t = None
        if time is not None and exists(self.pos_mlp):
            t = self.pos_mlp(time)
        
        original_mean = torch.mean(x, [1, 2, 3], keepdim=True)
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        print(x.shape)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.residual:
            return self.final_conv(x) + orig_x

        out = self.final_conv(x)
        if self.output_mean_scale:
            out_mean = torch.mean(out, [1,2,3], keepdim=True)
            out = out - original_mean + out_mean

        return out
    

    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class UnetConvNextBlock2(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int = 3,
        out_channels: int = 256,
        dim_mults: Tuple =(1, 2, 4, 8),
        extended_final_conv: bool =True,
        residual: bool = False,
        prediction_heads = 1, 
        final_relu: bool = True,
        final_rff: Optional[Dict] = None,
        meta_data_embedding: bool = False
    ):
        """
        code from here: https://github.com/arpitbansal297/Cold-Diffusion-Models/blob/main/snowification/diffusion/model/unet_convnext.py
        Simpler version - no positional embedding


        Args:
            final_rff: {sigma: float, encoded_size: int}
        """
        
        super().__init__()

        if meta_data_embedding:
            self.embedder = MetadataEmbedding()
            self.emb_dim = self.embedder.total_length
        else:
            self.embedder = None
            self.emb_dim = None

        self.in_channels = in_channels # seviri channels
        self.out_channels = out_channels # height levels
        self.prediction_heads = prediction_heads # target variables

        dims = [self.in_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, emb_dim=self.emb_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim=self.emb_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim=self.emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim=self.emb_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, emb_dim=self.emb_dim),
                ConvNextBlock(dim_in, dim_in, emb_dim=self.emb_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

      
        # one prediction head per variables
        if prediction_heads > 1:
            self.final_conv = nn.ModuleList([])
            for prediction_head in range(prediction_heads):
                self.final_conv.append(nn.Sequential(
                                        ConvNextBlock(dim, dim),
                                        nn.Conv2d(dim, self.out_channels, 1),
                                        nn.ReLU()))

        else:
           self.final_conv = nn.Sequential(
                ConvNextBlock(dim, dim),
                nn.Conv2d(dim, self.out_channels, 1),
                nn.ReLU()
            )

    def forward(self, x, meta_data=None):
        orig_x = x
        meta_data_embedding = None

        if meta_data is not None and exists(self.embedder):
            meta_data_embedding = self.embedder(meta_data)

        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, meta_data_embedding)
            x = convnext2(x, meta_data_embedding)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, meta_data_embedding)
        x = self.mid_attn(x)
        x = self.mid_block2(x, meta_data_embedding)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, meta_data_embedding)
            x = convnext2(x, meta_data_embedding)
            x = attn(x)
            x = upsample(x)

        # prediction heads
        if self.prediction_heads == 1:
            out = self.final_conv(x)
        else:
            # multiple prediction heads
            out = [self.final_conv[i](x).unsqueeze(1) for i in range(self.prediction_heads)]
            out = torch.concat(out,dim=1)

        return out

class UnetConvNextBlockERA(nn.Module):
    def __init__(
        self,
        dim,
        in_channels = 3,
        out_channels = None,
        dim_mults=(1, 2, 4, 8),
        with_era5_emb = False,
    ):        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        print("Is era5 embed used ? ", with_era5_emb)


        dims = [self.in_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_era5_emb:
            era5_dim = 256 # todo include in args
            era5_embed_dim = int(era5_dim/8)
            era5_channels = 1 # todo include in args
            # self.era5_encoder = nn.Sequential(
            #     nn.Conv1d(era5_channels,era5_dim*4,1,stride=1,padding=0),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(era5_dim*4,era5_dim,1,stride=1,padding=0),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(era5_dim,int(era5_dim/2),1,stride=1,padding=0),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(int(era5_dim/2),int(era5_dim/4),1,stride=1,padding=0),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(int(era5_dim/4),1,1,stride=1,padding=0)
            # )
            self.era5_encoder = nn.Sequential(
                nn.Linear(era5_dim, era5_dim * 2),
                nn.GELU(),
                nn.Linear(era5_dim * 2, era5_dim * 4),
                nn.GELU(),
                nn.Linear(era5_dim * 4, era5_dim * 2),
                nn.GELU(),
                nn.Linear(era5_dim * 2, int(era5_dim/2)),
                nn.GELU(),
                nn.Linear(int(era5_dim/2), int(era5_dim/4)),
                nn.GELU(),
                nn.Linear(int(era5_dim/4), int(era5_embed_dim))
            )
        else:
            era5_dim = None
            era5_embed_dim = None
            self.era5_encoder = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, emb_dim = era5_embed_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim = era5_embed_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim = era5_embed_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim = era5_embed_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, emb_dim = era5_embed_dim),
                ConvNextBlock(dim_in, dim_in, emb_dim = era5_embed_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, self.out_channels, 1),
            #nn.Tanh() # ADDED
        )

    def forward(self, x, era5=None):
        era5_embed = None
        if era5 is not None and exists(self.era5_encoder):
            era5_embed = self.era5_encoder(era5).squeeze()
        
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, era5_embed)
            x = convnext2(x, era5_embed)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, era5_embed)
        x = self.mid_attn(x)
        x = self.mid_block2(x, era5_embed)

        for convnext, convnext2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, era5_embed)
            x = convnext2(x, era5_embed)
            x = attn(x)
            x = upsample(x)
    
        out = self.final_conv(x)

        return out



class EncoderConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        with_pos_emb = True
    ):
        super().__init__()
        self.channels = channels
        print("Positional embedding: ", with_pos_emb)

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_pos_emb:
            pos_dim = dim
            self.pos_emb = SinusoidalPosEmb2d(dim)
            self.pos_mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            pos_dim = None
            self.pos_mlp = None

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, emb_dim = pos_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, emb_dim = pos_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # analog to middle part of Unet where dimensions dont change
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, emb_dim = pos_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, emb_dim = pos_dim)

    def forward(self, x, pos=None):
        t = None
        if pos is not None and exists(self.pos_mlp):
            t = self.pos_mlp(self.pos_emb(pos))
        
        h = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        return x