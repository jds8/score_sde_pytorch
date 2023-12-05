# provenance: https://github.com/jannerm/diffuser/blob/main/diffuser/models/diffusion.py

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from . import utils, layers, normalization

from .toy_helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    Attention,
)


class DiffusionBlock(nn.Module):
    def __init__(self, nunits):
        super(DiffusionBlock, self).__init__()
        self.linear = nn.Linear(nunits, nunits)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = nn.functional.relu(x)
        return x


@utils.register_model(name='toy_model')
class DiffusionModel(nn.Module):
    def __init__(self, config):
        super(DiffusionModel, self).__init__()

        nfeatures = config.model.nf
        nblocks= 2
        nunits = 64

        self.inblock = nn.Linear(nfeatures+1, nunits)
        self.midblocks = nn.ModuleList([DiffusionBlock(nunits) for _ in range(nblocks)])
        self.outblock = nn.Linear(nunits, nfeatures)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond=None) -> torch.Tensor:
        val = torch.hstack([x, time])
        val = self.inblock(val)
        for midblock in self.midblocks:
            val = midblock(val)
        val = self.outblock(val)
        return val


class ResidualBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x traj_length ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x traj_length ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.bool_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cemb, bool_emb):
        '''
            x : [ batch_size x inp_channels x traj_length ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x traj_length ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t) + self.cond_mlp(cemb) + self.bool_mlp(bool_emb)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


@utils.register_model(name='nnet')
class TemporalNNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        d_model = config.model.d_model
        cond_dim = config.model.cond_dim
        dim = 32
        dim_mults = (1, 2, 4, 8)
        attention = False
        device = None

        dims = [d_model, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.d_model = d_model

        time_dim = dim
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.bool_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Upsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Downsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, d_model, 1),
        )

    def forward(self, x, time, cond=None):
        try:
            b_dim, h_dim, t_dim = x.shape
        except:
            import pdb; pdb.set_trace()
            b_dim, h_dim, t_dim = x.shape
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        cond = cond if cond is not None else torch.ones(t.shape[0], 1, device=x.device) * -1
        cemb = self.cond_mlp(cond).reshape(cond.shape[0], -1)

        use_cond = (cond > 0.).to(torch.float).reshape(cond.shape)
        bool_emb = self.bool_mlp(use_cond).reshape(cemb.shape)
        h = []

        for idx, (resnet, resnet2, attn, upsample) in enumerate(self.ups):
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            h.append(x)
            x = upsample(x)

        x = self.mid_block1(x, t, cemb, bool_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cemb, bool_emb)

        for idx, (resnet, resnet2, attn, downsample) in enumerate(self.downs):
            hpop = h.pop()
            try:
                x = torch.cat((x, hpop), dim=1)
            except:
                import pdb; pdb.set_trace()
                x = torch.cat((x, hpop), dim=1)
            x = resnet(x, t, cemb, bool_emb)
            x = resnet2(x, t, cemb, bool_emb)
            x = attn(x)
            x = downsample(x)
        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x[:b_dim, :h_dim, :t_dim]
