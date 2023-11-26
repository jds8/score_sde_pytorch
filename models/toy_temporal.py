# provenance: https://github.com/jannerm/diffuser/blob/main/diffuser/models/diffusion.py

import torch
import torch.nn as nn

from . import utils, layers, normalization


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
