# Leo Feng et al. https://arxiv.org/abs/2410.01201

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module, ModuleList, RMSNorm

from x_mlps_pytorch import Feedforwards

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(
    log_coeffs,
    log_values
):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 1.,
        proj_out = None,
        prenorm = True
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.norm = RMSNorm(dim) if prenorm else Identity()
        self.to_memories_and_gate = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim) if proj_out else Identity()

    def forward(
        self,
        x,
        memories = None,
        return_memories = False
    ):
        x = self.norm(x)
        seq_len = x.shape[1]
        hidden, gate = self.to_memories_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(memories, hidden, gate) if exists(memories) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(memories):
                log_values = torch.cat((memories.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_memories = out[:, -1:]

        out = self.to_out(out)

        if not return_memories:
            return out

        return out, next_memories

class minGRUBlocks(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 1.,
        prenorm = True,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                minGRU(
                    dim,
                    expansion_factor = expansion_factor,
                    prenorm = prenorm
                ),
                Feedforwards(
                    dim,
                    depth = 1,
                    expansion_factor = ff_mult
                )
            ]))

    def forward(
        self,
        x,
        memories = None,
        return_memories = False
    ):
        next_memories = []
        memories = default(memories, [None] * len(self.layers))

        for (gru, ff), memory in zip(self.layers, memories):
            gru_out, next_memory = gru(x, memories = memory, return_memories = True)
            x = x + gru_out
            x = ff(x)

            next_memories.append(next_memory)

        if not return_memories:
            return x

        return x, next_memories
