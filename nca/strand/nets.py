"""The strand CA's update networks (nca/strand/train2.py).

StrandNCA: nca.model.HexNCA's step ("taps" perception) with a deeper per-cell MLP. One step:

    x = concat(state, consts)                         # C + n_in channels
    h = relu(conv3x3_hexmasked(x) + b1)               # the 7 hex taps (self + 6 neighbours) -> hidden
    h = relu(conv1x1(h)) x (depth - 1)                # --depth: hidden layers of the per-cell update
    d = conv1x1(h) + b2                               # -> C, zero-initialised: step 0 is a no-op
    state = clamp(state + d, lo, hi);  state[ch0] = walls;  state *= mask (consts[:, 0])

At depth 1 this is HexNCA's taps step exactly (same parameters, same order). Width scales with --channels
and --hidden; depth adds hidden x hidden 1x1 layers (default torch init) between the perception and the
zero-initialised output, so a deeper model also starts as the identity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hexgrid import KERNEL_MASK


class StrandNCA(nn.Module):
    def __init__(self, channels: int, hidden: int, depth: int, clamp, n_in: int):
        super().__init__()
        assert depth >= 1
        self.channels, self.hidden, self.depth, self.n_consts = channels, hidden, depth, n_in
        self.clamp = tuple(clamp) if clamp is not None else None
        self.register_buffer("kmask", torch.tensor(KERNEL_MASK).view(1, 1, 3, 3))
        w1 = torch.empty(hidden, channels + n_in, 3, 3)
        nn.init.kaiming_uniform_(w1, a=5 ** 0.5)  # torch's default conv init; corners masked
        self.w1 = nn.Parameter(w1 * self.kmask)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.mids = nn.ModuleList(nn.Conv2d(hidden, hidden, 1) for _ in range(depth - 1))
        self.w2 = nn.Parameter(torch.zeros(channels, hidden, 1, 1))  # zero-init: an untrained model is the identity
        self.b2 = nn.Parameter(torch.zeros(channels))

    def update(self, state, consts):
        """d [B,C,S,S]: the update before the residual."""
        B = state.shape[0]
        x = torch.cat([state, consts.expand(B, -1, -1, -1)], 1)
        h = F.relu(F.conv2d(x, self.w1 * self.kmask, self.b1, padding=1))
        for m in self.mids:
            h = F.relu(m(h))
        return F.conv2d(h, self.w2, self.b2)

    def step(self, state, walls, consts):
        """One synchronous step. state [B,C,S,S]; walls [B or 1,1,S,S]; consts [B or 1,n_in,S,S] (mask first)."""
        B = state.shape[0]
        state = state + self.update(state, consts)
        if self.clamp is not None:
            state = state.clamp(self.clamp[0], self.clamp[1])
        state = torch.cat([walls.expand(B, -1, -1, -1), state[:, 1:]], 1)
        return state * consts[:, :1]
