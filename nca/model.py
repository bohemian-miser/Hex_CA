"""The hex neural cellular automaton (distill.pub "Growing NCA" style, on a hex grid).

State [B, 16, S, S]: ch0 = wall (input, re-imposed every step), ch1 = fill
(output, filled iff > 0.5), ch2..15 hidden. One step:

    x = concat(state, mask)                    # 17 channels
    h = relu(conv3x3_hexmasked(x) + b1)        # 17 -> H, corner taps k=0,8 always zero
    d = conv1x1(h) + b2                        # H -> 16, zero-initialised: step 0 is a no-op
    state = state + d * fire                   # fire = 1 everywhere when fire_rate == 1
    if clamp: state = clamp(state, lo, hi)
    state[ch0] = wall;  state *= mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hexgrid import KERNEL_MASK


class HexNCA(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 64, clamp=(-1.0, 1.0), fire_rate: float = 1.0):
        super().__init__()
        self.channels, self.hidden = channels, hidden
        self.clamp = tuple(clamp) if clamp is not None else None
        self.fire_rate = fire_rate
        kmask = torch.tensor(KERNEL_MASK).view(1, 1, 3, 3)
        self.register_buffer("kmask", kmask)

        w1 = torch.empty(hidden, channels + 1, 3, 3)
        nn.init.kaiming_uniform_(w1, a=5 ** 0.5)  # torch's default conv init
        # Corners start at zero too; they get zero gradient (masked in
        # step()), so Adam never moves them either.
        self.w1 = nn.Parameter(w1 * kmask)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(channels, hidden, 1, 1))  # zero-init: untrained model is identity
        self.b2 = nn.Parameter(torch.zeros(channels))

    def step(self, state, walls, mask):
        """One synchronous step. state [B,C,S,S]; walls, mask [B or 1,1,S,S] float."""
        x = torch.cat([state, mask.expand(state.shape[0], -1, -1, -1)], 1)
        h = F.relu(F.conv2d(x, self.w1 * self.kmask, self.b1, padding=1))
        d = F.conv2d(h, self.w2, self.b2)
        if self.fire_rate < 1.0:
            d = d * (torch.rand_like(d[:, :1]) < self.fire_rate).float()
        state = state + d
        if self.clamp is not None:
            state = state.clamp(self.clamp[0], self.clamp[1])
        state = torch.cat([walls.expand(state.shape[0], -1, -1, -1), state[:, 1:]], 1)
        return state * mask

    def forward(self, state, walls, mask, steps: int):
        for _ in range(steps):
            state = self.step(state, walls, mask)
        return state


def fresh_state(walls, channels: int = 16):
    """Fresh state: all zeros except ch0 = walls. walls [B,1,S,S] float."""
    B, _, S, _ = walls.shape
    state = torch.zeros(B, channels, S, S, dtype=walls.dtype)
    state[:, 0:1] = walls
    return state
