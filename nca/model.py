"""The hex neural cellular automaton (distill.pub "Growing NCA" style, on a hex grid).

State [B, C, S, S] (C = 16): ch0 = wall (input, re-imposed every step), ch1 =
fill (output, filled iff > 0.5), ch2.. hidden. The perception also sees
n_consts constant planes (hexgrid.consts): mask, theta1, theta2 (spec v3; a v1
model has only the mask), then src1, src1c, src2, src2c (spec v6). With floods
(spec v6) channels 2..7 are hand-written exact floods, frozen (install_floods). With perception "taps+pool" (spec v4, the default) it
also sees, for every state channel, its max and its min over the cell's on-board
taps among the 7 (self + 6 neighbours; no corners, no off-board cells). One step:

    x = concat(state, consts)                  # C + n_consts channels
    p = concat(hexmax(state), hexmin(state))   # 2C channels ("taps+pool" only)
    h = relu(conv3x3_hexmasked(x) + w1pool . p + b1)   # -> H, corner taps k=0,8 always zero
    d = conv1x1(h) + b2                        # H -> C, zero-initialised: step 0 is a no-op
    state = state + d * fire                   # fire = 1 everywhere when fire_rate == 1
    if clamp: state = clamp(state, lo, hi)
    state[ch0] = wall;  state *= mask          # mask = consts[0]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hexgrid import CONST_NAMES, KERNEL_MASK, consts as hex_consts

PERCEPTIONS = ("taps", "taps+pool")


def hex_max_negmin(state, mask, lo, hi):
    """[B,2C,S,S]: the max of each channel, then MINUS the min of each channel, over the on-board taps among
    a cell's 7 (self + 6 neighbours; the corner taps are not neighbours).

    Off-board taps (and the array's outside) read min(lo, -hi), which is neutral when the state lies in
    [lo, hi] (it is clamped) since self is on board; off-board cells themselves get junk (the step masks
    them). The 7 hex taps are the union of two 2x2 blocks, rows {-1,0} x cols {0,+1} and rows {0,+1} x
    cols {-1,0}, so one max_pool2d and one maximum do both pools (and min = -max(-x)): on the Pi this
    is several times cheaper than stacking the 7 taps.
    """
    S = state.shape[-1]
    fill = min(lo, -hi)
    x = torch.cat([state, -state], 1).masked_fill(mask == 0, fill)
    q = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=fill), 2, stride=1)  # q[a,b] = max of padded [a..a+1, b..b+1]
    return torch.maximum(q[:, :, :S, 1:], q[:, :, 1:, :S])


def hex_pool(state, mask, lo, hi):
    """(max, min) [B,C,S,S] of each channel over the on-board taps among a cell's 7; 0 on off-board cells."""
    C = state.shape[1]
    p = hex_max_negmin(state, mask, lo, hi)
    return p[:, :C] * mask, -p[:, C:] * mask


class HexNCA(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 64, clamp=(-1.0, 1.0), fire_rate: float = 1.0,
                 n_consts: int = 3, perception: str = "taps+pool", floods: bool = False):
        super().__init__()
        assert 1 <= n_consts <= len(CONST_NAMES) and perception in PERCEPTIONS
        self.channels, self.hidden, self.n_consts, self.perception = channels, hidden, n_consts, perception
        self.clamp = tuple(clamp) if clamp is not None else None
        self.fire_rate = fire_rate
        kmask = torch.tensor(KERNEL_MASK).view(1, 1, 3, 3)
        self.register_buffer("kmask", kmask)

        w1 = torch.empty(hidden, channels + n_consts, 3, 3)
        nn.init.kaiming_uniform_(w1, a=5 ** 0.5)  # torch's default conv init
        # Corners start at zero too; they get zero gradient (masked in
        # step()), so Adam never moves them either.
        self.w1 = nn.Parameter(w1 * kmask)
        if perception == "taps+pool":  # same scale as w1's init: U(+-1/sqrt(fan_in of the 3x3 conv))
            bound = 1.0 / ((channels + n_consts) * 9) ** 0.5
            self.w1pool = nn.Parameter(torch.empty(hidden, 2 * channels, 1, 1).uniform_(-bound, bound))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.zeros(channels, hidden, 1, 1))  # zero-init: untrained model is identity
        self.b2 = nn.Parameter(torch.zeros(channels))
        self.floods = floods
        if floods:
            install_floods(self)

    @property
    def const_names(self):
        return list(CONST_NAMES[:self.n_consts])

    def step(self, state, walls, consts):
        """One synchronous step. state [B,C,S,S]; walls [B or 1,1,S,S]; consts [B or 1,n_consts,S,S]
        (const_stack: the mask first)."""
        B = state.shape[0]
        mask = consts[:, :1]
        x = torch.cat([state, consts.expand(B, -1, -1, -1)], 1)
        pre = F.conv2d(x, self.w1 * self.kmask, self.b1, padding=1)
        if self.perception == "taps+pool":
            lo, hi = self.clamp if self.clamp is not None else (-1e9, 1e9)
            C = self.channels
            w = torch.cat([self.w1pool[:, :C], -self.w1pool[:, C:]], 1)  # the pool gives -min: fold the sign in
            pre = pre + F.conv2d(hex_max_negmin(state, mask, lo, hi), w)
        h = F.relu(pre)
        d = F.conv2d(h, self.w2, self.b2)
        if self.fire_rate < 1.0:
            d = d * (torch.rand_like(d[:, :1]) < self.fire_rate).float()
        state = state + d
        if self.clamp is not None:
            state = state.clamp(self.clamp[0], self.clamp[1])
        state = torch.cat([walls.expand(B, -1, -1, -1), state[:, 1:]], 1)
        return state * mask

    @torch.no_grad()
    def restore_floods(self):
        """Write the frozen hand-written values back (a no-op unless something other than the gradient moved
        them: weight decay, say). Call it after every optimiser step."""
        for name, m in getattr(self, "frozen", {}).items():
            getattr(self, name)[m] = self.frozen_values[name][m]

    def forward(self, state, walls, consts, steps: int):
        for _ in range(steps):
            state = self.step(state, walls, consts)
        return state


# --------------------------------------------------------------------------
# Spec v6: the floods written by hand into the first FLOOD_UNITS hidden units and the output rows of
# channels 2..7, then frozen. Per cell, a_j = hex max of state channel j over the on-board taps (the pooled
# perception), wall = ch0 (centre tap), K = FLOOD_K (above any |value|: values live in [-1, 1]):
#
#   units 5f .. 5f+4, f = 0..3: channel j = 2 + f, source s = const src1, src1c, src2, src2c (index 3 + f)
#       5f+0  relu(a_j)                    (= a_j: a_j >= 0)
#       5f+1  relu(s - a_j)
#       5f+2  relu(a_j + K*wall - K)       (= a_j on a wall, 0 elsewhere)
#       5f+3  relu(s - a_j + K*wall - K)   (= relu(s - a_j) on a wall, 0 elsewhere)
#       5f+4  relu(ch_j)                   (= ch_j: ch_j >= 0)
#     delta_j = u0 + u1 - u2 - u3 - u4, so new_j = (1 - wall) * max(a_j, s): a gated max-flood from the rim.
#   units 20..22: channel 7 = span, one step behind:  p = ch2 + ch3 - 1, q = ch4 + ch5 - 1 (both >= -1)
#       20 relu(ch2 + ch3 + 1) (= p + 2)    21 relu(ch2 + ch3 - ch4 - ch5) (= relu(p - q))    22 relu(ch7 + 2)
#     delta_7 = u20 - u21 - u22, so new_7 = p - relu(p - q) = min(p, q).
#   units 23..25: channel 6 = the largest span on the board, an ungated max-flood of ch7 (two steps behind)
#       23 relu(a_6) (= a_6)    24 relu(ch7 - a_6)    25 relu(ch6) (= ch6)
#     delta_6 = u23 + u24 - u25, so new_6 = max(a_6, ch7).
# Frozen: those units' rows of w1, w1pool, b1 (every input) and the whole output rows w2[2..7, :], b2[2..7]
# (so no free unit can write into channels 2..7). Free: units FLOOD_UNITS.. and the output rows of
# channels 1 and 8.., which may read the flood units too.
# --------------------------------------------------------------------------

FLOOD_UNITS = 26
FLOOD_K = 3.0


def install_floods(model: HexNCA):
    """Write spec v6's hand-written floods into `model` (needs the 7 consts, the pooled perception, 8+
    channels and more than FLOOD_UNITS hidden units) and freeze them: a gradient hook zeroes the gradient
    of every frozen entry (so grad normalisation and Adam's moments never see it), and model.restore_floods()
    writes the values back after a step (in case anything but the gradient moved them)."""
    C, H, n = model.channels, model.hidden, model.n_consts
    assert model.perception == "taps+pool" and list(CONST_NAMES[:n]) == list(CONST_NAMES) and C >= 8 \
        and H > FLOOD_UNITS, "install_floods needs the 7 consts, taps+pool, 8+ channels, > 26 hidden units"
    U, K = FLOOD_UNITS, FLOOD_K
    w1 = torch.zeros(U, C + n)    # centre-tap weights; column c < C is state channel c, then the consts
    pool = torch.zeros(U, 2 * C)  # column j < C: the hex max of channel j (the mins stay unused)
    b1 = torch.zeros(U)
    w2 = torch.zeros(C, H)        # only rows 2..7 are used
    WALL = 0
    for f in range(4):
        j, s, u = 2 + f, C + 3 + f, 5 * f
        pool[u + 0, j] = 1
        w1[u + 1, s], pool[u + 1, j] = 1, -1
        pool[u + 2, j], w1[u + 2, WALL], b1[u + 2] = 1, K, -K
        w1[u + 3, s], pool[u + 3, j], w1[u + 3, WALL], b1[u + 3] = 1, -1, K, -K
        w1[u + 4, j] = 1
        w2[j, u:u + 5] = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])
    w1[20, 2], w1[20, 3], b1[20] = 1, 1, 1
    w1[21, 2], w1[21, 3], w1[21, 4], w1[21, 5] = 1, 1, -1, -1
    w1[22, 7], b1[22] = 1, 2
    w2[7, 20:23] = torch.tensor([1.0, -1.0, -1.0])
    pool[23, 6] = 1
    w1[24, 7], pool[24, 6] = 1, -1
    w1[25, 6] = 1
    w2[6, 23:26] = torch.tensor([1.0, 1.0, -1.0])

    frozen = {name: torch.zeros_like(getattr(model, name), dtype=torch.bool) for name in ("w1", "w1pool", "b1", "w2", "b2")}
    with torch.no_grad():
        model.w1[:U] = 0
        model.w1[:U, :, 1, 1] = w1
        model.w1pool[:U] = pool.view(U, 2 * C, 1, 1)
        model.b1[:U] = b1
        model.w2[2:8] = w2[2:8].view(6, H, 1, 1)
        model.b2[2:8] = 0
    for name in ("w1", "w1pool", "b1"):
        frozen[name][:U] = True
    for name in ("w2", "b2"):
        frozen[name][2:8] = True
    model.frozen = frozen
    model.frozen_values = {name: getattr(model, name).detach().clone() for name in frozen}
    for name, m in frozen.items():
        getattr(model, name).register_hook(lambda g, keep=(~m).to(torch.float32): g * keep.to(g.device))


def const_stack(R: int, n_consts: int = 3):
    """float32 tensor [1, n_consts, S, S]: the first n_consts of hexgrid.consts (mask, theta1, theta2, src1,
    src1c, src2, src2c) for radius R. Build it once per radius."""
    return torch.from_numpy(np.ascontiguousarray(hex_consts(R)[:n_consts]))[None]


def load_expanded(model: HexNCA, sd) -> int:
    """Load state dict `sd` into `model`, which may take more consts than `sd` was trained with, and may
    pool where `sd` doesn't: the old w1 columns are copied, the new const columns and w1pool are zero, so
    the model starts exactly at the old behaviour (the new inputs have no effect until training moves
    them). Returns the old n_consts."""
    sd = dict(sd)
    if model.perception == "taps+pool" and "w1pool" not in sd:
        sd["w1pool"] = torch.zeros_like(model.w1pool)
    elif model.perception == "taps" and "w1pool" in sd:
        raise ValueError("can't load a taps+pool model into a taps one")
    w1 = sd["w1"]
    old = w1.shape[1] - model.channels
    if old > model.n_consts or w1.shape[0] != model.hidden:
        raise ValueError(f"can't load w1 {tuple(w1.shape)} into a model with channels={model.channels}, "
                         f"hidden={model.hidden}, n_consts={model.n_consts}")
    if old < model.n_consts:
        new = torch.zeros_like(model.w1)
        new[:, :w1.shape[1]] = w1
        sd["w1"] = new
    model.load_state_dict(sd)
    return old


def fresh_state(walls, channels: int = 16):
    """Fresh state: all zeros except ch0 = walls. walls [B,1,S,S] float; the state is on walls' device."""
    B, _, S, _ = walls.shape
    state = torch.zeros(B, channels, S, S, dtype=walls.dtype, device=walls.device)
    state[:, 0:1] = walls
    return state
