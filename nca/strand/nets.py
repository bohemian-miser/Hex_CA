"""The strand CA's update networks (nca/strand/train2.py): StrandNCA (inputs a, c, c-bc, d, d-cs) and FrameNCA
(option E, local-frame perception; at the end of this file).

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

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


# ---------------------------------------------------------------- option E: local-frame perception

DROW = (0, -1, -1, 0, 1, 1)  # walker.DIRS as (drow, dcol)
DCOL = (1, 1, 0, -1, -1, 0)
N_FRAMES = 12


def frame_sigma():
    """int [12, 6]: sigma[f, k] = the board direction of local direction k in frame f = rot * 2 + mb, i.e.
    (m k + rot) mod 6 with m = 1 - 2 mb (rules.py's geo % 12; strand-data.md's tile_rot convention)."""
    return np.array([[((1 - 2 * mb) * k + rot) % 6 for k in range(6)] for rot in range(6) for mb in (0, 1)])


class FrameNCA(nn.Module):
    """Option E (docs/spectacle-nca-options.md §3): every cell runs the same update in its OWN frame -- its 7
    taps ordered by its local edges and every directional channel read in local directions -- so the weights
    never see the board's frame (the RoPE / steerable-NCA idea; on hex a 60-degree turn is a cyclic shift of
    the six taps, a mirror reverses them).

    Channels are scalar or directional: a directional group is 6 channels, one per board direction (the edge
    planes ch1-6 and the closed planes ch7-12 are such groups, the hidden state has `dir_groups` more after
    them, and the tap's chord-edge planes are one among the consts). The state stays in the board frame; a
    cell in frame f sees board tap 1 + d as its local tap 1 + sigma_f^-1(d) and board channel g + d as local
    channel g + sigma_f^-1(d), and its update's directional outputs go back the same way. Rather than
    permuting the features per cell, each of the 12 frames gets the shared weights permuted to the board frame
    (W1_f, W2_f): the same FLOPs as the masked 3x3 conv, the cells grouped by frame.

    consts: [mask, the input planes ..., frame] -- the last plane is the cell's frame index (0..11), used only
    to pick W_f, never as an input feature. `dir_in` lists the first channel of every directional group of
    x = concat(state, consts without the frame).
    """

    def __init__(self, channels: int, hidden: int, depth: int, clamp, n_in: int, dir_in):
        super().__init__()
        assert depth >= 1
        self.channels, self.hidden, self.depth, self.n_consts = channels, hidden, depth, n_in
        self.clamp = tuple(clamp) if clamp is not None else None
        Cx = channels + n_in
        self.dir_in = sorted(int(g) for g in dir_in)
        sig = frame_sigma()
        inv = np.argsort(sig, 1)  # inv[f, d] = the local direction of board direction d
        tap_local = np.array([[0] + [1 + inv[f, d] for d in range(6)] for f in range(N_FRAMES)])
        chan = np.tile(np.arange(Cx), (N_FRAMES, 1))
        for g in self.dir_in:
            assert g + 6 <= Cx and (g + 6 <= channels or g >= channels)
            chan[:, g:g + 6] = g + inv
        self.register_buffer("tap_local", torch.from_numpy(tap_local).long())  # [12, 7]
        self.register_buffer("chan_x", torch.from_numpy(chan).long())         # [12, Cx]
        self.register_buffer("chan_out", torch.from_numpy(chan[:, :channels].copy()).long())  # [12, C]
        w = torch.empty(hidden, Cx, 3, 3)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)  # HexNCA's init, then its 7 taps in local order
        self.w1 = nn.Parameter(torch.stack([w[:, :, 1, 1]] + [w[:, :, 1 + DROW[k], 1 + DCOL[k]] for k in range(6)], 1))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.mids = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth - 1))
        self.w2 = nn.Parameter(torch.zeros(channels, hidden))  # zero-init: an untrained model is the identity
        self.b2 = nn.Parameter(torch.zeros(channels))

    def frame_weights(self):
        """(W1 [12, 7*Cx, hidden], W2 [12, C, hidden], b2 [12, C]): the shared weights in each frame's board order."""
        H, _, Cx = self.w1.shape
        W1 = self.w1[:, self.tap_local]  # [H, 12, 7, Cx]: board tap t -> local tap
        W1 = W1.gather(3, self.chan_x[None, :, None, :].expand(H, N_FRAMES, 7, Cx))  # board channel -> local
        return W1.permute(1, 2, 3, 0).reshape(N_FRAMES, 7 * Cx, H), self.w2[self.chan_out], self.b2[self.chan_out]

    def update(self, x, frame):
        B, Cx, S, _ = x.shape
        xp = F.pad(x, (1, 1, 1, 1))
        taps = [x] + [xp[:, :, 1 + DROW[d]:1 + DROW[d] + S, 1 + DCOL[d]:1 + DCOL[d] + S] for d in range(6)]
        rows = torch.stack(taps, 1).permute(0, 3, 4, 1, 2).reshape(B * S * S, 7 * Cx)
        W1, W2, b2 = self.frame_weights()
        fr = frame.reshape(-1)
        out = x.new_zeros(B * S * S, self.channels)
        for f in range(N_FRAMES):
            idx = (fr == f).nonzero(as_tuple=True)[0]
            if not len(idx):
                continue
            h = F.relu(rows[idx] @ W1[f] + self.b1)
            for m in self.mids:
                h = F.relu(m(h))
            out = out.index_copy(0, idx, h @ W2[f].T + b2[f])
        return out.view(B, S, S, self.channels).permute(0, 3, 1, 2)

    def step(self, state, walls, consts):
        B = state.shape[0]
        frame = consts[:, -1].round().long().expand(B, -1, -1)
        cin = consts[:, :-1]
        x = torch.cat([state, cin.expand(B, -1, -1, -1)], 1)
        if torch.is_grad_enabled():  # recompute the 7-tap rows in backward: they are 7x the state's size
            d = torch.utils.checkpoint.checkpoint(self.update, x, frame, use_reentrant=False)
        else:
            d = self.update(x, frame)
        state = state + d
        if self.clamp is not None:
            state = state.clamp(self.clamp[0], self.clamp[1])
        state = torch.cat([walls.expand(B, -1, -1, -1), state[:, 1:]], 1)
        return state * cin[:, :1]
