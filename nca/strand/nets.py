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

import weakref

import numpy as np
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


# ---------------------------------------------------------------- option E: local-frame perception

DROW = (0, -1, -1, 0, 1, 1)  # walker.DIRS as (drow, dcol)
DCOL = (1, 1, 0, -1, -1, 0)
N_FRAMES = 12


def frame_sigma():
    """int [12, 6]: sigma[f, k] = the board direction of local direction k in frame f = rot * 2 + mb, i.e.
    (m k + rot) mod 6 with m = 1 - 2 mb (rules.py's geo % 12; strand-data.md's tile_rot convention)."""
    return np.array([[((1 - 2 * mb) * k + rot) % 6 for k in range(6)] for rot in range(6) for mb in (0, 1)])


class _LocalTaps(torch.autograd.Function):
    """FrameNCA's first layer, pre [n, H] = rows @ W + b1, with rows [n, 7 Cx] = every cell's 7 taps in its own
    frame, picked out of the zero-padded board by the plan's per-cell indices.

    The padded board is laid out as rows: xs [B (S+2)^2, Cs] (the scalar channels, a row per padded cell) and
    xd [B (S+2)^2 6, G] (the directional groups, a row per padded cell and board direction). idx_s picks the
    cell's 7 scalar rows (local tap j = the neighbour across local edge j), idx_d its 7 x 6 directional rows
    (local direction k of tap j = board direction sigma_f(k)); the gathered rows are in (tap, channel) order, and
    W = w1's columns in that order (wcol; wcol_inv takes the gradient back). Backward keeps xs and xd (about the
    input's size) and gathers again, rather than keeping the gathered rows (7x the input, at every step of the
    backprop window): what a 3x3 conv does with its im2col. All in a few large ops: no per-frame loop."""

    @staticmethod
    def forward(ctx, x, w1, b1, sca, dir_k, wcol, wcol_inv, idx_s, idx_d):
        B, Cx, Hh, Ww = x.shape
        H, n, ns = w1.shape[0], B * Hh * Ww, 7 * sca.numel()
        xp = F.pad(x, (1, 1, 1, 1)).permute(0, 2, 3, 1)       # [B, S+2, S+2, Cx]
        xs = xp.index_select(3, sca).view(-1, sca.numel())    # scalar channels: a row per padded cell
        xd = xp.index_select(3, dir_k).view(-1, dir_k.numel() // 6)  # direction-major: 6 rows per padded cell
        W = w1.reshape(H, 7 * Cx).index_select(1, wcol).t()   # [7 Cx, H], the gathered rows' order
        pre = torch.addmm(b1, xs.index_select(0, idx_s).view(n, -1), W[:ns])
        pre.addmm_(xd.index_select(0, idx_d).view(n, -1), W[ns:])
        ctx.save_for_backward(xs, xd, W, sca, dir_k, wcol_inv, idx_s, idx_d)
        ctx.shape = x.shape
        return pre

    @staticmethod
    def backward(ctx, g):
        xs, xd, W, sca, dir_k, wcol_inv, idx_s, idx_d = ctx.saved_tensors
        B, Cx, Hh, Ww = ctx.shape
        n, ns = g.shape[0], 7 * sca.numel()
        need = ctx.needs_input_grad
        gx = gw1 = gb1 = None
        if need[1]:  # the gathered rows again, then dW and back to w1's layout
            gW = torch.cat([g.t().mm(xs.index_select(0, idx_s).view(n, -1)),
                            g.t().mm(xd.index_select(0, idx_d).view(n, -1))], 1)  # [H, 7 Cx]
            gw1 = gW.index_select(1, wcol_inv).view(-1, 7, Cx)
        if need[0]:  # each padded board row gets the sum over the rows gathered from it; then x's layout
            gs = torch.zeros_like(xs).index_add_(0, idx_s, g.mm(W[:ns].t()).view(-1, xs.shape[1]))
            gd = torch.zeros_like(xd).index_add_(0, idx_d, g.mm(W[ns:].t()).view(-1, xd.shape[1]))
            gp = g.new_empty(B, Hh + 2, Ww + 2, Cx)  # sca and dir_k cover every channel once
            gp.index_copy_(3, sca, gs.view(B, Hh + 2, Ww + 2, -1)).index_copy_(3, dir_k, gd.view(B, Hh + 2, Ww + 2, -1))
            gx = gp[:, 1:-1, 1:-1].permute(0, 3, 1, 2)
        if need[2]:
            gb1 = g.sum(0)
        return gx, gw1, gb1, None, None, None, None, None, None


class FrameNCA(nn.Module):
    """Option E (docs/spectacle-nca-options.md §3): every cell runs the same update in its OWN frame -- its 7
    taps ordered by its local edges and every directional channel read in local directions -- so the weights
    never see the board's frame (the RoPE / steerable-NCA idea; on hex a 60-degree turn is a cyclic shift of
    the six taps, a mirror reverses them).

    Channels are scalar or directional: a directional group is 6 channels, one per board direction (the edge
    planes ch1-6 and the closed planes ch7-12 are such groups, the hidden state has `dir_groups` more after
    them, and the tap's chord-edge planes are one among the consts). The state stays in the board frame. A cell
    in frame f reads its local tap 1 + k at the neighbour in board direction sigma_f(k), and local channel g + k
    of a directional group as board channel g + sigma_f(k); its update's directional outputs go back the same
    way (board channel g + sigma_f(k) <- local g + k).

    The permutation is done on the features, per cell, by indexing: every cell's 7 taps x (scalar channels, and
    the directional groups with their 6 directions) are picked out of the zero-padded board by a per-cell index
    built from its frame (cached per consts tensor, so once per rollout), straight into the [cells, 7 Cx] rows of
    the shared per-cell MLP (_LocalTaps). So a step is C's step -- one 7-tap matmul, the hidden layers, the
    output -- plus that gather and one for the outputs; no per-frame loop, no host syncs, no per-cell weights.
    The parameters (w1 [H, 7 local taps, Cx local channels], b1, mids, w2 [C local, H], b2) and the buffers are
    those of the first, per-frame-weights version (checkpoints load either way); nca/strand/selftest2.py keeps
    that version as the parity reference.

    consts: [mask, the input planes ..., frame] -- the last plane is the cell's frame index (0..11), used only
    to permute, never as an input feature. `dir_in` lists the first channel of every directional group of
    x = concat(state, consts without the frame).
    """

    def __init__(self, channels: int, hidden: int, depth: int, clamp, n_in: int, dir_in):
        super().__init__()
        assert depth >= 1
        self.channels, self.hidden, self.depth, self.n_consts = channels, hidden, depth, n_in
        self.clamp = tuple(clamp) if clamp is not None else None
        Cx = channels + n_in
        self.dir_in = sorted(int(g) for g in dir_in)
        assert self.dir_in, "FrameNCA needs at least one directional group"
        sig = frame_sigma()
        inv = np.argsort(sig, 1)  # inv[f, d] = the local direction of board direction d
        tap_local = np.array([[0] + [1 + inv[f, d] for d in range(6)] for f in range(N_FRAMES)])
        chan = np.tile(np.arange(Cx), (N_FRAMES, 1))
        is_dir = np.zeros(Cx, bool)
        for g in self.dir_in:
            assert g + 6 <= Cx and (g + 6 <= channels or g >= channels) and not is_dir[g:g + 6].any()
            chan[:, g:g + 6] = g + inv
            is_dir[g:g + 6] = True
        # board -> local, per frame: tap_local [12, 7], chan_x [12, Cx]; chan_out [12, C] gives every board output
        # channel the local output it takes
        self.register_buffer("tap_local", torch.from_numpy(tap_local).long())
        self.register_buffer("chan_x", torch.from_numpy(chan).long())
        self.register_buffer("chan_out", torch.from_numpy(chan[:, :channels].copy()).long())
        # local -> board, for the gather: tap_board[f, j] = the board tap local tap j reads, sigma[f, k] = the
        # board direction of local direction k
        self.register_buffer("sigma", torch.from_numpy(sig).long(), persistent=False)
        self.register_buffer("tap_board", torch.from_numpy(np.argsort(tap_local, 1)).long(), persistent=False)
        sca = np.flatnonzero(~is_dir)
        dir_k = np.array([g + k for k in range(6) for g in self.dir_in])  # direction-major: rows of 6 per cell
        self.register_buffer("sca", torch.from_numpy(sca).long(), persistent=False)
        self.register_buffer("dir_k", torch.from_numpy(dir_k).long(), persistent=False)
        # w1's columns (local tap j, local channel) in the order of the gathered rows: the scalar block (j, c),
        # then the directional block (j, k, group)
        wcol = [j * Cx + c for j in range(7) for c in sca] + \
               [j * Cx + g + k for j in range(7) for k in range(6) for g in self.dir_in]
        self.register_buffer("wcol", torch.tensor(wcol).long(), persistent=False)
        self.register_buffer("wcol_inv", torch.from_numpy(np.argsort(wcol)).long(), persistent=False)
        w = torch.empty(hidden, Cx, 3, 3)
        nn.init.kaiming_uniform_(w, a=5 ** 0.5)  # HexNCA's init, then its 7 taps in local order
        self.w1 = nn.Parameter(torch.stack([w[:, :, 1, 1]] + [w[:, :, 1 + DROW[k], 1 + DCOL[k]] for k in range(6)], 1))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.mids = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(depth - 1))
        self.w2 = nn.Parameter(torch.zeros(channels, hidden))  # zero-init: an untrained model is the identity
        self.b2 = nn.Parameter(torch.zeros(channels))
        self._plan_cache = None

    def __getstate__(self):  # the cache holds a weak reference (no pickling / deepcopy of it)
        state = self.__dict__.copy()
        state["_plan_cache"] = None
        return state

    def plan(self, consts, B):
        """The gather indices for consts' frames (its last plane), batch B: (idx_s, flat [B S S 7]; idx_d, flat
        [B S S 7 6]; the output index [B S S, C]). Cached for the consts tensor last seen (the same object,
        unmodified), so a rollout over one consts builds them once."""
        ver = -1 if consts.is_inference() else consts._version
        key = (B, tuple(consts.shape), consts.device, ver)
        hit = self._plan_cache
        if hit is not None and ver >= 0 and hit[0]() is consts and hit[1] == key:
            return hit[2]
        Hh, Ww = consts.shape[-2:]
        Hp, Wp = Hh + 2, Ww + 2
        dev = self.sigma.device
        f = consts[:, -1].round().long().to(dev).expand(B, -1, -1).reshape(B, Hh * Ww)
        cell = ((torch.arange(Hh, device=dev)[:, None] + 1) * Wp + torch.arange(Ww, device=dev) + 1).view(1, -1)
        base = torch.arange(B, device=dev)[:, None] * (Hp * Wp) + cell                   # [B, N]: padded rows
        off = torch.tensor([0] + [DROW[d] * Wp + DCOL[d] for d in range(6)], device=dev)  # board tap -> row offset
        rows = base[:, :, None] + off[self.tap_board[f]]                                 # [B, N, 7]: local taps
        idx_d = rows[..., None] * 6 + self.sigma[f][:, :, None, :]                       # [B, N, 7, 6]
        out = (rows.reshape(-1), idx_d.reshape(-1), self.chan_out[f].reshape(B * Hh * Ww, self.channels))
        if ver >= 0:
            self._plan_cache = (weakref.ref(consts), key, out)
        return out

    def update(self, x, plan):
        """d [B,C,S,S]: the update before the residual; x = concat(state, consts without the frame)."""
        B, _, Hh, Ww = x.shape
        idx_s, idx_d, idx_out = plan
        h = F.relu(_LocalTaps.apply(x, self.w1, self.b1, self.sca, self.dir_k, self.wcol, self.wcol_inv, idx_s, idx_d))
        for m in self.mids:
            h = F.relu(m(h))
        out = torch.addmm(self.b2, h, self.w2.t())  # [n, C], local output channels
        return out.gather(1, idx_out).view(B, Hh, Ww, self.channels).permute(0, 3, 1, 2)

    def step(self, state, walls, consts):
        """One synchronous step. state [B,C,S,S]; walls [B or 1,1,S,S]; consts [B or 1,n_in + 1,S,S] (mask first,
        the frame last)."""
        B = state.shape[0]
        cin = consts[:, :-1]
        x = torch.cat([state, cin.expand(B, -1, -1, -1)], 1)
        state = state + self.update(x, self.plan(consts, B))
        if self.clamp is not None:
            state = state.clamp(self.clamp[0], self.clamp[1])
        state = torch.cat([walls.expand(B, -1, -1, -1), state[:, 1:]], 1)
        return state * cin[:, :1]
