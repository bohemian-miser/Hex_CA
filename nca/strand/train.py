"""Train the M1 strand CA (docs/spectacle-nca-plan.md §4 M1a / M1b, §6; data: docs/strand-data.md): nca.model's
HexNCA, drawing Spectacle's line patterns on the hex lattice from the 15 chord bits of a rule.

    python -m nca.strand.train --name m1a-s --task m1a --levels 2 --hidden 64 --minutes 8
    python -m nca.strand.train --name m1a-s --resume --minutes 8           # exactly where it stopped
    python -m nca.strand.train --name m1b-s --task m1b --init runs/m1a-s/best.pt --levels 2 --minutes 8

TASKS
  m1a  a strand from a tap. The tap = the tapped chord's two edge bits on its cell, held (the plan's input).
       The line grows both ways ONE CHORD PER STEP: after t steps every edge of a chord within t-1 steps of the
       tap along the strand must be drawn (state ch1..6 > 0.5: edge d of the cell is crossed); chords further
       along are don't-care (the front cannot know them yet); every other edge must be 0 -- except an edge
       of a strand from before a "move" / "edit" damage, which must be 0 only once the erasing wave could have
       got there (its old distance from the tap < steps since). Fixed point: the whole strand, round the
       circuit or tail to tail, nothing beyond a tail. Ideal steps to complete = grow + 1 (walker.grow_steps).
  m1b  the whole pattern, no tap: ch7..12 = the chord using edge d belongs to a circuit. Per edge, not per
       cell (docs/strand-data.md, correction 4: a cell can hold a circuit's chord and a tail's). A circuit's
       edges must say 1 from the first step; a tail's edge must say 0 once the "open" news from its nearer
       end could have got there (distance along the strand to that end < steps), before that it is
       don't-care; every other edge 0. Ideal steps: 1 for a circuit, (n-1)//2 + 1 for an n-chord tail.
  The don't-care regions are each answer's light cone along the strand, not a hint about how to compute it.

MODEL. HexNCA as it is (masked 3x3 = 7 hex taps; --perception taps+pool adds the pooled max/min; ReLU
hidden; zero-init residual update; clamp), with 22 consts: the board mask, the 15 chord bits (walker.PAIRS
order), the 6 tap bits (mask first: HexNCA.step reads consts[:, :1] as the mask; that order differs from
loader.M1aBatcher's). State: ch0 = the mask (HexNCA re-imposes ch0 as its "walls"), ch1-6 the m1a edges,
ch7-12 the m1b closed edges, ch13.. hidden (--channels, default 32). Boards are padded to a square S per
level (L2 12, L3 37, level-4 crops 41; HexNCA's pooled perception assumes S x S). Level 4 = crops only.

POOL. One per training level (--levels), --pool-size slots, each a board, a tap (m1a), its state, and its
age (CA steps since the tap / the board), so a strand longer than one rollout completes across visits. Each
iteration draws a level (uniformly) and --batch slots:
  - max(1, batch/8) slots get a brand-new board + tap (fresh state);
  - the worst slot of the batch (by its loss at its last visit) starts over from the fresh state, same tap;
  - each of the rest, with probability --damage, gets one damage:
      state  zero every channel but ch0 in a hex disc of radius 1..S/6 round a drawn strand cell (m1a: only
             once the strand is complete; the strand must regrow)
      move   (m1a) the tap moves to a random chord of the board; age 0; the old strand must vanish
      edit   1-3 cells (never the tapped one) get their chords rotated by 1-5 sixths (a valid rule edit):
             the strand re-routes (m1a: age 0, the dropped part must vanish; m1b: closed flags recomputed, age 0)
  - T ~ U[a*S, b*S] steps (a, b = --steps-mult, default 3 6: the plan's U[3,6] x diameter): the first T -
    --bptt run WITHOUT gradient on the slots as they were (they just carry on), THEN the events above (new
    boards, the restart, damage) happen, then the last --bptt steps run with gradient -- so the response to
    every event (growth from a fresh tap, an erasure, a re-route, a repair) is inside the backpropagated
    window (truncated backprop: the pool carries the state anyway). Loss = the masked, don't-care-weighted MSE
    of the task's 6 planes (summed over edges / 6 x board cells), mean of the last --last-k steps, each step
    against its own age's target; Adam with per-parameter gradient normalisation (as nca.train). A
    non-finite step is skipped and its slots start over (20 in a row stop). dataSec = the event work.

QUICK CHECK (iteration 0 of a fresh / --init run, then every --eval-every): HELD-OUT rules (eval/ files) at
--eval-levels (default 3 4; level 4 = its crops), from the fresh state, read out after H = max(--eval-mult x S,
the set's largest ideal + 8) steps (capped at --eval-cap), so every strand has the time one chord per step
needs. m1a: --eval-n exported taps per level, stratified by strand length (short <= 10, medium 11-60, long >
60 chords) and weighted back to the natural mix. m1b: --eval-n boards per level, every strand counted.
The log line gets "q": {exact (natural mix, mean over levels), balanced (mean over levels x length buckets),
iou (edge IoU: m1a per tap against the strand, m1b per board on the closed edges), steps {ratio: median
settle/ideal, excess: mean settle - ideal, over exact ones; settle = 1 + the last step it was wrong},
byLevel, bySubset (15, 128, 258), byLen (short, medium, long), and for m1b board (all edges right)}, the top-
level "exact" (= q.exact, for nca.progress), "score" = q.balanced (best.pt keeps the best), "evalSec", "evalN"
(= samples per level x levels). The start line has "trivial": m1a draw only the tapped chord, m1b one
constant answer (all closed / all open), and "evalSets" (per level: n, H, counts per length bucket).

Every log line (runs/<name>/log.jsonl, one per 50 iterations) has loss, secPerIter, dataSec, modelSec, lr,
maxRssMB, damage counts, "pool" (per level: share of slots settled = age >= ideal, median age), "schedFrac"
with --schedule time, "time", "device"; stops: {"stopped": "time" | "done" | "collapsed"}. As nca.train:
--minutes stops before an iteration that would end past the limit (slowest iteration + a due check's
time); --schedule time puts the lr steps (x0.3, x0.1) at 60% / 85% of the time box (counted across
--resume), else of --iters; a linear warm-up over --warmup iterations after every fresh optimiser; the
COLLAPSE GUARD (a window's loss non-finite or > --collapse-loss-x x the median of the 10 before, or a check
below --collapse-frac x the best once the best is >= 0.4) rolls back to best.pt with a fresh optimiser, half
the lr scale and fresh pool states, logged {"rollback", "lrScale", "rollbacks", "reason", "restored",
"best"}; past --max-rollbacks it ends {"stopped": "collapsed"}. ckpt.pt (every --ckpt-every, and at a stop)
holds weights, optimiser, iteration, config, rng, pools (CPU), best, evalSec and the guard's state; best.pt
the best check's weights and score. --init: another checkpoint's weights (same --channels / --hidden; fresh
optimiser and pools: a curriculum stage or m1a -> m1b). --data: the dataset dir, or a .tgz of it
(tar -C data -czf strand.tgz strand) unpacked once into data/strand. runs/<name>/pool.npz (--snap-every):
nca.train's snapshot contract, each level's S x S boards embedded in a hexagon of radius S-1 (the key "R"),
walls := cells with chords, fill := the task's planes' max, target := the strand's / circuits' cells.
"""

import argparse
import json
import math
import os
import resource
import shutil
import tarfile
import time

import numpy as np
import torch
import torch.nn as nn

from ..model import HexNCA
from .loader import StrandFile, default_dir, file_paths, load_meta
from .walker import PAIR_INDEX, PAIRS, exit_table, walk

N_IN = 22                         # consts: mask, chords 15, tap 6
OUT = {"m1a": slice(1, 7), "m1b": slice(7, 13)}
N_FIXED = 13                      # ch0 mask + 6 edges + 6 closed; hidden channels after
INF = 32767                       # "never" in the int16 distance planes (strands are <= 3,689 chords)
AGE_CAP = 30000                   # a slot's age stops here (above every distance)
BUCKETS = ("short", "medium", "long")
SUBSETS = ("15", "128", "258")
DAMAGE = {"m1a": ("state", "move", "edit"), "m1b": ("state", "edit")}
EVAL_SEED = 20261006
LR = 5e-4
DECAY_AT = (0.6, 0.85)
STOP_SKIPS = 20
COLLAPSE_MIN_BEST = 0.4
COLLAPSE_HISTORY = 10
SNAP_N, SNAP_M = 48, 6


def bucket_of(n):
    """0 short (<= 10 chords), 1 medium (11-60), 2 long (> 60); works on arrays too."""
    return np.where(np.asarray(n) <= 10, 0, np.where(np.asarray(n) <= 60, 1, 2))


# ---------------------------------------------------------------- the model

def make_model(channels, hidden, clamp, perception="taps"):
    """HexNCA with N_IN consts (HexNCA's own n_consts is for the flood model's named planes, at most 7)."""
    model = HexNCA(channels, hidden, clamp, 1.0, 1, perception)
    w1 = torch.empty(hidden, channels + N_IN, 3, 3)
    nn.init.kaiming_uniform_(w1, a=5 ** 0.5)
    model.w1 = nn.Parameter(w1 * model.kmask)
    model.n_consts = N_IN
    if perception == "taps+pool":
        bound = 1.0 / ((channels + N_IN) * 9) ** 0.5
        model.w1pool = nn.Parameter(torch.empty(hidden, 2 * channels, 1, 1).uniform_(-bound, bound))
    return model


def fresh(cs, channels):
    """[B,C,S,S] zeros with ch0 = the mask (consts[:, :1])."""
    st = torch.zeros(cs.shape[0], channels, cs.shape[2], cs.shape[3], device=cs.device)
    st[:, 0:1] = cs[:, 0:1]
    return st


# ---------------------------------------------------------------- boards and planes

class Boards:
    """The boards of one (split, level), padded to S x S (level 4: the crops only)."""

    def __init__(self, data_dir, split, level, subset_of):
        self.level, self.subset_of = level, subset_of
        self.files = [StrandFile(p) for p in file_paths(data_dir, split, (level,))]
        if not self.files:
            raise SystemExit(f"no {split} files at level {level} under {data_dir}")
        self.boards = [(k, b) for k, f in enumerate(self.files) for b in range(f.n_boards)
                       if level != 4 or int(f["board_kind"][b]) == 1]
        self.S = max(max(self.files[k].hw(b)) for k, b in self.boards)

    def planes(self, k, b):
        f, S = self.files[k], self.S
        h, w = f.hw(b)
        mask = np.zeros((S, S), np.uint8)
        mask[:h, :w] = f.mask(b)
        ch = np.zeros((15, S, S), np.uint8)
        ch[:, :h, :w] = f.chords(b)
        return mask, ch


def consts_of(mask, ch, tap=None):
    x = np.zeros((N_IN,) + mask.shape, np.uint8)
    x[0], x[1:16] = mask, ch
    if tap is not None:
        row, col, d0, d1 = tap
        x[16 + d0, row, col] = x[16 + d1, row, col] = 1
    return x


def dist_planes(s, shape):
    """int16 [6,H,W] (shape = (H, W)): each edge of the strand's chords -> the chord's steps from the tap (a
    circuit: the shorter way round); INF elsewhere."""
    n, idx = len(s), s.index
    d = np.minimum(idx, n - idx) if s.closed else np.abs(idx)
    out = np.full((6,) + tuple(shape), INF, np.int16)
    out[s.ins, s.rows, s.cols] = d
    out[s.outs, s.rows, s.cols] = d
    return out


def pattern_planes(mask, ch):
    """Every strand of a board (no tap): closed int16 [6,S,S] (the edge's chord is a circuit's), care int16
    [6,S,S] (a tail's edge: its chord's distance to the nearer end; -1 elsewhere), sid int32 [6,S,S] (strand
    of the edge's chord, -1 none), and per strand length, closed, ideal (steps to settle)."""
    shape = mask.shape
    ex = exit_table(ch)
    closed = np.zeros((6,) + shape, np.int16)
    care = np.full((6,) + shape, -1, np.int16)
    sid = np.full((6,) + shape, -1, np.int32)
    seen = np.zeros((15,) + shape, bool)
    lens, cl, ideal = [], [], []
    for p in range(15):
        a, b = PAIRS[p]
        for r, c in zip(*np.nonzero((ch[p] > 0) & (mask > 0))):
            if seen[p, r, c]:
                continue
            s = walk(ex, mask, int(r), int(c), a, b)
            k, n = len(lens), len(s)
            seen[PAIR_INDEX[s.ins, s.outs], s.rows, s.cols] = True
            sid[s.ins, s.rows, s.cols] = k
            sid[s.outs, s.rows, s.cols] = k
            if s.closed:
                closed[s.ins, s.rows, s.cols] = 1
                closed[s.outs, s.rows, s.cols] = 1
            else:
                pos = s.index - s.index.min()
                od = np.minimum(pos, n - 1 - pos)
                care[s.ins, s.rows, s.cols] = od
                care[s.outs, s.rows, s.cols] = od
            lens.append(n)
            cl.append(bool(s.closed))
            ideal.append(1 if s.closed else (n - 1) // 2 + 1)
    return {"closed": closed, "care": care, "sid": sid, "length": np.array(lens, np.int64),
            "closedS": np.array(cl, bool), "ideal": np.array(ideal, np.int64)}


def rotate_cells(rng, mask, ch, avoid=None):
    """A copy of ch with 1-3 random chord cells (never `avoid` = (row, col)) rotated by 1-5 sixths."""
    ch = ch.copy()
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero((mask > 0) & ch.any(0))) if (r, c) != avoid]
    if not cells:
        return ch
    for j in rng.choice(len(cells), min(len(cells), int(rng.integers(1, 4))), replace=False):
        r, c = cells[j]
        k = int(rng.integers(1, 6))
        pairs = [PAIRS[p] for p in np.nonzero(ch[:, r, c])[0]]
        ch[:, r, c] = 0
        for a, b in pairs:
            ch[PAIR_INDEX[(a + k) % 6, (b + k) % 6], r, c] = 1
    return ch


def m1a_slot(mask, ch, tap, src, rule):
    s = walk(exit_table(ch), mask, *tap)
    return {"consts": consts_of(mask, ch, tap), "a": dist_planes(s, mask.shape),
            "b": np.full((6,) + mask.shape, -1, np.int16), "tap": tap, "src": src, "rule": rule,
            "length": len(s), "ideal": s.grow_steps() + 1}


def m1b_slot(mask, ch, src, rule):
    pl = pattern_planes(mask, ch)
    return {"consts": consts_of(mask, ch), "a": pl["closed"], "b": pl["care"], "tap": (-1, -1, -1, -1),
            "src": src, "rule": rule, "length": int(pl["length"].max(initial=0)),
            "ideal": int(pl["ideal"].max(initial=1))}


def new_slot(rng, task, bd):
    k, b = bd.boards[int(rng.integers(len(bd.boards)))]
    f = bd.files[k]
    mask, ch = bd.planes(k, b)
    if task == "m1a":
        _, row, col, d0, d1 = f.random_tap(rng, b)
        return m1a_slot(mask, ch, (row, col, d0, d1), (k, b), f.rule_id)
    return m1b_slot(mask, ch, (k, b), f.rule_id)


SLOT_KEYS = ("consts", "a", "b", "tap", "src", "rule", "length", "ideal")


def new_pool(rng, task, bd, n, channels, device):
    slots = [new_slot(rng, task, bd) for _ in range(n)]
    P = {k: np.stack([np.asarray(s[k]) for s in slots]) for k in SLOT_KEYS}
    P.update({"age": np.zeros(n, np.int64), "loss": np.full(n, np.nan, np.float32), "born": np.zeros(n, np.int64),
              "edits": np.zeros(n, np.int32), "last": np.full(n, -1, np.int8)})
    P["state"] = fresh(torch.from_numpy(P["consts"]).float(), channels).to(device)
    return P


def put(P, i, slot):
    for k in SLOT_KEYS:
        P[k][i] = slot[k]


def erase_after(old_a, old_b, age, new_a):
    """m1a: the erase plane after a move / edit: an edge of the old strand (or still pending from an earlier
    one) that is not on the new strand must be 0 once its old distance < steps since now."""
    old = np.where(old_a < INF, old_a.astype(np.int32), -1)
    pend = np.where(old_b >= 0, old_b.astype(np.int32) - int(age), -1)
    e = np.where(new_a < INF, -1, np.maximum(old, pend))
    return np.clip(e, -1, INF).astype(np.int16)


def hex_disc(S, r0, c0, rad):
    rr, cc = np.mgrid[0:S, 0:S]
    dr, dc = rr - r0, cc - c0
    return (np.abs(dr) + np.abs(dc) + np.abs(dr + dc)) <= 2 * rad


def damage(rng, task, P, i, bd, kind):
    """One damage to pool slot i (numpy planes; the state's disc zeroing is returned as a mask, or None).
    Returns (applied, disc)."""
    mask, ch = P["consts"][i, 0], P["consts"][i, 1:16]
    S = mask.shape[0]
    k, b = (int(x) for x in P["src"][i])
    if kind == "state":
        if task == "m1a":
            if P["age"][i] < P["ideal"][i]:
                return False, None  # only a complete strand is damaged
            on = (P["a"][i] < INF).any(0)
        else:
            on = ch.any(0)
        rows, cols = np.nonzero(on & (mask > 0))
        if not len(rows):
            return False, None
        j = int(rng.integers(len(rows)))
        return True, hex_disc(S, rows[j], cols[j], int(rng.integers(1, max(1, S // 6) + 1)))
    if task == "m1a":
        old_a, old_b, age = P["a"][i].copy(), P["b"][i].copy(), P["age"][i]
        if kind == "move":
            f = bd.files[k]
            h, w = f.hw(b)
            p, rows, cols = np.nonzero(ch[:, :h, :w])
            j = int(rng.integers(len(p)))
            d0, d1 = PAIRS[p[j]]
            if rng.integers(2):
                d0, d1 = d1, d0
            slot = m1a_slot(mask, ch, (int(rows[j]), int(cols[j]), d0, d1), (k, b), int(P["rule"][i]))
        else:  # edit
            tap = tuple(int(x) for x in P["tap"][i])
            slot = m1a_slot(mask, rotate_cells(rng, mask, ch, tap[:2]), tap, (k, b), int(P["rule"][i]))
        slot["b"] = erase_after(old_a, old_b, age, slot["a"])
    else:  # m1b edit
        slot = m1b_slot(mask, rotate_cells(rng, mask, ch), (k, b), int(P["rule"][i]))
    put(P, i, slot)
    P["age"][i] = 0
    return True, None


# ---------------------------------------------------------------- loss

def target_weight(task, a, b, age):
    """(target, weight) [B,6,S,S] at `age` ([B,1,1,1] float) from the slot planes a, b (float)."""
    if task == "m1a":
        tgt = (a < age).float()
        return tgt, torch.where(a < INF, tgt, (b < age).float())
    return a, (b < age).float()


def step_loss(state, task, a, b, age, mk, ncell):
    """Per sample: the weighted MSE of the task's planes, summed over edges and board cells / (6 x cells)."""
    tgt, w = target_weight(task, a, b, age)
    return (((state[:, OUT[task]] - tgt) ** 2) * w * mk).sum((1, 2, 3)) / ncell


# ---------------------------------------------------------------- the quick check

class EvalLevel:
    """Fixed held-out samples of one level: consts [n,22,S,S], a (m1a: dist; m1b: closed) and per item (a tap /
    a strand) level, bucket, subset, weight (natural mix within the level), ideal; m1b also sid (global)."""

    def __init__(self, task, bd, n, mult, cap):
        self.task, self.level, self.S = task, bd.level, bd.S
        rng = np.random.default_rng(EVAL_SEED + bd.level)
        allowed = set(bd.boards)
        cons, aa, sids, items = [], [], [], {k: [] for k in ("bucket", "subset", "w", "ideal", "length")}
        if task == "m1a":
            cands = [(k, i, int(ln[i])) for k, f in enumerate(bd.files)
                     for ln, tb in [(np.diff(f["tap_ptr"]), f["tap_board"])]
                     for i in range(f.n_taps) if (k, int(tb[i])) in allowed]
            bk = bucket_of([c[2] for c in cands])
            share = [float(np.mean(bk == j)) for j in range(3)]
            per = [n // 3 + (j < n % 3) for j in range(3)]
            for j in range(3):
                pool = np.nonzero(bk == j)[0]
                m = min(per[j], len(pool))
                for c in rng.choice(pool, m, replace=False) if m else []:
                    k, i, _ = cands[c]
                    f = bd.files[k]
                    b, row, col, d0, d1 = f.tap_args(i)
                    s = f.tap(i)  # the exported walkStrand walk
                    mask, ch = bd.planes(k, b)
                    cons.append(consts_of(mask, ch, (row, col, d0, d1)))
                    aa.append(dist_planes(s, (bd.S, bd.S)))
                    for key, v in (("bucket", j), ("subset", bd.subset_of[f.rule_id]), ("w", share[j] / m),
                                   ("ideal", s.grow_steps() + 1), ("length", len(s))):
                        items[key].append(v)
        else:
            off = 0
            for c in rng.choice(len(bd.boards), min(n, len(bd.boards)), replace=False):
                k, b = bd.boards[c]
                mask, ch = bd.planes(k, b)
                pl = pattern_planes(mask, ch)
                cons.append(consts_of(mask, ch))
                aa.append(pl["closed"])
                sids.append(np.where(pl["sid"] >= 0, pl["sid"] + off, -1))
                off += len(pl["length"])
                sub = bd.subset_of[bd.files[k].rule_id]
                items["bucket"] += list(bucket_of(pl["length"]))
                items["subset"] += [sub] * len(pl["length"])
                items["ideal"] += list(pl["ideal"])
                items["length"] += list(pl["length"])
            items["w"] = [1.0 / max(1, off)] * off
            self.sid = np.stack(sids)
        self.consts, self.a = np.stack(cons), np.stack(aa)
        self.items = {k: np.array(v) for k, v in items.items()}
        self.H = int(min(cap, max(mult * bd.S, int(self.items["ideal"].max(initial=1)) + 8)))

    def counts(self):
        return {"n": len(self.consts), "items": len(self.items["w"]), "H": self.H,
                "byLen": {BUCKETS[j]: int((self.items["bucket"] == j).sum()) for j in range(3)}}

    def trivial(self):
        """m1a: only the tapped chord drawn (exact iff the strand is one chord); m1b: one constant answer."""
        if self.task == "m1a":
            return self.items["length"] == 1
        ok = self._closed_strands()
        return ok if ok.mean() >= 0.5 else ~ok

    def _closed_strands(self):
        n = len(self.items["w"])
        out = np.zeros(n, bool)
        on = self.sid >= 0
        out[self.sid[on]] = self.a[on] > 0
        return out


@torch.no_grad()
def evaluate(stepper, ev, channels, device, batch):
    """Roll each sample of ev from the fresh state for ev.H steps; per item: exact at the end, settle step, and
    (m1a per tap, m1b per board) edge IoU; m1b also board exactness. stepper(state, walls, cs, t, sl)."""
    out = slice(1, 7) if ev.task == "m1a" else slice(7, 13)
    n, H = len(ev.consts), ev.H
    if ev.task == "m1b":
        n_items = len(ev.items["w"])
        last_s = torch.zeros(n_items + 1, dtype=torch.long, device=device)
    last_all, iou_all = [], []
    for s0 in range(0, n, batch):
        sl = slice(s0, min(n, s0 + batch))
        cs = torch.from_numpy(ev.consts[sl]).to(device).float()
        walls, mk = cs[:, :1], cs[:, :1] > 0
        a = torch.from_numpy(ev.a[sl]).to(device)
        tgt = (a < INF) if ev.task == "m1a" else (a > 0)
        tgt = tgt & mk
        if ev.task == "m1b":
            sid = (torch.from_numpy(ev.sid[sl]).to(device).long() + 1).flatten()
        last = torch.zeros(cs.shape[0], dtype=torch.long, device=device)
        state = fresh(cs, channels)
        for t in range(1, H + 1):
            state = stepper(state, walls, cs, t, sl)
            pred = (state[:, out] > 0.5) & mk
            wrong = pred ^ tgt
            last = torch.where(wrong.flatten(1).any(1), t, last)
            if ev.task == "m1b":
                bad = torch.zeros(n_items + 1, device=device).scatter_reduce_(0, sid, wrong.flatten().float(), "amax")
                last_s = torch.where(bad > 0, t, last_s)
        inter = (pred & tgt).flatten(1).sum(1).float()
        union = (pred | tgt).flatten(1).sum(1).float()
        iou_all.append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).cpu().numpy())
        last_all.append(last.cpu().numpy())
    last_b, iou = np.concatenate(last_all), np.concatenate(iou_all)
    last_i = last_b if ev.task == "m1a" else last_s[1:].cpu().numpy()
    return {"exact": last_i < H, "settle": last_i + 1, "iou": iou, "board": last_b < H}


def summarise(task, evs, results):
    """The log's "q" dict from per-level results (see the module docstring)."""
    byL, ex_all, w_all, sub_all, bk_all, ratio_all, excess_all, lv_all = {}, [], [], [], [], [], [], []
    for ev, r in zip(evs, results):
        it, ex = ev.items, r["exact"]
        w = it["w"] / it["w"].sum()
        bal = [float(ex[it["bucket"] == j].mean()) for j in range(3) if (it["bucket"] == j).any()]
        iou = float((w * r["iou"]).sum()) if task == "m1a" else float(r["iou"].mean())
        byL[str(ev.level)] = {"exact": round(float((w * ex).sum()), 4), "balanced": round(float(np.mean(bal)), 4),
                              "iou": round(iou, 4)}
        if task == "m1b":
            byL[str(ev.level)]["board"] = round(float(r["board"].mean()), 4)
        ex_all.append(ex), w_all.append(w), sub_all.append(it["subset"]), bk_all.append(it["bucket"])
        lv_all.append(np.full(len(ex), ev.level))
        ratio_all.append(np.where(ex, r["settle"] / np.maximum(1, it["ideal"]), np.nan))
        excess_all.append(np.where(ex, r["settle"] - it["ideal"], np.nan))
    ex, w, sub, bk, lv = (np.concatenate(x) for x in (ex_all, w_all, sub_all, bk_all, lv_all))
    ratio, excess = np.concatenate(ratio_all), np.concatenate(excess_all)
    nl = len(evs)
    rnd = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)  # noqa: E731
    med = lambda x: rnd(np.nanmedian(x)) if np.isfinite(x).any() else None  # noqa: E731
    avg = lambda x: rnd(np.nanmean(x)) if np.isfinite(x).any() else None  # noqa: E731
    q = {"exact": rnd(np.mean([v["exact"] for v in byL.values()])),
         "balanced": rnd(np.mean([v["balanced"] for v in byL.values()])),
         "iou": rnd(np.mean([v["iou"] for v in byL.values()])),
         "steps": {"ratio": med(ratio), "excess": avg(excess)}}
    if task == "m1b":
        q["board"] = rnd(np.mean([v["board"] for v in byL.values()]))
    q["byLevel"] = byL
    q["bySubset"] = {s: {"exact": rnd((w * ex)[sub == s].sum() / max(1e-12, w[sub == s].sum()))}
                     for s in SUBSETS if (sub == s).any()}
    q["byLen"] = {}
    for j, name in enumerate(BUCKETS):
        sel = bk == j
        if sel.any():
            per_level = [ex[sel & (lv == L)].mean() for L in np.unique(lv) if (sel & (lv == L)).any()]
            q["byLen"][name] = {"exact": rnd(np.mean(per_level)), "steps": med(ratio[sel])}
    q["levels"] = nl
    return q


def trivial_of(task, evs):
    out = {}
    for ev in evs:
        ok = ev.trivial()
        w = ev.items["w"] / ev.items["w"].sum()
        bal = [float(ok[ev.items["bucket"] == j].mean()) for j in range(3) if (ev.items["bucket"] == j).any()]
        out[str(ev.level)] = {"exact": round(float((w * ok).sum()), 4), "balanced": round(float(np.mean(bal)), 4)}
    out["what"] = "only the tapped chord drawn" if task == "m1a" else "one constant answer for every edge"
    return out


# ---------------------------------------------------------------- the run's plumbing (as nca.train)

def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: torch sees no CUDA device")
    return torch.device(name)


def to_cpu(x):
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, dict):
        y = type(x)((k, to_cpu(v)) for k, v in x.items())
        if hasattr(x, "_metadata"):
            y._metadata = x._metadata
        return y
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def lr_at(it, iters, lr, scale=1.0, floor=0.0, warm_from=None, warmup=0, frac=None):
    """nca.train.lr_at: x1 to 60%, x0.3 to 85%, x0.1 after (of it/iters, or of the time box), x scale, >= floor;
    then a linear warm-up over `warmup` iterations from warm_from."""
    f = it / max(1, iters) if frac is None else frac
    x = max(floor, lr * scale * (1.0 if f < DECAY_AT[0] else 0.3 if f < DECAY_AT[1] else 0.1))
    if warm_from is not None and warmup > 0 and 0 <= it - warm_from < warmup:
        x *= (it - warm_from + 1) / warmup
    return x


def collapse_reason(score, best, frac, loss, history, skipped=0, steps=1, loss_x=4.0):
    """nca.train.collapse_reason (copied: that file is being edited)."""
    if loss is None or not math.isfinite(loss) or skipped > 0.1 * steps:
        return f"loss not finite ({skipped} of {steps} steps skipped)"
    if len(history) >= COLLAPSE_HISTORY:
        med = float(np.median(history[-COLLAPSE_HISTORY:]))
        if loss > loss_x * med:
            return f"loss {loss:.4g} > {loss_x:g}x the median {med:.4g} of the previous {COLLAPSE_HISTORY} windows"
    if score is not None and best >= COLLAPSE_MIN_BEST and score < frac * best:
        return f"score {score:.4f} < {frac:g} x best {best:.4f}"
    return None


def best_score(run_dir):
    try:
        b = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    s = b.get("best")
    return None if s is None or s < 0 else (float(s), b.get("iteration", 0))


def save_ckpt(path, model, opt, it, cfg, rng, pools=None, best=-1.0, eval_sec=0.0, guard=None):
    tmp = path + ".tmp"
    torch.save({"model": to_cpu(model.state_dict()), "opt": to_cpu(opt.state_dict()), "iteration": it,
                "config": cfg, "rng": rng.bit_generator.state, "pool": to_cpu(pools), "best": best,
                "evalSec": eval_sec, "steer": None, **(guard or {})}, tmp)
    os.replace(tmp, path)


def ensure_data(arg):
    """The dataset dir: `arg` itself, or for a .tgz (tar -C data -czf strand.tgz strand) data/strand, unpacked
    from it once (atomically: several runs may start at once)."""
    if not arg:
        return default_dir()
    if not arg.endswith((".tgz", ".tar.gz")):
        return arg
    dest = default_dir()
    if os.path.isfile(os.path.join(dest, "meta.json")):
        return dest
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, f".strand-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(arg) as t:
        try:
            t.extractall(tmp, filter="data")
        except TypeError:  # a Python without extraction filters
            t.extractall(tmp)
    src = os.path.join(tmp, "strand")
    if not os.path.isfile(os.path.join(src, "meta.json")):
        raise SystemExit(f"{arg} has no strand/meta.json (make it with: tar -C data -czf strand.tgz strand)")
    try:
        os.rename(src, dest)
    except OSError:  # another run got there first
        if not os.path.isfile(os.path.join(dest, "meta.json")):
            raise
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def write_snapshot(path, it, pools, task, last_level, last_idx):
    """pool.npz in nca.train's snapshot contract; each level's S x S boards sit in a hexagon of radius R = S-1
    (rows 0..S-1, cols S-1..2S-2 of a (2S-1)^2 array: q in 0..S-1, r in -(S-1)..0, so |q+r| <= S-1). Also, for
    the dashboard's chord-level strand view (nca/dashboard.py): tgt_edges_R uint8 / pred_edges_R float16
    [n,6,S2,S2] (the target / predicted edge planes, direction d at ch[1+d]) and tap_R int16 [n,4] (row, col,
    d0, d1 in the UN-embedded board coordinates; -1 where a slot has no tap, e.g. m1b). Cheap: these come
    straight out of arrays write_snapshot already builds (P["a"], P["tap"], the state slice st)."""
    out = {"iteration": np.int64(it), "radii": np.array(sorted(P["consts"].shape[-1] - 1 for P in pools.values())),
           "last_R": np.int64(pools[last_level]["consts"].shape[-1] - 1), "last_idx": np.asarray(last_idx, np.int64)}
    for P in pools.values():
        n, S = min(len(P["age"]), SNAP_N), P["consts"].shape[-1]
        R, S2 = S - 1, 2 * S - 1

        def emb(x, dtype):
            y = np.zeros(x.shape[:-2] + (S2, S2), dtype)
            y[..., :S, S - 1:] = x
            return y
        st = P["state"][:n].cpu().numpy()
        on = (P["a"][:n] < INF) if task == "m1a" else (P["a"][:n] > 0)
        out[f"walls_{R}"] = emb(P["consts"][:n, 1:16].any(1), np.uint8)
        out[f"mask_{R}"] = emb(P["consts"][:n, 0], np.uint8)
        out[f"fill_{R}"] = emb(st[:, OUT[task]].max(1), np.float16)
        out[f"target_{R}"] = emb(on.any(1), np.uint8)
        out[f"tgt_edges_{R}"] = emb(on, np.uint8)
        out[f"pred_edges_{R}"] = emb(st[:, 1:7], np.float16)
        out[f"tap_{R}"] = P["tap"][:n].astype(np.int16)
        out[f"ntargets_{R}"] = np.ones(n, np.uint8)
        out[f"loss_{R}"] = P["loss"][:n].astype(np.float32)
        out[f"age_{R}"] = (it - P["born"][:n]).astype(np.int32)
        out[f"edits_{R}"] = P["edits"][:n].astype(np.int32)
        out[f"damage_{R}"] = P["last"][:n].astype(np.int8)
        out[f"state_{R}"] = emb(st[:min(n, SNAP_M)], np.float16)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main

def main(argv=None):
    t_start = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--task", choices=("m1a", "m1b"), default="m1a")
    p.add_argument("--levels", type=int, nargs="+", default=[2], help="training levels (2, 3, 4 = level-4 crops)")
    p.add_argument("--eval-levels", type=int, nargs="+", default=[3, 4], help="quick-check levels (held-out rules)")
    p.add_argument("--data", default=None, help="dataset dir (default data/strand) or a .tgz of it")
    p.add_argument("--steps-mult", type=float, nargs=2, default=[3, 6], metavar=("A", "B"),
                   help="steps per iteration ~ U[A*S, B*S], S = the level's board side")
    p.add_argument("--bptt", type=int, default=48, help="backprop through the last this many steps only")
    p.add_argument("--last-k", type=int, default=8, help="loss = mean over the last K steps")
    p.add_argument("--iters", type=int, default=None, help="total iterations (default 100000)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--pool-size", type=int, default=256, help="slots per training level")
    p.add_argument("--damage", type=float, default=0.3, help="probability of one damage per kept slot")
    p.add_argument("--lr", type=float, default=None, help=f"peak lr (default {LR})")
    p.add_argument("--lr-floor", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--collapse-frac", type=float, default=0.6)
    p.add_argument("--collapse-loss-x", type=float, default=4.0)
    p.add_argument("--max-rollbacks", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--perception", choices=("taps", "taps+pool"), default="taps")
    p.add_argument("--clamp", type=float, nargs=2, default=[-2.0, 2.0], metavar=("LO", "HI"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", help="start from this checkpoint's weights (same --channels/--hidden)")
    p.add_argument("--eval-every", type=int, default=200, help="a multiple of 50")
    p.add_argument("--eval-n", type=int, default=None, help="m1a taps / m1b boards per eval level (default 64)")
    p.add_argument("--eval-mult", type=float, default=8, help="readout after max(mult*S, largest ideal + 8) steps")
    p.add_argument("--eval-cap", type=int, default=2000, help="... at most this many")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--minutes", type=float, default=0)
    p.add_argument("--schedule", choices=("iters", "time"), default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--snap-every", type=int, default=50, help="pool.npz every this many iterations; 0 = never")
    p.add_argument("--force-collapse", type=int, default=0, help=argparse.SUPPRESS)  # selftest: collapse at this check
    args = p.parse_args(argv)
    if args.eval_every % 50:
        p.error("--eval-every must be a multiple of 50 (the log's period)")
    if args.schedule == "time" and not args.minutes and not args.resume:
        p.error("--schedule time needs --minutes")

    torch.set_num_threads(args.threads or min(4, os.cpu_count() or 1))
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path, best_path = os.path.join(run_dir, "ckpt.pt"), os.path.join(run_dir, "best.pt")

    ckpt = init = None
    start_it = 0
    if args.resume:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg, start_it = ckpt["config"], ckpt["iteration"]
        cfg["iters"] = args.iters or cfg["iters"]
        if args.lr:
            cfg["lr"] = args.lr
        if args.data:
            cfg["data"] = args.data
        if args.schedule and args.schedule != cfg["schedule"]:
            cfg["schedule"] = args.schedule
            if args.schedule == "time" and not cfg.get("scheduleMin"):
                if not args.minutes:
                    p.error("--schedule time needs --minutes")
                cfg["scheduleMin"] = args.minutes
    else:
        if args.init:
            init = torch.load(args.init, map_location="cpu", weights_only=False)
            ic = init["config"]
            if (ic["channels"], ic["hidden"]) != (args.channels, args.hidden) or ic.get("nIn") != N_IN:
                p.error(f"--init {args.init} has channels={ic['channels']} hidden={ic['hidden']} nIn={ic.get('nIn')}; "
                        f"this run asks for --channels {args.channels} --hidden {args.hidden} (and {N_IN} consts)")
        cfg = {"task": args.task, "levels": sorted(args.levels), "evalLevels": sorted(args.eval_levels),
               "data": args.data, "channels": args.channels, "hidden": args.hidden, "nIn": N_IN,
               "perception": init["config"]["perception"] if init else args.perception,
               "clamp": init["config"]["clamp"] if init else list(args.clamp), "fireRate": 1.0,
               "stepsMult": list(args.steps_mult), "bptt": args.bptt, "lastK": args.last_k,
               "lr": args.lr or LR, "lrFloor": args.lr_floor, "warmup": args.warmup,
               "collapseFrac": args.collapse_frac, "collapseLossX": args.collapse_loss_x,
               "maxRollbacks": args.max_rollbacks, "batch": args.batch, "iters": args.iters or 100000,
               "seed": args.seed, "poolSize": args.pool_size, "damage": args.damage,
               "damageKinds": list(DAMAGE[args.task]), "evalMult": args.eval_mult, "evalCap": args.eval_cap,
               "schedule": args.schedule or "iters", "scheduleMin": args.minutes if args.schedule == "time" else None,
               "init": args.init,
               "prevIterations": (init["config"].get("prevIterations", 0) + init["iteration"]) if init else 0,
               "loss": "don't-care-weighted MSE of the task's 6 planes over board cells, mean of the last lastK steps"}
    task, levels, C = cfg["task"], cfg["levels"], cfg["channels"]
    eval_n = cfg["evalN1"] = args.eval_n or cfg.get("evalN1", 64)
    data_dir = ensure_data(cfg.get("data"))
    meta = load_meta(data_dir)
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}

    torch.manual_seed(cfg["seed"])
    model = make_model(C, cfg["hidden"], cfg["clamp"], cfg["perception"])
    rng = np.random.default_rng(cfg["seed"])
    if ckpt:
        model.load_state_dict(ckpt["model"])
        rng.bit_generator.state = ckpt["rng"]
    elif init:
        model.load_state_dict(init["model"])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if ckpt:
        opt.load_state_dict(ckpt["opt"])
    lr_scale = ckpt.get("lrScale", 1.0) if ckpt else 1.0
    rollbacks = ckpt.get("rollbacks", 0) if ckpt else 0
    warm_from = ckpt.get("warmFrom") if ckpt else 0
    prior_sec = ckpt.get("elapsedSec", 0.0) if ckpt else 0.0
    elapsed = lambda: round(prior_sec + time.time() - t_start, 1)  # noqa: E731
    guard = lambda: {"lrScale": lr_scale, "rollbacks": rollbacks, "warmFrom": warm_from, "elapsedSec": elapsed()}  # noqa: E731
    sched_frac = lambda: (elapsed() / (60.0 * cfg["scheduleMin"])  # noqa: E731
                          if cfg.get("schedule") == "time" and cfg.get("scheduleMin") else None)
    lr_of = lambda i: lr_at(i, cfg["iters"], cfg["lr"], lr_scale, cfg["lrFloor"], warm_from, cfg["warmup"],  # noqa: E731
                            sched_frac())

    train_bd = {L: Boards(data_dir, "train", L, subset_of) for L in levels}
    evs = [EvalLevel(task, Boards(data_dir, "eval", L, subset_of), eval_n, cfg["evalMult"], cfg["evalCap"])
           for L in cfg["evalLevels"]]
    stepper = lambda st, w, cs, t, sl: model.step(st, w, cs)  # noqa: E731

    def check():
        res = [evaluate(stepper, ev, C, device, args.eval_batch) for ev in evs]
        q = summarise(task, evs, res)
        return {"q": q, "exact": q["exact"], "score": q["balanced"], "evalN": eval_n * len(evs)}

    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    def emit(rec):
        rec["time"], rec["device"] = round(time.time(), 2), device.type
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if ckpt and ckpt.get("pool"):
        pools = {int(L): P for L, P in ckpt["pool"].items()}
        for P in pools.values():
            P["state"] = P["state"].to(device)
    else:
        pools = {L: new_pool(rng, task, train_bd[L], cfg["poolSize"], C, device) for L in levels}
        for P in pools.values():
            P["born"][:] = start_it
    start = {"config": cfg, "startIteration": start_it, "evalN": eval_n * len(evs), "threads": torch.get_num_threads(),
             "torch": torch.__version__, "lrScale": lr_scale, "rollbacks": rollbacks,
             "params": sum(x.numel() for x in model.parameters()),
             "boardSide": {str(L): bd.S for L, bd in train_bd.items()},
             "evalSets": {str(ev.level): ev.counts() for ev in evs}, "trivial": trivial_of(task, evs),
             "schedule": {"mode": cfg["schedule"], "decayAt": list(DECAY_AT),
                          **({"boxMin": cfg["scheduleMin"], "usedMin": round(prior_sec / 60, 2)}
                             if cfg["schedule"] == "time" else {"iters": cfg["iters"]})}}
    if device.type == "cuda":
        start["gpu"] = torch.cuda.get_device_name(device)
    emit(start)

    snap_path, last = os.path.join(run_dir, "pool.npz"), [None]

    def snapshot(it):
        if not args.snap_every or last[0] is None:
            return
        try:
            write_snapshot(snap_path, it, pools, task, *last[0])
        except Exception as e:  # noqa: BLE001 -- never worth a training run
            emit({"iteration": it, "snapshotError": repr(e)})

    best = ckpt.get("best", -1.0) if ckpt else -1.0
    on_disk = best_score(run_dir) if ckpt else None
    if on_disk and on_disk[0] > best:
        best = on_disk[0]
    eval_sec = ckpt.get("evalSec", 0.0) if ckpt else 0.0
    n_checks = [0]
    if not ckpt:
        t_eval = time.time()
        rec = {"iteration": 0, **check()}
        eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
        best = rec["best"] = rec["score"]
        save_ckpt(best_path, model, opt, 0, cfg, rng, best=best, eval_sec=eval_sec)
        emit(rec)

    t_win, losses, t_data = time.time(), [], []
    kinds = dict.fromkeys(DAMAGE[task] + ("new", "worst"), 0)
    skipped = skip_run = win_skipped = win_steps = 0
    history, collapsed, slowest = [], False, 0.0
    a_mult, b_mult = cfg["stepsMult"]
    B = cfg["batch"]
    it = start_it
    while it < cfg["iters"]:
        due = (it + 1) % args.eval_every == 0 or it + 1 == cfg["iters"]
        if args.minutes and time.time() - t_start + slowest + (eval_sec if due else 0) > 60 * args.minutes:
            if it > start_it:
                save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
                snapshot(it)
            emit({"stopped": "time", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2),
                  "slowestIterSec": round(slowest, 2), "evalSec": round(eval_sec, 1)})
            break
        t_it = time.time()
        # ---- T, then the no-gradient prefix: the slots as they were, carried on (settling spans visits)
        L = int(levels[int(rng.integers(len(levels)))])
        P, bd = pools[L], train_bd[L]
        S = bd.S
        n_pool = len(P["age"])
        idx = rng.choice(n_pool, min(B, n_pool), replace=False)
        T = int(rng.integers(round(a_mult * S), max(round(a_mult * S), round(b_mult * S)) + 1))
        G = min(T, cfg["bptt"])
        K = min(cfg["lastK"], G)
        tidx = torch.from_numpy(idx).to(device)
        state = P["state"][tidx].clone()
        if T > G:
            cs_pre = torch.from_numpy(P["consts"][idx]).to(device).float()
            with torch.no_grad():
                for _ in range(T - G):
                    state = model.step(state, cs_pre[:, :1], cs_pre)
            P["age"][idx] = np.minimum(P["age"][idx] + (T - G), AGE_CAP)
        # ---- the events, at the start of the gradient window (so the response to each is backpropagated):
        # new boards, the worst slot's restart, damage (numpy board work, inline: ~1 ms a slot)
        t_ev = time.time()
        cur = np.nan_to_num(P["loss"][idx], nan=-1.0)
        j_worst = int(cur.argmax())
        new = set(rng.choice(len(idx), max(1, len(idx) // 8), replace=False).tolist())
        restart = new | {j_worst}
        discs = []
        for pos, i in enumerate(idx):
            if pos in new:
                put(P, i, new_slot(rng, task, bd))
                kinds["new"] += 1
            elif pos == j_worst:
                if task == "m1a":
                    P["b"][i] = -1  # from the fresh state: nothing drawn, nothing to erase
                kinds["worst"] += 1
            elif rng.random() < cfg["damage"]:
                kind = DAMAGE[task][int(rng.integers(len(DAMAGE[task])))]
                ok, disc = damage(rng, task, P, i, bd, kind)
                if ok:
                    kinds[kind] += 1
                    P["edits"][i] += 1
                    P["last"][i] = DAMAGE[task].index(kind)
                    if disc is not None:
                        discs.append((pos, disc))
            if pos in restart:
                P["age"][i], P["born"][i], P["edits"][i], P["last"][i] = 0, it, 0, -1
        cs = torch.from_numpy(P["consts"][idx]).to(device).float()
        a = torch.from_numpy(P["a"][idx]).to(device).float()
        b = torch.from_numpy(P["b"][idx]).to(device).float()
        rs = torch.tensor(sorted(restart), device=device)
        state[rs] = fresh(cs[rs], C)
        for pos, disc in discs:
            state[pos, 1:] *= torch.from_numpy(~disc).to(device).float()
        state[:, 0:1] = cs[:, 0:1]
        age0 = torch.from_numpy(P["age"][idx]).to(device).float()
        walls, mk = cs[:, :1], cs[:, :1]
        ncell = 6 * mk.sum((1, 2, 3)).clamp(min=1)
        t_data.append(time.time() - t_ev)
        # ---- the gradient window: G steps, the loss over the last K, each against its own age's target
        acc = 0.0
        for t in range(G):
            state = model.step(state, walls, cs)
            if t >= G - K:
                age = (age0 + t + 1).clamp(max=AGE_CAP).view(-1, 1, 1, 1)
                acc = acc + step_loss(state, task, a, b, age, mk, ncell) / K
        loss = acc.mean()
        opt.zero_grad()
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(q.grad).all())
                                                    for q in model.parameters() if q.grad is not None)
        if finite:
            skip_run = 0
            for q in model.parameters():
                if q.grad is not None:
                    q.grad /= q.grad.norm() + 1e-8
            for g in opt.param_groups:
                g["lr"] = lr_of(it)
            opt.step()
            losses.append(loss.item())
            P["state"][tidx] = state.detach()
            P["age"][idx] = np.minimum(P["age"][idx] + G, AGE_CAP)
            P["loss"][idx] = acc.detach().cpu().numpy()
        else:
            skipped, skip_run, win_skipped = skipped + 1, skip_run + 1, win_skipped + 1
            P["state"][tidx] = fresh(cs, C)
            P["age"][idx], P["born"][idx], P["loss"][idx] = 0, it, np.nan
        it += 1
        win_steps += 1
        last[0] = (L, idx)
        slowest = max(slowest, time.time() - t_it)
        if skip_run >= STOP_SKIPS:
            emit({"iteration": it, "stopped": f"{STOP_SKIPS} non-finite steps in a row", "skipped": skipped})
            raise SystemExit(f"{STOP_SKIPS} non-finite steps in a row; the last checkpoint is {ckpt_path}")

        if it % 50 == 0 or it == cfg["iters"]:
            spi = (time.time() - t_win) / max(1, win_steps)
            rec = {"iteration": it, "loss": float(np.mean(losses)) if losses else None, "secPerIter": spi,
                   "dataSec": round(float(np.mean(t_data)), 5), "modelSec": round(spi - float(np.mean(t_data)), 5),
                   "lr": lr_of(it - 1), "maxRssMB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                   "damage": kinds,
                   "pool": {str(Lp): {"settled": round(float((Pp["age"] >= Pp["ideal"]).mean()), 3),
                                      "age": int(np.median(Pp["age"]))} for Lp, Pp in pools.items()}}
            if cfg["schedule"] == "time":
                rec["schedFrac"] = round(sched_frac(), 4)
            if skipped:
                rec["skipped"] = skipped
            why = collapse_reason(None, best, cfg["collapseFrac"], rec["loss"], history, win_skipped, win_steps,
                                  cfg["collapseLossX"])
            if it % args.eval_every == 0 or it == cfg["iters"]:
                t_eval = time.time()
                rec.update(check())
                eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
                n_checks[0] += 1
                if args.force_collapse and n_checks[0] == args.force_collapse:
                    why = why or "forced (--force-collapse, a test)"
                if rec["score"] > best and not why:
                    best = rec["best"] = rec["score"]
                    save_ckpt(best_path, model, opt, it, cfg, rng, best=best, eval_sec=eval_sec)
                why = why or collapse_reason(rec["score"], best, cfg["collapseFrac"], 0.0, [], 0, 1)
            emit(rec)
            if rec["loss"] is not None and math.isfinite(rec["loss"]):
                history.append(rec["loss"])
            t_win, losses, t_data = time.time(), [], []
            kinds = dict.fromkeys(kinds, 0)
            win_skipped = win_steps = 0
            if why:
                back = best_score(run_dir)
                if rollbacks >= cfg["maxRollbacks"] or back is None:
                    save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
                    emit({"iteration": it, "stopped": "collapsed", "reason": why, "rollbacks": rollbacks,
                          "lrScale": lr_scale, "best": best,
                          **({} if back else {"note": "no readable best.pt to roll back to"})})
                    collapsed = True
                    break
                bk = torch.load(best_path, map_location="cpu", weights_only=False)
                model.load_state_dict(bk["model"])
                opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
                lr_scale, rollbacks, warm_from, history = lr_scale / 2, rollbacks + 1, it, []
                for Pp in pools.values():
                    Pp["state"] = fresh(torch.from_numpy(Pp["consts"]).float(), C).to(device)
                    Pp["age"][:], Pp["born"][:], Pp["edits"][:], Pp["last"][:], Pp["loss"][:] = 0, it, 0, -1, np.nan
                    if task == "m1a":
                        Pp["b"][:] = -1  # nothing drawn any more: nothing to erase
                emit({"iteration": it, "rollback": it, "lrScale": lr_scale, "rollbacks": rollbacks, "reason": why,
                      "restored": back[1], "best": best})
                save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
                snapshot(it)
                continue
        if it % max(1, args.ckpt_every) == 0 or it == cfg["iters"]:
            save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
            snapshot(it)
        elif args.snap_every and it % args.snap_every == 0:
            snapshot(it)
    if it >= cfg["iters"] and not collapsed:
        emit({"stopped": "done", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2)})
    log.close()


if __name__ == "__main__":
    main()
