"""Train the strand CA with the rule at the tap (docs/spectacle-nca-options.md §0, §3-§5; data: nca/strand/rules.py,
docs/strand-data.md "v2"): a HexNCA-style CA (nca/strand/nets.py) where each cell sees only static board facts and,
on the tapped cell, the tap. Supersedes nca/strand/train.py (the pre-rendered-chord M1, kept for launch 5's runs).

    python -m nca.strand.train2 --name c-l2 --inputs c --levels 2 --minutes 25
    python -m nca.strand.train2 --name c-l3 --inputs c --levels 2 3 --init runs/c-l2/best.pt --lr 2e-4 --minutes 240
    python -m nca.strand.train2 --name c-m1b --inputs c --task m1b --levels 2 3 4 --init runs/c-l4/best.pt

INPUTS (the consts, mask first; --inputs; the static planes are rules.py's):
  a     mask 1 + A's 16 (type one-hot 9, rotation one-hot 6, mirror 1)                          + tap 59 =  76
  c     mask 1 + C's 55 (per direction the facing edge's class 6 x 8, anchor 6, mirror 1)        + tap 59 = 115
  c-bc  C, plus the rule's 53-bit code on EVERY board cell (the diagnostic ceiling, not the product) + 59 = 168
  d     mask 1 + D's 11 (type one-hot 9, rot / 6 as one number, mirror 1)                        + tap 59 =  71
  d-cs  mask 1 + D-cs's 12 (type one-hot 9, cos and sin of 2 pi rot / 6, mirror 1)               + tap 59 =  72
  e     mask 1 + E's 9 (type one-hot only)                                                       + tap 59 =  69
        and nets.FrameNCA: each cell runs the update in its own frame (option E: rotation and mirror are not
        inputs but which permutation of its taps and directional channels the cell reads; the consts carry the
        frame index as a last plane for that, never as a feature). Hidden state: --dir-groups directional
        groups of 6 channels (default (channels - 13) // 12) after ch1-12, scalars after them.
  e-bc  E, plus the rule's 53-bit code on EVERY board cell (E's diagnostic, as c-bc is C's)          + 59 = 122
  The tap, in every arm, on the tapped cell only and held every step: the rule's code (53: 8 class bits, then
  9 x 5 digit one-hot; rules.py) and the tapped chord's two edge directions (6). No owner slot (one player in
  M1). A chord is never an input.
RULES: a new pool slot draws a TRAIN rule (rules.RuleTable.sample: the subset uniform over the 7, the rule
uniform within it; never a held-out rule), a board of its level (L2 / L3: the nine patches; L4:
the 2,048 training crops) and a uniformly random chord of it as the tap; the target strand is walked from the
rule rendered by rules.py (= Spectacle's walkStrand, `python -m nca.strand.rules --parity`).
TASKS
  m1a  ch1..6 = edge d of the cell is crossed by the tapped strand, growing one chord per step both ways from
       the tap: train.py's targets and light cones, unchanged.
  m1b  m1a's six planes AND ch7..12 = the chord through edge d is on a circuit, for the tapped strand only
       (the tap is held; docs/spectacle-nca-options.md §0.5). A circuit's edge must say 1 once it is drawn; a
       tail's edge must say 0 once the "open" news could have got there -- the front reaches an end at its
       distance from the tap and the news comes back one chord per step, so chord i of n (tap at p) by
       min(p + i, 2(n-1) - p - i) -- don't-care before that; every other edge 0 (after a move / edit, once
       the erase wave could have got there). After an edit, the new strand's closed planes are don't-care
       until its whole settle time has passed (a re-route can open or close it far from the tap). 12 output
       planes; start it from an m1a checkpoint (--init).
  Ideal steps: m1a grow + 1 (walker.Strand.grow_steps); m1b max(grow + 1, the latest open news + 1).
MODEL: nets.StrandNCA (nets.FrameNCA for e, e-bc) = HexNCA's taps step with --depth hidden layers (default 1 =
HexNCA); --channels (96): ch0 the mask, ch1-6 edges, ch7-12 closed, ch13.. hidden (83 at 96); --hidden 128; clamp
[-2, 2]. Boards padded to S x S per level (L2 12, L3 37, L4 crops 42).
POOL, DAMAGE, LOSS, SCHEDULE, COLLAPSE GUARD, --resume, --init, ckpt.pt / best.pt: as nca/strand/train.py's
docstring, except: damage "edit" = 1-3 board cells with chords (never the tapped one) re-typed to another
random (type, rotation) -- a static-input edit, so their chords change under the slot's rule and the strand
re-routes; "move" = the tap moves to a random chord of the board, same rule; m1b gets all three
(it has a tap now). The loss is the masked, don't-care-weighted MSE of the task's planes / (planes x board
cells). --ckpt-pool half keeps the pool states as float16 in ckpt.pt (a third the size; a resume is then not
bit-exact). --data: data/strand-v2 or a .tgz of it (tar -C data -czf strand-v2.tgz strand-v2), unpacked once
into data/strand-v2; --legacy likewise for data/strand (tar -C data -czf strand.tgz strand).
QUICK CHECK (held-out rules only; --eval-sets, at --eval-levels):
  legacy  the v1 split's 20 held-out rules on the v1 eval boards (level 4: the crops), the SAME taps as
          train.py's check (same EVAL_SEED and stratified draw), inputs built from those boards' types and
          rotations: comparable with the overnight runs
  wide    the held-out rules, 40 per subset (all of a subset with fewer: 1 / 6 / 13 / 2 / 3 -- in 15 / 128 / 258
          the legacy rules, on other boards and taps), fixed seed;
          L2 / L3 patches and the 64 L4 eval crops; --eval-n taps per level stratified by length bucket
          (short <= 10, medium 11-60, long > 60 chords), round-robin over the subsets within a bucket
  Read out after max(--eval-mult x S, the set's largest ideal + 8) steps (capped at --eval-cap), from the
  fresh state. "q": exact (natural mix) and balanced (bucket mean; the score best.pt keeps), each the mean
  over (set, level); iou; steps {ratio, excess}; byLevel; bySet {legacy, wide: exact, balanced, byLevel,
  byLen}; bySubset (exact, n); byLen; m1b adds parts {edges, closed} (the shares with the six edge planes /
  the six closed planes right). And "code", the CODE-FIDELITY PROBE (§5.4): a ridge linear probe from the
  hidden channels (13..) of drawn strand cells (never the tapped one) to the 53 code bits, fitted on the
  training pools (train rules), tested on the quick check's read-out states (held-out rules): bits (per-bit
  accuracy, digits decoded by argmax), classes / digits / exact (shares of cells with the 8 class bits / the
  9 digits / everything right), byDist (exact by chords from the tap: 1-5, 6-20, 21-60, 61+), chance
  {classes, digits} (the same shares for the fit's mean code, a constant guess), n.
LOG (runs/<name>/log.jsonl): train.py's fields -- loss, secPerIter, dataSec (events + building the inputs),
modelSec, lr, maxRssMB, damage, pool {settled, age}, q, exact, score, evalSec, evalN; the start line has the
config, params, boardSide, evalSets, trivial (only the tapped chord drawn) and rules (split sizes). pool.npz:
train.py's snapshot contract (walls = cells with chords under the slot's rule; state = the first 32 channels).
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

from . import train as V1
from .loader import load_meta
from .rules import CODE_BITS, MAJORS, N_DIGITS, N_GEO, N_TYPES, STATIC, Boards as GeoBoards, RuleTable, \
    chord_planes, random_chord
from .nets import FrameNCA, StrandNCA
from .rules import default_dir as v2_default_dir
from .walker import PAIRS, walk

INPUTS = ("a", "c", "c-bc", "d", "d-cs", "e", "e-bc")
BROADCAST = ("c-bc", "e-bc")  # the rule's code on every board cell (diagnostics)
TAP = CODE_BITS + 6                # 59 planes on the tapped cell: the code, the tapped chord's two edges
OUT = {"m1a": slice(1, 7), "m1b": slice(1, 13)}
N_FIXED = 13                       # ch0 mask + 6 edges + 6 closed; hidden channels after
INF, AGE_CAP, BUCKETS = V1.INF, V1.AGE_CAP, V1.BUCKETS
DAMAGE = ("state", "move", "edit")
EVAL_SEED = V1.EVAL_SEED
LR = 5e-4
SNAP_C = 32                        # state channels in pool.npz
GROUP = {2: "L2", 3: "L3", 4: "L4"}
EVAL_GROUP = {2: "L2", 3: "L3", 4: "L4eval"}
DIST_BINS = (("1-5", 1, 5), ("6-20", 6, 20), ("21-60", 21, 60), ("61+", 61, 1 << 30))
PROBE_CELLS = 30000


def static_of(inputs):
    """rules.py's static table behind an --inputs option."""
    return {"c-bc": "c", "e-bc": "e"}.get(inputs, inputs)


def framed(inputs):
    """Option E's nets.FrameNCA (local frames; the consts end with the frame plane)."""
    return static_of(inputs) == "e"


def n_inputs(inputs):
    return 1 + STATIC[static_of(inputs)] + (CODE_BITS if inputs in BROADCAST else 0) + TAP


def dir_groups_of(channels, groups):
    """Hidden directional groups for option E: `groups`, or by default about half the hidden channels."""
    return (channels - N_FIXED) // 12 if groups is None or groups < 0 else groups


def make_model(channels, hidden, clamp, n_in, depth=1, inputs="c", dir_groups=None):
    if not framed(inputs):
        return StrandNCA(channels, hidden, depth, clamp, n_in)
    G = dir_groups_of(channels, dir_groups)
    # directional groups of concat(state, consts): edges, closed, G hidden groups; the tap's two chord edges
    dir_in = [1, 7] + [N_FIXED + 6 * j for j in range(G)] + [channels + n_in - 6]
    return FrameNCA(channels, hidden, depth, clamp, n_in, dir_in)


def fresh(mask, channels):
    """[B,C,S,S] zeros with ch0 = mask ([B,1,S,S] float)."""
    st = torch.zeros(mask.shape[0], channels, mask.shape[2], mask.shape[3], device=mask.device)
    st[:, 0:1] = mask
    return st


# ---------------------------------------------------------------- inputs

class Codes:
    """The 53-bit code of rules given as int rows [s, digit_0 .. digit_8] (vectorised rules.RuleTable.code)."""

    def __init__(self, tab):
        self.cls = np.stack([tab.code(s, np.zeros(N_TYPES, np.int64))[:len(MAJORS)] for s in range(tab.n_sub)])

    def __call__(self, rules):
        rules = np.asarray(rules, np.int64).reshape(-1, 1 + N_TYPES)
        out = np.zeros((len(rules), CODE_BITS), np.uint8)
        out[:, :len(MAJORS)] = self.cls[rules[:, 0]]
        rows = np.arange(len(rules))[:, None]
        out[rows, len(MAJORS) + np.arange(N_TYPES)[None] * N_DIGITS + rules[:, 1:]] = 1
        return out


class Planes:
    """Builds the consts [B, n_in, S, S] on the device from per-slot geo (int16 [B,S,S], -1 off board), code
    (uint8 [B,53]) and tap (int [B,4]: row, col, d0, d1)."""

    def __init__(self, tab, inputs, device):
        st = tab.static[static_of(inputs)]
        lut = np.zeros((N_GEO + 1, st.shape[1]), np.float32)
        lut[:N_GEO] = st
        self.lut = torch.from_numpy(lut).to(device)
        self.bc = inputs in BROADCAST
        self.frames = framed(inputs)  # E: one more plane, the cell's frame (FrameNCA permutes by it)
        self.n_in = n_inputs(inputs)
        self.device = device
        self.tap_at = 1 + st.shape[1] + (CODE_BITS if self.bc else 0)  # first tap plane

    def __call__(self, geo, code, tap):
        dev = self.device
        g = torch.as_tensor(np.asarray(geo), device=dev).long()
        B, S = g.shape[0], g.shape[-1]
        on = g >= 0
        mask = on.float()[:, None]
        st = self.lut[torch.where(on, g, N_GEO)].permute(0, 3, 1, 2)
        code_t = torch.as_tensor(np.asarray(code), device=dev).float()
        tap_t = torch.as_tensor(np.asarray(tap, np.int64), device=dev)
        vals = torch.zeros(B, TAP, device=dev)
        vals[:, :CODE_BITS] = code_t
        bi = torch.arange(B, device=dev)
        vals[bi, CODE_BITS + tap_t[:, 2]] = 1
        vals[bi, CODE_BITS + tap_t[:, 3]] = 1
        tp = torch.zeros(B, TAP, S * S, device=dev)
        tp[bi, :, tap_t[:, 0] * S + tap_t[:, 1]] = vals
        parts = [mask, st]
        if self.bc:
            parts.append(code_t[:, :, None, None] * mask)
        parts.append(tp.view(B, TAP, S, S))
        if self.frames:
            parts.append(torch.where(on, g % 12, 0).float()[:, None])
        return torch.cat(parts, 1)


# ---------------------------------------------------------------- slots and targets

def pad_geo(geo, S):
    out = np.full((S, S), -1, np.int16)
    out[:geo.shape[0], :geo.shape[1]] = geo
    return out


def strand_targets(st, shape, task):
    """(a, oc, ideal): a = train.dist_planes (each strand edge -> its chord's steps from the tap, INF elsewhere);
    oc = the closed planes' care-from time on the strand's edges (a circuit: = a; a tail: when the open news gets
    there, min(p + i, 2(n-1) - p - i)), INF elsewhere; ideal = steps to settle."""
    a = V1.dist_planes(st, shape)
    n = len(st)
    if st.closed:
        ov = np.minimum(st.index, n - st.index)
    else:
        pos = st.index - st.index.min()
        p = st.tap_pos
        ov = np.minimum(p + pos, 2 * (n - 1) - p - pos)
    oc = np.full((6,) + tuple(shape), INF, np.int16)
    oc[st.ins, st.rows, st.cols] = ov
    oc[st.outs, st.rows, st.cols] = ov
    ideal = st.grow_steps() + 1
    if task == "m1b":
        ideal = max(ideal, int(ov.max()) + 1)
    return a, oc, ideal


SLOT_KEYS = ("geo", "rule", "tap", "src", "a", "b", "oc", "closed", "length", "ideal")


def make_slot(tab, geo, rule, tap, src, task):
    s, digits = int(rule[0]), np.asarray(rule[1:], np.int64)
    st = walk(tab.exits(s, digits, geo), geo >= 0, *tap)
    a, oc, ideal = strand_targets(st, geo.shape, task)
    return {"geo": geo, "rule": np.asarray(rule, np.int16), "tap": np.asarray(tap, np.int16), "src": int(src), "a": a, "b": np.full(a.shape, -1, np.int16), "oc": oc, "closed": bool(st.closed),
            "length": len(st), "ideal": ideal}


def new_slot(rng, tab, bd, group, S, task):
    """A training board of `group`, a train rule, a random chord as the tap."""
    nb = len(bd.hw[group])
    while True:
        i = int(rng.integers(nb))
        geo = pad_geo(bd.board(group, i), S)
        s, digits = tab.sample(rng)
        tap = random_chord(rng, tab.render_bits(s, digits, geo))
        if tap is not None:  # a rule can draw nothing on a small patch (e.g. none of its drawing types there)
            return make_slot(tab, geo, np.concatenate([[s], digits]), tap, i, task)


def new_pool(rng, tab, bd, group, S, n, channels, task, device):
    slots = [new_slot(rng, tab, bd, group, S, task) for _ in range(n)]
    P = {k: np.stack([np.asarray(s[k]) for s in slots]) for k in SLOT_KEYS}
    P.update({"age": np.zeros(n, np.int64), "loss": np.full(n, np.nan, np.float32), "born": np.zeros(n, np.int64),
              "edits": np.zeros(n, np.int32), "last": np.full(n, -1, np.int8)})
    P["state"] = fresh(torch.from_numpy(P["geo"] >= 0).float()[:, None], channels).to(device)
    return P


def put(P, i, slot):
    for k in SLOT_KEYS:
        P[k][i] = slot[k]


def damage(rng, tab, task, P, i, kind):
    """One damage to pool slot i. Returns (applied, disc): disc = the state-damage disc (bool [S,S]) or None."""
    geo = P["geo"][i]
    S = geo.shape[0]
    if kind == "state":
        if P["age"][i] < P["ideal"][i]:
            return False, None  # only a complete strand is damaged
        rows, cols = np.nonzero((P["a"][i] < INF).any(0) & (geo >= 0))
        if not len(rows):
            return False, None
        j = int(rng.integers(len(rows)))
        return True, V1.hex_disc(S, rows[j], cols[j], int(rng.integers(1, max(1, S // 6) + 1)))
    rule = P["rule"][i]
    s, digits = int(rule[0]), rule[1:].astype(np.int64)
    tap = tuple(int(x) for x in P["tap"][i])
    old_a, old_b, age = P["a"][i].copy(), P["b"][i].copy(), int(P["age"][i])
    if kind == "move":
        tap = random_chord(rng, tab.render_bits(s, digits, geo))
    else:  # edit: re-type 1-3 cells with chords, never the tapped one
        bits = tab.render_bits(s, digits, geo)
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(bits != 0)) if (r, c) != tap[:2]]
        if not cells:
            return False, None
        geo = geo.copy()
        for j in rng.choice(len(cells), min(len(cells), int(rng.integers(1, 4))), replace=False):
            r, c = cells[j]
            g = int(geo[r, c])
            new = g
            while new == g:  # another (type, rotation); the patch's mirror sign stays
                new = int(rng.integers(N_TYPES)) * 12 + int(rng.integers(6)) * 2 + g % 2
            geo[r, c] = new
    slot = make_slot(tab, geo, rule, tap, int(P["src"][i]), task)
    slot["b"] = V1.erase_after(old_a, old_b, age, slot["a"])
    if kind == "edit" and task == "m1b":
        slot["oc"] = np.where(slot["oc"] < INF, slot["ideal"] - 1, INF).astype(np.int16)
    put(P, i, slot)
    P["age"][i] = 0
    return True, None


# ---------------------------------------------------------------- loss

def target_weight(task, a, b, oc, closed, age):
    """(target, weight) [B,6 or 12,S,S] at `age` ([B,1,1,1] float); a, b, oc float [B,6,S,S]; closed bool [B]."""
    on = a < INF
    tgt = (a < age).float()
    w = torch.where(on, tgt, (b < age).float())
    if task == "m1a":
        return tgt, w
    ctgt = (on & closed.view(-1, 1, 1, 1)).float()
    cw = torch.where(on, (oc < age).float(), (b < age).float())
    return torch.cat([tgt, ctgt], 1), torch.cat([w, cw], 1)


def step_loss(state, task, a, b, oc, closed, age, mk, ncell):
    tgt, w = target_weight(task, a, b, oc, closed, age)
    return (((state[:, OUT[task]] - tgt) ** 2) * w * mk).sum((1, 2, 3)) / ncell


# ---------------------------------------------------------------- the quick check

class EvalSet:
    """Fixed held-out taps of one (set, level): geo [n,S,S], rule [n,10], tap [n,4], a / oc [n,6,S,S],
    closed [n]; items: bucket, subset (key), w (natural mix within the set-level), ideal, length."""

    def __init__(self, name, level, S, rows, task, mult, cap, natural=None):
        self.name, self.level, self.S, self.task = name, level, S, task
        self.geo = np.stack([r["geo"] for r in rows])
        self.rule = np.stack([r["rule"] for r in rows]).astype(np.int64)
        self.tap = np.stack([r["tap"] for r in rows]).astype(np.int64)
        self.a = np.stack([r["a"] for r in rows])
        self.oc = np.stack([r["oc"] for r in rows])
        self.closed = np.array([r["closed"] for r in rows], bool)
        self.items = {k: np.array([r[k] for r in rows]) for k in ("bucket", "subset", "w", "ideal", "length")}
        self.H = int(min(cap, max(mult * S, int(self.items["ideal"].max(initial=1)) + 8)))

    def counts(self):
        return {"n": len(self.geo), "H": self.H, "S": self.S,
                "byLen": {BUCKETS[j]: int((self.items["bucket"] == j).sum()) for j in range(3)}}


def _row(tab, geo, rule, tap, st, task, bucket, w):
    a, oc, ideal = strand_targets(st, geo.shape, task)
    return {"geo": geo, "rule": np.asarray(rule, np.int64), "tap": np.asarray(tap, np.int64), "a": a,
            "oc": oc, "closed": bool(st.closed), "bucket": bucket, "subset": tab.keys[int(rule[0])], "w": w,
            "ideal": ideal, "length": len(st)}


def legacy_set(task, tab, legacy_dir, level, n, mult, cap):
    """train.py's quick-check taps (the same draw), with this trainer's inputs."""
    meta = load_meta(legacy_dir)
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}
    rule_of = {}
    for r in meta["rules"]:
        s = tab.keys.index(r["subset"])
        rule_of[r["id"]] = [s] + [tab.options[s][t].index(m) for t, m in enumerate(r["matching"])]
    bd = V1.Boards(legacy_dir, "eval", level, subset_of)
    rng = np.random.default_rng(EVAL_SEED + bd.level)  # --- the draw of train.EvalLevel (m1a), verbatim
    allowed = set(bd.boards)
    cands = [(k, i, int(ln[i])) for k, f in enumerate(bd.files)
             for ln, tb in [(np.diff(f["tap_ptr"]), f["tap_board"])]
             for i in range(f.n_taps) if (k, int(tb[i])) in allowed]
    bk = V1.bucket_of([c[2] for c in cands])
    share = [float(np.mean(bk == j)) for j in range(3)]
    per = [n // 3 + (j < n % 3) for j in range(3)]
    rows = []
    for j in range(3):
        pool = np.nonzero(bk == j)[0]
        m = min(per[j], len(pool))
        for c in rng.choice(pool, m, replace=False) if m else []:
            k, i, _ = cands[c]
            f = bd.files[k]
            b, row, col, d0, d1 = f.tap_args(i)
            h, w = f.hw(b)
            t, r = f["tile_type"][b, :h, :w].astype(np.int16), f["tile_rot"][b, :h, :w].astype(np.int16)
            geo = pad_geo(np.where(t >= 0, t * 12 + r * 2 + int(f["board_mirror"][b] < 0), -1).astype(np.int16), bd.S)
            rows.append(_row(tab, geo, rule_of[f.rule_id], (row, col, d0, d1), f.tap(i), task, j, share[j] / m))
    return EvalSet("legacy", level, bd.S, rows, task, mult, cap)


def wide_set(task, tab, bd, level, n, mult, cap, per_subset=40, boards_per_rule=9, taps_per_board=4):
    """v2 held-out rules: up to per_subset per subset (fixed seed), on the level's eval boards; n taps stratified
    by length bucket, round-robin over the subsets in each bucket. Natural mix: subsets uniform, then taps."""
    rng = np.random.default_rng(EVAL_SEED + 100 + level)
    group = EVAL_GROUP[level]
    S = int(bd.hw[group].max())
    nb = len(bd.hw[group])
    cands = {s: [] for s in range(tab.n_sub)}
    for s in range(tab.n_sub):
        held = np.nonzero(tab.held_out(s))[0]
        pick = held if len(held) <= per_subset else np.sort(rng.choice(held, per_subset, replace=False))
        for idx in pick:
            digits = tab.digits_of(s, int(idx))
            boards = range(nb) if nb <= boards_per_rule else rng.choice(nb, boards_per_rule, replace=False)
            for b in boards:
                geo = pad_geo(bd.board(group, int(b)), S)
                p, rows, cols = np.nonzero(chord_planes(tab.render_bits(s, digits, geo)))
                if not len(p):
                    continue
                ex = tab.exits(s, digits, geo)
                for k in rng.choice(len(p), min(taps_per_board, len(p)), replace=False):
                    d0, d1 = PAIRS[p[k]]
                    if rng.integers(2):
                        d0, d1 = d1, d0
                    tap = (int(rows[k]), int(cols[k]), int(d0), int(d1))
                    st = walk(ex, geo >= 0, *tap)
                    cands[s].append((geo, np.concatenate([[s], digits]), tap, st, int(V1.bucket_of(len(st)))))
    subs = [s for s in range(tab.n_sub) if cands[s]]
    share = [float(np.mean([np.mean([c[4] == j for c in cands[s]]) for s in subs])) for j in range(3)]
    per = [n // 3 + (j < n % 3) for j in range(3)]
    rows = []
    for j in range(3):
        left = {s: [c for c in cands[s] if c[4] == j] for s in subs}
        for s in left:
            rng.shuffle(left[s])
        got = []
        while len(got) < per[j] and any(left.values()):
            for s in subs:
                if left[s] and len(got) < per[j]:
                    got.append(left[s].pop())
        for geo, rule, tap, st, _ in got:
            rows.append(_row(tab, geo, rule, tap, st, task, j, share[j] / len(got)))
    return EvalSet("wide", level, S, rows, task, mult, cap)


@torch.no_grad()
def evaluate(stepper, planes, ev, channels, device, batch, codes, probe=False):
    """Roll each tap of ev from the fresh state for ev.H steps. Per tap: exact at the end, the settle step, edge
    IoU (the six edge planes); m1b also edges / closed (each half right); probe: per drawn strand cell (not
    the tapped one) its hidden channels, the code and its distance from the tap.
    stepper(state, walls, consts, t, sl) -> state (t = 1..H, sl = the slice of ev's taps)."""
    task = ev.task
    out = OUT[task]
    n, H = len(ev.geo), ev.H
    res = {k: [] for k in ("last", "last_e", "last_c", "iou")}
    cells = {"x": [], "y": [], "dist": []}
    for s0 in range(0, n, batch):
        sl = slice(s0, min(n, s0 + batch))
        cs = planes(ev.geo[sl], codes(ev.rule[sl]), ev.tap[sl])
        walls = cs[:, :1]
        mk = walls > 0
        a = torch.from_numpy(ev.a[sl]).to(device)
        on = a < INF
        tgt = on
        if task == "m1b":
            tgt = torch.cat([on, on & torch.from_numpy(ev.closed[sl]).to(device).view(-1, 1, 1, 1)], 1)
        tgt = tgt & mk
        B = cs.shape[0]
        last = torch.zeros(B, dtype=torch.long, device=device)
        last_e, last_c = last.clone(), last.clone()
        state = fresh(walls, channels)
        for t in range(1, H + 1):
            state = stepper(state, walls, cs, t, sl)
            wrong = ((state[:, out] > 0.5) & mk) ^ tgt
            last = torch.where(wrong.flatten(1).any(1), t, last)
            if task == "m1b":
                last_e = torch.where(wrong[:, :6].flatten(1).any(1), t, last_e)
                last_c = torch.where(wrong[:, 6:].flatten(1).any(1), t, last_c)
        pred = (state[:, 1:7] > 0.5) & mk
        inter = (pred & on).flatten(1).sum(1).float()
        union = (pred | on).flatten(1).sum(1).float()
        res["iou"].append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).cpu().numpy())
        res["last"].append(last.cpu().numpy())
        res["last_e"].append(last_e.cpu().numpy())
        res["last_c"].append(last_c.cpu().numpy())
        if probe:
            dist = torch.where(on, a, torch.full_like(a, INF)).amin(1)  # [B,S,S]: the cell's nearest chord
            sel = (dist < INF) & (dist > 0)
            b_i, r_i, c_i = torch.nonzero(sel, as_tuple=True)
            cells["x"].append(state[b_i, N_FIXED:, r_i, c_i].cpu().numpy())
            cells["y"].append(codes(ev.rule[sl])[b_i.cpu().numpy()])
            cells["dist"].append(dist[b_i, r_i, c_i].cpu().numpy())
    last = np.concatenate(res["last"])
    out_d = {"exact": last < H, "settle": last + 1, "iou": np.concatenate(res["iou"])}
    if task == "m1b":
        out_d["edges"] = np.concatenate(res["last_e"]) < H
        out_d["closed"] = np.concatenate(res["last_c"]) < H
    if probe:
        out_d["cells"] = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in cells.items()}
    return out_d


def summarise(task, evs, results, tab):
    """The log's "q" dict from per-(set, level) results (see the module docstring)."""
    rnd = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)  # noqa: E731
    med = lambda x: rnd(np.nanmedian(x)) if np.isfinite(x).any() else None  # noqa: E731
    avg = lambda x: rnd(np.nanmean(x)) if np.isfinite(x).any() else None  # noqa: E731
    rows = []
    for ev, r in zip(evs, results):
        it, ex = ev.items, r["exact"]
        w = it["w"] / it["w"].sum()
        bk = it["bucket"]
        row = {"set": ev.name, "level": ev.level, "exact": float((w * ex).sum()),
               "balanced": float(np.mean([ex[bk == j].mean() for j in range(3) if (bk == j).any()])),
               "iou": float((w * r["iou"]).sum()),
               "byLen": {j: float(ex[bk == j].mean()) for j in range(3) if (bk == j).any()},
               "ex": ex, "w": w, "sub": it["subset"], "bk": bk,
               "ratio": np.where(ex, r["settle"] / np.maximum(1, it["ideal"]), np.nan),
               "excess": np.where(ex, r["settle"] - it["ideal"], np.nan)}
        if task == "m1b":
            row["edges"], row["closed"] = float((w * r["edges"]).sum()), float((w * r["closed"]).sum())
        rows.append(row)
    mean = lambda key, rs: rnd(np.mean([x[key] for x in rs])) if rs else None  # noqa: E731
    q = {"exact": mean("exact", rows), "balanced": mean("balanced", rows), "iou": mean("iou", rows),
         "steps": {"ratio": med(np.concatenate([x["ratio"] for x in rows])),
                   "excess": avg(np.concatenate([x["excess"] for x in rows]))}}
    if task == "m1b":
        q["parts"] = {"edges": mean("edges", rows), "closed": mean("closed", rows)}
    levels = sorted({x["level"] for x in rows})
    q["byLevel"] = {str(L): {k: mean(k, [x for x in rows if x["level"] == L]) for k in ("exact", "balanced", "iou")}
                    for L in levels}
    q["bySet"] = {}
    for name in dict.fromkeys(x["set"] for x in rows):
        rs = [x for x in rows if x["set"] == name]
        q["bySet"][name] = {"exact": mean("exact", rs), "balanced": mean("balanced", rs),
                            "byLevel": {str(x["level"]): {"exact": rnd(x["exact"]), "balanced": rnd(x["balanced"])}
                                        for x in rs},
                            "byLen": {BUCKETS[j]: rnd(np.mean([x["byLen"][j] for x in rs if j in x["byLen"]]))
                                      for j in range(3) if any(j in x["byLen"] for x in rs)}}
    ex, w, sub = (np.concatenate([x[k] for x in rows]) for k in ("ex", "w", "sub"))
    q["bySubset"] = {k: {"exact": rnd((w * ex)[sub == k].sum() / max(1e-12, w[sub == k].sum())),
                         "n": int((sub == k).sum())} for k in tab.keys if (sub == k).any()}
    q["byLen"] = {}
    for j, name in enumerate(BUCKETS):
        rs = [x for x in rows if j in x["byLen"]]
        if rs:
            ratio = np.concatenate([x["ratio"][x["bk"] == j] for x in rs])
            q["byLen"][name] = {"exact": rnd(np.mean([x["byLen"][j] for x in rs])), "steps": med(ratio)}
    q["levels"] = len(levels)
    return q


def trivial_of(evs):
    """Only the tapped chord drawn (exact iff the strand is one chord): per set and level."""
    out = {}
    for ev in evs:
        ok = ev.items["length"] == 1
        w = ev.items["w"] / ev.items["w"].sum()
        bk = ev.items["bucket"]
        out[f"{ev.name}{ev.level}"] = {"exact": round(float((w * ok).sum()), 4),
                                       "balanced": round(float(np.mean([ok[bk == j].mean() for j in range(3)
                                                                       if (bk == j).any()])), 4)}
    out["what"] = "only the tapped chord drawn (nothing closed)"
    return out


# ---------------------------------------------------------------- the code-fidelity probe

@torch.no_grad()
def probe_fit(pools, codes, max_cells=PROBE_CELLS, lam=1e-3, seed=0):
    """Ridge regression from the hidden channels of drawn strand cells of the training pools (not the tapped
    cell) to their slots' 53-bit codes. Returns (W [hidden+1, 53] float64, mean code)."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for P in pools.values():
        age = np.minimum(P["age"], AGE_CAP)[:, None, None, None]
        dist = np.where(P["a"] < age, P["a"], INF).min(1)  # drawn chords' distance per cell
        dist[np.arange(len(dist)), P["tap"][:, 0], P["tap"][:, 1]] = INF
        sl, r, c = np.nonzero(dist < INF)
        if not len(sl):
            continue
        k = rng.choice(len(sl), min(len(sl), max_cells // len(pools)), replace=False)
        sl, r, c = sl[k], r[k], c[k]
        st = P["state"]
        idx = torch.from_numpy(sl).to(st.device)
        xs.append(st[idx, N_FIXED:, torch.from_numpy(r).to(st.device), torch.from_numpy(c).to(st.device)]
                  .double().cpu())
        ys.append(torch.from_numpy(codes(P["rule"][sl])).double())
    if not xs:
        return None, None
    X, Y = torch.cat(xs), torch.cat(ys)
    Xb = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    A = Xb.T @ Xb
    A += lam * (A.diagonal().mean() + 1e-12) * torch.eye(A.shape[0], dtype=A.dtype)
    W = torch.linalg.solve(A, Xb.T @ Y)
    return W, Y.mean(0)


def probe_score(W, mean, cells):
    """code fidelity on held-out cells: see the module docstring."""
    x, y, dist = cells["x"], cells["y"], cells["dist"]
    if W is None or not len(x):
        return {"n": int(len(x))}
    pred = torch.cat([torch.from_numpy(x).double(), torch.ones(len(x), 1, dtype=torch.float64)], 1) @ W
    pred = pred.numpy()

    def decode(p):
        out = np.zeros(p.shape, np.uint8)
        out[:, :len(MAJORS)] = p[:, :len(MAJORS)] > 0.5
        dg = p[:, len(MAJORS):].reshape(len(p), N_TYPES, N_DIGITS).argmax(-1)
        out[np.arange(len(p))[:, None], len(MAJORS) + np.arange(N_TYPES)[None] * N_DIGITS + dg] = 1
        return out
    d, chance = decode(pred), decode(np.broadcast_to(mean.numpy(), pred.shape))
    ok = d == y
    cls, dig = ok[:, :len(MAJORS)].all(1), ok[:, len(MAJORS):].all(1)
    exact = cls & dig
    return {"bits": round(float(ok.mean()), 4), "classes": round(float(cls.mean()), 4),
            "digits": round(float(dig.mean()), 4), "exact": round(float(exact.mean()), 4),
            "byDist": {name: round(float(exact[(dist >= lo) & (dist <= hi)].mean()), 4)
                       for name, lo, hi in DIST_BINS if ((dist >= lo) & (dist <= hi)).any()},
            "chance": {"classes": round(float((chance == y)[:, :len(MAJORS)].all(1).mean()), 4),
                       "digits": round(float((chance == y)[:, len(MAJORS):].all(1).mean()), 4)},
            "n": int(len(x))}


# ---------------------------------------------------------------- plumbing

def ensure_dir(arg, name, marker):
    """A dataset dir: `arg` itself, or for a .tgz (tar -C data -czf X.tgz NAME) data/NAME, unpacked from it once
    (atomically: several runs may start at once)."""
    data_root = os.path.dirname(v2_default_dir())
    dest = os.path.join(data_root, name)
    if not arg:
        return dest
    if not arg.endswith((".tgz", ".tar.gz")):
        return arg
    if os.path.isfile(os.path.join(dest, marker)):
        return dest
    os.makedirs(data_root, exist_ok=True)
    tmp = os.path.join(data_root, f".{name}-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(arg) as t:
        try:
            t.extractall(tmp, filter="data")
        except TypeError:  # a Python without extraction filters
            t.extractall(tmp)
    src = os.path.join(tmp, name)
    if not os.path.isfile(os.path.join(src, marker)):
        raise SystemExit(f"{arg} has no {name}/{marker} (make it with: tar -C data -czf X.tgz {name})")
    try:
        os.rename(src, dest)
    except OSError:  # another run got there first
        if not os.path.isfile(os.path.join(dest, marker)):
            raise
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def pools_for_ckpt(pools, half):
    out = {}
    for L, P in pools.items():
        Q = dict(P)
        Q["state"] = P["state"].detach().cpu()
        if half:
            Q["state"] = Q["state"].half()
        out[L] = Q
    return out


def save_ckpt(path, model, opt, it, cfg, rng, pools=None, best=-1.0, eval_sec=0.0, guard=None):
    tmp = path + ".tmp"
    torch.save({"model": V1.to_cpu(model.state_dict()), "opt": V1.to_cpu(opt.state_dict()), "iteration": it,
                "config": cfg, "rng": rng.bit_generator.state,
                "pool": pools_for_ckpt(pools, cfg.get("ckptPool") == "half") if pools else None, "best": best,
                "evalSec": eval_sec, "steer": None, **(guard or {})}, tmp)
    os.replace(tmp, path)


def write_snapshot(path, it, pools, task, tab, last_level, last_idx):
    """pool.npz in nca.train's snapshot contract (train.write_snapshot's layout): walls = cells with chords under
    the slot's rule, fill = max of the task's planes, target = the strand's cells, state = channels 0..31."""
    out = {"iteration": np.int64(it), "radii": np.array(sorted(P["geo"].shape[-1] - 1 for P in pools.values())),
           "last_R": np.int64(pools[last_level]["geo"].shape[-1] - 1), "last_idx": np.asarray(last_idx, np.int64)}
    for P in pools.values():
        n, S = min(len(P["age"]), V1.SNAP_N), P["geo"].shape[-1]
        R, S2 = S - 1, 2 * S - 1

        def emb(x, dtype):
            y = np.zeros(x.shape[:-2] + (S2, S2), dtype)
            y[..., :S, S - 1:] = x
            return y
        st = P["state"][:n].float().cpu().numpy()
        walls = np.stack([tab.render_bits(int(P["rule"][i, 0]), P["rule"][i, 1:].astype(np.int64), P["geo"][i]) != 0
                          for i in range(n)])
        out[f"walls_{R}"] = emb(walls, np.uint8)
        out[f"mask_{R}"] = emb(P["geo"][:n] >= 0, np.uint8)
        out[f"fill_{R}"] = emb(st[:, OUT[task]].max(1), np.float16)
        out[f"target_{R}"] = emb((P["a"][:n] < INF).any(1), np.uint8)
        out[f"ntargets_{R}"] = np.ones(n, np.uint8)
        out[f"loss_{R}"] = P["loss"][:n].astype(np.float32)
        out[f"age_{R}"] = (it - P["born"][:n]).astype(np.int32)
        out[f"edits_{R}"] = P["edits"][:n].astype(np.int32)
        out[f"damage_{R}"] = P["last"][:n].astype(np.int8)
        out[f"state_{R}"] = emb(st[:min(n, V1.SNAP_M), :SNAP_C], np.float16)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, path)


# ---------------------------------------------------------------- main

def main(argv=None):
    t_start = time.time()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--inputs", choices=INPUTS, default="c", help="static input option (§3); c-bc, e-bc = diagnostics")
    p.add_argument("--task", choices=("m1a", "m1b"), default="m1a")
    p.add_argument("--levels", type=int, nargs="+", default=[2], help="training levels (2, 3, 4 = level-4 crops)")
    p.add_argument("--eval-levels", type=int, nargs="+", default=[3, 4], help="quick-check levels (held-out rules)")
    p.add_argument("--eval-sets", nargs="+", choices=("legacy", "wide"), default=["legacy", "wide"])
    p.add_argument("--data", default=None, help="data/strand-v2 (default) or a .tgz of it")
    p.add_argument("--legacy", default=None, help="data/strand (default) or a .tgz of it: the legacy eval set")
    p.add_argument("--steps-mult", type=float, nargs=2, default=[3, 6], metavar=("A", "B"),
                   help="steps per iteration ~ U[A*S, B*S], S = the level's board side")
    p.add_argument("--bptt", type=int, default=48, help="backprop through the last this many steps only")
    p.add_argument("--last-k", type=int, default=8, help="loss = mean over the last K steps")
    p.add_argument("--iters", type=int, default=None, help="total iterations (default 100000)")
    p.add_argument("--batch", type=int, default=32)
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
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--depth", type=int, default=1, help="hidden layers of the per-cell update MLP (1 = HexNCA's)")
    p.add_argument("--dir-groups", type=int, default=-1,
                   help="--inputs e, e-bc: hidden directional groups (6 channels each); default (channels - 13) // 12")
    p.add_argument("--clamp", type=float, nargs=2, default=[-2.0, 2.0], metavar=("LO", "HI"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", help="start from this checkpoint's weights (same --inputs/--channels/--hidden)")
    p.add_argument("--eval-every", type=int, default=200, help="a multiple of 50")
    p.add_argument("--eval-n", type=int, default=None, help="taps per (set, level) (default 120)")
    p.add_argument("--eval-mult", type=float, default=8, help="readout after max(mult*S, largest ideal + 8) steps")
    p.add_argument("--eval-cap", type=int, default=2000, help="... at most this many")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--no-probe", action="store_true", help="skip the code-fidelity probe in the quick check")
    p.add_argument("--minutes", type=float, default=0)
    p.add_argument("--schedule", choices=("iters", "time"), default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--ckpt-pool", choices=("full", "half"), default="full",
                   help="pool states in ckpt.pt as float32 (bit-exact resume) or float16 (a third the size)")
    p.add_argument("--snap-every", type=int, default=50, help="pool.npz every this many iterations; 0 = never")
    p.add_argument("--force-collapse", type=int, default=0, help=argparse.SUPPRESS)  # selftest: collapse at this check
    args = p.parse_args(argv)
    if args.eval_every % 50:
        p.error("--eval-every must be a multiple of 50 (the log's period)")
    if args.schedule == "time" and not args.minutes and not args.resume:
        p.error("--schedule time needs --minutes")

    torch.set_num_threads(args.threads or min(4, os.cpu_count() or 1))
    device = V1.pick_device(args.device)
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
        for k, v in (("lr", args.lr), ("data", args.data), ("legacy", args.legacy)):
            if v:
                cfg[k] = v
        if args.schedule and args.schedule != cfg["schedule"]:
            cfg["schedule"] = args.schedule
            if args.schedule == "time" and not cfg.get("scheduleMin"):
                if not args.minutes:
                    p.error("--schedule time needs --minutes")
                cfg["scheduleMin"] = args.minutes
    else:
        n_in = n_inputs(args.inputs)
        if args.init:
            init = torch.load(args.init, map_location="cpu", weights_only=False)
            ic = init["config"]
            if (ic["channels"], ic["hidden"], ic.get("depth", 1), ic.get("nIn"), ic.get("inputs"),
                    ic.get("dirGroups")) != (args.channels, args.hidden, args.depth, n_in, args.inputs,
                                             dir_groups_of(args.channels, args.dir_groups)):
                p.error(f"--init {args.init} has inputs={ic.get('inputs')} channels={ic['channels']} "
                        f"hidden={ic['hidden']} depth={ic.get('depth', 1)} nIn={ic.get('nIn')}; this run asks for "
                        f"--inputs {args.inputs} --channels {args.channels} --hidden {args.hidden} --depth "
                        f"{args.depth} ({n_in} consts)")
        if args.channels <= N_FIXED:
            p.error(f"--channels must be > {N_FIXED}")
        if framed(args.inputs) and N_FIXED + 6 * dir_groups_of(args.channels, args.dir_groups) > args.channels:
            p.error("--dir-groups: 13 + 6 x groups must fit in --channels")
        cfg = {"task": args.task, "inputs": args.inputs, "levels": sorted(args.levels),
               "evalLevels": sorted(args.eval_levels), "evalSets": list(args.eval_sets), "data": args.data,
               "legacy": args.legacy, "channels": args.channels, "hidden": args.hidden, "depth": args.depth,
               "dirGroups": dir_groups_of(args.channels, args.dir_groups),
               "nIn": n_in, "perception": "taps",
               "clamp": init["config"]["clamp"] if init else list(args.clamp), "fireRate": 1.0,
               "stepsMult": list(args.steps_mult), "bptt": args.bptt, "lastK": args.last_k,
               "lr": args.lr or LR, "lrFloor": args.lr_floor, "warmup": args.warmup,
               "collapseFrac": args.collapse_frac, "collapseLossX": args.collapse_loss_x,
               "maxRollbacks": args.max_rollbacks, "batch": args.batch, "iters": args.iters or 100000,
               "seed": args.seed, "poolSize": args.pool_size, "damage": args.damage, "damageKinds": list(DAMAGE),
               "evalMult": args.eval_mult, "evalCap": args.eval_cap, "probe": not args.no_probe,
               "schedule": args.schedule or "iters", "scheduleMin": args.minutes if args.schedule == "time" else None,
               "ckptPool": args.ckpt_pool, "init": args.init,
               "prevIterations": (init["config"].get("prevIterations", 0) + init["iteration"]) if init else 0,
               "loss": "don't-care-weighted MSE of the task's planes over board cells, mean of the last lastK steps"}
    task, levels, C = cfg["task"], cfg["levels"], cfg["channels"]
    eval_n = cfg["evalN1"] = args.eval_n or cfg.get("evalN1", 120)
    data_dir = ensure_dir(cfg.get("data"), "strand-v2", "rules-hex.json")
    tab, bd = RuleTable(data_dir), GeoBoards(data_dir)
    codes = Codes(tab)

    torch.manual_seed(cfg["seed"])
    model = make_model(C, cfg["hidden"], cfg["clamp"], cfg["nIn"], cfg.get("depth", 1), cfg["inputs"],
                       cfg.get("dirGroups"))
    rng = np.random.default_rng(cfg["seed"])
    if ckpt:
        model.load_state_dict(ckpt["model"])
        rng.bit_generator.state = ckpt["rng"]
    elif init:
        model.load_state_dict(init["model"])
    model.to(device)
    planes = Planes(tab, cfg["inputs"], device)
    assert planes.n_in == cfg["nIn"]
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
    lr_of = lambda i: V1.lr_at(i, cfg["iters"], cfg["lr"], lr_scale, cfg["lrFloor"], warm_from, cfg["warmup"],  # noqa: E731
                               sched_frac())

    side = {L: int(bd.hw[GROUP[L]].max()) for L in levels}
    evs = []
    for name in cfg["evalSets"]:
        for L in cfg["evalLevels"]:
            if name == "legacy":
                legacy_dir = ensure_dir(cfg.get("legacy"), "strand", "meta.json")
                evs.append(legacy_set(task, tab, legacy_dir, L, eval_n, cfg["evalMult"], cfg["evalCap"]))
            else:
                evs.append(wide_set(task, tab, bd, L, eval_n, cfg["evalMult"], cfg["evalCap"]))
    stepper = lambda st, w, cs, t, sl: model.step(st, w, cs)  # noqa: E731

    def check():
        res = [evaluate(stepper, planes, ev, C, device, args.eval_batch, codes, probe=cfg["probe"]) for ev in evs]
        q = summarise(task, evs, res, tab)
        if cfg["probe"]:
            W, mean = probe_fit(pools, codes, seed=cfg["seed"])
            cells = {k: np.concatenate([r["cells"][k] for r in res]) for k in ("x", "y", "dist")}
            q["code"] = probe_score(W, mean, cells)
        return {"q": q, "exact": q["exact"], "score": q["balanced"], "evalN": int(sum(len(ev.geo) for ev in evs))}

    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    def emit(rec):
        rec["time"], rec["device"] = round(time.time(), 2), device.type
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if ckpt and ckpt.get("pool"):
        pools = {int(L): P for L, P in ckpt["pool"].items()}
        for P in pools.values():
            P["state"] = P["state"].float().to(device)
    else:
        pools = {L: new_pool(rng, tab, bd, GROUP[L], side[L], cfg["poolSize"], C, task, device) for L in levels}
        for P in pools.values():
            P["born"][:] = start_it
    held = [int(tab.held_out(s).sum()) for s in range(tab.n_sub)]
    start = {"config": cfg, "startIteration": start_it, "evalN": int(sum(len(ev.geo) for ev in evs)),
             "threads": torch.get_num_threads(), "torch": torch.__version__, "lrScale": lr_scale,
             "rollbacks": rollbacks, "params": sum(x.numel() for x in model.parameters()),
             "boardSide": {str(L): S for L, S in side.items()},
             "evalSets": {f"{ev.name}{ev.level}": ev.counts() for ev in evs}, "trivial": trivial_of(evs),
             "rules": {"subsets": tab.keys, "count": [int(x) for x in tab.count], "heldout": held,
                       "legacyHeldout": len(tab.legacy_heldout())},
             "schedule": {"mode": cfg["schedule"], "decayAt": list(V1.DECAY_AT),
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
            write_snapshot(snap_path, it, pools, task, tab, *last[0])
        except Exception as e:  # noqa: BLE001 -- never worth a training run
            emit({"iteration": it, "snapshotError": repr(e)})

    best = ckpt.get("best", -1.0) if ckpt else -1.0
    on_disk = V1.best_score(run_dir) if ckpt else None
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

    n_planes = OUT[task].stop - OUT[task].start
    t_win, losses, t_data = time.time(), [], []
    kinds = dict.fromkeys(DAMAGE + ("new", "worst"), 0)
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
        P = pools[L]
        S = side[L]
        n_pool = len(P["age"])
        idx = rng.choice(n_pool, min(B, n_pool), replace=False)
        T = int(rng.integers(round(a_mult * S), max(round(a_mult * S), round(b_mult * S)) + 1))
        G = min(T, cfg["bptt"])
        K = min(cfg["lastK"], G)
        tidx = torch.from_numpy(idx).to(device)
        state = P["state"][tidx].clone()
        t_ev = time.time()
        if T > G:
            cs_pre = planes(P["geo"][idx], codes(P["rule"][idx]), P["tap"][idx])
            t_pre = time.time()
            with torch.no_grad():
                for _ in range(T - G):
                    state = model.step(state, cs_pre[:, :1], cs_pre)
            P["age"][idx] = np.minimum(P["age"][idx] + (T - G), AGE_CAP)
            t_ev += time.time() - t_pre  # only the input building counts as data time
        # ---- the events, at the start of the gradient window: new boards, the worst slot's restart, damage
        cur = np.nan_to_num(P["loss"][idx], nan=-1.0)
        j_worst = int(cur.argmax())
        new = set(rng.choice(len(idx), max(1, len(idx) // 8), replace=False).tolist())
        restart = new | {j_worst}
        discs = []
        for pos, i in enumerate(idx):
            if pos in new:
                put(P, i, new_slot(rng, tab, bd, GROUP[L], S, task))
                kinds["new"] += 1
            elif pos == j_worst:
                P["b"][i] = -1  # from the fresh state: nothing drawn, nothing to erase
                kinds["worst"] += 1
            elif rng.random() < cfg["damage"]:
                kind = DAMAGE[int(rng.integers(len(DAMAGE)))]
                ok, disc = damage(rng, tab, task, P, i, kind)
                if ok:
                    kinds[kind] += 1
                    P["edits"][i] += 1
                    P["last"][i] = DAMAGE.index(kind)
                    if disc is not None:
                        discs.append((pos, disc))
            if pos in restart:
                P["age"][i], P["born"][i], P["edits"][i], P["last"][i] = 0, it, 0, -1
        cs = planes(P["geo"][idx], codes(P["rule"][idx]), P["tap"][idx])
        a = torch.from_numpy(P["a"][idx]).to(device).float()
        b = torch.from_numpy(P["b"][idx]).to(device).float()
        oc = torch.from_numpy(P["oc"][idx]).to(device).float()
        closed = torch.from_numpy(P["closed"][idx]).to(device)
        rs = torch.tensor(sorted(restart), device=device)
        state[rs] = fresh(cs[rs, :1], C)
        for pos, disc in discs:
            state[pos, 1:] *= torch.from_numpy(~disc).to(device).float()
        state[:, 0:1] = cs[:, 0:1]
        age0 = torch.from_numpy(P["age"][idx]).to(device).float()
        walls, mk = cs[:, :1], cs[:, :1]
        ncell = n_planes * mk.sum((1, 2, 3)).clamp(min=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_data.append(time.time() - t_ev)
        # ---- the gradient window: G steps, the loss over the last K, each against its own age's target
        acc = 0.0
        for t in range(G):
            state = model.step(state, walls, cs)
            if t >= G - K:
                age = (age0 + t + 1).clamp(max=AGE_CAP).view(-1, 1, 1, 1)
                acc = acc + step_loss(state, task, a, b, oc, closed, age, mk, ncell) / K
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
            P["state"][tidx] = fresh(cs[:, :1], C)
            P["age"][idx], P["born"][idx], P["loss"][idx] = 0, it, np.nan
        it += 1
        win_steps += 1
        last[0] = (L, idx)
        slowest = max(slowest, time.time() - t_it)
        if skip_run >= V1.STOP_SKIPS:
            emit({"iteration": it, "stopped": f"{V1.STOP_SKIPS} non-finite steps in a row", "skipped": skipped})
            raise SystemExit(f"{V1.STOP_SKIPS} non-finite steps in a row; the last checkpoint is {ckpt_path}")

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
            why = V1.collapse_reason(None, best, cfg["collapseFrac"], rec["loss"], history, win_skipped, win_steps,
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
                why = why or V1.collapse_reason(rec["score"], best, cfg["collapseFrac"], 0.0, [], 0, 1)
            emit(rec)
            if rec["loss"] is not None and math.isfinite(rec["loss"]):
                history.append(rec["loss"])
            t_win, losses, t_data = time.time(), [], []
            kinds = dict.fromkeys(kinds, 0)
            win_skipped = win_steps = 0
            if why:
                back = V1.best_score(run_dir)
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
                    Pp["state"] = fresh(torch.from_numpy(Pp["geo"] >= 0).float()[:, None], C).to(device)
                    Pp["age"][:], Pp["born"][:], Pp["edits"][:], Pp["last"][:], Pp["loss"][:] = 0, it, 0, -1, np.nan
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
