"""Train the strand CA with the tap as a one-time event: lines that last, lines that die
(docs/spectacle-nca-taps.md; train2.py's successor, which stays as it was for launches 6-9).

    python -m nca.strand.train3 --name i-t1l2 --tap impulse --stage T1 --levels 2 --minutes 45 \\
        --channels 96 --hidden 256 --depth 2 --init runs/tap-e2-w-l3/best.pt
    python -m nca.strand.train3 --name i-t2 --tap impulse --stage T2 --levels 2 3 --init runs/i-t1l3/best.pt
    python -m nca.strand.train3 --name pi-i --tap impulse --overfit 4 --levels 2 --channels 64 --dir-groups 1

Everything not said here is train2.py's (its docstring): the inputs (default --inputs e, nets.FrameNCA; the consts
layout is train2's, so a train2 checkpoint loads with --init), the pool with ages, the worst slot's restart and
fresh slots, truncated backprop (--bptt, the loss over the last --last-k steps), --schedule time, the collapse
guard and its rollbacks, --resume (bit-exact), --init, --ckpt-every, --ckpt-pool, pool.npz and gallery.npz for
the dashboard, the quick check's legacy / wide sets, q.exit and the code probe. Only m1a (the six edge planes).

THE TAP (--tap; §1, decision 1). A tap is (cell, chord, rule, t): it happens at age t of its slot (episodes.py's
conventions) and is then gone. Nothing lingers: a line must keep itself alive.
  impulse  the tap's 59 planes (code 53, the chord's two edges 6; train2's tap planes) are consts on the tapped
           cell for --tap-steps steps (t + 1 .. t + tapSteps), then zero
  fixed    no tap planes ever (they stay zero, so the consts layout is train2's): at age t, before step t + 1, the
           host writes the tapped cell's state: edge planes 1 + d0, 1 + d1 <- max(itself, 1) (additive: a tap
           beside a line of one's own keeps it, decision 7), and the code in its 35-plane form (8 class bits, then
           each of the 9 digits as 3 bits, most significant first; +-1) into the last 35 channels, which must be
           scalar (FrameNCA reads its directional groups rotated by the cell's frame): 13 + 6 x dirGroups <= C - 35
  held     train2's: the tap's planes from age t on, every step (the baseline the overfits compare with)
TARGETS (§2; nca/strand/sim.py owns them, episodes.py adapts): per edge up to 4 intervals [on_at, off_at) in
  nominal steps (sim's planes: chord k of a tap at age t at t + speed x k; a wipe at the step the dying wave
  reaches it; INF never; a don't-care-only interval where the timing is uncertain). --speed = sim's period, CA
  steps per chord: 2 (default) is decision 4's cap; 1 is train2's growth. The target at age a is sim.target_weight's:
  1 from on_at + slack to off_at, don't-care from on_at - early and from off_at to off_at + slack, 0 everywhere
  else (--slack, --early: sim's 2 and 1; --early -1: no not-before, an edge is don't-care until due, train2's
  light cone; so --speed 1 --slack 1 --early -1 is train2's target exactly for one tap). A slot's ideal = sim's
  settle (the last change) + slack. Loss: the don't-care-weighted MSE of the six edge planes / (6 x board cells),
  the mean over the last --last-k steps of the window, each step against its own age.
EPISODES (§4.3-4.4; sim.draw_taps through episodes.draw: taps the host refuses are dropped, the CA never sees
  them). A slot holds up to 4 taps with their times. Kinds: t1 one tap at age 0 (sim "single"); t2 2-4 taps of
  different codes (sim "collide": each later one on a cell the first line will pass, before it gets there; times
  <= --stagger, default bptt / 2), drawn again unless a line is hit by --hit-max (default bptt), or with probability
  --control a control pair (sim "control": two codes whose strands share no cell); t3 2-3 taps of one code (sim
  "own": a join ahead of the first tip, or a chord beside the first line), drawn again unless two taps stood.
STAGES (--stage; §4.4) set the mix of new slots: T1 all t1; T2 t2 (1 - remix), t1 remix; T3 t3 (1 - remix), t1 and
  t2 remix / 2 each (--remix, default 1/3: each stage re-mixes a third of its slots from the earlier ones);
  --mix t1=.5,t3=.5 overrides it.
POOL: train2's, with two knobs for how old slots get (§5's "longer pool ages"): --new-share (1/8 of each batch made
  new) and --no-worst-restart (train2 restarts the batch's worst slot every iteration, so a slot that drifts at an
  age it was never trained at is reset before it is trained there).
DAMAGE (decision 6; train2's state / move / edit are gone): --noise P (default 0.3) per kept slot: Gaussian noise
  of sigma U[--noise-sigma] (0.1-0.3) on the hidden channels (13..) of its line cells (cells with an edge due now),
  the target unchanged; --midtap P (default 0 in T1, 0.1 in T2 / T3) per kept slot: a new tap at an age in the
  window's first half (episodes.midtap; T2 a rival on a strand through its lines, T3 one of its own code or, half
  the time, a rival) -- the targets re-derived for the whole episode (the past is unchanged); plus the collisions
  and staggered taps of the episodes themselves.
--overfit K (§1's Pi test): one train rule (--stage T1; T2: one pair of rules), K episodes on level-2 boards of
  --overfit-len chords (default 4-24); new slots draw from those K and the quick check rolls the same K: exact as
  train2 reads it (after max(evalMult x S, ideal + 8) steps), onTime (at each one's ideal), and at the ages
  --persist-ages (default 100 300 600 here); steps = the settle step / ideal; the score is exact.
QUICK CHECK (held-out rules): train2's legacy and wide sets with the tap at age 0 under --tap, targets at --speed
  (ideal = 1 + speed x the strand's growth steps + slack; read-out after max(evalMult x S, largest ideal + 8)):
  exact, balanced, iou, steps, byLevel, bySet, bySubset, byLen, exit, trivial -- as train2's. New (§4.5):
  speed    {rate, n}: per way out of each tap with >= 9 chords, (8 - 1) / (the age the drawn run from the tap first
           reaches 8 chords - the age it first reaches 1); expect 1 / --speed
  persist  {x1, x2, x4, ratio, n}: a wide set (--persist-n taps per --persist-levels level whose 4 x ideal <=
           --persist-cap)
           rolled to 4 x ideal: exact at 1 x, 2 x, 4 x its own ideal (the tap long gone); ratio = x4 / x1
  collide  {exact, exactHit, exactCtrl, wipe, grown, falseWipe, regrow, worm, wormMax, n, nHit, byLevel}
           (--eval-sets collide; default in T2 / T3): --episode-n t2 episodes per --episode-levels level (a
           --control share of them controls; ideal + 2 S <= --eval-cap), rolled to rest = ideal + 2 S. wipe: of
           the edges due to go that the model drew while they were due, the share off at ideal (a model that draws
           nothing has none: None, not 1); grown: the share of the doomed edges it drew at all while due;
           falseWipe: of the edges of lines nobody hit that were drawn while due, the share off at rest; regrow: of
           the wiped edges, the share drawn at any age in (ideal, rest]; worm: chords of hit lines first drawn
           after the first hit, per episode with a hit (wormMax: the most the targets allow, chords due after the
           hit less the slack: what a fleeing tip may still lay); exact: the drawing = the targets at ideal
  own      {exact, collateral, phantom, stray, n} (--eval-sets own; default in T3): t3 episodes. collateral: of the
           union's edges drawn while due, the share off at rest (an own meeting taken for a hit); phantom: the share
           of episodes whose drawing still changes after ideal (a tip still running: the CA has no tip output);
           stray: edges drawn at rest off the union, per union edge
  code     train2's linear probe, fitted on the pools' line cells (the code of each cell's own line) + byAge
           {x1, x2, x4}: the probe on the persist set's line cells at 1 x / 2 x / 4 x ideal
  codeMlp  the same with a 2-layer MLP (hidden 256, --mlp-steps Adam steps; class bits by sigmoid, digits by
           softmax): if it reads the code whole where the linear one does not, the code is there and the readout
           is the problem (§3); off with --no-mlp-probe
  score (best.pt, the collapse guard) = the mean of balanced, persist.x4, collide.wipe x (1 - collide.falseWipe)
  and own.exact, those that are measured.
LOG: train2's fields; damage counts {noise, midtap, new, worst}; pool {settled, age, kinds}. pool.npz: train2's
  contract (the target = the edges due at the slot's age, the tap = its first). The checkpoint's config carries
  tap / tapSteps / speed / slack / early / fixedChannels for nca/strand/export.py (the play page's tap mode).
"""

import argparse
import json
import math
import os
import resource
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import episodes as E
from . import sim as SIM
from . import train as V1
from . import train2 as T2
from .nets import FrameNCA
from .rules import CODE_BITS, MAJORS, N_DIGITS, N_TYPES, Boards as GeoBoards, RuleTable
from .walker import PAIRS

TAP_MODES = ("impulse", "fixed", "held")
STAGES = ("T1", "T2", "T3")
KINDS = ("t1", "t2", "ctrl", "t3")       # a slot's kind, stored as its index + 1 (0: none)
N_FIXED, TAP, INF, AGE_CAP = T2.N_FIXED, T2.TAP, E.INF, V1.AGE_CAP
CODE35 = len(MAJORS) + 3 * N_TYPES       # the fixed write's code planes
SPEED_M = 8                              # the speed metric's run length (chords)
DAMAGE = ("noise", "midtap")
GROUP, EVAL_GROUP = T2.GROUP, T2.EVAL_GROUP
SNAP_C = 32


# ---------------------------------------------------------------- the model's step, the tap's write

class Runner:
    """The model's step with the FrameNCA gather plan computed once per board batch (the consts change from step
    to step while a tap's impulse is on, the frames never), and the --tap mode's write."""

    def __init__(self, model, cfg, device):
        self.model, self.cfg, self.device = model, cfg, device
        self.framed = isinstance(model, FrameNCA)
        self.mode, self.tap_steps = cfg["tap"], int(cfg.get("tapSteps") or 1)
        self.C = cfg["channels"]
        self.code_ch = torch.tensor(cfg["fixedChannels"], device=device) if cfg.get("fixedChannels") else None

    def plan(self, cs, B):
        return self.model.plan(cs, B) if self.framed else None

    def step(self, state, walls, cs, plan, age_before=None, events=None):
        """One step (age_before / events: for selftest3's oracles, which draw the ideal schedule)."""
        m = self.model
        if not self.framed:
            return m.step(state, walls, cs)
        B = state.shape[0]  # FrameNCA.step with the plan given (nets.py unchanged; selftest3 checks they agree)
        cin = cs[:, :-1]
        x = torch.cat([state, cin.expand(B, -1, -1, -1)], 1)
        state = state + m.update(x, plan)
        if m.clamp is not None:
            state = state.clamp(m.clamp[0], m.clamp[1])
        state = torch.cat([walls.expand(B, -1, -1, -1), state[:, 1:]], 1)
        return state * cin[:, :1]


def fixed_channels(channels, dir_groups):
    """The fixed write's code channels: the last 35, if they are scalar (after ch1-12 and the directional groups)."""
    lo = channels - CODE35
    return list(range(lo, channels)) if lo >= N_FIXED + 6 * dir_groups else None


def code35(rules):
    """float32 [n, 35] (+-1): the fixed write's code: the 8 class bits, then each digit as 3 bits (MSB first)."""
    rules = np.asarray(rules, np.int64).reshape(-1, 1 + N_TYPES)
    out = np.zeros((len(rules), CODE35), np.float32)
    out[:, :len(MAJORS)] = CODES[0](rules)[:, :len(MAJORS)]
    for j in range(3):
        out[:, len(MAJORS) + j::3] = (rules[:, 1:] >> (2 - j)) & 1
    return 2 * out - 1


CODES = [None]  # the run's Codes (rules -> 53 bits), set in main (and by the selftest)


def base_consts(planes, geo):
    """train2.Planes' consts with no tap anywhere: [B, n_in (+frame), S, S]."""
    B = len(geo)
    cs = planes(geo, np.zeros((B, CODE_BITS), np.uint8), np.zeros((B, 4), np.int64))
    cs[:, planes.tap_at:planes.tap_at + TAP] = 0
    return cs


class Events:
    """The taps of a batch of slots (int16 [B, MAXT, 15], episodes.TAP_COLS) as events by age, under --tap."""

    def __init__(self, taps, runner, planes, S):
        dev = runner.device
        taps = np.asarray(taps)
        self.taps = taps
        self.runner, self.planes, self.S = runner, planes, S
        self.B = len(taps)
        self.valid = taps[..., 0] >= 0
        self.t = np.where(self.valid, taps[..., 14].astype(np.int64), 1 << 40)
        rows, cols = np.maximum(taps[..., 0], 0).astype(np.int64), np.maximum(taps[..., 1], 0).astype(np.int64)
        self.rc = (rows, cols)
        self.pos = rows * S + cols
        self.d0, self.d1 = np.maximum(taps[..., 2], 0).astype(np.int64), np.maximum(taps[..., 3], 0).astype(np.int64)
        flat = taps[..., 4:14].reshape(-1, 1 + N_TYPES).astype(np.int64)
        flat = np.where(flat >= 0, flat, 0)
        vals = np.zeros((self.B * taps.shape[1], TAP), np.float32)
        vals[:, :CODE_BITS] = CODES[0](flat)
        ix = np.arange(len(flat))
        vals[ix, CODE_BITS + self.d0.ravel()] = 1
        vals[ix, CODE_BITS + self.d1.ravel()] = 1
        self.vals = torch.from_numpy(vals.reshape(self.B, -1, TAP)).to(dev)
        self.c35 = torch.from_numpy(code35(flat).reshape(self.B, -1, CODE35)).to(dev) if runner.mode == "fixed" else None
        self._cache = {}

    def active(self, age_before):
        """bool [B, MAXT]: the taps whose planes are on in the step from age_before to age_before + 1."""
        a = np.asarray(age_before)[:, None]
        if self.runner.mode == "impulse":
            return self.valid & (self.t <= a) & (self.t > a - self.runner.tap_steps)
        if self.runner.mode == "held":
            return self.valid & (self.t <= a)
        return np.zeros_like(self.valid)

    def consts(self, cs, age_before):
        act = self.active(age_before)
        if not act.any():
            return cs
        key = act.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        b, i = np.nonzero(act)
        S2 = self.S * self.S
        tp = torch.zeros(self.B, S2, TAP, device=cs.device)
        bt = torch.from_numpy(b).to(cs.device)
        tp.index_put_((bt, torch.from_numpy(self.pos[b, i]).to(cs.device)), self.vals[bt, torch.from_numpy(i).to(cs.device)],
                      accumulate=True)
        at = self.planes.tap_at
        out = torch.cat([cs[:, :at], tp.permute(0, 2, 1).reshape(self.B, TAP, self.S, self.S), cs[:, at + TAP:]], 1)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = out
        return out

    def write(self, state, age_before):
        """--tap fixed: the state with the taps due at age_before written (a new tensor; unchanged if none)."""
        if self.runner.mode != "fixed":
            return state
        w = self.valid & (self.t == np.asarray(age_before)[:, None])
        if not w.any():
            return state
        edge = torch.zeros_like(state, dtype=torch.bool)
        code = torch.zeros_like(state, dtype=torch.bool)
        val = torch.zeros_like(state)
        ch = self.runner.code_ch
        for b, i in zip(*np.nonzero(w)):
            r, c = int(self.rc[0][b, i]), int(self.rc[1][b, i])
            edge[b, 1 + int(self.d0[b, i]), r, c] = edge[b, 1 + int(self.d1[b, i]), r, c] = True
            code[b, ch, r, c] = True
            val[b, ch, r, c] = self.c35[b, i]
        # out of place: the state may be mid-window (autograd); max(itself, 1) on the edges, the code written
        return torch.where(code, val, torch.where(edge, state.clamp(min=1.0), state))

    def step(self, state, walls, cs, plan, age_before):
        state = self.write(state, age_before)
        return self.runner.step(state, walls, self.consts(cs, age_before), plan, age_before, self)


# ---------------------------------------------------------------- targets and loss

def target_weight(on, off, age, slack, early):
    """sim.target_weight: (target, care) bool [..., 6, S, S] at `age` from interval planes [..., K, 6, S, S] (torch
    float or numpy; INF never; age broadcasts, e.g. [B,1,1,1,1]): an interval wants the edge drawn from on + slack
    until off, is don't-care from on - early to on + slack and from off to off + slack; 0 everywhere else. One
    extension: early < 0, no not-before (an edge with an interval is don't-care until it is due: train2's light
    cone)."""
    one = (on + slack <= age) & (age < off)
    grow = ((on - early <= age) if early >= 0 else (on < INF)) & (age < on + slack)
    dc = grow | ((off <= age) & (age < off + slack))
    tgt = one.any(-4)
    return tgt, tgt | ~dc.any(-4)


def step_loss(state, on, off, age, slack, early, mk, ncell):
    tgt, care = target_weight(on, off, age, slack, early)
    return (((state[:, 1:7] - tgt.float()) ** 2) * care.float() * mk).sum((1, 2, 3)) / ncell


def due_np(on, off, age, slack, early=1):
    """bool [..., 6, S, S]: the edges due (target 1) at `age` (a number, or one per leading row)."""
    a = np.asarray(age).reshape((-1,) + (1,) * (on.ndim - 1)) if np.ndim(age) else age
    return target_weight(on.astype(np.int32), off.astype(np.int32), a, slack, early)[0]


# ---------------------------------------------------------------- slots

SLOT_KEYS = ("geo", "taps", "on", "off", "ideal", "kind", "length", "hit", "codes", "src")


def slot_of(sl, geo, src, slack):
    """A pool slot (dict) from an episodes.Slot."""
    return {"geo": geo, "taps": E.taps_array(sl.taps), "on": sl.on, "off": sl.off, "ideal": sl.settle + slack,
            "kind": KINDS.index(sl.kind) + 1, "length": sl.length, "hit": sl.hit, "codes": sl.codes, "src": int(src)}


class Generator:
    """New episodes of each kind for a level's boards (or the --overfit list)."""

    def __init__(self, tab, bd, cfg, group, S, split="train"):
        self.tab, self.bd, self.cfg, self.group, self.S, self.split = tab, bd, cfg, group, S, split
        self.fixed = None

    def board(self, rng):
        i = int(rng.integers(len(self.bd.hw[self.group])))
        return i, T2.pad_geo(self.bd.board(self.group, i), self.S)

    def episode(self, rng, kind):
        cfg = self.cfg
        if kind == "t2" and rng.random() < cfg["control"]:
            kind = "ctrl"
        for _ in range(100):
            i, geo = self.board(rng)
            sl = E.draw(rng, self.tab, geo, kind, cfg["speed"], cfg["slack"], self.split, cfg["stagger"], cfg["hitMax"],
                        hit_min=cfg.get("hitMin", 0))
            if sl is not None:
                return sl, geo, i
        raise RuntimeError(f"no {kind} episode in 100 boards of {self.group}")

    def new_slot(self, rng, kind):
        if self.fixed is not None:
            sl, geo, i = self.fixed[int(rng.integers(len(self.fixed)))]
        else:
            sl, geo, i = self.episode(rng, kind)
        return slot_of(sl, geo, i, self.cfg["slack"])


def overfit_episodes(tab, bd, cfg, K, S, seed):
    """§1's Pi test: K episodes on level-2 boards, each of --overfit-len chords (every line's, together): one train
    rule's single taps (T1), one pair of train rules colliding (T2) or one rule's own meetings (T3)."""
    lo, hi = cfg.get("overfitLen") or (4, 1 << 30)
    rng = np.random.default_rng(seed)
    group = GROUP[2]
    nb = len(bd.hw[group])
    kind = {"T1": "t1", "T2": "t2", "T3": "t3"}[cfg["stage"]]
    for _ in range(200):
        s, digits = tab.sample(rng, "train")
        s2, digits2 = tab.sample(rng, "train")
        rules = [SIM.rule_key((s, *digits)), SIM.rule_key((s2, *digits2))]
        if rules[0] == rules[1]:
            continue
        out, seen = [], set()
        for _ in range(40 * K):
            if len(out) == K:
                break
            i = int(rng.integers(nb))
            geo = T2.pad_geo(bd.board(group, i), S)
            sl = E.draw(rng, tab, geo, kind, cfg["speed"], cfg["slack"], "train", cfg["stagger"], cfg["hitMax"],
                        rules=rules[:1] if kind != "t2" else rules, tries=3, n=2 if kind == "t2" else None,
                        hit_min=cfg.get("hitMin", 0))
            if sl is None or not lo <= sl.length <= hi:
                continue
            key = (i, tuple(tuple(tp) for tp in sl.taps))
            if key not in seen:
                seen.add(key)
                out.append((sl, geo, i))
        if len(out) == K:
            return out
    raise RuntimeError("--overfit: no rule with enough episodes")


def new_pool(rng, gens, mix, n, channels, device):
    slots = [gens.new_slot(rng, pick_kind(rng, mix)) for _ in range(n)]
    P = {k: np.stack([np.asarray(s[k]) for s in slots]) for k in SLOT_KEYS}
    P.update({"age": np.zeros(n, np.int64), "loss": np.full(n, np.nan, np.float32), "born": np.zeros(n, np.int64),
              "edits": np.zeros(n, np.int32), "last": np.full(n, -1, np.int8)})
    P["state"] = T2.fresh(torch.from_numpy(P["geo"] >= 0).float()[:, None], channels).to(device)
    return P


def put(P, i, slot):
    for k in SLOT_KEYS:
        P[k][i] = slot[k]


def pick_kind(rng, mix):
    kinds, w = zip(*mix.items())
    return kinds[int(rng.choice(len(kinds), p=np.asarray(w) / sum(w)))]


def stage_mix(stage, remix, override=None):
    if override:
        out = {}
        for part in override.split(","):
            k, v = part.split("=")
            if k not in ("t1", "t2", "t3"):
                raise ValueError(f"--mix: unknown kind {k}")
            out[k] = float(v)
        return {k: v for k, v in out.items() if v > 0}
    if stage == "T1":
        return {"t1": 1.0}
    if stage == "T2":
        return {k: v for k, v in {"t2": 1 - remix, "t1": remix}.items() if v > 0}
    return {k: v for k, v in {"t3": 1 - remix, "t1": remix / 2, "t2": remix / 2}.items() if v > 0}


def noise_damage(rng, P, i, pos, state, cfg):
    """Gaussian noise on the hidden channels of slot i's line cells (an edge due at its age); in place on the batch
    state row `pos`. Returns whether anything was damaged."""
    due = due_np(P["on"][i], P["off"][i], int(P["age"][i]), cfg["slack"], cfg["early"]).any(0) & (P["geo"][i] >= 0)
    r, c = np.nonzero(due)
    if not len(r):
        return False
    lo, hi = cfg["noiseSigma"]
    sigma = float(rng.uniform(lo, hi))
    C = state.shape[1]
    z = rng.standard_normal((C - N_FIXED, len(r))).astype(np.float32) * sigma
    dev = state.device
    rt, ct = torch.from_numpy(r).to(dev), torch.from_numpy(c).to(dev)
    state[pos, N_FIXED:, rt, ct] += torch.from_numpy(z).to(dev)
    return True


def midtap_damage(rng, tab, P, i, age_t, cfg, stage):
    """A new tap on kept slot i at age age_t (episodes.midtap); the slot's targets re-simulated. Returns whether it
    was applied."""
    taps = E.taps_of(P["taps"][i])
    rival = stage == "T2" or (stage == "T3" and rng.random() < 0.5)
    sl = E.midtap(rng, tab, P["geo"][i], taps, int(age_t), rival, cfg["speed"], cfg["slack"])
    if sl is None:
        return False
    put(P, i, slot_of(sl, P["geo"][i], int(P["src"][i]), cfg["slack"]))
    return True


# ---------------------------------------------------------------- the quick check

def rescale(ev, cfg):
    """A train2 EvalSet (legacy / wide: one tap at age 0) with train3's timing: the last chord nominally at speed x its
    growth steps, so ideal = speed x (train2's ideal - 1) + slack; and H."""
    ev.items["ideal"] = cfg["speed"] * (ev.items["ideal"] - 1) + cfg["slack"]
    ev.H = int(min(cfg["evalCap"], max(cfg["evalMult"] * ev.S, int(ev.items["ideal"].max(initial=1)) + 8)))
    ev.taps = np.full((len(ev.geo), E.MAXT, E.TAP_COLS), -1, np.int16)
    ev.taps[:, 0, :4] = ev.tap
    ev.taps[:, 0, 4:14] = ev.rule
    ev.taps[:, 0, 14] = 0
    return ev


def speed_ways(walks, M=SPEED_M):
    """Per tap, the first M chords of each way out of it ([2, M] (row, col, in, out), or None if a way is shorter)."""
    out = []
    for rr, cc, ii, oo, _, p, cl in walks:
        n = len(rr)
        ways = []
        for way in (1, -1):
            ks = [p + way * j for j in range(1, M + 1)]
            if cl:
                ks = [k % n for k in ks] if M < n // 2 else None
            elif any(k < 0 or k >= n for k in ks):
                ks = None
            ways.append(None if ks is None else np.array([(rr[k], cc[k], ii[k], oo[k]) for k in ks], np.int64))
        out.append(ways)
    return out


@torch.no_grad()
def evaluate_single(runner, planes, ev, device, batch, probe=False):
    """train2.evaluate under --tap and --speed (the tap at age 0, then the events), plus the speed metric's run
    ages. Same output keys as train2.evaluate, and "speed": per way (age the run first reaches 1, M chords)."""
    C = runner.C
    n, H = len(ev.geo), ev.H
    res = {k: [] for k in ("last", "iou", "tries", "hits", "tapDrawn", "speed")}
    cells = {"x": [], "y": [], "dist": []}
    ways_all = speed_ways(ev.walks)
    for s0 in range(0, n, batch):
        sl = slice(s0, min(n, s0 + batch))
        cs = base_consts(planes, ev.geo[sl])
        walls = cs[:, :1]
        mk = walls > 0
        B = cs.shape[0]
        plan = runner.plan(cs, B)
        evs = Events(ev.taps[sl], runner, planes, ev.S)
        a = torch.from_numpy(ev.a[sl]).to(device)
        on = a < INF
        tgt = on & mk
        last = torch.zeros(B, dtype=torch.long, device=device)
        ways = ways_all[sl]
        wb, wk, wr, wc, wi, wo, wj = [], [], [], [], [], [], []
        for b, wy in enumerate(ways):
            for k, w in enumerate(wy):
                if w is not None:
                    for j in range(SPEED_M):
                        wb.append(b), wk.append(k), wj.append(j)
                        wr.append(w[j, 0]), wc.append(w[j, 1]), wi.append(w[j, 2]), wo.append(w[j, 3])
        has_ways = bool(wb)
        if has_ways:
            ix = [torch.tensor(v, device=device) for v in (wb, wi, wr, wc, wo)]
            wk_t, wj_t = torch.tensor(wk, device=device), torch.tensor(wj, device=device)
            reach = torch.full((B, 2, 2), -1, dtype=torch.long, device=device)  # [b, way, (1, M)]
        state = T2.fresh(walls, C)
        for t in range(1, H + 1):
            state = evs.step(state, walls, cs, plan, np.full(B, t - 1))
            drawn = (state[:, 1:7] > 0.5) & mk
            last = torch.where((drawn ^ tgt).flatten(1).any(1), t, last)
            if has_ways:
                ch = (drawn[ix[0], ix[1], ix[2], ix[3]] & drawn[ix[0], ix[4], ix[2], ix[3]]).float()
                run = torch.zeros(B, 2, SPEED_M, device=device)
                run[ix[0], wk_t, wj_t] = ch
                front = run.cumprod(2).sum(2)                                      # [B, 2]
                for m, col in ((1, 0), (SPEED_M, 1)):
                    reach[..., col] = torch.where((reach[..., col] < 0) & (front >= m), t, reach[..., col])
        pred = (state[:, 1:7] > 0.5) & mk
        pn = pred.cpu().numpy()
        for i, (rr, cc, ii, oo, chs, p, cl) in enumerate(ev.walks[sl]):
            dr = pn[i, ii, rr, cc] & pn[i, oo, rr, cc]
            tr, ht = T2.exit_hops(dr, chs, p, cl)
            res["tries"].append(tr)
            res["hits"].append(ht)
            res["tapDrawn"].append(bool(dr[p]))
        if has_ways:
            rc = reach.cpu().numpy()
            for b, wy in enumerate(ways):
                for k, w in enumerate(wy):
                    if w is not None:
                        res["speed"].append(rc[b, k])
        inter = (pred & on).flatten(1).sum(1).float()
        union = (pred | on).flatten(1).sum(1).float()
        res["iou"].append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union)).cpu().numpy())
        res["last"].append(last.cpu().numpy())
        if probe:
            dist = torch.where(on, a, torch.full_like(a, INF)).amin(1)
            sel = (dist < INF) & (dist > 0)
            b_i, r_i, c_i = torch.nonzero(sel, as_tuple=True)
            cells["x"].append(state[b_i, N_FIXED:, r_i, c_i].cpu().numpy())
            cells["y"].append(CODES[0](ev.rule[sl])[b_i.cpu().numpy()])
            cells["dist"].append(dist[b_i, r_i, c_i].cpu().numpy())
    last = np.concatenate(res["last"])
    out = {"exact": last < H, "settle": last + 1, "iou": np.concatenate(res["iou"]),
           "tries": np.array(res["tries"]).reshape(-1, 3), "hits": np.array(res["hits"]).reshape(-1, 3),
           "tapDrawn": np.array(res["tapDrawn"], bool), "speed": np.array(res["speed"]).reshape(-1, 2)}
    if probe:
        out["cells"] = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in cells.items()}
    return out


class EpisodeSet:
    """Fixed episodes of one (set, level): geo, taps, on, off, line [n, ...], ideal, kind, hit, length [n]; reads
    [n, R] ages (-1 none) at which exact is read (the first is ideal) and probe cells kept; rest [n]."""

    def __init__(self, name, level, S, rows, reads, cfg, cap):
        self.name, self.level, self.S = name, level, S
        for k in SLOT_KEYS:
            setattr(self, k, np.stack([np.asarray(r[k]) for r in rows]) if rows else np.zeros((0,)))
        self.reads = np.asarray(reads, np.int64).reshape(len(rows), -1)
        self.rest = np.minimum(cap, np.maximum(self.reads.max(1, initial=0), self.ideal + 2 * S)).astype(np.int64)
        self.reads[self.reads > self.rest[:, None]] = -1  # past the rollout: not read

    def __len__(self):
        return len(self.geo)

    def counts(self):
        return {"n": len(self), "S": self.S, "rest": int(self.rest.max(initial=0)),
                "kinds": {k: int((self.kind == KINDS.index(k) + 1).sum()) for k in KINDS if (self.kind == KINDS.index(k) + 1).any()}}


@torch.no_grad()
def evaluate_episodes(runner, planes, es, device, batch, slack, early, probe=False):
    """Roll each episode from the fresh state with its taps as events to its rest age. Per episode: exact [R] at
    its read-out ages (the drawing = the target wherever it cares), settle (1 + the last age <= ideal at which the
    drawing differs from the final one), and the edge counts behind collide / own (the module docstring: an edge
    is V if it is drawn at rest, W if it was drawn and is gone at rest, Z if never drawn; don't-care-only intervals
    count as nothing); probe: line cells at each read-out (single-code episodes: x, the code, the read-out's
    index)."""
    C = runner.C
    n = len(es)
    R = es.reads.shape[1]
    keys = ("exact", "settle", "W", "Wever", "Wleft", "Wregrow", "V", "Vever", "Vlost", "Z", "stray", "changed",
            "worm", "wormMax")
    res = {k: [] for k in keys}
    cells = {"x": [], "y": [], "dist": [], "age": []}
    for s0 in range(0, n, batch):
        sl = slice(s0, min(n, s0 + batch))
        cs = base_consts(planes, es.geo[sl])
        walls = cs[:, :1]
        mk = walls > 0
        B = cs.shape[0]
        plan = runner.plan(cs, B)
        evs = Events(es.taps[sl], runner, planes, es.S)
        on = torch.from_numpy(es.on[sl].astype(np.float32)).to(device)       # [B,K,6,S,S]
        off = torch.from_numpy(es.off[sl].astype(np.float32)).to(device)
        ideal = torch.from_numpy(es.ideal[sl].astype(np.int64)).to(device)
        rest = torch.from_numpy(es.rest[sl]).to(device)
        reads = torch.from_numpy(es.reads[sl]).to(device)
        hit = torch.from_numpy(es.hit[sl].astype(np.int64)).to(device)
        real = on < off
        V = (real & (off >= INF)).any(1) & mk
        W = (real & (off < INF)).any(1) & ~V & mk
        Z = ~real.any(1) & mk
        final = target_weight(on, off, ideal.view(-1, 1, 1, 1, 1).float(), slack, early)[0] & mk
        exact = torch.zeros(B, R, dtype=torch.bool, device=device)
        settle = torch.zeros(B, dtype=torch.long, device=device)
        z = torch.zeros_like(W)
        Wleft, Wregrow, Wever, Vever, Vlost, stray = z.clone(), z.clone(), z.clone(), z.clone(), z.clone(), z.clone()
        changed = torch.zeros(B, dtype=torch.bool, device=device)
        seen_h, post_h = z.clone(), z.clone()
        prev = z.clone()
        state = T2.fresh(walls, C)
        steps = int(es.rest[sl].max(initial=0))
        tap_cells = torch.zeros(B, es.S, es.S, dtype=torch.bool, device=device)
        for b in range(B):
            for tp in E.taps_of(es.taps[s0 + b]):
                tap_cells[b, tp.row, tp.col] = True
        hv = hit.view(-1, 1, 1, 1)
        for t in range(1, steps + 1):
            state = evs.step(state, walls, cs, plan, np.full(B, t - 1))
            drawn = (state[:, 1:7] > 0.5) & mk
            due, care = target_weight(on, off, torch.full((B, 1, 1, 1, 1), float(t), device=device), slack, early)
            due &= mk
            ok = ~((drawn ^ due) & care).flatten(1).any(1)
            live = t <= rest                                              # [B]
            at = reads == t                                               # [B, R]
            exact |= at & ok[:, None]
            lv = live.view(-1, 1, 1, 1)
            settle = torch.where((t <= reads[:, 0]) & (drawn ^ final).flatten(1).any(1), t, settle)
            Wleft = torch.where((t == ideal).view(-1, 1, 1, 1), drawn & W, Wleft)
            Wregrow |= drawn & W & (t > ideal).view(-1, 1, 1, 1) & lv
            changed |= ((drawn ^ prev).flatten(1).any(1)) & (t > ideal + slack) & live
            Vever |= drawn & due & V & lv
            Wever |= drawn & due & W & lv
            at_rest = (t == rest).view(-1, 1, 1, 1)
            Vlost = torch.where(at_rest, V & ~drawn, Vlost)
            stray = torch.where(at_rest, Z & drawn, stray)
            seen_h |= drawn & (t <= hv) & lv
            post_h |= drawn & (t > hv) & (hv >= 0) & lv
            prev = drawn
            if probe and at.any():
                for b, r in zip(*torch.nonzero(at, as_tuple=True)):
                    b, r = int(b), int(r)
                    if int(es.codes[s0 + b]) != 1:
                        continue
                    sel = due[b].any(0) & ~tap_cells[b]
                    rr, cc = torch.nonzero(sel, as_tuple=True)
                    if not len(rr):
                        continue
                    cells["x"].append(state[b, N_FIXED:, rr, cc].T.cpu().numpy())
                    cells["y"].append(np.repeat(CODES[0](es.taps[s0 + b][:1, 4:14]), len(rr), 0))
                    first = torch.where(real[b], on[b], float(INF)).amin(0).amin(0)[rr, cc]
                    cells["dist"].append((first // max(1, runner.cfg["speed"])).cpu().numpy())
                    cells["age"].append(np.full(len(rr), r))
        worm = (W & post_h & ~seen_h).flatten(1).sum(1)
        allowed = (real & (off < INF) & (on + slack > hv[:, None])).any(1) & W & (hv >= 0)  # the most allowed
        cnt = lambda x: x.flatten(1).sum(1).cpu().numpy()  # noqa: E731
        res["exact"].append(exact.cpu().numpy())
        res["settle"].append(settle.cpu().numpy() + 1)
        for k, v in (("W", W), ("Wever", Wever), ("Wleft", Wleft & Wever), ("Wregrow", Wregrow), ("V", V), ("Vever", Vever),
                     ("Vlost", Vlost & Vever), ("Z", Z), ("stray", stray)):
            res[k].append(cnt(v))
        res["changed"].append(changed.cpu().numpy())
        res["worm"].append(worm.cpu().numpy() / 2)
        res["wormMax"].append(cnt(allowed) / 2)
    out = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in res.items()}
    if probe:
        out["cells"] = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in cells.items()}
    return out


def episode_set(name, tab, bd, cfg, level, n, seed, kinds, reads_fn, split="heldout"):
    """n held-out episodes of `kinds` ({kind: share}) on the level's eval boards (fixed seed)."""
    rng = np.random.default_rng(seed)
    group = EVAL_GROUP[level]
    S = int(bd.hw[group].max())
    gen = Generator(tab, bd, cfg, group, S, split)
    rows = []
    while len(rows) < n:
        r = gen.new_slot(rng, pick_kind(rng, kinds))
        if r["ideal"] + 2 * S <= cfg["evalCap"]:  # its read-out and the window after it inside the rollout
            rows.append(r)
    rows.sort(key=lambda r: int(r["ideal"]))  # batches of like lengths: none waits on a long one
    reads = [reads_fn(r) for r in rows]
    return EpisodeSet(name, level, S, rows, reads, cfg, cfg["evalCap"])


def persist_set(tab, bd, cfg, level, n, mults, ages, cap):
    """--persist-n wide-set taps of the level whose last read-out fits under cap, as single-tap episodes."""
    ev = rescale(T2.wide_set("m1a", tab, bd, level, max(3, 2 * n), cfg["evalMult"], cfg["evalCap"]), cfg)
    rows, reads = [], []
    for i in range(len(ev.geo)):
        ideal = int(ev.items["ideal"][i])
        rd = [ideal] + [int(round(m * ideal)) for m in mults] + list(ages)
        if max(rd) > cap or len(rows) >= n:
            continue
        sl = E.single(tab, ev.geo[i], ev.tap[i], ev.rule[i], cfg["speed"], cfg["slack"])
        rows.append(slot_of(sl, ev.geo[i], 0, cfg["slack"]))
        reads.append(rd)
    order = np.argsort([r["ideal"] for r in rows], kind="stable")
    return EpisodeSet("persist", level, ev.S, [rows[i] for i in order], [reads[i] for i in order], cfg, cap)


def summarise_episodes(es_list, results, read_names):
    """The q entries of the episode sets: persist / collide / own / overfit (see the module docstring)."""
    rnd = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)  # noqa: E731
    q = {}
    by = {}
    for es, r in zip(es_list, results):
        by.setdefault(es.name, []).append((es, r))
    for name, pairs in by.items():
        cat = lambda k: np.concatenate([r[k] for _, r in pairs])  # noqa: E731
        kind = np.concatenate([es.kind for es, _ in pairs])
        ideal = np.concatenate([es.ideal for es, _ in pairs])
        reads = np.concatenate([es.reads for es, _ in pairs])
        exact = cat("exact")
        out = {"n": int(len(kind))}
        sh = lambda a, b: rnd(a.sum() / b.sum()) if b.sum() else None  # noqa: E731
        hitk = kind == KINDS.index("t2") + 1
        if name in ("persist", "overfit"):
            for j, rn in enumerate(read_names[name]):
                ok = reads[:, j] >= 0
                out[rn] = rnd(exact[ok, j].mean()) if ok.any() else None
            if name == "persist" and out.get("x1"):
                out["ratio"] = rnd((out.get("x4") or 0) / out["x1"])
            if name == "overfit":  # the settle step (the drawing final from then to the read-out) / ideal
                out["steps"] = rnd(np.median(cat("settle")[exact[:, 0]] / np.maximum(1, ideal[exact[:, 0]]))) \
                    if exact[:, 0].any() else None
            if hitk.any():  # an overfit of collisions (--stage T2)
                out.update({"wipe": rnd(1 - cat("Wleft")[hitk].sum() / cat("Wever")[hitk].sum())
                            if cat("Wever")[hitk].sum() else None, "grown": sh(cat("Wever")[hitk], cat("W")[hitk]),
                            "regrow": sh(cat("Wregrow")[hitk], cat("W")[hitk]),
                            "falseWipe": sh(cat("Vlost"), cat("Vever")), "worm": rnd(cat("worm")[hitk].mean()),
                            "wormMax": rnd(cat("wormMax")[hitk].mean())})
        else:
            ctrl = kind == KINDS.index("ctrl") + 1
            out["exact"] = rnd(exact[:, 0].mean())
            if name == "collide":
                out.update({"exactHit": rnd(exact[hitk, 0].mean()) if hitk.any() else None,
                            "exactCtrl": rnd(exact[ctrl, 0].mean()) if ctrl.any() else None,
                            "wipe": rnd(1 - cat("Wleft")[hitk].sum() / cat("Wever")[hitk].sum())
                            if cat("Wever")[hitk].sum() else None, "grown": sh(cat("Wever")[hitk], cat("W")[hitk]),
                            "falseWipe": sh(cat("Vlost"), cat("Vever")),
                            "regrow": sh(cat("Wregrow")[hitk], cat("W")[hitk]),
                            "worm": rnd(cat("worm")[hitk].mean()) if hitk.any() else None,
                            "wormMax": rnd(cat("wormMax")[hitk].mean()) if hitk.any() else None,
                            "nHit": int(hitk.sum())})
            else:
                out.update({"collateral": sh(cat("Vlost"), cat("Vever")), "phantom": rnd(cat("changed").mean()),
                            "stray": sh(cat("stray"), cat("V"))})
            out["byLevel"] = {str(es.level): rnd(r["exact"][:, 0].mean()) for es, r in pairs if len(es)}
        q[name] = out
    return q


# ---------------------------------------------------------------- the code probes

def probe_cells(pools, max_cells, slack, early, seed=0):
    """(x [n, hidden], y [n, 53]): line cells of the training pools' single-code slots (an edge due at the slot's
    age; never a tapped cell), each with its code."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for P in pools.values():
        age = np.minimum(P["age"], AGE_CAP)
        due = due_np(P["on"], P["off"], age, slack, early)             # [n,6,S,S]
        any_ = due.any(1) & (P["codes"] == 1)[:, None, None]
        for i in range(len(age)):
            for tp in E.taps_of(P["taps"][i]):
                any_[i, tp.row, tp.col] = False
        sl, r, c = np.nonzero(any_)
        if not len(sl):
            continue
        k = rng.choice(len(sl), min(len(sl), max_cells // len(pools)), replace=False)
        sl, r, c = sl[k], r[k], c[k]
        rules = P["taps"][sl, 0, 4:14]
        st = P["state"]
        dev = st.device
        xs.append(st[torch.from_numpy(sl).to(dev), N_FIXED:, torch.from_numpy(r).to(dev), torch.from_numpy(c).to(dev)]
                  .float().cpu().numpy())
        ys.append(CODES[0](rules))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def ridge_fit(X, Y, lam=1e-3):
    X, Y = torch.from_numpy(X).double(), torch.from_numpy(Y).double()
    Xb = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)
    A = Xb.T @ Xb
    A += lam * (A.diagonal().mean() + 1e-12) * torch.eye(A.shape[0], dtype=A.dtype)
    W = torch.linalg.solve(A, Xb.T @ Y)
    return lambda x: (torch.cat([torch.from_numpy(x).double(), torch.ones(len(x), 1, dtype=torch.float64)], 1) @ W).numpy()


def mlp_fit(X, Y, device, hidden=256, steps=400, lr=1e-3, batch=1024, seed=0):
    """A 2-layer MLP probe (hidden -> 256 -> 53): class bits by BCE, each digit by a 5-way cross-entropy. The global
    torch RNG is left as it was (a resume stays bit-exact)."""
    keep = torch.random.get_rng_state()
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, CODE_BITS)).to(device)
    torch.random.set_rng_state(keep)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    x = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(Y).float().to(device)
    nc = len(MAJORS)
    dig = y[:, nc:].view(-1, N_TYPES, N_DIGITS).argmax(-1)
    rng = np.random.default_rng(seed)
    with torch.enable_grad():
        for _ in range(steps):
            k = torch.from_numpy(rng.integers(len(x), size=min(batch, len(x)))).to(device)
            out = net(x[k])
            loss = F.binary_cross_entropy_with_logits(out[:, :nc], y[k, :nc]) + \
                F.cross_entropy(out[:, nc:].reshape(-1, N_DIGITS), dig[k].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(xn):
        o = net(torch.from_numpy(xn).float().to(device))
        cls = torch.sigmoid(o[:, :nc])
        d = torch.softmax(o[:, nc:].reshape(-1, N_TYPES, N_DIGITS), -1).reshape(len(o), -1)
        return torch.cat([cls, d], 1).cpu().double().numpy()
    return predict


def probe_score(predict, mean, cells, byage=None, age_names=()):
    """train2.probe_score's fields for a predictor (linear or MLP) + byAge {name: exact} on the persist cells."""
    x, y, dist = cells["x"], cells["y"], cells["dist"]
    if predict is None or not len(x):
        return {"n": int(len(x))}

    def decode(p):
        out = np.zeros(p.shape, np.uint8)
        out[:, :len(MAJORS)] = p[:, :len(MAJORS)] > 0.5
        dg = p[:, len(MAJORS):].reshape(len(p), N_TYPES, N_DIGITS).argmax(-1)
        out[np.arange(len(p))[:, None], len(MAJORS) + np.arange(N_TYPES)[None] * N_DIGITS + dg] = 1
        return out

    def exact_of(xx, yy):
        ok = decode(predict(xx)) == yy
        return ok, ok[:, :len(MAJORS)].all(1), ok[:, len(MAJORS):].all(1)
    ok, cls, dig = exact_of(x, y)
    exact = cls & dig
    chance = decode(np.broadcast_to(mean, (len(y), CODE_BITS)))
    out = {"bits": round(float(ok.mean()), 4), "classes": round(float(cls.mean()), 4),
           "digits": round(float(dig.mean()), 4), "exact": round(float(exact.mean()), 4),
           "byDist": {name: round(float(exact[(dist >= lo) & (dist <= hi)].mean()), 4)
                      for name, lo, hi in T2.DIST_BINS if ((dist >= lo) & (dist <= hi)).any()},
           "chance": {"classes": round(float((chance == y)[:, :len(MAJORS)].all(1).mean()), 4),
                      "digits": round(float((chance == y)[:, len(MAJORS):].all(1).mean()), 4)},
           "n": int(len(x))}
    if byage is not None and len(byage["x"]):
        _, c2, d2 = exact_of(byage["x"], byage["y"])
        ex2 = c2 & d2
        out["byAge"] = {nm: round(float(ex2[byage["age"] == j].mean()), 4) for j, nm in enumerate(age_names)
                        if (byage["age"] == j).any()}
    return out


# ---------------------------------------------------------------- snapshot

def write_snapshot(path, it, pools, tab, last_level, last_idx, slack, early):
    """pool.npz in train2.write_snapshot's contract; the target = the edges due at each slot's age, walls = cells
    with chords under any of its taps' rules, the tap = its first."""
    out = {"iteration": np.int64(it), "radii": np.array(sorted(P["geo"].shape[-1] - 1 for P in pools.values())),
           "last_R": np.int64(pools[last_level]["geo"].shape[-1] - 1), "last_idx": np.asarray(last_idx, np.int64),
           "damage_names": np.array(DAMAGE)}
    for P in pools.values():
        n, S = min(len(P["age"]), V1.SNAP_N), P["geo"].shape[-1]
        R, S2 = S - 1, 2 * S - 1

        def emb(x, dtype):
            y = np.zeros(x.shape[:-2] + (S2, S2), dtype)
            y[..., :S, S - 1:] = x
            return y
        st = P["state"][:n].float().cpu().numpy()
        walls = np.zeros((n, S, S), bool)
        for i in range(n):
            for tp in E.taps_of(P["taps"][i]):
                walls[i] |= tab.render_bits(tp.rule[0], np.asarray(tp.rule[1:], np.int64), P["geo"][i]) != 0
        on = due_np(P["on"][:n], P["off"][:n], np.minimum(P["age"][:n], AGE_CAP), slack, early)
        out[f"walls_{R}"] = emb(walls, np.uint8)
        out[f"mask_{R}"] = emb(P["geo"][:n] >= 0, np.uint8)
        out[f"fill_{R}"] = emb(st[:, 1:7].max(1), np.float16)
        out[f"target_{R}"] = emb(on.any(1), np.uint8)
        out[f"tgt_edges_{R}"] = emb(on, np.uint8)
        out[f"pred_edges_{R}"] = emb(st[:, 1:7], np.float16)
        out[f"tap_{R}"] = P["taps"][:n, 0, :4].astype(np.int16)
        out[f"ntargets_{R}"] = (P["taps"][:n, :, 0] >= 0).sum(1).astype(np.uint8)
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
    p.add_argument("--inputs", choices=[x for x in T2.INPUTS if x not in T2.BROADCAST], default="e")
    p.add_argument("--tap", choices=TAP_MODES, default="impulse", help="how a tap enters (module docstring)")
    p.add_argument("--tap-steps", type=int, default=1, help="--tap impulse: steps the tap planes are on (1 or 2)")
    p.add_argument("--speed", type=int, default=2, help="CA steps per chord (2: decision 4's cap; 1: train2's)")
    p.add_argument("--slack", type=int, default=SIM.SLACK, help="don't-care steps after on_at / off_at (sim.SLACK)")
    p.add_argument("--early", type=int, default=SIM.EARLY, help="don't-care steps before on_at (sim.EARLY); -1: "
                                                                 "none, train2's light cone")
    p.add_argument("--stage", choices=STAGES, default="T1")
    p.add_argument("--remix", type=float, default=1 / 3, help="share of new slots from the earlier stages")
    p.add_argument("--mix", default=None, help="kinds of new slots, e.g. t1=0.5,t3=0.5 (overrides --stage's)")
    p.add_argument("--control", type=float, default=0.25, help="t2: share of control pairs (no collision)")
    p.add_argument("--stagger", type=int, default=None, help="later taps by this age (default bptt / 2)")
    p.add_argument("--hit-max", type=int, default=None, help="t2: the first hit by this age (default bptt)")
    p.add_argument("--hit-min", type=int, default=0, help="t2: the first hit no earlier than this age (sim's collisions "
                                                          "mostly hit at 2-3, before the lines have grown)")
    p.add_argument("--noise", type=float, default=0.3, help="probability of noise damage per kept slot")
    p.add_argument("--new-share", type=float, default=0.125, help="share of each batch made new slots (train2's 1/8)")
    p.add_argument("--no-worst-restart", action="store_true",
                   help="don't restart the batch's worst slot (train2 does): old, drifting slots stay to be trained")
    p.add_argument("--noise-sigma", type=float, nargs=2, default=[0.1, 0.3], metavar=("LO", "HI"))
    p.add_argument("--midtap", type=float, default=None, help="probability of a new tap per kept slot (default 0 "
                                                                 "in T1, 0.1 in T2 / T3)")
    p.add_argument("--overfit", type=int, default=0, help="K fixed episodes of one rule (§1's Pi test)")
    p.add_argument("--overfit-seed", type=int, default=0)
    p.add_argument("--overfit-len", type=int, nargs=2, default=[4, 24], metavar=("LO", "HI"),
                   help="--overfit: each episode's chords, every line's together")
    p.add_argument("--levels", type=int, nargs="+", default=[2])
    p.add_argument("--eval-levels", type=int, nargs="+", default=[3, 4])
    p.add_argument("--eval-sets", nargs="+", choices=("legacy", "wide", "persist", "collide", "own"), default=None,
                   help="default legacy wide persist, + collide in T2, + collide own in T3")
    p.add_argument("--episode-levels", type=int, nargs="+", default=[2, 3], help="levels of the collide / own sets")
    p.add_argument("--episode-n", type=int, default=48, help="episodes per collide / own set and level")
    p.add_argument("--persist-n", type=int, default=48, help="taps per persist set and level")
    p.add_argument("--persist-levels", type=int, nargs="+", default=None, help="the persist sets' levels (default "
                                                                                "--eval-levels)")
    p.add_argument("--persist-mult", type=float, nargs="*", default=[2, 4], help="persist read-outs, x ideal")
    p.add_argument("--persist-ages", type=int, nargs="*", default=None,
                   help="persist read-outs at these ages too (--overfit's default: 100 300 600)")
    p.add_argument("--persist-cap", type=int, default=1200, help="the persist set's longest rollout")
    p.add_argument("--data", default=None)
    p.add_argument("--legacy", default=None)
    p.add_argument("--steps-mult", type=float, nargs=2, default=[3, 6], metavar=("A", "B"))
    p.add_argument("--bptt", type=int, default=48)
    p.add_argument("--last-k", type=int, default=48)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--pool-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-floor", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--collapse-frac", type=float, default=0.6)
    p.add_argument("--collapse-loss-x", type=float, default=4.0)
    p.add_argument("--max-rollbacks", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--channels", type=int, default=96)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dir-groups", type=int, default=-1)
    p.add_argument("--clamp", type=float, nargs=2, default=[-2.0, 2.0], metavar=("LO", "HI"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-n", type=int, default=None)
    p.add_argument("--eval-mult", type=float, default=8)
    p.add_argument("--eval-cap", type=int, default=2000)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--no-mlp-probe", action="store_true")
    p.add_argument("--mlp-steps", type=int, default=400)
    p.add_argument("--minutes", type=float, default=0)
    p.add_argument("--schedule", choices=("iters", "time"), default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--ckpt-pool", choices=("full", "half"), default="full")
    p.add_argument("--snap-every", type=int, default=50)
    p.add_argument("--no-gallery", action="store_true")
    p.add_argument("--force-collapse", type=int, default=0, help=argparse.SUPPRESS)
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
        n_in = T2.n_inputs(args.inputs)
        G = T2.dir_groups_of(args.channels, args.dir_groups)
        if args.init:
            init = torch.load(args.init, map_location="cpu", weights_only=False)
            ic = init["config"]
            if (ic["channels"], ic["hidden"], ic.get("depth", 1), ic.get("nIn"), ic.get("inputs"),
                    ic.get("dirGroups")) != (args.channels, args.hidden, args.depth, n_in, args.inputs, G):
                p.error(f"--init {args.init} has inputs={ic.get('inputs')} channels={ic['channels']} "
                        f"hidden={ic['hidden']} depth={ic.get('depth', 1)} nIn={ic.get('nIn')} "
                        f"dirGroups={ic.get('dirGroups')}; this run asks for --inputs {args.inputs} --channels "
                        f"{args.channels} --hidden {args.hidden} --depth {args.depth} ({n_in} consts, {G} groups)")
        if args.channels <= N_FIXED:
            p.error(f"--channels must be > {N_FIXED}")
        if args.inputs in T2.BROADCAST:
            p.error(f"--inputs {args.inputs} holds the rule on every cell: not a one-time tap (train2 has it)")
        if T2.framed(args.inputs) and N_FIXED + 6 * G > args.channels:
            p.error("--dir-groups: 13 + 6 x groups must fit in --channels")
        fixed_ch = fixed_channels(args.channels, G if T2.framed(args.inputs) else 0)
        if args.tap == "fixed" and fixed_ch is None:
            p.error(f"--tap fixed writes the code into the last {CODE35} channels, which must be scalar: "
                    f"13 + 6 x {G} directional groups > {args.channels} - {CODE35} (more --channels or fewer --dir-groups)")
        if args.tap_steps < 1:
            p.error("--tap-steps must be >= 1")
        stagger = args.stagger if args.stagger is not None else args.bptt // 2
        persist_ages = args.persist_ages if args.persist_ages is not None else ([100, 300, 600] if args.overfit else [])
        eval_sets = args.eval_sets or (["legacy", "wide", "persist"] + (["collide"] if args.stage in ("T2", "T3") else [])
                                       + (["own"] if args.stage == "T3" else []))
        cfg = {"task": "m1a", "trainer": "train3", "inputs": args.inputs, "levels": sorted(args.levels),
               "evalLevels": sorted(args.eval_levels), "evalSets": eval_sets,
               "episodeLevels": sorted(args.episode_levels), "episodeN": args.episode_n,
               "persistN": args.persist_n, "persistMult": list(args.persist_mult), "persistAges": list(persist_ages),
               "persistLevels": sorted(args.persist_levels or args.eval_levels),
               "persistCap": args.persist_cap, "data": args.data, "legacy": args.legacy,
               "channels": args.channels, "hidden": args.hidden, "depth": args.depth, "dirGroups": G, "nIn": n_in,
               "perception": "taps", "clamp": init["config"]["clamp"] if init else list(args.clamp),
               "tap": args.tap, "tapSteps": args.tap_steps if args.tap == "impulse" else None,
               "fixedChannels": fixed_ch if args.tap == "fixed" else None, "code35": "8 class bits, 9 digits x 3 bits MSB first, +-1",
               "speed": args.speed, "slack": args.slack,
               "early": args.early,
               "stage": args.stage, "remix": args.remix, "mix": stage_mix(args.stage, args.remix, args.mix),
               "control": args.control,
               "stagger": stagger, "hitMax": args.hit_max if args.hit_max is not None else args.bptt,
               "hitMin": args.hit_min,
               "noise": args.noise, "noiseSigma": list(args.noise_sigma), "newShare": args.new_share,
               "worstRestart": not args.no_worst_restart,
               "midtap": args.midtap if args.midtap is not None else (0.0 if args.stage == "T1" else 0.1),
               "overfit": args.overfit, "overfitSeed": args.overfit_seed, "overfitLen": list(args.overfit_len),
               "stepsMult": list(args.steps_mult), "bptt": args.bptt, "lastK": args.last_k,
               "lr": args.lr or T2.LR, "lrFloor": args.lr_floor, "warmup": args.warmup,
               "collapseFrac": args.collapse_frac, "collapseLossX": args.collapse_loss_x,
               "maxRollbacks": args.max_rollbacks, "batch": args.batch, "iters": args.iters or 100000,
               "seed": args.seed, "poolSize": args.pool_size, "damageKinds": list(DAMAGE),
               "evalMult": args.eval_mult, "evalCap": args.eval_cap, "probe": not args.no_probe,
               "mlpProbe": not args.no_mlp_probe, "mlpSteps": args.mlp_steps,
               "schedule": args.schedule or "iters", "scheduleMin": args.minutes if args.schedule == "time" else None,
               "ckptPool": args.ckpt_pool, "init": args.init,
               "prevIterations": (init["config"].get("prevIterations", 0) + init["iteration"]) if init else 0,
               "loss": "don't-care-weighted MSE of the six edge planes (on_at / off_at windows) over board cells, "
                       "mean of the last lastK steps"}
    levels, C, slack, early = cfg["levels"], cfg["channels"], cfg["slack"], cfg["early"]
    eval_n = cfg["evalN1"] = args.eval_n or cfg.get("evalN1", 120)
    data_dir = T2.ensure_dir(cfg.get("data"), "strand-v2", "rules-hex.json")
    tab, bd = RuleTable(data_dir), GeoBoards(data_dir)
    CODES[0] = T2.Codes(tab)
    mix = cfg["mix"]

    torch.manual_seed(cfg["seed"])
    model = T2.make_model(C, cfg["hidden"], cfg["clamp"], cfg["nIn"], cfg.get("depth", 1), cfg["inputs"],
                          cfg.get("dirGroups"))
    rng = np.random.default_rng(cfg["seed"])
    if ckpt:
        model.load_state_dict(ckpt["model"])
        rng.bit_generator.state = ckpt["rng"]
    elif init:
        model.load_state_dict(init["model"])
    model.to(device)
    planes = T2.Planes(tab, cfg["inputs"], device)
    assert planes.n_in == cfg["nIn"]
    runner = Runner(model, cfg, device)
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
    gens = {L: Generator(tab, bd, cfg, GROUP[L], side[L]) for L in levels}
    overfit = None
    if cfg["overfit"]:
        if levels != [2]:
            p.error("--overfit trains on level 2 only (--levels 2)")
        overfit = overfit_episodes(tab, bd, cfg, cfg["overfit"], side[2], cfg["overfitSeed"])
        gens[2].fixed = overfit
    legacy_dir = T2.ensure_dir(cfg.get("legacy"), "strand", "meta.json")
    evs, ess, read_names = [], [], {}
    persist_names = ["x1"] + [f"x{m:g}" for m in cfg["persistMult"]] + [f"a{a}" for a in cfg["persistAges"]]
    if overfit:
        rows = [slot_of(sl, geo, i, slack) for sl, geo, i in overfit]
        # exact as train2 reads it (after max(evalMult x S, ideal + 8) steps), on time (at ideal), then the ages
        reads = [[int(min(cfg["evalCap"], max(cfg["evalMult"] * side[2], r["ideal"] + 8))), r["ideal"]]
                 + list(cfg["persistAges"]) for r in rows]
        ess.append(EpisodeSet("overfit", 2, side[2], rows, reads, cfg, max([cfg["evalCap"]] + cfg["persistAges"]) + 8))
        read_names["overfit"] = ["exact", "onTime"] + [f"a{a}" for a in cfg["persistAges"]]
    else:
        for name in cfg["evalSets"]:
            if name in ("legacy", "wide"):
                for L in cfg["evalLevels"]:
                    ev = (T2.legacy_set("m1a", tab, legacy_dir, L, eval_n, cfg["evalMult"], cfg["evalCap"])
                          if name == "legacy" else T2.wide_set("m1a", tab, bd, L, eval_n, cfg["evalMult"], cfg["evalCap"]))
                    evs.append(rescale(ev, cfg))
            elif name == "persist":
                for L in cfg.get("persistLevels") or cfg["evalLevels"]:
                    es = persist_set(tab, bd, cfg, L, cfg["persistN"], cfg["persistMult"], cfg["persistAges"],
                                     cfg["persistCap"])
                    if len(es):
                        ess.append(es)
                read_names["persist"] = persist_names
            else:
                kinds = {"t2": 1.0} if name == "collide" else {"t3": 1.0}
                for L in cfg["episodeLevels"]:
                    ess.append(episode_set(name, tab, bd, cfg, L, cfg["episodeN"], V1.EVAL_SEED + 300 + L
                                           + (0 if name == "collide" else 50), kinds, lambda r: [r["ideal"]]))
    if not evs and not ess:
        p.error("no quick-check sets")
    gallery_evs = {} if args.no_gallery or overfit else \
        {L: rescale(T2.legacy_set("m1a", tab, legacy_dir, L, V1.GALLERY_PER_LEVEL, 8, 2000), cfg) for L in V1.GALLERY_LEVELS}
    gallery_path = os.path.join(run_dir, "gallery.npz")

    @torch.no_grad()
    def gallery_rollout(ev):
        cs = base_consts(planes, ev.geo)
        walls = cs[:, :1]
        plan = runner.plan(cs, len(ev.geo))
        evts = Events(ev.taps, runner, planes, ev.S)
        state = T2.fresh(walls, C)
        for t in range(1, ev.H + 1):
            state = evts.step(state, walls, cs, plan, np.full(len(ev.geo), t - 1))
        return ((state[:, 1:7] > 0.5) & (walls > 0)).cpu().numpy()

    def write_gallery_now(it):
        groups = {}
        for L, ev in gallery_evs.items():
            rot = np.where(ev.geo >= 0, (ev.geo % 12) // 2, -1).astype(np.int8)
            rule = [f"{s}/{'-'.join(str(int(d)) for d in r[1:])}" for s, r in zip(ev.items["subset"], ev.rule)]
            groups[L] = {"mask": ev.geo >= 0, "rot": rot, "tap": ev.tap, "tgt": ev.a < INF, "pred": gallery_rollout(ev),
                         "rule": rule, "length": ev.items["length"], "ideal": ev.items["ideal"]}
        if groups:
            V1.write_gallery(gallery_path, it, groups)

    def check(it=0):
        q = {}
        if evs:
            res = [evaluate_single(runner, planes, ev, device, args.eval_batch, probe=cfg["probe"]) for ev in evs]
            q = T2.summarise("m1a", evs, res, tab)
            sp = np.concatenate([r["speed"] for r in res])
            ok = (sp >= 0).all(1)
            q["speed"] = {"rate": round(float(np.mean((SPEED_M - 1) / np.maximum(1, sp[ok, 1] - sp[ok, 0]))), 4)
                          if ok.any() else None, "n": int(ok.sum()), "of": int(len(sp))}
        eres = [evaluate_episodes(runner, planes, es, device, args.eval_batch, slack, early,
                                  probe=cfg["probe"] and es.name == "persist") for es in ess]
        q.update(summarise_episodes(ess, eres, read_names))
        if overfit:
            q["exact"] = q["overfit"]["exact"]
            q["balanced"] = q["overfit"]["exact"]
        if cfg["probe"] and evs:
            cells = {k: np.concatenate([r["cells"][k] for r in res]) for k in ("x", "y", "dist")}
            per = [r["cells"] for r in eres if "cells" in r and len(r["cells"]["x"])]
            byage = {k: np.concatenate([c[k] for c in per]) for k in ("x", "y", "age")} if per else None
            X, Y = probe_cells(pools, T2.PROBE_CELLS, slack, early, seed=cfg["seed"])
            if X is not None:
                mean = Y.mean(0)
                q["code"] = probe_score(ridge_fit(X, Y), mean, cells, byage, persist_names)
                if cfg["mlpProbe"]:
                    q["codeMlp"] = probe_score(mlp_fit(X, Y, device, steps=cfg["mlpSteps"], seed=cfg["seed"]), mean,
                                               cells, byage, persist_names)
        try:
            write_gallery_now(it)
        except Exception:  # noqa: BLE001 -- a nice-to-have, never worth a training run
            pass
        parts = [q.get("balanced")]
        if "persist" in q:
            parts.append(q["persist"].get("x4"))
        if "collide" in q and q["collide"].get("wipe") is not None:
            parts.append(q["collide"]["wipe"] * (1 - (q["collide"].get("falseWipe") or 0)))
        if "own" in q:
            parts.append(q["own"].get("exact"))
        parts = [x for x in parts if x is not None]
        score = q["overfit"]["exact"] if overfit else (round(float(np.mean(parts)), 4) if parts else 0.0)
        return {"q": q, "exact": q.get("exact"), "score": score,
                "evalN": int(sum(len(ev.geo) for ev in evs) + sum(len(es) for es in ess))}

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
        pools = {L: new_pool(rng, gens[L], mix, cfg["poolSize"], C, device) for L in levels}
        for P in pools.values():
            P["born"][:] = start_it
    held = [int(tab.held_out(s).sum()) for s in range(tab.n_sub)]
    start = {"config": cfg, "startIteration": start_it, "evalN": int(sum(len(ev.geo) for ev in evs) + sum(len(es) for es in ess)),
             "threads": torch.get_num_threads(), "torch": torch.__version__, "lrScale": lr_scale,
             "rollbacks": rollbacks, "params": sum(x.numel() for x in model.parameters()),
             "boardSide": {str(L): S for L, S in side.items()},
             "evalSets": {f"{ev.name}{ev.level}": ev.counts() for ev in evs} | {f"{es.name}{es.level}": es.counts() for es in ess},
             "trivial": T2.trivial_of(evs) if evs else None,
             "rules": {"subsets": tab.keys, "count": [int(x) for x in tab.count], "heldout": held,
                       "legacyHeldout": len(tab.legacy_heldout())},
             "schedule": {"mode": cfg["schedule"], "decayAt": list(V1.DECAY_AT),
                          **({"boxMin": cfg["scheduleMin"], "usedMin": round(prior_sec / 60, 2)}
                             if cfg["schedule"] == "time" else {"iters": cfg["iters"]})}}
    if overfit:
        start["overfit"] = [{"taps": [[tp.row, tp.col, tp.d0, tp.d1, tp.t] for tp in sl.taps],
                             "rules": [tab.describe(tp.rule[0], tp.rule[1:]) for tp in sl.taps], "length": sl.length,
                             "kind": sl.kind, "hit": sl.hit, "ideal": sl.settle + slack} for sl, _, _ in overfit]
    if device.type == "cuda":
        start["gpu"] = torch.cuda.get_device_name(device)
    emit(start)

    snap_path, last = os.path.join(run_dir, "pool.npz"), [None]

    def snapshot(it):
        if not args.snap_every or last[0] is None:
            return
        try:
            write_snapshot(snap_path, it, pools, tab, *last[0], slack, early)
        except Exception as e:  # noqa: BLE001
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
        T2.save_ckpt(best_path, model, opt, 0, cfg, rng, best=best, eval_sec=eval_sec)
        emit(rec)

    t_win, losses, t_data = time.time(), [], []
    kinds = dict.fromkeys(DAMAGE + ("new", "worst"), 0)
    skipped = skip_run = win_skipped = win_steps = 0
    history, collapsed, slowest = [], False, 0.0
    a_mult, b_mult = cfg["stepsMult"]
    B = cfg["batch"]
    it = start_it
    while it < cfg["iters"]:
        due_check = (it + 1) % args.eval_every == 0 or it + 1 == cfg["iters"]
        if args.minutes and time.time() - t_start + slowest + (eval_sec if due_check else 0) > 60 * args.minutes:
            if it > start_it:
                T2.save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
                snapshot(it)
            emit({"stopped": "time", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2),
                  "slowestIterSec": round(slowest, 2), "evalSec": round(eval_sec, 1)})
            break
        t_it = time.time()
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
        if T > G:  # ---- the no-gradient prefix: the slots carry on, their taps firing as their ages come
            cs = base_consts(planes, P["geo"][idx])
            plan = runner.plan(cs, len(idx))
            evts = Events(P["taps"][idx], runner, planes, S)
            age = P["age"][idx].copy()
            t_pre = time.time()
            with torch.no_grad():
                for s in range(T - G):
                    state = evts.step(state, cs[:, :1], cs, plan, np.minimum(age + s, AGE_CAP))
            P["age"][idx] = np.minimum(age + (T - G), AGE_CAP)
            t_ev += time.time() - t_pre
        # ---- the events at the start of the gradient window: new episodes, the worst slot's restart, damage
        cur = np.nan_to_num(P["loss"][idx], nan=-1.0)
        j_worst = int(cur.argmax()) if cfg.get("worstRestart", True) else -1
        n_new = max(1, int(round(len(idx) * cfg.get("newShare", 0.125)))) if cfg.get("newShare", 0.125) > 0 else 0
        new = set(rng.choice(len(idx), n_new, replace=False).tolist()) if n_new else set()
        restart = new | ({j_worst} if j_worst >= 0 else set())
        noisy = []
        for pos, i in enumerate(idx):
            if pos in new:
                put(P, i, gens[L].new_slot(rng, pick_kind(rng, mix)))
                kinds["new"] += 1
            elif pos == j_worst:
                kinds["worst"] += 1
            else:
                if cfg["midtap"] and rng.random() < cfg["midtap"] and not overfit:
                    if midtap_damage(rng, tab, P, i, int(P["age"][i]) + int(rng.integers(0, G // 2 + 1)), cfg,
                                     cfg["stage"]):
                        kinds["midtap"] += 1
                        P["edits"][i] += 1
                        P["last"][i] = DAMAGE.index("midtap")
                if rng.random() < cfg["noise"]:
                    noisy.append((pos, i))
            if pos in restart:
                P["age"][i], P["born"][i], P["edits"][i], P["last"][i] = 0, it, 0, -1
        cs = base_consts(planes, P["geo"][idx])
        walls = cs[:, :1]
        plan = runner.plan(cs, len(idx))
        if restart:
            rs = torch.tensor(sorted(restart), device=device)
            state[rs] = T2.fresh(cs[rs, :1], C)
        for pos, i in noisy:
            if noise_damage(rng, P, i, pos, state, cfg):
                kinds["noise"] += 1
                P["edits"][i] += 1
                P["last"][i] = DAMAGE.index("noise")
        state[:, 0:1] = walls
        evts = Events(P["taps"][idx], runner, planes, S)
        on = torch.from_numpy(P["on"][idx].astype(np.float32)).to(device)
        off = torch.from_numpy(P["off"][idx].astype(np.float32)).to(device)
        age0 = P["age"][idx].copy()
        age0_t = torch.from_numpy(age0).to(device).float()
        mk = walls
        ncell = 6 * mk.sum((1, 2, 3)).clamp(min=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_data.append(time.time() - t_ev)
        # ---- the gradient window: G steps, the loss over the last K, each against its own age's target
        acc = 0.0
        for t in range(G):
            state = evts.step(state, walls, cs, plan, np.minimum(age0 + t, AGE_CAP))
            if t >= G - K:
                age = (age0_t + t + 1).clamp(max=AGE_CAP).view(-1, 1, 1, 1, 1)
                acc = acc + step_loss(state, on, off, age, slack, early, mk, ncell) / K
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
            P["age"][idx] = np.minimum(age0 + G, AGE_CAP)
            P["loss"][idx] = acc.detach().cpu().numpy()
        else:
            skipped, skip_run, win_skipped = skipped + 1, skip_run + 1, win_skipped + 1
            P["state"][tidx] = T2.fresh(cs[:, :1], C)
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
                                      "age": int(np.median(Pp["age"])),
                                      "kinds": {k: int((Pp["kind"] == j + 1).sum()) for j, k in enumerate(KINDS)
                                                if (Pp["kind"] == j + 1).any()}} for Lp, Pp in pools.items()}}
            if cfg["schedule"] == "time":
                rec["schedFrac"] = round(sched_frac(), 4)
            if skipped:
                rec["skipped"] = skipped
            why = V1.collapse_reason(None, best, cfg["collapseFrac"], rec["loss"], history, win_skipped, win_steps,
                                     cfg["collapseLossX"])
            if it % args.eval_every == 0 or it == cfg["iters"]:
                t_eval = time.time()
                rec.update(check(it))
                eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
                n_checks[0] += 1
                if args.force_collapse and n_checks[0] == args.force_collapse:
                    why = why or "forced (--force-collapse, a test)"
                if rec["score"] > best and not why:
                    best = rec["best"] = rec["score"]
                    T2.save_ckpt(best_path, model, opt, it, cfg, rng, best=best, eval_sec=eval_sec)
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
                    T2.save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
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
                    Pp["state"] = T2.fresh(torch.from_numpy(Pp["geo"] >= 0).float()[:, None], C).to(device)
                    Pp["age"][:], Pp["born"][:], Pp["edits"][:], Pp["last"][:], Pp["loss"][:] = 0, it, 0, -1, np.nan
                emit({"iteration": it, "rollback": it, "lrScale": lr_scale, "rollbacks": rollbacks, "reason": why,
                      "restored": back[1], "best": best})
                T2.save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
                snapshot(it)
                continue
        if it % max(1, args.ckpt_every) == 0 or it == cfg["iters"]:
            T2.save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, guard())
            snapshot(it)
        elif args.snap_every and it % args.snap_every == 0:
            snapshot(it)
    if it >= cfg["iters"] and not collapsed:
        emit({"stopped": "done", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2)})
    log.close()


if __name__ == "__main__":
    main()
