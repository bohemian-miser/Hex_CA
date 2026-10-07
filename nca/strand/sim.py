"""The multi-strand event simulator: targets for a strand CA whose taps are one-time events, whose lines keep
themselves alive and whose lines die when another pattern hits them (docs/spectacle-nca-taps.md §0, §2, §4.3).

API (small on purpose; the rest of this file is internal):

    from nca.strand.sim import Tap, simulate, target_weight, draw_taps, INF

    taps = [Tap(t=0, row=4, col=7, d0=1, d1=4, rule=(s, *digits)), Tap(t=6, ...)]
    ep = simulate(tab, geo, taps)       # tab: rules.RuleTable; geo: int16 [H,W] (-1 off board, padding is fine)
    ep.on_at, ep.off_at                 # int16 [K,6,H,W]: K intervals [on, off) per edge plane, INF = never
    tgt, care = target_weight(ep.on_at, ep.off_at, age)   # bool [6,H,W]; numpy or torch, batched or not
    ep.accepted                         # bool per tap: the host's refusals (the CA never sees a refused tap)
    ep.settle                           # the time of the last change; ep.ideal = settle + SLACK
    ep.lines, ep.tips, ep.chords, ep.events   # numpy record arrays: fates and the timeline, for the metrics
    taps = draw_taps(rng, tab, geo, "collide")  # an episode's taps: single, collide, control, own

TIME. `t` / `age` count CA updates from the episode's start (the trainer's slot age). A tap at time t is given to
the update from age t to t + 1 (impulse or fixed write: the trainer's business); its chord is "drawn at t" here,
so the earliest the CA can show it is age t + 1. Every time below is such a nominal time; the trainer's window
around it (target_weight) is what makes it a light cone.

WHAT HAPPENS (one step t: moves, then dying waves, then taps; the doc's §2 table):
  - A tap (t, cell, chord (d0, d1), rule) starts a line on that chord with two tips, one leaving across d1
    ("ahead"), one across d0 ("behind"). Refused (`accepted` False, nothing drawn) if the cell has no such chord
    under the rule, holds any chord of another rule (a rival owns its whole tile), or the chord is already drawn
    under this rule. Rules are unique per player, so the rule IS the identity: same rule = own line.
  - A tip moves one chord every PERIOD (2) steps (decision 4): chord k from the tap is drawn at t_tap + 2k. Off
    the board or into a cell without its chord: a tail, the tip stops.
  - Into a cell holding a chord of another rule: a HIT (decision 3, the tile rule; Spectacle's `crossingMode:
    'tile'` + `mutualCut`). Every chord on the hit cell goes at once (off_at = h) and a dying wave runs out
    along each of them; the hitter's tip dies without drawing and a wave runs back along the hitter from its
    last chord (off_at = h + 1, h + 2, ...). Tips of different rules entering one empty cell on the same step
    (case 3) all hit there; so does any tip entering a cell that is hit on that step.
  - A DYING WAVE moves one chord per step (wave=1) along drawn chords of its rule, following the strand: chord j
    along the line from the hit goes at h + j, whichever line of that rule drew it (lines of one rule that
    touch end to end are one line, as Spectacle's `join` makes them). It never jumps to another chord of the
    cell (case 9). A tip whose chord the wave erases dies; tips move before waves in a step, so a fleeing tip
    draws until the wave catches it (case 6: 2d steps for a tip d ahead) and those chords go too.
  - Onto a chord already drawn under its own rule: ABSORBED, the tip stops and nothing changes (case 4: a join
    at a loose end; the two tips of a loop meeting). Into a cell with another chord of its rule: it draws beside
    it. Two tips of one rule onto one chord on one step: one draws, the other is absorbed.
  - A line is never refreshed by its tap: after the tap only its own tips and waves change it.

TARGETS. Every chord the sim draws is an interval [on, off) on both its edges (off = INF if it survives). Edges
are the CA's ch1-6 (edge d of the cell is crossed); they carry no rule, so intervals of different lines on one
edge simply add up (K > 1: an edge drawn, wiped, drawn again). target_weight turns intervals into the loss's
target and care masks at an age, with the doc's windows: an interval requires 1 from on + SLACK until off; it is
don't-care from on - EARLY to on + SLACK (growth: not before 2k - 1, by 2k + slack) and from off to off + SLACK
(a wipe at h + k: drawn until h + k, don't-care to h + k + slack, 0 after); 0 everywhere else. Where the timing
itself is uncertain the sim adds don't-care-only intervals (on = off; `chords.maybe`): the chord a hitter would
have drawn into the hit cell (Spectacle draws it for one tick; case 3), and the next chord of a tip a wave kills
when the CA's tip could still have drawn it inside the window (case 6).

OUTPUTS (record arrays; `line` = the index of an accepted tap's line, in `lines`):
  lines   tap (index into taps), code (index into ep.codes), hit (first step it was hit: as hitter or victim; INF
          none), role (bits: 1 hitter, 2 victim (its cell was hit), 4 contact (erased by a wave from a line of its
          rule touching it)), by (the other side's line, -1), len (chords drawn), gone (step its last chord went;
          INF if any is left)
  tips    line, side (+1 ahead, -1 behind), end (step it stopped; INF still growing), why (0 growing, 1 tail,
          2 absorbed, 3 hit, 4 killed: its chord erased), row, col (its last cell), k (its chord's k)
  chords  row, col, a, b (its two edges), line, k (signed chords from the tap: + ahead, - behind), on, off,
          maybe
  events  t, kind (TAP REFUSE TAIL ABSORB HIT KILL: KINDS), line, row, col, other (HIT: the victim line, -1 for an
          empty cell (case 3); REFUSE: the tap index)

SWITCHES (for the parity check against Spectacle's engine: python -m nca.strand.sim --parity): wave=0 wipes a hit
line whole at once (the game's instant `dropPath`); sequential=True resolves moves one tip at a time in the
engine's order (players by first tap, then lines by tap, ahead tip before behind) instead of all at once. The
CA is wave=1, sequential=False.

    python -m nca.strand.sim --test        # unit tests (nca/strand/test_sim.py): the walker at double time, ...
    python -m nca.strand.sim --bench       # episodes per second per level and kind (the trainer's data path)
    python -m nca.strand.sim --parity      # against data/strand-v2/collide.json (scripts/strand-export.ts --collide)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np

from .walker import DCOL, DROW, PAIR_INDEX, PAIRS, walk

INF = 32767          # "never" (= train.INF)
PERIOD = 2           # steps per chord a tip grows (decision 4)
EARLY = 1            # an interval is don't-care from on - EARLY ...
SLACK = 2            # ... to on + SLACK, and from off to off + SLACK
OPP = (3, 4, 5, 0, 1, 2)
KINDS = ("TAP", "REFUSE", "TAIL", "ABSORB", "HIT", "KILL")
TAP, REFUSE, TAIL, ABSORB, HIT, KILL = range(6)
GROWING, W_TAIL, W_ABSORBED, W_HIT, W_KILLED = range(5)
HITTER, VICTIM, CONTACT = 1, 2, 4


class Tap(NamedTuple):
    """A tap: at step t, chord (d0, d1) of cell (row, col), under rule (s, digit_0 .. digit_8)."""
    t: int
    row: int
    col: int
    d0: int
    d1: int
    rule: tuple


def rule_key(rule) -> tuple:
    """A rule as a hashable (s, digit_0, .., digit_8) of ints (from a tuple, list or numpy row)."""
    return tuple(int(x) for x in np.asarray(rule).reshape(-1))


_LINE_T = np.dtype([("tap", "i4"), ("code", "i4"), ("hit", "i4"), ("role", "i1"), ("by", "i4"), ("len", "i4"),
                    ("gone", "i4")])
_TIP_T = np.dtype([("line", "i4"), ("side", "i1"), ("end", "i4"), ("why", "i1"), ("row", "i2"), ("col", "i2"),
                   ("k", "i4")])
_CHORD_T = np.dtype([("row", "i2"), ("col", "i2"), ("a", "i1"), ("b", "i1"), ("line", "i4"), ("k", "i4"),
                     ("on", "i4"), ("off", "i4"), ("maybe", "?")])
_EVENT_T = np.dtype([("t", "i4"), ("kind", "i1"), ("line", "i4"), ("row", "i2"), ("col", "i2"), ("other", "i4")])


@dataclass
class Episode:
    """What `simulate` returns; see the module docstring."""
    shape: tuple
    taps: list
    codes: list                     # code index -> rule key
    accepted: np.ndarray            # bool [n taps]
    reason: list                    # per tap: '' or why it was refused
    lines: np.ndarray
    tips: np.ndarray
    chords: np.ndarray
    events: np.ndarray
    settle: int                     # the last step anything changed
    _planes: tuple | None = field(default=None, repr=False)

    @property
    def ideal(self) -> int:
        """Steps until every target is fixed: the last change plus the slack."""
        return self.settle + SLACK

    @property
    def on_at(self) -> np.ndarray:
        return self.planes()[0]

    @property
    def off_at(self) -> np.ndarray:
        return self.planes()[1]

    def planes(self, K: int | None = None, after: int | None = None):
        """(on_at, off_at) int16 [K,6,H,W], each edge's intervals in order of `on`, INF-padded. K=None: as many as
        the busiest edge needs (at least 1); a fixed K pads, and raises if an edge needs more. `after`: drop the
        intervals already over by then (off + SLACK <= after), for a pool slot that only looks forward."""
        if K is None and after is None and self._planes is not None:
            return self._planes
        H, W = self.shape
        c = self.chords
        if after is not None:
            c = c[c["off"].astype(np.int64) + SLACK > after]
        cell = c["row"].astype(np.int64) * W + c["col"]
        e = np.concatenate([c["a"].astype(np.int64) * (H * W) + cell, c["b"].astype(np.int64) * (H * W) + cell])
        on = np.concatenate([c["on"], c["on"]])
        off = np.concatenate([c["off"], c["off"]])
        o = np.lexsort((on, e))
        e, on, off = e[o], on[o], off[o]
        start = np.r_[True, e[1:] != e[:-1]] if len(e) else np.zeros(0, bool)
        first = np.maximum.accumulate(np.where(start, np.arange(len(e)), 0)) if len(e) else np.zeros(0, np.int64)
        rank = np.arange(len(e)) - first
        need = int(rank.max()) + 1 if len(e) else 1
        if K is None:
            K = need
        elif need > K:
            raise ValueError(f"an edge has {need} intervals, more than K={K}")
        on_at = np.full((K, 6 * H * W), INF, np.int16)
        off_at = np.full((K, 6 * H * W), INF, np.int16)
        on_at[rank, e] = np.minimum(on, INF)
        off_at[rank, e] = np.minimum(off, INF)
        out = on_at.reshape(K, 6, H, W), off_at.reshape(K, 6, H, W)
        if after is None and K == need:
            self._planes = out
        return out


def target_weight(on_at, off_at, age, early: int = EARLY, slack: int = SLACK):
    """(target, care), bool [..., 6, H, W], at `age` from interval planes [..., K, 6, H, W] (numpy arrays or torch
    tensors; `age` a number or anything that broadcasts against [..., K, 6, H, W], e.g. a [B,1,1,1,1] tensor).
    target: some interval requires the edge drawn; care: target, or no interval's window makes it don't-care.
    The loss is then the masked MSE of (state - target) over care, as train2's."""
    if isinstance(on_at, np.ndarray):
        on, off = on_at.astype(np.int32), off_at.astype(np.int32)
    else:  # torch: int16 + slack would overflow
        on, off = on_at.int() if on_at.dtype.itemsize < 4 and not on_at.is_floating_point() else on_at, \
            off_at.int() if off_at.dtype.itemsize < 4 and not off_at.is_floating_point() else off_at
    one = (on + slack <= age) & (age < off)
    dc = ((on - early <= age) & (age < on + slack)) | ((off <= age) & (age < off + slack))
    tgt = one.any(-4)
    return tgt, tgt | ~dc.any(-4)


# ---------------------------------------------------------------- the simulation

_NBR = {}


def _nbr(H: int, W: int) -> list:
    """Flat neighbour table: [d * H * W + cell] -> the cell across edge d, -1 off the array."""
    key = (H, W)
    if key not in _NBR:
        r, c = np.divmod(np.arange(H * W), W)
        out = []
        for d in range(6):
            nr, nc = r + DROW[d], c + DCOL[d]
            ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            out.append(np.where(ok, nr * W + nc, -1))
        _NBR[key] = np.concatenate(out).tolist()
    return _NBR[key]


class _Sim:
    def __init__(self, tab, geo, taps, period, wave, sequential, maybe, slack, players, exits):
        self.H, self.W = H, W = geo.shape
        self.HW = HW = H * W
        self.nb = _nbr(H, W)
        self.period, self.wave, self.seq, self.maybe, self.slack = period, wave, sequential, maybe, slack
        self.taps = [Tap(int(p.t), int(p.row), int(p.col), int(p.d0), int(p.d1), rule_key(p.rule)) for p in taps]
        self.codes = [rule_key(r) for r in players] if players is not None else []
        index = {r: i for i, r in enumerate(self.codes)}
        for p in self.taps:
            if p.rule not in index:
                index[p.rule] = len(self.codes)
                self.codes.append(p.rule)
        self.code_of = index
        self.ex = []
        for r in self.codes:
            e = exits.get(r) if exits is not None else None
            if e is None:
                e = tab.exits(r[0], np.asarray(r[1:], np.int64), geo)
            self.ex.append(e.reshape(-1).tolist())
        self.edge_rec = [-1] * (6 * HW)
        self.cell_code = [-1] * HW
        self.cell_n = [0] * HW
        # an erased chord lingers (the CA may still show it until off + slack): its cell stays a rival's tile and
        # its chord can't be drawn again until then (wave=0, the game: nothing lingers)
        self.linger = slack if wave else 0
        self.dead_code = [-1] * HW
        self.cell_dead = [0] * HW        # the cell is a dying rival's tile while t < this
        self.edge_dead = [0] * (6 * HW)  # the chord on this edge is dying while t < this
        self.now = 0
        # chords (records), lines, tips: parallel lists
        self.rc_cell, self.rc_a, self.rc_b, self.rc_line, self.rc_k, self.rc_on, self.rc_off = [], [], [], [], [], [], []
        self.mb = []                     # maybe records: (cell, a, b, line, k, on, off)
        self.rec_tips = {}               # record -> tips whose current chord it is
        self.ln_tap, self.ln_code, self.ln_hit, self.ln_role, self.ln_by = [], [], [], [], []
        self.ln_tips, self.ln_slot = [], []
        self.tp_line, self.tp_side, self.tp_cell, self.tp_out, self.tp_rec, self.tp_k = [], [], [], [], [], []
        self.tp_next, self.tp_alive, self.tp_end, self.tp_why = [], [], [], []
        self.due = {}
        self.fronts, self.born = [], []
        self.events = []
        self.last = 0
        self.accepted = [False] * len(self.taps)
        self.reason = [""] * len(self.taps)
        self.player_lines = [0] * len(self.codes)

    # -- state changes
    def draw(self, cell, a, b, line, k, t):
        r = len(self.rc_cell)
        self.rc_cell.append(cell)
        self.rc_a.append(a)
        self.rc_b.append(b)
        self.rc_line.append(line)
        self.rc_k.append(k)
        self.rc_on.append(t)
        self.rc_off.append(INF)
        HW = self.HW
        self.edge_rec[a * HW + cell] = r
        self.edge_rec[b * HW + cell] = r
        code = self.ln_code[line]
        assert self.cell_code[cell] in (-1, code), "two rules on one cell"
        self.cell_code[cell] = code
        self.cell_n[cell] += 1
        self.last = t
        return r

    def erase(self, r, t, cause_line=-1):
        """Erase record r at t; kill the tips standing on it. Returns the line it belonged to."""
        self.rc_off[r] = t
        cell = self.rc_cell[r]
        HW = self.HW
        self.edge_rec[self.rc_a[r] * HW + cell] = -1
        self.edge_rec[self.rc_b[r] * HW + cell] = -1
        self.cell_n[cell] -= 1
        if self.cell_n[cell] == 0:
            self.cell_code[cell] = -1
        if self.linger:
            self.dead_code[cell] = self.ln_code[self.rc_line[r]]
            self.cell_dead[cell] = self.edge_dead[self.rc_a[r] * HW + cell] = self.edge_dead[self.rc_b[r] * HW + cell] \
                = t + self.linger
        self.last = t
        for tip in self.rec_tips.pop(r, ()):
            if self.tp_alive[tip]:
                self.kill(tip, t)
        return self.rc_line[r]

    def stop(self, tip, t, why, kind):
        self.tp_alive[tip] = False
        self.tp_end[tip] = t
        self.tp_why[tip] = why
        if kind is not None:
            cell = self.tp_cell[tip]
            self.events.append((t, kind, self.tp_line[tip], cell // self.W, cell % self.W, -1))

    def kill(self, tip, t):
        """A tip whose chord was erased at t; maybe the chord it could still have drawn inside the window."""
        self.stop(tip, t, W_KILLED, KILL)
        tm = self.tp_next[tip]
        if self.maybe and tm - EARLY < t + self.slack:
            nxt = self.peek(tip)
            if nxt is not None and nxt[3] == "draw":
                j, ein, eout = nxt[:3]
                self.mb.append((j, ein, eout, self.tp_line[tip], self.tp_k[tip] + self.tp_side[tip], tm, tm))

    def peek(self, tip):
        """(cell, ein, eout, what) of tip's next move now, what in draw / absorb / hit / dying (its next chord is
        being wiped: the tip meets the wave and dies); None at a tail."""
        cell, d = self.tp_cell[tip], self.tp_out[tip]
        HW = self.HW
        j = self.nb[d * HW + cell]
        if j < 0:
            return None
        ein = OPP[d]
        code = self.ln_code[self.tp_line[tip]]
        eout = self.ex[code][ein * HW + j]
        if eout < 0:
            return None
        cc = self.cell_code[j]
        if cc >= 0 and cc != code or self.cell_dead[j] > self.now and self.dead_code[j] != code:
            return j, ein, eout, "hit"
        if self.edge_rec[ein * HW + j] >= 0:
            return j, ein, eout, "absorb"
        if self.edge_dead[ein * HW + j] > self.now:
            return j, ein, eout, "dying"
        return j, ein, eout, "draw"

    def advance_tip(self, tip, j, ein, eout, t):
        line = self.tp_line[tip]
        k = self.tp_k[tip] + self.tp_side[tip]
        old = self.tp_rec[tip]
        lst = self.rec_tips.get(old)
        if lst is not None:
            lst.remove(tip)
            if not lst:
                del self.rec_tips[old]
        r = self.draw(j, ein, eout, line, k, t)
        self.rec_tips.setdefault(r, []).append(tip)
        self.tp_cell[tip], self.tp_out[tip], self.tp_rec[tip], self.tp_k[tip] = j, eout, r, k
        self.schedule(tip, t + self.period)

    def schedule(self, tip, t):
        self.tp_next[tip] = t
        self.due.setdefault(t, []).append(tip)

    # -- hits and waves
    def hit(self, hits, t):
        """Cells hit at t, {cell: [(tip, ein, eout)]}: the hitters stop; every chord on a hit cell goes and waves
        run out of it; waves run back along the hitters from the cells they hit. All at once (head-on tips hit
        each other's cells on one step)."""
        HW, W = self.HW, self.W
        for j in hits:
            for tip, _, _ in hits[j]:
                if self.tp_alive[tip]:
                    self.stop(tip, t, W_HIT, None)
        victims = {}
        for j in sorted(hits):
            recs = []
            for e in range(6):
                r = self.edge_rec[e * HW + j]
                if r >= 0 and r not in recs:
                    recs.append(r)
            vs = victims[j] = []
            for r in recs:
                a, b, code = self.rc_a[r], self.rc_b[r], self.ln_code[self.rc_line[r]]
                line = self.erase(r, t)
                if line not in vs:
                    vs.append(line)
                self.start_wave(j, a, code, t)
                self.start_wave(j, b, code, t)
        for j in sorted(hits):
            vs = victims[j]
            for tip, ein, eout in hits[j]:
                line = self.tp_line[tip]
                for v in vs or [-1]:
                    self.events.append((t, HIT, line, j // W, j % W, v))
                self.mark(line, HITTER, t, vs[0] if vs else -1)
                for v in vs:
                    self.mark(v, VICTIM, t, line)
                if self.maybe:
                    self.mb.append((j, ein, eout, line, self.tp_k[tip] + self.tp_side[tip], t, t))
                self.start_wave(j, OPP[self.tp_out[tip]], self.ln_code[line], t)

    def mark(self, line, role, t, by):
        if self.ln_hit[line] == INF:
            self.ln_hit[line] = t
            self.ln_by[line] = by
        self.ln_role[line] |= role

    def start_wave(self, cell, edge, code, t):
        """A wave that erased (or would have erased) the chord at `cell` at t goes out across `edge`."""
        if self.wave == 0:
            self.flood([(cell, edge, code)], t)
        else:
            self.born.append((cell, edge, code))

    def step_front(self, front, t):
        cell, e, code = front
        HW = self.HW
        j = self.nb[e * HW + cell]
        if j < 0:
            return None
        ei = OPP[e]
        r = self.edge_rec[ei * HW + j]
        if r < 0 or self.ln_code[self.rc_line[r]] != code:
            return None
        line = self.erase(r, t)
        if self.ln_hit[line] == INF:
            self.ln_role[line] |= CONTACT
        return j, (self.rc_b[r] if self.rc_a[r] == ei else self.rc_a[r]), code

    def flood(self, fronts, t):
        while fronts:
            nxt = []
            for f in fronts:
                g = self.step_front(f, t)
                if g is not None:
                    nxt.append(g)
            fronts = nxt

    def advance_waves(self, t):
        if self.fronts:
            nxt = []
            for f in self.fronts:
                g = self.step_front(f, t)
                if g is not None:
                    nxt.append(g)
            self.fronts = nxt
        if self.born:
            self.fronts.extend(self.born)
            self.born = []

    # -- moves
    def moves(self, t, tips):
        tips = [tp for tp in tips if self.tp_alive[tp] and self.tp_next[tp] == t]
        if not tips:
            return
        tips.sort(key=lambda tp: (self.ln_slot[self.tp_line[tp]], -self.tp_side[tp]))
        if self.seq:
            for tip in tips:
                if not self.tp_alive[tip]:
                    continue
                nxt = self.peek(tip)
                if nxt is None:
                    self.stop(tip, t, W_TAIL, TAIL)
                elif nxt[3] == "hit":
                    self.hit({nxt[0]: [(tip, nxt[1], nxt[2])]}, t)
                elif nxt[3] == "absorb":
                    self.stop(tip, t, W_ABSORBED, ABSORB)
                elif nxt[3] == "dying":
                    self.stop(tip, t, W_KILLED, KILL)
                else:
                    self.advance_tip(tip, nxt[0], nxt[1], nxt[2], t)
            return
        hits, cands = {}, {}
        for tip in tips:
            nxt = self.peek(tip)
            if nxt is None:
                self.stop(tip, t, W_TAIL, TAIL)
            elif nxt[3] == "hit":
                hits.setdefault(nxt[0], []).append((tip, nxt[1], nxt[2]))
            elif nxt[3] == "absorb":
                self.stop(tip, t, W_ABSORBED, ABSORB)
            elif nxt[3] == "dying":
                self.stop(tip, t, W_KILLED, KILL)
            else:
                cands.setdefault(nxt[0], []).append((tip, nxt[1], nxt[2]))
        for j, group in cands.items():
            if j in hits or len({self.ln_code[self.tp_line[tp]] for tp, _, _ in group}) > 1:
                hits.setdefault(j, []).extend(group)  # case 3, or into a cell hit this step
                continue
            done = set()
            for tip, ein, eout in group:
                if ein in done:  # two tips of one rule onto one chord (from either end): one draws it
                    self.stop(tip, t, W_ABSORBED, ABSORB)
                else:
                    done.update((ein, eout))
                    self.advance_tip(tip, j, ein, eout, t)
        if hits:
            self.hit(hits, t)

    # -- taps
    def tap(self, i, t):
        p = self.taps[i]
        H, W, HW = self.H, self.W, self.HW
        code = self.code_of[p.rule]
        why = ""
        if not (0 <= p.row < H and 0 <= p.col < W):
            why = "off the board"
        else:
            cell = p.row * W + p.col
            if self.ex[code][p.d0 * HW + cell] != p.d1:
                why = "no such chord"
            elif self.cell_code[cell] not in (-1, code) or self.cell_dead[cell] > t and self.dead_code[cell] != code:
                why = "a rival's tile"
            elif self.edge_rec[p.d0 * HW + cell] >= 0:
                why = "your own line"
            elif self.edge_dead[p.d0 * HW + cell] > t:
                why = "your own line, dying"
        if why:
            self.reason[i] = why
            self.events.append((t, REFUSE, -1, p.row, p.col, i))
            return
        self.accepted[i] = True
        line = len(self.ln_tap)
        self.ln_tap.append(i)
        self.ln_code.append(code)
        self.ln_hit.append(INF)
        self.ln_role.append(0)
        self.ln_by.append(-1)
        self.ln_slot.append((code, self.player_lines[code]))
        self.player_lines[code] += 1
        r = self.draw(cell, p.d0, p.d1, line, 0, t)
        self.events.append((t, TAP, line, p.row, p.col, i))
        tips = []
        for side, out in ((1, p.d1), (-1, p.d0)):
            tip = len(self.tp_line)
            tips.append(tip)
            self.tp_line.append(line)
            self.tp_side.append(side)
            self.tp_cell.append(cell)
            self.tp_out.append(out)
            self.tp_rec.append(r)
            self.tp_k.append(0)
            self.tp_next.append(INF)
            self.tp_alive.append(True)
            self.tp_end.append(INF)
            self.tp_why.append(GROWING)
            self.rec_tips.setdefault(r, []).append(tip)
            self.schedule(tip, t + self.period)
        self.ln_tips.append(tips)

    # -- the loop
    def run(self, horizon):
        order = sorted(range(len(self.taps)), key=lambda i: (self.taps[i].t, i))
        ti = 0
        t = self.taps[order[0]].t if order else 0
        while True:
            if horizon is not None and t > horizon:
                break
            self.now = t
            lst = self.due.pop(t, None)
            if lst:
                self.moves(t, lst)
            self.advance_waves(t)
            while ti < len(order) and self.taps[order[ti]].t <= t:
                self.tap(order[ti], t)
                ti += 1
            if self.fronts:
                t += 1
                continue
            nt = INF
            while self.due:
                m = min(self.due)
                if any(self.tp_alive[tp] and self.tp_next[tp] == m for tp in self.due[m]):
                    nt = m
                    break
                del self.due[m]
            if ti < len(order):
                nt = min(nt, self.taps[order[ti]].t)
            if nt >= INF:
                break
            t = nt
        return self.result()

    def result(self) -> Episode:
        W = self.W
        n = len(self.rc_cell)
        recs = [(self.rc_cell[r] // W, self.rc_cell[r] % W, self.rc_a[r], self.rc_b[r], self.rc_line[r], self.rc_k[r],
                 self.rc_on[r], self.rc_off[r], False) for r in range(n)]
        recs += [(c // W, c % W, a, b, line, k, on, off, True) for c, a, b, line, k, on, off in self.mb]
        chords = np.array(recs, _CHORD_T) if recs else np.zeros(0, _CHORD_T)
        L = len(self.ln_tap)
        length = np.bincount(np.asarray(self.rc_line, np.int64), minlength=L) if n else np.zeros(L, np.int64)
        gone = np.full(L, -1, np.int64)
        alive = np.zeros(L, bool)
        for r in range(n):
            line = self.rc_line[r]
            if self.rc_off[r] >= INF:
                alive[line] = True
            else:
                gone[line] = max(gone[line], self.rc_off[r])
        gone[alive] = INF
        lines = np.zeros(L, _LINE_T)
        if L:
            lines["tap"], lines["code"], lines["hit"] = self.ln_tap, self.ln_code, self.ln_hit
            lines["role"], lines["by"], lines["len"] = self.ln_role, self.ln_by, length
            lines["gone"] = gone
        T = len(self.tp_line)
        tips = np.zeros(T, _TIP_T)
        if T:
            tips["line"], tips["side"], tips["end"], tips["why"] = self.tp_line, self.tp_side, self.tp_end, self.tp_why
            cells = np.asarray(self.tp_cell, np.int64)
            tips["row"], tips["col"], tips["k"] = cells // W, cells % W, self.tp_k
        events = np.array(self.events, _EVENT_T) if self.events else np.zeros(0, _EVENT_T)
        return Episode((self.H, self.W), self.taps, self.codes, np.array(self.accepted, bool), self.reason, lines,
                       tips, chords, events, self.last)


def simulate(tab, geo, taps: Sequence[Tap], *, horizon: int | None = None, period: int = PERIOD, wave: int = 1,
             sequential: bool = False, maybe: bool = True, slack: int = SLACK, players=None, exits=None) -> Episode:
    """Run `taps` on board `geo` (int16 [H,W], -1 off board) to rest (or to step `horizon`).

    period: steps per chord (2; 1 is the doc's "worm" fallback). wave: 1 = the CA's dying wave, a chord a step;
    0 = the game's instant wipe. sequential: resolve moves one tip at a time in the engine's order (the parity
    check's game mode) instead of all at once. maybe: add the don't-care-only intervals. slack: the window the
    maybe-chords are judged against (keep = target_weight's). players: rule keys in the engine's player order
    (default: by first tap). exits: {rule key: tab.exits(...)} to reuse."""
    geo = np.asarray(geo)
    return _Sim(tab, geo, taps, period, wave, sequential, maybe, slack, players, exits).run(horizon)


# ---------------------------------------------------------------- drawing episodes

def strand_of(tab, geo, rule, row, col, d0, d1, ex=None):
    """walker.Strand of the chord (d0, d1) at (row, col) under `rule` (a rule key)."""
    rule = rule_key(rule)
    if ex is None:
        ex = tab.exits(rule[0], np.asarray(rule[1:], np.int64), geo)
    return walk(ex, geo >= 0, row, col, d0, d1)


def _chords_of(bits, row, col):
    v = int(bits[row, col])
    return [PAIRS[p] for p in range(15) if v >> p & 1]


def draw_taps(rng: np.random.Generator, tab, geo, kind: str = "collide", n: int | None = None, t_max: int = 24,
              split: str = "train", rules=None) -> list:
    """Taps for one episode of `kind` on board `geo` (rules from `split`, or the given rule keys):
      single   one rule, one random chord at t = 0 (today's slot)
      collide  n (default 2-4) rules; the first taps at 0, each later one taps a chord of its rule on a cell
               the first line will pass, at a random time before it gets there, so the lines meet
      control  2 rules whose strands share no cell (nothing can hit: the false-wipe control); None if the board
               has no such pair after a few tries
      own      one rule, 2-3 taps: on the first strand ahead of its tip (a join at a loose end), or on another
               chord of a cell it passes (lines beside each other)
    Taps may still be refused or meet differently (the sim decides); times are in [0, t_max]."""
    def new_rule():
        s, digits = tab.sample(rng, split)
        return (s, *[int(x) for x in digits])

    def chord_at(rule, rc=None):
        bits = tab.render_bits(rule[0], np.asarray(rule[1:], np.int64), geo)
        if rc is None:
            p, rows, cols = np.nonzero((bits[None].astype(np.int32) >> np.arange(15)[:, None, None]) & 1)
            if not len(p):
                return None
            k = int(rng.integers(len(p)))
            a, b = PAIRS[p[k]]
            r0, c0 = int(rows[k]), int(cols[k])
        else:
            ch = _chords_of(bits, *rc)
            if not ch:
                return None
            a, b = ch[int(rng.integers(len(ch)))]
            r0, c0 = rc
        if rng.integers(2):
            a, b = b, a
        return r0, c0, int(a), int(b)

    pick = list(rules) if rules is not None else None
    for _ in range(20):
        rule0 = rule_key(pick[0]) if pick else new_rule()
        c0 = chord_at(rule0)
        if c0 is None:
            continue
        taps = [Tap(0, *c0, rule0)]
        if kind == "single":
            return taps
        st = strand_of(tab, geo, rule0, *c0)
        dist = np.abs(st.index) if not st.closed else np.minimum(st.index, len(st) - st.index)
        if kind == "own":
            for _ in range(int(rng.integers(1, 3)) if n is None else n - 1):
                j = int(rng.integers(len(st)))
                rc = (int(st.rows[j]), int(st.cols[j]))
                if rng.integers(2) and dist[j] >= 2:  # same strand, ahead of the tip: a join
                    taps.append(Tap(int(rng.integers(0, min(t_max, 2 * int(dist[j]) - 1) + 1)), rc[0], rc[1],
                                    int(st.ins[j]), int(st.outs[j]), rule0))
                else:  # another chord of the cell (if it has one): beside it
                    c = chord_at(rule0, rc)
                    if c is not None:
                        taps.append(Tap(int(rng.integers(0, t_max + 1)), *c, rule0))
            return taps
        if kind == "control":
            cells0 = set(zip(st.rows.tolist(), st.cols.tolist()))
            for _ in range(10):
                rule1 = rule_key(pick[1]) if pick and len(pick) > 1 else new_rule()
                if rule1 == rule0:
                    continue
                c1 = chord_at(rule1)
                if c1 is None:
                    continue
                st1 = strand_of(tab, geo, rule1, *c1)
                if cells0.isdisjoint(zip(st1.rows.tolist(), st1.cols.tolist())):
                    return taps + [Tap(int(rng.integers(0, t_max + 1)), *c1, rule1)]
            return None
        if kind != "collide":
            raise ValueError(kind)
        m = int(rng.integers(2, 5)) if n is None else n
        used = {rule0}
        for q in range(1, m):
            rule = rule_key(pick[q]) if pick and len(pick) > q else new_rule()
            if rule in used:
                continue
            used.add(rule)
            far = np.nonzero(dist >= 1)[0]
            if not len(far):
                break
            j = int(far[rng.integers(len(far))])
            c = chord_at(rule, (int(st.rows[j]), int(st.cols[j])))
            if c is None:
                continue
            taps.append(Tap(int(rng.integers(0, min(t_max, 2 * int(dist[j]) - 1) + 1)), *c, rule))
        if len(taps) >= 2:
            return taps
    return None


# ---------------------------------------------------------------- the bench

def bench(data_dir=None, seconds=5.0, out=print):
    """Fresh episodes per second, per level and kind: rules + taps (draw_taps) + simulate + planes."""
    from .rules import Boards, RuleTable

    tab, bd = RuleTable(data_dir), Boards(data_dir)
    rng = np.random.default_rng(0)
    tab.sample(rng)  # the split, once
    res = {}
    for g, S in (("L2", 12), ("L3", 37), ("L4", 42)):
        for kind in ("single", "collide", "own"):
            n = chords = steps = 0
            t_sim = 0.0
            t0 = time.time()
            while time.time() - t0 < seconds:
                geo = np.full((S, S), -1, np.int16)
                b = bd.board(g, int(rng.integers(len(bd.hw[g]))))
                geo[:b.shape[0], :b.shape[1]] = b
                taps = draw_taps(rng, tab, geo, kind)
                if taps is None:
                    continue
                t1 = time.time()
                ep = simulate(tab, geo, taps)
                ep.planes()
                t_sim += time.time() - t1
                n, chords, steps = n + 1, chords + len(ep.chords), steps + ep.settle
            dt = time.time() - t0
            res[f"{g}/{kind}"] = round(n / dt, 1)
            out(f"{g} {kind:8s}: {n / dt:7.1f} episodes/s ({1000 * dt / max(1, n):.2f} ms each, sim + planes "
                f"{1000 * t_sim / max(1, n):.2f} ms); mean {chords / max(1, n):.0f} chords, settle {steps / max(1, n):.0f}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data/strand-v2")
    ap.add_argument("--test", action="store_true", help="the unit tests (nca/strand/test_sim.py)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--parity", action="store_true", help="against Spectacle's engine (collide.json)")
    ap.add_argument("--collide", default=None, help="the fixture (default <data>/collide.json)")
    ap.add_argument("--show", type=int, default=12, help="--parity: differences to print per class")
    args = ap.parse_args(argv)
    bad = 0
    if args.test:
        from . import test_sim
        bad += test_sim.main([] if args.data is None else ["--data", args.data])
    if args.parity:
        from .parity_sim import parity
        bad += parity(args.data, args.collide, show=args.show)
    if args.bench:
        bench(args.data, args.seconds)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
