"""Episodes for nca/strand/train3.py: a pool slot's taps and what the CA must draw, and when -- the adapter between
the trainer and nca/strand/sim.py (docs/spectacle-nca-taps.md §2, §4.3), which owns every rule of what happens
(tips, the 1/2 speed, hits, dying waves, absorption, refusals) and the targets' timing:

  sim.Tap(t, row, col, d0, d1, rule): a tap at age t of the slot, given to the CA's update from t to t + 1
  ep.planes(K) -> on_at, off_at int16 [K,6,H,W]: each edge's intervals [on, off) in nominal steps (chord k of a
      tap's strand at t + period * k), INF = never; an interval with on == off is don't-care only
  target / care at an age: sim.target_weight (1 from on + slack to off; don't-care from on - early and from off
      to off + slack); train3.target_weight is the same with one extension (early < 0)
  ep.settle (the last change), ep.lines (fates: hit, ...), ep.accepted (the host's refusals)

Here: a slot's tap rows (int16 [MAXT, 15]: row, col, d0, d1, s, digit_0..digit_8, t; row -1 unused), the
episode draws of train3's kinds on top of sim.draw_taps (refused taps dropped: the CA never sees one; a collision
whose first hit is past hit_max drawn again; an own meeting needs two taps that stood), and a tap landing on a
kept slot mid-episode (midtap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sim as SIM
from .walker import DCOL, DROW, PAIRS

INF = SIM.INF
MAXT = 4             # taps a slot holds
K_MAX = 4            # intervals an edge can hold in a slot (sim: drawn, wiped, drawn again; don't-care-only ones)
TAP_COLS = 15
SIM_KIND = {"t1": "single", "t2": "collide", "ctrl": "control", "t3": "own"}


def tap_row(tp: SIM.Tap) -> np.ndarray:
    return np.array([tp.row, tp.col, tp.d0, tp.d1, *tp.rule, tp.t], np.int16)


def taps_array(taps) -> np.ndarray:
    """int16 [MAXT, 15]: a slot's tap rows (unused rows all -1)."""
    out = np.full((MAXT, TAP_COLS), -1, np.int16)
    for i, tp in enumerate(taps):
        out[i] = tap_row(tp)
    return out


def taps_of(arr) -> list:
    """sim.Taps from a slot's tap rows."""
    out = []
    for r in np.asarray(arr):
        if int(r[0]) >= 0:
            r = [int(x) for x in r]
            out.append(SIM.Tap(r[14], r[0], r[1], r[2], r[3], tuple(r[4:14])))
    return out


@dataclass
class Slot:
    """An episode as the pool holds it."""
    taps: list            # the taps that stood (sim.Tap)
    on: np.ndarray        # int16 [K_MAX,6,H,W]
    off: np.ndarray
    settle: int
    hit: int              # the first hit's step (-1 none)
    kind: str             # t1 / t2 / ctrl / t3, by what happened
    length: int           # chords drawn (by every line, wiped or not)
    codes: int            # distinct rules


def run(tab, geo, taps, period, slack, exits=None):
    """sim.simulate, then again without the taps the host refused (they changed nothing; the CA never sees them).
    Returns (Slot, sim episode)."""
    ep = SIM.simulate(tab, geo, taps, period=period, slack=slack, exits=exits)
    if not ep.accepted.all():
        taps = [tp for tp, ok in zip(taps, ep.accepted) if ok]
        ep = SIM.simulate(tab, geo, taps, period=period, slack=slack, exits=exits)
    on, off = ep.planes(K_MAX)
    h = ep.lines["hit"][ep.lines["hit"] < INF] if len(ep.lines) else np.zeros(0)
    codes = len({SIM.rule_key(tp.rule) for tp in taps})
    hit = int(h.min()) if len(h) else -1
    kind = "t2" if hit >= 0 else ("t1" if len(taps) == 1 else ("t3" if codes == 1 else "ctrl"))
    return Slot(list(taps), on, off, int(ep.settle), hit, kind, int((~ep.chords["maybe"]).sum()), codes), ep


def draw(rng, tab, geo, kind, period, slack, split="train", t_max=24, hit_max=48, rules=None, tries=20):
    """A Slot of train3's `kind` (t1 one tap; t2 a collision with its first hit by hit_max; ctrl a control pair;
    t3 an own meeting: two or more taps that stood) via sim.draw_taps, or None."""
    for _ in range(tries):
        taps = SIM.draw_taps(rng, tab, geo, SIM_KIND[kind], t_max=t_max, split=split, rules=rules)
        if taps is None:
            continue
        try:
            slot, _ = run(tab, geo, taps, period, slack)
        except ValueError:  # an edge with more than K_MAX intervals
            continue
        if kind == "t2" and not 0 <= slot.hit <= hit_max:
            continue
        if kind == "t3" and (len(slot.taps) < 2 or slot.kind != "t3"):
            continue
        if kind == "ctrl" and slot.kind != "ctrl":
            continue
        return slot
    return None


def single(tab, geo, tap4, rule, period, slack):
    """One tap at age 0 (the quick check's legacy / wide / persist taps)."""
    return run(tab, geo, [SIM.Tap(0, *[int(x) for x in tap4], SIM.rule_key(rule))], period, slack)[0]


def neighbours(cells: np.ndarray) -> np.ndarray:
    """bool [H,W]: the cells next to `cells` (not in it)."""
    H, W = cells.shape
    out = np.zeros_like(cells)
    rows, cols = np.nonzero(cells)
    for d in range(6):
        r, c = rows + DROW[d], cols + DCOL[d]
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        out[r[ok], c[ok]] = True
    return out & ~cells


def midtap(rng, tab, geo, taps, t, rival, period, slack, split="train", tries=20):
    """A tap landing on a kept slot at age t (decision 6's "new taps landing mid-episode"): rival, a new rule's
    chord on a cell next to or ahead of the slot's lines (one of them will hit it, or it them); own, a chord of the
    slot's first rule on or next to its lines (a join, or a line beside). The episode is simulated again with it (the
    past is unchanged); a tap the host refuses is not taken. Returns the new Slot, or None."""
    if len(taps) >= MAXT or not taps:
        return None
    base, _ = run(tab, geo, taps, period, slack)
    real = base.on < base.off
    alive = (real & (base.off > t)).any((0, 1))                 # cells a line holds at t or will
    cand = alive | neighbours(alive)
    rows, cols = np.nonzero(cand & (geo >= 0))
    if not len(rows):
        return None
    used = {SIM.rule_key(tp.rule) for tp in taps}
    for _ in range(tries):
        j = int(rng.integers(len(rows)))
        r, c = int(rows[j]), int(cols[j])
        if rival:
            s, digits = tab.sample(rng, split)
            rule = SIM.rule_key((s, *digits))
            if rule in used:
                continue
        else:
            rule = SIM.rule_key(taps[0].rule)
        bits = int(tab.render_bits(rule[0], np.asarray(rule[1:], np.int64), geo)[r, c])
        ps = [p for p in range(15) if bits >> p & 1]
        if not ps:
            continue
        d0, d1 = PAIRS[ps[int(rng.integers(len(ps)))]]
        if rng.integers(2):
            d0, d1 = d1, d0
        new = SIM.Tap(int(t), r, c, int(d0), int(d1), rule)
        try:
            slot, ep = run(tab, geo, list(taps) + [new], period, slack)
        except ValueError:
            continue
        if len(slot.taps) <= len(taps):
            continue  # refused
        if rival and not (ep.lines["hit"][-1] < INF):
            continue  # a rival that never meets anything is a control, not a hit
        return slot
    return None
