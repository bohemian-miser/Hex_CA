"""Unit tests of the event simulator (nca/strand/sim.py): `python -m nca.strand.test_sim` or
`python -m nca.strand.sim --test` (~1 min on the Pi; needs data/strand-v2). Plain functions named test_*, so pytest
runs them too.

  1. one tap, nothing else = the walker's targets at double time: on_at = t_tap + 2 x train.dist_planes on the
     strand, INF elsewhere, off_at INF, one interval; settle = 2 x grow_steps; and with the default window the
     edge is required from 2 (k + 1) -- today's due age k + 1, doubled (L2 / L3 / L4, every subset, loops too)
  2. invariants on random episodes of every kind, CA and game modes: never two rules on one cell; every chord
     drawn at t_tap + 2|k| as a contiguous stretch of its line's strand; nothing lasts on a hit line and
     nothing is lost by a line no wave touched; every tip stops; refused taps draw nothing
  3. the dying wave: on episodes with one hit and lines that touch no other line of their rule, every erased
     chord goes at h + its distance along the line from the hit (the hitter's from the cell it hit), the fleeing
     part (drawn after h) included; game mode (wave=0) erases a hit line whole at h
  4. hand-placed cases: a hit at a known step, head-on tips, two tips into one empty cell (case 3), taps refused
     on a rival's tile and on an own chord but not on another chord of an own cell, a join at a loose end (two
     taps of one rule on one strand cover the whole strand), an edge drawn twice (K = 2)
  5. target_weight: the windows on hand-made intervals, numpy = torch
"""

import argparse
import sys

import numpy as np

from . import train as V1
from .rules import PAIR_INDEX, Boards, RuleTable
from .sim import (ABSORB, CONTACT, EARLY, HIT, HITTER, INF, SLACK, VICTIM, Tap, draw_taps, simulate, strand_of,
                  target_weight)
from .walker import PAIRS

_CTX = {}
FAILS = []


def ctx(data=None):
    if "tab" not in _CTX:
        _CTX["tab"], _CTX["bd"] = RuleTable(data), Boards(data)
    return _CTX["tab"], _CTX["bd"]


def check(msg, ok):
    print(("OK   " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)
    return ok


def padded(bd, group, i, S):
    b = bd.board(group, i)
    geo = np.full((S, S), -1, np.int16)
    geo[:b.shape[0], :b.shape[1]] = b
    return geo


def boards(rng, bd, n):
    out = []
    for _ in range(n):
        g, S = (("L2", 12), ("L3", 37), ("L4", 42))[int(rng.integers(3))]
        out.append(padded(bd, g, int(rng.integers(len(bd.hw[g]))), S))
    return out


def drawn_at(ep, t):
    """{(row, col, a, b): line} of the real chords drawn at time t."""
    c = ep.chords
    on = (~c["maybe"]) & (c["on"] <= t) & (c["off"] > t)
    return {(int(r), int(q), int(min(a, b)), int(max(a, b))): int(l)
            for r, q, a, b, l in zip(c["row"][on], c["col"][on], c["a"][on], c["b"][on], c["line"][on])}


# ---------------------------------------------------------------- 1. the walker at double time

def test_single_strand_is_walker_doubled(n=300, seed=1):
    tab, bd = ctx()
    rng = np.random.default_rng(seed)
    bad, loops, n_ok = [], 0, 0
    for geo in boards(rng, bd, n):
        taps = draw_taps(rng, tab, geo, "single")
        if taps is None:
            continue
        t0 = int(rng.integers(0, 5))
        p = taps[0]._replace(t=t0)
        st = strand_of(tab, geo, p.rule, p.row, p.col, p.d0, p.d1)
        loops += st.closed
        ep = simulate(tab, geo, [p])
        a = V1.dist_planes(st, geo.shape)
        want_on = np.where(a < INF, t0 + 2 * a.astype(np.int32), INF)
        ok = (ep.on_at.shape[0] == 1 and np.array_equal(ep.on_at[0], want_on) and (ep.off_at == INF).all()
              and ep.settle == t0 + 2 * st.grow_steps() and len(ep.chords) == len(st))
        # the window: required from 2 (k + 1) after the tap = today's due age (k + 1, "a < age"), doubled
        for age in (t0 + 1, t0 + 2, t0 + 7, t0 + 2 * st.grow_steps() + 3):
            tgt, care = target_weight(ep.on_at, ep.off_at, age)
            old_due = (a < INF) & (2 * (a.astype(np.int32) + 1) <= age - t0)
            ok &= np.array_equal(tgt, old_due) and bool(care[a == INF].all())
        if ok:
            n_ok += 1
        else:
            bad.append(f"{p}")
    check(f"one tap = walker targets at double time: {n_ok}/{n_ok + len(bad)} (loops {loops}; on_at = t + 2 dist, "
          f"off_at INF, settle = t + 2 grow_steps, required from 2 (k + 1))" + (f"; first bad {bad[0]}" if bad else ""),
          not bad and loops > 0)


# ---------------------------------------------------------------- 2. invariants

def invariants(tab, geo, ep, wave):
    """Problems with an episode (a list of strings)."""
    errs = []
    c = ep.chords
    real = c[~c["maybe"]]
    taps = ep.taps
    lines = ep.lines
    # chord times and contiguity along the line's strand
    for li in range(len(lines)):
        p = taps[lines["tap"][li]]
        mine = real[real["line"] == li]
        sched = p.t + 2 * np.abs(mine["k"])
        if not ((mine["on"] == sched).all() if wave else (mine["on"] >= sched).all()):  # game: the engine's heads lag
            errs.append(f"line {li}: a chord off the 2|k| schedule")
        st = strand_of(tab, geo, p.rule, p.row, p.col, p.d0, p.d1)
        pos = {(int(r), int(q), int(PAIR_INDEX[a, b])): int(i) for r, q, a, b, i in
               zip(st.rows, st.cols, st.ins, st.outs, st.index)}
        n = len(st)
        for ch in mine:
            i = pos.get((int(ch["row"]), int(ch["col"]), int(PAIR_INDEX[ch["a"], ch["b"]])))
            if i is None:
                errs.append(f"line {li}: chord not on its strand")
                break
            k = int(ch["k"])
            if not (i == k or (st.closed and (i - k) % n == 0)):
                errs.append(f"line {li}: chord at strand index {i} but k = {k}")
                break
        if len(set(mine["k"].tolist())) != len(mine):
            errs.append(f"line {li}: a k twice")
    # one rule per cell, at every change time
    code_of = lines["code"]
    for t in sorted(set(c["on"].tolist()) | set(c["off"][c["off"] < INF].tolist())):
        seen = {}
        for (r, q, _, _), li in drawn_at(ep, t).items():
            if seen.setdefault((r, q), code_of[li]) != code_of[li]:
                errs.append(f"t={t}: two rules on cell ({r}, {q})")
                break
    # fates at rest
    for li in range(len(lines)):
        left = (real["line"] == li) & (real["off"] >= INF)
        if lines["role"][li] & (HITTER | VICTIM) and left.any():
            errs.append(f"line {li}: hit at {lines['hit'][li]} but {int(left.sum())} chords left")
        if lines["role"][li] == 0 and ((real["line"] == li) & (real["off"] < INF)).any():
            errs.append(f"line {li}: lost chords without a hit or a wave")
    if (ep.tips["end"] >= INF).any():
        errs.append("a tip still growing at rest")
    for i, ok in enumerate(ep.accepted):
        if not ok and (lines["tap"] == i).any():
            errs.append(f"refused tap {i} has a line")
    if wave == 0:
        for li in np.nonzero(lines["role"] & (HITTER | VICTIM))[0]:
            mine = real[real["line"] == li]
            h = lines["hit"][li]
            if not ((mine["off"] == h) | (mine["on"] > h)).all() and not (lines["role"][li] & CONTACT):
                errs.append(f"game mode: line {li} not erased whole at its hit {h}")
    return errs


def test_invariants(n=240, seed=2):
    tab, bd = ctx()
    rng = np.random.default_rng(seed)
    counts = {}
    bad = []
    kinds = ("single", "collide", "control", "own")
    for geo in boards(rng, bd, n):
        kind = kinds[int(rng.integers(len(kinds)))]
        taps = draw_taps(rng, tab, geo, kind)
        if taps is None:
            continue
        for wave, seq in ((1, False), (0, True)):
            ep = simulate(tab, geo, taps, wave=wave, sequential=seq)
            errs = invariants(tab, geo, ep, wave)
            k = f"{kind}/{'ca' if wave else 'game'}"
            counts[k] = counts.get(k, 0) + 1
            counts["hits"] = counts.get("hits", 0) + int((ep.events["kind"] == HIT).sum())
            counts["absorbs"] = counts.get("absorbs", 0) + int((ep.events["kind"] == ABSORB).sum())
            if errs:
                bad.append(f"{k}: {errs[0]} ({taps})")
    check(f"invariants on {sum(v for k, v in counts.items() if '/' in k)} episodes ({counts}): "
          f"{len(bad)} bad" + (f"; first: {bad[0]}" if bad else ""), not bad and counts.get("hits", 0) > 20)


# ---------------------------------------------------------------- 3. the wave

def test_wave_timing(n=600, seed=3):
    """Lines in a single hit, touching nothing else of their rule: off = h + distance along the line."""
    tab, bd = ctx()
    rng = np.random.default_rng(seed)
    checked = fleeing = loops = 0
    bad = []
    for geo in boards(rng, bd, n):
        taps = draw_taps(rng, tab, geo, "collide", n=2)
        if taps is None:
            continue
        ep = simulate(tab, geo, taps)
        ev = ep.events
        hits = ev[ev["kind"] == HIT]
        if len(hits) != 1 or (ep.lines["role"] & CONTACT).any() or len(set(ep.lines["code"].tolist())) != len(ep.lines):
            continue
        h, hl, row, col = int(hits["t"][0]), int(hits["line"][0]), int(hits["row"][0]), int(hits["col"][0])
        real = ep.chords[~ep.chords["maybe"]]
        for li in range(len(ep.lines)):
            if not ep.lines["role"][li] & (HITTER | VICTIM):
                continue
            mine = real[real["line"] == li]
            if li == hl:  # the hitter: its origin is the cell it hit, one beyond the tip that hit
                tip = ep.tips[(ep.tips["line"] == li) & (ep.tips["end"] == h) & (ep.tips["why"] == 3)]
                k0 = [int(tip["k"][0]) + int(tip["side"][0])]
            else:
                k0 = mine["k"][(mine["row"] == row) & (mine["col"] == col) & (mine["on"] < h)].tolist()
            if not k0:
                bad.append(f"line {li}: no origin")
                continue
            p = ep.taps[ep.lines["tap"][li]]
            if strand_of(tab, geo, p.rule, p.row, p.col, p.d0, p.d1).closed:
                # round a loop the fleeing tip can run back into the cell the hit emptied: only "all gone"
                loops += 1
                if (mine["off"] >= INF).any():
                    bad.append(f"loop line {li} hit at {h}: chords left")
                continue
            want = h + np.abs(mine["k"][:, None] - np.asarray(k0)[None]).min(1)
            fleeing += int((mine["on"] > h).sum())
            if not np.array_equal(mine["off"], want):
                bad.append(f"line {li} hit at {h}: off {mine['off'].tolist()} want {want.tolist()} ({taps})")
            checked += 1
    check(f"dying wave: {checked} hit lines on open strands, every chord goes at h + its distance from the hit "
          f"({fleeing} drawn after the hit by a fleeing tip); {loops} on loops all gone: {len(bad)} bad" + (f"; first: {bad[0]}" if bad else ""),
          not bad and checked > 50 and fleeing > 0)


# ---------------------------------------------------------------- 4. hand-placed cases

def chords_at(tab, rule, geo, row, col):
    bits = int(tab.render_bits(rule[0], np.asarray(rule[1:], np.int64), geo)[row, col])
    return [PAIRS[p] for p in range(15) if bits >> p & 1]


def find_pair(tab, geo, rng, tries=400):
    """Two rules A, B and chords: A's strand (open, >= 8 chords) passes a cell at distance d >= 3 from A's tap where
    B has a chord. Returns (tapA, tapB at that cell, d, strand A) or None."""
    for _ in range(tries):
        a = draw_taps(rng, tab, geo, "single")
        if a is None:
            continue
        pa = a[0]
        st = strand_of(tab, geo, pa.rule, pa.row, pa.col, pa.d0, pa.d1)
        if st.closed or len(st) < 8:
            continue
        idx = [i for i in range(len(st)) if st.index[i] >= 3]
        if not idx:
            continue
        i = idx[int(rng.integers(len(idx)))]
        # the cell must be visited once by A's strand
        if ((st.rows == st.rows[i]) & (st.cols == st.cols[i])).sum() != 1:
            continue
        s, digits = tab.sample(rng)
        rb = (s, *[int(x) for x in digits])
        if rb == pa.rule:
            continue
        ch = chords_at(tab, rb, geo, int(st.rows[i]), int(st.cols[i]))
        if not ch:
            continue
        b0, b1 = ch[0]
        return pa, Tap(0, int(st.rows[i]), int(st.cols[i]), int(b0), int(b1), rb), int(st.index[i]), st
    return None


def test_hand_cases(seed=4):
    tab, bd = ctx()
    rng = np.random.default_rng(seed)
    geo = padded(bd, "L3", 0, 37)
    # (a) A taps at 0, B is already on a cell A reaches at 2d: A hits B there at 2d (or earlier, if B's line,
    # growing, met A first) -- in any case both lines are gone at rest and nothing else is
    ok_a = ok_ab = 0
    for _ in range(40):
        f = find_pair(tab, geo, rng)
        if f is None:
            continue
        pa, pb, d, st = f
        ep = simulate(tab, geo, [pa, pb])
        if not ep.accepted.all():
            continue
        ok_ab += 1
        hits = ep.events[ep.events["kind"] == HIT]
        left = drawn_at(ep, ep.settle + 1)
        first = int(hits["t"].min()) if len(hits) else INF
        ok_a += len(hits) >= 1 and first <= 2 * d and not left
    check(f"(a) A meets B on its strand: hit by 2d and both gone at rest in {ok_a}/{ok_ab}", ok_ab >= 10 and ok_a == ok_ab)

    # (b) head-on: A's tip and B's tip on adjacent cells, both moving at once -> both hit at the same step
    # (c) case 3: two rules' tips into one empty cell at once -> both hit there, nothing drawn on it
    head_on = case3 = 0
    for _ in range(300):
        taps = draw_taps(rng, tab, geo, "collide", n=2)
        if taps is None:
            continue
        ep = simulate(tab, geo, taps)
        hits = ep.events[ep.events["kind"] == HIT]
        for t in set(hits["t"].tolist()):
            ht = hits[hits["t"] == t]
            if len(ht) == 2 and set(ht["line"].tolist()) == {0, 1}:
                if (ht["other"] == -1).all() and len({(r, q) for r, q in zip(ht["row"], ht["col"])}) == 1:
                    real = ep.chords[~ep.chords["maybe"]]
                    cell = (real["row"] == ht["row"][0]) & (real["col"] == ht["col"][0])
                    case3 += not ((real["on"][cell] <= t) & (real["off"][cell] > t)).any()
                elif set(ht["other"].tolist()) == {0, 1}:
                    head_on += 1
    check(f"(b, c) head-on hits ({head_on}) and case-3 hits into one empty cell ({case3}) both seen, nothing drawn "
          f"on a case-3 cell", head_on > 0 and case3 > 0)

    # (d) taps: refused on a rival's tile and on an own chord; another chord of an own cell is fine
    for _ in range(400):
        f = find_pair(tab, geo, rng)
        if f is None:
            continue
        pa, pb = f[0], f[1]
        others = [c for c in chords_at(tab, pa.rule, geo, pa.row, pa.col) if c != tuple(sorted((pa.d0, pa.d1)))]
        theirs = chords_at(tab, pb.rule, geo, pa.row, pa.col)
        if others and theirs:
            break
    taps = [pa, pa._replace(t=1), Tap(1, pa.row, pa.col, *others[0], pa.rule), Tap(1, pa.row, pa.col, *theirs[0], pb.rule)]
    ep = simulate(tab, geo, taps)
    check(f"(d) refusals: the own chord again -> '{ep.reason[1]}', another chord of the own cell accepted, a rival "
          f"on the tile -> '{ep.reason[3]}'", ep.accepted.tolist() == [True, False, True, False])

    # (e) a join at a loose end: two taps of one rule on one open strand, the second ahead of the first's tip ->
    # the strand drawn whole, one tip of each absorbed, no hit
    ok_e = n_e = 0
    for _ in range(200):
        taps = draw_taps(rng, tab, geo, "single")
        if taps is None:
            continue
        p = taps[0]
        st = strand_of(tab, geo, p.rule, p.row, p.col, p.d0, p.d1)
        far = np.nonzero(st.index >= 4)[0]
        if st.closed or not len(far):
            continue
        j = int(far[0]) + int(rng.integers(len(far)))
        j = min(j, len(st) - 1)
        q = Tap(int(rng.integers(0, 2 * int(st.index[j]) - 1)), int(st.rows[j]), int(st.cols[j]), int(st.ins[j]),
                int(st.outs[j]), p.rule)
        ep = simulate(tab, geo, [p, q])
        if not ep.accepted.all():
            continue
        n_e += 1
        left = drawn_at(ep, ep.settle + 1)
        ok_e += (len(left) == len(st) and not (ep.events["kind"] == HIT).any()
                 and int((ep.events["kind"] == ABSORB).sum()) >= 1)
    check(f"(e) joins: two taps on one strand draw it whole with an absorption, {ok_e}/{n_e}", n_e >= 20 and ok_e == n_e)

    # (f) an edge drawn twice: a line that was hit, then a later tap of its rule on one of its chords -> K = 2
    ok_f, info = False, "none found"
    for _ in range(300):
        taps = draw_taps(rng, tab, geo, "collide", n=2)
        if taps is None:
            continue
        ep = simulate(tab, geo, taps)
        real = ep.chords[~ep.chords["maybe"]]
        gone = real[real["off"] < INF]
        if not len(gone):
            continue
        g = gone[0]
        rule = ep.taps[ep.lines["tap"][g["line"]]].rule
        t2 = max(int(ep.settle) + 3, int(g["off"]) + SLACK + EARLY + 2)
        ep2 = simulate(tab, geo, list(taps) + [Tap(t2, int(g["row"]), int(g["col"]), int(g["a"]), int(g["b"]), rule)])
        if not ep2.accepted[-1]:
            continue
        K = ep2.on_at.shape[0]
        e = (int(g["a"]), int(g["row"]), int(g["col"]))
        tgt_mid, _ = target_weight(ep2.on_at, ep2.off_at, int(g["off"]) - 1)
        tgt_gap, care_gap = target_weight(ep2.on_at, ep2.off_at, t2 - 1 - EARLY)
        tgt_new, _ = target_weight(ep2.on_at, ep2.off_at, t2 + SLACK)
        ok_f = K >= 2 and (tgt_mid[e] or int(g["off"]) - 1 < int(g["on"]) + SLACK) and not tgt_gap[e] and care_gap[e] \
            and bool(tgt_new[e])
        info = (f"K {K}, edge {e}: on {ep2.on_at[(slice(None),) + e].tolist()} off {ep2.off_at[(slice(None),) + e].tolist()}"
                f", target at {int(g['off']) - 1} / {t2 - 1 - EARLY} / {t2 + SLACK}: {bool(tgt_mid[e])} / "
                f"{bool(tgt_gap[e])} (care {bool(care_gap[e])}) / {bool(tgt_new[e])}")
        break
    check(f"(f) an edge drawn, wiped and drawn again: K >= 2 and the target follows both intervals ({info})", ok_f)


# ---------------------------------------------------------------- 5. the window

def test_target_weight():
    on = np.full((2, 6, 1, 1), INF, np.int16)
    off = np.full((2, 6, 1, 1), INF, np.int16)
    on[0, 0], off[0, 0] = 10, 20        # an interval
    on[0, 1], off[0, 1] = 10, 11        # too short to be required: don't-care only
    on[0, 2], off[0, 2] = 5, 7          # two intervals on one edge
    on[1, 2], off[1, 2] = 30, INF
    exp = {}
    for age in range(0, 40):
        tgt, care = target_weight(on, off, age)
        exp[age] = (tgt[:, 0, 0].copy(), care[:, 0, 0].copy())
    t = lambda a, e: (bool(exp[a][0][e]), bool(exp[a][1][e]))  # noqa: E731
    ok = (t(8, 0) == (False, True) and t(9, 0) == (False, False) and t(11, 0) == (False, False)
          and t(12, 0) == (True, True) and t(19, 0) == (True, True) and t(20, 0) == (False, False)
          and t(21, 0) == (False, False) and t(22, 0) == (False, True))
    ok &= all(not exp[a][0][1] for a in range(40)) and t(8, 1) == (False, True) and t(9, 1) == (False, False) \
        and t(12, 1) == (False, False) and t(13, 1) == (False, True)
    ok &= t(3, 2) == (False, True) and t(4, 2) == (False, False) and t(7, 2) == (False, False) \
        and t(9, 2) == (False, True) and t(28, 2) == (False, True) and t(29, 2) == (False, False) \
        and t(32, 2) == (True, True)
    ok &= all(exp[a][1][3] and not exp[a][0][3] for a in range(40))  # never drawn: 0, always cared about
    try:
        import torch
        tt, tc = target_weight(torch.from_numpy(on)[None].expand(3, -1, -1, -1, -1),
                               torch.from_numpy(off)[None].expand(3, -1, -1, -1, -1),
                               torch.tensor([8, 12, 21]).view(3, 1, 1, 1, 1))
        same = all(np.array_equal(tt[i].numpy(), target_weight(on, off, a)[0]) and
                   np.array_equal(tc[i].numpy(), target_weight(on, off, a)[1]) for i, a in enumerate((8, 12, 21)))
    except ImportError:
        same = True
    check(f"target_weight: the windows (0 before on - {EARLY}, don't-care to on + {SLACK}, 1 to off, don't-care to "
          f"off + {SLACK}, 0 after; short intervals don't-care only; two intervals on one edge); torch = numpy {same}",
          ok and same)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    args = ap.parse_args(argv)
    ctx(args.data)
    for f in (test_target_weight, test_single_strand_is_walker_doubled, test_invariants, test_wave_timing,
              test_hand_cases):
        f()
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
