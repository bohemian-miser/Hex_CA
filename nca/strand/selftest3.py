"""Self-check of the tap-as-event trainer (nca/strand/train3.py) and its episodes (nca/strand/episodes.py):
`python -m nca.strand.selftest3` (a few minutes on the Pi; needs data/strand-v2 and data/strand, as selftest2).
`python -m nca.strand.selftest3 --overfit` runs §1's Pi overfits instead (docs/spectacle-nca-taps.md; ~1-2 h).

  1. episodes (the adapter over nca/strand/sim.py): one tap at speed 1 = train2's a, at speed 2 = 2a, nothing goes,
     settle = the last chord; the draws are what they say (t2: codes differ and a line is hit by hit_max; ctrl: two
     codes, no hit; t3: one code, two taps or more; every tap stood); a rival midtap hits, an own one doesn't; the
     slot's planes = sim's; train3.target_weight = sim.target_weight on those episodes at every age
  2. targets: the windows on a hand-made interval (don't-care from on - early to on + slack and from off to off +
     slack); --speed 1 --slack 1 --early -1 is train2.target_weight exactly, every age, every slot of a pool
  3. taps: impulse planes on the tapped cell exactly in steps t + 1 .. t + tapSteps (held: from t + 1 on; fixed:
     never), a slot's later taps as theirs come; the fixed write sets the two edges and the 35 code channels and
     nothing else, and code35 decodes back to the rule; Runner.step = FrameNCA.step bit for bit
  4. the quick check: an oracle that draws the targets (each edge as late as allowed) scores exact 1.0, exit 1.0,
     speed 1 / speed, persist 1.0 at x1 x2 x4, own exact 1.0 with collateral / phantom / stray 0, sim's collisions
     and controls exact with wipe 1.0, falseWipe 0, regrow 0 and worm <= wormMax; one that lets lines decay after
     2 x ideal keeps x1 x2 and loses x4; one that drops part of an own meeting late: collateral, phantom > 0;
     hand-made hit episodes (off = on + 12): worm = wormMax exactly; never wiping: wipe 0; drawing again: regrow
     1.0; the probes read an injected code back exactly (linear and MLP), byAge too, and a random state at chance
  5. runs: --tap impulse / fixed / held (T1), T3 with midtaps, T2, --overfit; a forced collapse rolls back;
     --resume is bit-exact; --init from a train2 checkpoint; pool.npz reads through nca.dashboard.pool_to_json;
     nca.strand.export writes the tap mode (version 2), its self-check passes and its event rollout with a held tap
     = the held rollout
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import episodes as E
from . import sim as SIM
from . import train as V1
from . import train2 as T2
from . import train3 as T
from .rules import CODE_BITS, Boards, RuleTable

FAILS = []


def check(msg, ok):
    print(("OK   " if ok else "FAIL ") + msg, file=sys.__stdout__, flush=True)
    if not ok:
        FAILS.append(msg)


def cfg_of(**kw):
    cfg = {"tap": "impulse", "tapSteps": 1, "speed": 2, "slack": 2, "early": 1, "channels": 32, "evalCap": 400,
           "evalMult": 4, "control": 0.25, "stagger": 12, "hitMax": 40, "fixedChannels": None}
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------- 1. episodes

def episode_checks(tab, bd):
    rng = np.random.default_rng(1)
    geo = T2.pad_geo(bd.board("L3", 1), 37)
    bad = 0
    for _ in range(30):
        tp = SIM.draw_taps(rng, tab, geo, "single")[0]
        slot = T2.make_slot(tab, geo, np.array(tp.rule), (tp.row, tp.col, tp.d0, tp.d1), 0, "m1a")
        a = slot["a"].astype(np.int32)
        for period in (1, 2):
            sl = E.single(tab, geo, (tp.row, tp.col, tp.d0, tp.d1), tp.rule, period, 2)
            bad += not np.array_equal(sl.on[0].astype(np.int32), np.where(a < E.INF, period * a, E.INF))
            bad += not (sl.on[1:] == E.INF).all() or not (sl.off == E.INF).all()
            bad += sl.settle != period * int(a[a < E.INF].max()) or sl.kind != "t1" or sl.hit != -1
    check("episodes: one tap at speed 1 = train2's a, at speed 2 = 2a; nothing goes; settle = the last chord", bad == 0)
    nb = len(bd.hw["L3"])
    got = {k: 0 for k in ("t2", "ctrl", "t3")}
    ok = True
    tw_bad = 0
    for k in range(30):
        geo = T2.pad_geo(bd.board("L3", k % nb), 37)
        for kind in got:
            sl = E.draw(rng, tab, geo, kind, 2, 2, t_max=12, hit_max=40)
            if sl is None:
                continue
            got[kind] += 1
            ep = SIM.simulate(tab, geo, sl.taps)
            on, off = ep.planes(E.K_MAX)
            ok &= bool(ep.accepted.all()) and np.array_equal(on, sl.on) and np.array_equal(off, sl.off)
            codes = len({t.rule for t in sl.taps})
            if kind == "t2":
                ok &= 0 <= sl.hit <= 40 and codes >= 2 and sl.kind == "t2"
            elif kind == "ctrl":
                ok &= sl.hit == -1 and codes == 2 and sl.kind == "ctrl"
            else:
                ok &= sl.hit == -1 and codes == 1 and len(sl.taps) >= 2
            ok &= all(0 <= t.t <= 12 for t in sl.taps)
            for age in range(0, sl.settle + 6, 3):
                a1 = SIM.target_weight(sl.on, sl.off, age)
                a2 = T.target_weight(sl.on.astype(np.int32), sl.off.astype(np.int32), age, SIM.SLACK, SIM.EARLY)
                tw_bad += not (np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1]))
    check(f"episodes: draws {got}: t2 hit by hit_max with 2+ codes, ctrl 2 codes no hit, t3 one code 2+ taps; every "
          f"tap stood; the slot's planes = sim's", ok and min(got.values()) >= 20)
    check("episodes: train3.target_weight = sim.target_weight on every drawn episode, every third age", tw_bad == 0)
    n_r = n_o = hit_r = 0
    for k in range(30):
        geo = T2.pad_geo(bd.board("L3", k % nb), 37)
        base = E.draw(rng, tab, geo, "t1", 2, 2)
        for rival in (True, False):
            sl = E.midtap(rng, tab, geo, base.taps, 20, rival, 2, 2)
            if sl is None:
                continue
            if rival:
                n_r += 1
                hit_r += sl.hit >= 0 and len({t.rule for t in sl.taps}) == 2
            else:
                n_o += 1
                hit_r += sl.hit == -1 and len({t.rule for t in sl.taps}) == 1
            hit_r -= not (sl.taps[-1].t == 20 and np.array_equal(  # the past is unchanged
                T.due_np(sl.on, sl.off, 15, 2), T.due_np(base.on, base.off, 15, 2)))
    check(f"episodes: midtaps at age 20: {n_r} rivals (each hits), {n_o} own (no hit); the targets before it unchanged",
          n_r >= 15 and n_o >= 15 and hit_r == n_r + n_o)


# ---------------------------------------------------------------- 2. targets

def target_checks(tab, bd):
    on = torch.full((1, 2, 6, 1, 1), float(E.INF))
    off = torch.full((1, 2, 6, 1, 1), float(E.INF))
    on[0, 0, 0], off[0, 0, 0] = 7, 20
    on[0, 0, 1], off[0, 0, 1] = 10, 10          # a don't-care-only interval
    want = {}
    for a in range(0, 26):
        tgt, w = T.target_weight(on, off, torch.tensor(float(a)).view(1, 1, 1, 1, 1), 2, 1)
        want[a] = tuple(int(x) for x in (tgt[0, 0, 0, 0], w[0, 0, 0, 0], tgt[0, 1, 0, 0], w[0, 1, 0, 0], tgt[0, 2, 0, 0], w[0, 2, 0, 0]))
    exp = {a: ((1 if 9 <= a < 20 else 0), (0 if 6 <= a < 9 or 20 <= a < 22 else 1),
               0, (0 if 9 <= a < 12 else 1), 0, 1) for a in range(26)}
    check("targets: an interval [7, 20), slack 2, early 1: 0 to age 5, don't-care 6-8, 1 at 9-19, don't-care 20-21, "
          "0 from 22; a don't-care-only one at 10: don't-care 9-11; no interval: 0 always", want == exp)
    rng = np.random.default_rng(2)
    P = T2.new_pool(rng, tab, bd, "L3", 37, 6, 16, "m1a", torch.device("cpu"))
    bad = 0
    for i in range(6):
        sl = E.single(tab, P["geo"][i], P["tap"][i], P["rule"][i], 1, 1)
        o = torch.from_numpy(sl.on.astype(np.float32))[None]
        f = torch.from_numpy(sl.off.astype(np.float32))[None]
        a = torch.from_numpy(P["a"][i:i + 1]).float()
        for age in range(0, 120, 3):
            t3, w3 = T.target_weight(o, f, torch.tensor(float(age)).view(1, 1, 1, 1, 1), 1, -1)
            t2, w2 = T2.target_weight("m1a", a, torch.full_like(a, -1), a, torch.zeros(1, dtype=torch.bool),
                                      torch.tensor(float(age)).view(1, 1, 1, 1))
            bad += not (torch.equal(t3.float(), t2) and torch.equal(w3.float(), w2))
    check("targets: --speed 1 --slack 1 --early -1 = train2.target_weight, every third age 0-120 of 6 level-3 slots",
          bad == 0)


# ---------------------------------------------------------------- 3. taps

def tap_checks(tab, bd):
    T.CODES[0] = T2.Codes(tab)
    dev = torch.device("cpu")
    planes = T2.Planes(tab, "e", dev)
    rng = np.random.default_rng(3)
    S = 12
    geo = T2.pad_geo(bd.board("L2", 0), S)
    t1 = SIM.draw_taps(rng, tab, geo, "single")[0]
    t2 = SIM.draw_taps(rng, tab, geo, "single")[0]._replace(t=5)
    while (t2.row, t2.col) == (t1.row, t1.col):
        t2 = SIM.draw_taps(rng, tab, geo, "single")[0]._replace(t=5)
    taps = E.taps_array([t1, t2])[None]
    cs = T.base_consts(planes, geo[None])
    at = planes.tap_at
    ok = not cs[:, at:at + T.TAP].any()
    res = {}
    for mode, steps in (("impulse", 1), ("impulse", 2), ("held", 1), ("fixed", 1)):
        runner = T.Runner(None, {"tap": mode, "tapSteps": steps, "channels": 64,
                                 "fixedChannels": list(range(29, 64))}, dev)
        ev = T.Events(taps, runner, planes, S)
        on_steps = {0: [], 1: []}
        for age in range(0, 12):
            c = ev.consts(cs, np.array([age]))
            tp = c[0, at:at + T.TAP]
            for k, t in enumerate((t1, t2)):
                here = tp[:, t.row, t.col]
                if here.any():
                    on_steps[k].append(age + 1)
                    full = planes(geo[None], T.CODES[0]([t.rule]), np.array([[t.row, t.col, t.d0, t.d1]]))
                    ok &= torch.equal(here, full[0, at:at + T.TAP, t.row, t.col])
            rest = tp.clone()
            rest[:, t1.row, t1.col] = 0
            rest[:, t2.row, t2.col] = 0
            ok &= not rest.any()
            ok &= torch.equal(torch.cat([c[:, :at], c[:, at + T.TAP:]], 1), torch.cat([cs[:, :at], cs[:, at + T.TAP:]], 1))
        res[(mode, steps)] = on_steps
    want = {("impulse", 1): {0: [1], 1: [6]}, ("impulse", 2): {0: [1, 2], 1: [6, 7]},
            ("held", 1): {0: list(range(1, 13)), 1: list(range(6, 13))}, ("fixed", 1): {0: [], 1: []}}
    check(f"taps: the tap planes are on in steps t + 1 .. t + tapSteps (impulse), from t + 1 (held), never (fixed), "
          f"on the tapped cell only, = train2.Planes' ({res[('impulse', 2)]})", ok and res == want)
    runner = T.Runner(None, {"tap": "fixed", "tapSteps": 1, "channels": 64, "fixedChannels": list(range(29, 64))}, dev)
    ev = T.Events(taps, runner, planes, S)
    st = torch.randn(1, 64, S, S) * 0.1
    w0 = ev.write(st, np.array([0]))
    w5 = ev.write(w0, np.array([5]))
    diff = (w0 != st)
    c35 = T.code35([t1.rule])[0]
    ok = bool(w0[0, 1 + t1.d0, t1.row, t1.col] == 1 and w0[0, 1 + t1.d1, t1.row, t1.col] == 1)
    ok &= torch.equal(w0[0, 29:64, t1.row, t1.col], torch.from_numpy(c35))
    ok &= int(diff.sum()) == 37 and ev.write(st, np.array([1])) is st
    ok &= bool((w5 != w0)[0, :, t2.row, t2.col].sum() == 37) and int((w5 != w0).sum()) == 37
    bits = (c35 > 0).astype(np.int64)
    digits = [int("".join(map(str, bits[8 + 3 * t:11 + 3 * t])), 2) for t in range(9)]
    ok &= digits == list(t1.rule[1:]) and np.array_equal(bits[:8], T.CODES[0]([t1.rule])[0, :8])
    st2 = st.clone()
    st2[0, 1 + t1.d0, t1.row, t1.col] = 1.7
    ok &= float(ev.write(st2, np.array([0]))[0, 1 + t1.d0, t1.row, t1.col]) == np.float32(1.7)
    check("taps: the fixed write sets the two chord edges to max(itself, 1) and the 35 code channels, nothing else, "
          "at the tap's age only; code35 decodes back to the rule", ok)
    torch.manual_seed(0)
    model = T2.make_model(40, 24, [-2, 2], T2.n_inputs("e"), 2, "e", 2)
    with torch.no_grad():
        model.w2.normal_(0, 0.3)
    runner = T.Runner(model, {"tap": "impulse", "tapSteps": 1, "channels": 40}, dev)
    c = planes(geo[None], T.CODES[0]([t1.rule]), np.array([[t1.row, t1.col, t1.d0, t1.d1]]))
    s0 = T2.fresh(c[:, :1], 40)
    a, b = s0, s0
    plan = runner.plan(c, 1)
    for _ in range(5):
        a = model.step(a, c[:, :1], c)
        b = runner.step(b, c[:, :1], c, plan)
    check("taps: Runner.step (the plan given) = FrameNCA.step, bit for bit, 5 steps", torch.equal(a, b))


# ---------------------------------------------------------------- 4. the quick check, against oracles

class Oracle(T.Runner):
    """Draws the targets of each batch item (looked up by its taps): an edge from on + slack (as late as allowed) + lag
    until off; variants: lines die after decay x ideal, wiped lines never go, wiped lines come back after ideal +
    regrow, the top half of the board goes after ideal + slack + 2 (drop)."""

    def __init__(self, cfg, table, lag=0, decay=None, no_wipe=False, regrow=None, drop=False):
        self.cfg, self.C, self.mode, self.tap_steps = cfg, cfg["channels"], cfg["tap"], 1
        self.code_ch, self.framed, self.device = None, False, torch.device("cpu")
        self.table, self.lag, self.decay, self.no_wipe, self.regrow, self.drop = table, lag, decay, no_wipe, regrow, drop

    def plan(self, cs, B):
        return None

    def step(self, state, walls, cs, plan, age_before=None, events=None):
        st = state.clone()
        sl, ea = self.cfg["slack"], self.cfg["early"]
        for b in range(st.shape[0]):
            on, off, ideal = self.table[events.taps[b].tobytes()]
            a = int(age_before[b]) + 1 - self.lag
            off_ = np.where(on < off, E.INF, off) if self.no_wipe else off
            d = T.target_weight(on.astype(np.int32), off_.astype(np.int32), a, sl, ea)[0]
            if self.decay is not None and a > self.decay * ideal:
                d[:] = False
            if self.regrow is not None and a > ideal + self.regrow:
                d |= ((on < off) & (off < E.INF)).any(0)
            if self.drop and a > ideal + sl + 2:
                d[:, : d.shape[1] // 2] = False
            st[b, 1:7] = torch.from_numpy(d.astype(np.float32))
        return st


def table_of(sets, cfg):
    tbl = {}
    for es in sets:
        if isinstance(es, T.EpisodeSet):
            for i in range(len(es)):
                tbl[es.taps[i].tobytes()] = (es.on[i], es.off[i], int(es.ideal[i]))
        else:
            for i in range(len(es.geo)):
                a = es.a[i].astype(np.int32)
                on = np.where(a < E.INF, cfg["speed"] * a, E.INF)[None]
                tbl[es.taps[i].tobytes()] = (on, np.full_like(on, E.INF), int(es.items["ideal"][i]))
    return tbl


def oracle_checks(tab, bd):
    T.CODES[0] = T2.Codes(tab)
    dev = torch.device("cpu")
    cfg = cfg_of()
    planes = T2.Planes(tab, "e", dev)
    legacy = os.path.join(os.path.dirname(E.__file__), "..", "..", "data", "strand")
    evs = [T.rescale(T2.legacy_set("m1a", tab, legacy, 2, 18, 4, 400), cfg),
           T.rescale(T2.wide_set("m1a", tab, bd, 2, 18, 4, 400), cfg)]
    pers = T.persist_set(tab, bd, cfg, 2, 12, [2, 4], [], 400)
    own = T.episode_set("own", tab, bd, cfg, 2, 12, 5, {"t3": 1.0}, lambda r: [r["ideal"]])
    ctl = T.episode_set("collide", tab, bd, dict(cfg, control=0.25), 3, 24, 6, {"t2": 1.0}, lambda r: [r["ideal"]])
    tbl = table_of(evs + [pers, own, ctl], cfg)
    names = {"persist": ["x1", "x2", "x4"]}

    def run(**kw):
        orc = Oracle(cfg, tbl, **kw)
        res = [T.evaluate_single(orc, planes, ev, dev, 8) for ev in evs]
        q = T2.summarise("m1a", evs, res, tab)
        sp = np.concatenate([r["speed"] for r in res])
        good = (sp >= 0).all(1)
        q["speed"] = float(np.mean((T.SPEED_M - 1) / np.maximum(1, sp[good, 1] - sp[good, 0]))) if good.any() else None
        er = [T.evaluate_episodes(orc, planes, es, dev, 8, cfg["slack"], cfg["early"]) for es in (pers, own, ctl)]
        q.update(T.summarise_episodes([pers, own, ctl], er, names))
        return q
    q = run()
    c = q["collide"]
    ok = q["exact"] == 1.0 and q["exit"]["rate"] == 1.0 and q["speed"] == 0.5 and \
        q["persist"]["x1"] == q["persist"]["x2"] == q["persist"]["x4"] == 1.0 and q["own"]["exact"] == 1.0 and \
        q["own"]["collateral"] == 0.0 and q["own"]["phantom"] == 0.0 and q["own"]["stray"] == 0.0 and \
        c["exact"] == 1.0 and c["wipe"] == 1.0 and c["falseWipe"] == 0.0 and c["regrow"] == 0.0 and \
        c["worm"] <= c["wormMax"] and c["nHit"] >= 12 and c["exactCtrl"] == 1.0
    check(f"quick check: the targets drawn score exact 1.0, exit 1.0, speed {q['speed']}, persist {q['persist']}, own "
          f"{q['own']}, collide {c}", ok)
    q = run(lag=-2)
    check(f"quick check: drawn 2 steps early, read-outs still exact (exact {q['exact']}, persist x1 {q['persist']['x1']})",
          q["exact"] == 1.0 and q["persist"]["x1"] == 1.0)
    q = run(decay=2)
    check(f"quick check: lines that die after 2 x ideal: persist x1 {q['persist']['x1']} x2 {q['persist']['x2']} x4 "
          f"{q['persist']['x4']} (ratio {q['persist'].get('ratio')})",
          q["persist"]["x1"] == 1.0 and q["persist"]["x2"] == 1.0 and q["persist"]["x4"] == 0.0)
    q = run(drop=True)
    check(f"quick check: own meetings that lose half the board late: collateral {q['own']['collateral']} > 0, "
          f"phantom {q['own']['phantom']} > 0", (q["own"]["collateral"] or 0) > 0 and (q["own"]["phantom"] or 0) > 0)
    # hand-made hit episodes (not sim.py's: only the metrics' arithmetic): every line goes 12 steps after it is due
    rows = []
    rng = np.random.default_rng(9)
    for k in range(8):
        geo = T2.pad_geo(bd.board("L2", k % 9), 12)
        sl = E.draw(rng, tab, geo, "t1", 2, cfg["slack"])
        sl.off = np.where(sl.on < E.INF, sl.on.astype(np.int32) + 12, E.INF).astype(np.int16)
        sl.hit, sl.kind = int(sl.on.min()) + 6, "t2"
        sl.settle = int(sl.off[sl.off < E.INF].max())
        rows.append(T.slot_of(sl, geo, 0, cfg["slack"]))
    hits = T.EpisodeSet("collide", 2, 12, rows, [[r["ideal"]] for r in rows], cfg, 400)
    tbl.update(table_of([hits], cfg))
    worm_max = float(np.mean([((r["off"] < E.INF) & (r["on"].astype(np.int32) + cfg["slack"] > r["hit"])
                                & (r["on"] < r["off"])).any(0).sum() / 2 for r in rows]))

    def hq(**kw):
        orc = Oracle(cfg, tbl, **kw)
        return T.summarise_episodes([hits], [T.evaluate_episodes(orc, planes, hits, dev, 8, cfg["slack"], cfg["early"])],
                                    {})["collide"]
    a, b, c = hq(), hq(no_wipe=True), hq(regrow=3)
    check(f"quick check: hit episodes (hand-made, off = on + 12): wipe {a['wipe']} regrow {a['regrow']} worm {a['worm']} "
          f"(drawn as late as allowed: = wormMax {a['wormMax']}, {worm_max}); never wiped: wipe {b['wipe']}; drawn "
          f"again after: regrow {c['regrow']}", a["wipe"] == 1.0 and a["regrow"] == 0.0
          and a["worm"] == a["wormMax"] == worm_max and b["wipe"] == 0.0 and c["regrow"] == 1.0 and a["exact"] == 1.0)
    # the probes: an injected code reads back (linear and MLP), a random state doesn't
    rng = np.random.default_rng(4)
    gen = T.Generator(tab, bd, cfg_of(channels=80), "L2", 12)
    P = T.new_pool(rng, gen, {"t1": 1.0}, 400, 80, dev)  # enough codes to span every (type, digit)
    P["age"][:] = 10 ** 4
    st = P["state"]
    for i in range(len(P["age"])):
        st[i, T.N_FIXED:T.N_FIXED + CODE_BITS] = torch.from_numpy(T.CODES[0]([E.taps_of(P["taps"][i])[0].rule])[0]).float()[:, None, None]
    X, Y = T.probe_cells({2: P}, 30000, 2, 1)
    xs, ys, ds = [], [], []
    for ev in evs:
        for i in range(len(ev.geo)):
            on = (ev.a[i] < E.INF).any(0)
            n = int(on.sum())
            v = np.zeros((n, 80 - T.N_FIXED), np.float32)
            v[:, :CODE_BITS] = T.CODES[0](ev.rule[i:i + 1])[0]
            xs.append(v)
            ys.append(np.repeat(T.CODES[0](ev.rule[i:i + 1]), n, 0))
            ds.append(np.full(n, 3))
    cells = {"x": np.concatenate(xs), "y": np.concatenate(ys), "dist": np.concatenate(ds)}
    byage = dict(cells, age=np.arange(len(cells["x"])) % 3)
    lin = T.probe_score(T.ridge_fit(X, Y), Y.mean(0), cells, byage, ["x1", "x2", "x4"])
    mlp = T.probe_score(T.mlp_fit(X, Y, dev, steps=300), Y.mean(0), cells, byage, ["x1", "x2", "x4"])
    rnd = dict(cells, x=np.random.default_rng(0).normal(size=cells["x"].shape).astype(np.float32))
    lin0 = T.probe_score(T.ridge_fit(X, Y), Y.mean(0), rnd)
    check(f"probes: an injected code reads back (linear exact {lin['exact']} byAge {lin.get('byAge')}, MLP exact "
          f"{mlp['exact']}, n {lin['n']}); a random state doesn't (linear {lin0['exact']})",
          lin["exact"] == 1.0 and mlp["exact"] >= 0.95 and lin["byAge"] == {"x1": 1.0, "x2": 1.0, "x4": 1.0}
          and lin0["exact"] < 0.05)


# ---------------------------------------------------------------- 5. runs

TINY = ["--levels", "2", "--eval-levels", "2", "--hidden", "8", "--channels", "16", "--depth", "1", "--batch", "4",
        "--pool-size", "8", "--eval-n", "6", "--eval-mult", "2", "--eval-cap", "60", "--steps-mult", "0.5", "1",
        "--bptt", "4", "--last-k", "2", "--threads", "1", "--snap-every", "0", "--persist-n", "4", "--persist-cap",
        "200", "--episode-n", "4", "--episode-levels", "2", "--mlp-steps", "20", "--no-gallery"]


def log_of(name):
    return [json.loads(x) for x in open(f"runs/{name}/log.jsonl")]


def runs():
    for name, tap, extra in (("t1-impulse", "impulse", []), ("t1-impulse-2", "impulse", ["--tap-steps", "2"]),
                             ("t1-held", "held", []), ("t1-fixed", "fixed", ["--channels", "48", "--dir-groups", "0"])):
        T.main(["--name", name, *TINY, "--tap", tap, *extra, "--iters", "3", "--eval-every", "50"])
        lg = log_of(name)
        q = lg[1]["q"]
        ok = lg[-1].get("stopped") == "done" and all(np.isfinite(x["loss"]) for x in lg if "loss" in x) and \
            {"persist", "speed", "exit"} <= set(q) and lg[0]["config"]["tap"] == tap
        check(f"run T1 --tap {tap} {' '.join(extra)}: check + 3 iterations, finite loss, q has persist / speed / exit", ok)
    T.main(["--name", "t3", *TINY, "--stage", "T3", "--iters", "3", "--eval-every", "50", "--midtap", "1"])
    lg = log_of("t3")
    q = [x for x in lg if "q" in x][-1]["q"]
    kinds = lg[-2]["pool"]["2"]["kinds"]
    check(f"run T3: own and collide sets in q ({sorted(q)}), pool kinds {kinds}, midtaps {lg[-2]['damage']['midtap']}",
          {"own", "collide", "persist"} <= set(q) and lg[-1].get("stopped") == "done" and "t3" in kinds)
    T.main(["--name", "t3f", *TINY, "--stage", "T3", "--tap", "fixed", "--channels", "48", "--dir-groups", "0",
            "--iters", "4", "--eval-every", "50", "--midtap", "1", "--steps-mult", "1", "2", "--bptt", "12"])
    lg = log_of("t3f")
    check("run T3 --tap fixed (staggered taps written mid-window, under autograd): finite loss",
          lg[-1].get("stopped") == "done" and all(np.isfinite(x["loss"]) for x in lg if "loss" in x))
    T.main(["--name", "t2", *TINY, "--stage", "T2", "--iters", "3", "--eval-every", "50", "--midtap", "1"])
    lg = log_of("t2")
    q = [x for x in lg if "q" in x][-1]["q"]
    kinds = lg[-2]["pool"]["2"]["kinds"]
    check(f"run T2: collide {q.get('collide')}, pool kinds {kinds}, midtaps {lg[-2]['damage']['midtap']}",
          lg[-1].get("stopped") == "done" and "collide" in q and q["collide"].get("nHit", 0) > 0 and "t2" in kinds)
    T.main(["--name", "of", *TINY, "--overfit", "3", "--iters", "3", "--eval-every", "50", "--persist-ages", "20", "40"])
    lg = log_of("of")
    q = lg[1]["q"]
    check(f"run --overfit 3: q {q.get('overfit')}", lg[-1].get("stopped") == "done" and
          set(q["overfit"]) >= {"exact", "onTime", "a20", "a40", "n"} and q["overfit"]["n"] == 3 and len(lg[0]["overfit"]) == 3)

    T.main(["--name", "rb", *TINY, "--iters", "50", "--eval-every", "50", "--force-collapse", "1"])
    rb = [x for x in log_of("rb") if "rollback" in x]
    best = torch.load("runs/rb/best.pt", weights_only=False)
    ck = torch.load("runs/rb/ckpt.pt", weights_only=False)
    same = all(torch.equal(best["model"][k], ck["model"][k]) for k in best["model"])
    check(f"forced collapse: a rollback (lrScale {rb[0]['lrScale'] if rb else '?'}), ckpt.pt's weights = best.pt's",
          len(rb) == 1 and rb[0]["lrScale"] == 0.5 and same and ck["rollbacks"] == 1)

    flat = ["--lr", "1e-3", "--lr-floor", "1e-3", "--warmup", "0", "--eval-every", "50", "--stage", "T3",
            "--midtap", "0.5"]
    T.main(["--name", "straight", *TINY, *flat, "--iters", "6"])
    T.main(["--name", "chunked", *TINY, *flat, "--iters", "3"])
    T.main(["--name", "chunked", "--resume", "--iters", "6", "--threads", "1", "--snap-every", "0", "--no-gallery"])
    a = torch.load("runs/straight/ckpt.pt", weights_only=False)
    b = torch.load("runs/chunked/ckpt.pt", weights_only=False)
    w = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    o = all(torch.equal(x, y) for x, y in zip(a["opt"]["state"][0].values(), b["opt"]["state"][0].values()))
    pa, pb = a["pool"][2], b["pool"][2]
    p = torch.equal(pa["state"], pb["state"]) and all(np.array_equal(pa[k], pb[k]) for k in ("geo", "taps", "on", "off", "age"))
    check(f"--resume (T3, midtaps): 3 + 3 iterations = 6 in one go (weights {w}, optimiser {o}, pool {p})", w and o and p)

    T2.main(["--name", "v2", "--inputs", "e", "--levels", "2", "--eval-levels", "2", "--hidden", "8", "--channels", "16",
             "--batch", "4", "--pool-size", "8", "--eval-n", "6", "--eval-mult", "2", "--eval-cap", "30",
             "--steps-mult", "0.5", "1", "--bptt", "4", "--last-k", "2", "--threads", "1", "--snap-every", "0",
             "--iters", "2", "--eval-every", "50", "--depth", "1", "--no-probe"])
    T.main(["--name", "warm", *TINY, "--iters", "2", "--eval-every", "50", "--init", "runs/v2/best.pt",
            "--snap-every", "1"])
    lg = log_of("warm")
    v2 = torch.load("runs/v2/best.pt", weights_only=False)
    w3 = torch.load("runs/warm/best.pt", weights_only=False)
    same = all(torch.equal(v2["model"][k], w3["model"][k]) for k in v2["model"])
    check(f"--init from a train2 checkpoint: loads (iteration 0's best.pt = its weights: {same}), prevIterations "
          f"{lg[0]['config']['prevIterations']}", same and lg[-1].get("stopped") == "done")
    try:
        from ..dashboard import pool_to_json
        js = pool_to_json(Path("runs/warm/pool.npz"))
        R = js["radii"][0]
        g = js["by_radius"][str(R)]
        check(f"pool.npz reads through nca.dashboard.pool_to_json (R {R}, S {g['S']}, n {g['n']}, C {g['C']}, damage "
              f"kinds {js.get('damageNames')})", g["S"] == 2 * R + 1 and g["n"] == 8 and g["C"] == 16
              and js.get("damageNames") == ["noise", "midtap"])
    except Exception as e:  # noqa: BLE001
        check(f"pool.npz reads through nca.dashboard.pool_to_json ({e!r})", False)
    from . import export as X
    for name in ("t1-impulse-2", "t1-fixed", "t1-held"):
        out = f"runs/{name}.json"
        X.main([f"runs/{name}/best.pt", "--out", out])
        j = json.load(open(out))
        ck = torch.load(f"runs/{name}/best.pt", weights_only=False)
        want = X.tap_spec(ck["config"])
        model, _ = X.model_from_json(j)
        tab = RuleTable()
        geo = Boards().board("L2", 0)
        rng = np.random.default_rng(0)
        taps = X.fixture_taps(tab, geo, rng, 2)
        held = X.rollout_events(model, dict(j, tap={"mode": "held"}), tab, geo, [(r, t, 0) for r, t in taps], 5)[0]
        ref = X.rollout(model, j, X.build_consts(j, tab, geo, taps), 5)[0]
        check(f"export {name}: tap {j['tap']}, version {j['version']}, speed {j['speed']}; held events = the held "
              f"rollout", j["tap"] == want and j["version"] == (1 if want["mode"] == "held" else 2) and j["speed"] == 2
              and np.array_equal(held, ref))


# ---------------------------------------------------------------- the overfits (§1)

OVERFIT = ["--overfit", "4", "--levels", "2", "--channels", "48", "--dir-groups", "0", "--hidden", "64",
           "--depth", "2", "--batch", "8", "--pool-size", "32", "--bptt", "48", "--last-k", "48", "--lr", "1e-3",
           "--warmup", "50", "--eval-every", "50", "--snap-every", "0", "--no-probe", "--no-gallery"]


def overfits(minutes, threads, only=None, stage="T1"):
    arms = [("held", ["--tap", "held"]), ("impulse", ["--tap", "impulse"]), ("fixed", ["--tap", "fixed"]),
            ("impulse-nocap", ["--tap", "impulse", "--speed", "1", "--slack", "1", "--early", "-1"]),
            ("held-nocap", ["--tap", "held", "--speed", "1", "--slack", "1", "--early", "-1"]),
            ("fixed-nocap", ["--tap", "fixed", "--speed", "1", "--slack", "1", "--early", "-1"]),
            ("impulse2", ["--tap", "impulse", "--tap-steps", "2"])]
    rows = []
    for name, extra in arms:
        if only and name not in only:
            continue
        run = f"pi3-{stage.lower()}-{name}"
        if not os.path.isfile(f"runs/{run}/log.jsonl") or not any('"stopped"' in x for x in open(f"runs/{run}/log.jsonl")):
            T.main(["--name", run, *OVERFIT, *extra, "--stage", stage, "--minutes", str(minutes), "--iters", "100000",
                    "--threads", str(threads)])
        lg = log_of(run)
        qs = [(x["iteration"], x["q"]["overfit"]) for x in lg if "q" in x]
        first = lambda key: next((it for it, q in qs if q.get(key) == 1.0), None)  # noqa: E731
        it, q = qs[-1]
        rows.append({"arm": name, "itersTo1": first("exact"), "onTimeTo1": first("onTime"), "a600To1": first("a600"),
                     "lastIter": it, **{k: v for k, v in q.items() if k != "n"},
                     "secPerIter": round(np.median([x["secPerIter"] for x in lg if "secPerIter" in x]), 3)})
        print(json.dumps(rows[-1]), file=sys.__stdout__, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overfit", action="store_true", help="§1's Pi overfits instead of the checks")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--stage", default="T1")
    ap.add_argument("--only", nargs="*", default=None, help="sections: episodes targets taps oracle runs")
    args = ap.parse_args()
    if args.overfit:
        overfits(args.minutes, args.threads, args.arms, args.stage)
        return
    torch.set_num_threads(1)
    tab, bd = RuleTable(), Boards()
    T.CODES[0] = T2.Codes(tab)
    want = set(args.only or ("episodes", "targets", "taps", "oracle", "runs"))
    if "episodes" in want:
        episode_checks(tab, bd)
    if "targets" in want:
        target_checks(tab, bd)
    if "taps" in want:
        tap_checks(tab, bd)
    if "oracle" in want:
        oracle_checks(tab, bd)
    if "runs" in want:
        here = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with open(os.devnull, "w") as null:
                    out, sys.stdout = sys.stdout, null
                    try:
                        runs()
                    finally:
                        sys.stdout = out
            finally:
                os.chdir(here)
    print(f"{'ALL OK' if not FAILS else f'{len(FAILS)} FAILED'}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
