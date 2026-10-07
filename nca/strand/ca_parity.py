"""The hybrid line CA (src/game/line-ca.ts, `LineCA`) against the multi-strand sim's CA mode and Spectacle's engine
(docs/spectacle-ca-hybrid.md §2.10 and §5, parity (b) and (c); D3):

    python -m nca.strand.ca_parity [--data data/strand-v2] [--n 200] [--levels 2 3] [--seed 1] [--show 6]
        [--classes tests/fixtures/ca-collide-classes.json] [--skip-sim] [--skip-engine]

LineCA runs in TypeScript: `npx tsx scripts/game-parity.ts --in FIXTURE --out CA` replays a fixture in collide.json's
format (boards with geo; episodes with players and timed taps) and writes what the CA's lines held, step by step.

(b) THE SIM. `--n` episodes of every draw_taps kind (single, collide, control, own) and of `busy_taps` (3-6 players,
    up to 4 taps each, as collide.json's) on the L2 / L3 patches
    (boards.npz), written as a fixture, replayed by LineCA and by sim.py's CA mode (simulate(wave=1,
    sequential=False)). Per kind: accepted taps equal, end state equal, and for the chords both drew how many
    intervals have their on and off within 1 step. Then the same against the sim with the CA's timing
    (`_CATiming`, all switches on), where every chord's on and off tick should be equal.
(c) THE ENGINE. collide.json (Spectacle's engine, `--collide`) replayed by LineCA: accepted taps, end state and per
    tap whether its line survives, at rest; and LineCA against the sim with the CA's timing, tick for tick.
    `--classes` writes each episode's class (below, or "agree") for tests/game-parity.test.ts.

Every difference at rest gets a class:
  as the sim     (c) only: LineCA = sim's CA mode exactly (taps and end state), so its difference from the engine is
                 the sim's own, split as sim_parity does: `waves` (the CA's dying wave takes a step a chord where the
                 engine wipes at once, so a join, a hit or a tap can race it: the doc's wave-vs-join race and its
                 relatives, counted by sim_parity.ca_events), `head timing` (the engine's second head), or both
  timing: ...    LineCA = the sim's CA mode with some of the CA's own timing switched on (`_CATiming`; the fewest
                 that make them agree are named): differences between the CA and the sim's CA mode in when things
                 happen, never in what happens
  unexplained    not even all of them: a rule difference, listed with the episode for a repro
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from .rules import Boards, RuleTable, default_dir, random_chord
from .sim import (ABSORB, INF, KILL, OPP, REFUSE, SLACK, TAIL, W_ABSORBED, W_KILLED, W_TAIL, Tap, _Sim, draw_taps,
                  rule_key, simulate, strand_of)
from .sim_parity import ca_events, engine_view, sim_view
from .walker import PAIRS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KINDS = ("single", "collide", "control", "own", "busy")
SIM_MODES = {"game": dict(wave=0, sequential=True), "CA": dict(wave=1, sequential=False),
             "wave0": dict(wave=0, sequential=False), "seq": dict(wave=1, sequential=True)}


# ---------------------------------------------------------------- LineCA, through the TypeScript script

def run_line_ca(fixture: str, out: str) -> dict:
    """Replay a fixture through LineCA (scripts/game-parity.ts); returns the parsed output."""
    subprocess.run(["npx", "tsx", "scripts/game-parity.ts", "--in", fixture, "--out", out, "--quiet"], cwd=ROOT,
                   check=True)
    with open(out) as f:
        return json.load(f)


def ca_view(e):
    """(end set, intervals, accepted) of a LineCA episode, as sim_parity.engine_view."""
    iv = sorted((p, r, c, a, b, on, INF if off < 0 else off) for p, r, c, a, b, on, off in e["chords"])
    return {x[:5] for x in iv if x[6] >= INF}, iv, [bool(x["ok"]) for x in e["taps"]]


def same(u, v) -> bool:
    """Two views agree at rest: the end state and the accepted taps."""
    return u[0] == v[0] and u[2] == v[2]


def timing(ca_iv, sim_iv):
    """Per chord key both drew: (on deltas, off deltas, keys whose interval counts differ). Deltas CA - sim;
    an off of INF on one side only counts as a large delta."""
    a, b = defaultdict(list), defaultdict(list)
    for x in ca_iv:
        a[x[:5]].append(x[5:])
    for x in sim_iv:
        b[x[:5]].append(x[5:])
    don, doff, count = [], [], 0
    for k in a.keys() & b.keys():
        if len(a[k]) != len(b[k]):
            count += 1
            continue
        for (on1, off1), (on2, off2) in zip(a[k], b[k]):
            don.append(on1 - on2)
            doff.append(0 if off1 >= INF and off2 >= INF else (99 if (off1 >= INF) != (off2 >= INF) else off1 - off2))
    return don, doff, count


# ---------------------------------------------------------------- the sim, and the CA's timing

OPTIONS = ("phase", "early", "hitter", "tap window", "wait", "cold", "cross")


class _CATiming(_Sim):
    """sim.py's CA mode with the hybrid CA's timing (docs/spectacle-ca-hybrid.md §2.3, §2.10), each a switch:
      phase       tips move on the CA's global go steps (even steps; period 2): a tap on an odd step grows its
                  first chord one step after it, not two
      early       collisions don't wait for a move: a tip collides the step after it is placed (one before its
                  move) when the tile ahead is a rival's, or empty with another rule's tip aimed at it too
                  (`hitAt`, `victim` and `crash` read `aimed`, which is not gated by `go`)
      hitter      a hitter's last chord goes on the hit step (`hitAt` erases it, both ends: a fresh tap's other
                  tip too) and W goes out of both its ends (the far one runs on along the line); the sim's wave
                  comes back from the hit tile a step later. A crash (an empty tile) is the same in both: the
                  hitters go a step later
      tap window  the CA's W pulses refuse a tap for one step: on what a wave erased on that step, on a crash tile
                  on its crash step, and where a wave of the tap's rule is about to cross into the tapped chord
                  (LineCA's `waveAt`). The sim refuses for its whole lingering window (slack 2, which is the CA's
                  one hot step as a tip sees it: a step later)
      wait        a tip aimed at a tile of its own rule that is being hit, or at its own rule's chord a wave is
                  erasing, waits for its next go step (`enter` needs not victim, not W); the sim kills it
      cold        the CA's W goes out of an erased end only if no wave came in by it on that step, and only an end
                  with W is hot: a tip may draw in at once across an end a wave came in by, two waves meeting in a
                  chord leave the tile free the next step, and a hit or a victim that a wave reaches on the same
                  step sends nothing back across the edge it came by. The sim's erased chord lingers whole
      cross       a wave reaching a tile on the step a tip of its rule draws a chord there finds nothing (it reads
                  the step before) and is spent, so the newcomer lives on; the sim moves tips before waves, and its
                  wave takes the new chord and the line behind it
    With all seven the sim is LineCA at rest; what differs then is a rule difference."""

    def __init__(self, *a, opts=(), **k):
        super().__init__(*a, **k)
        self.opts = frozenset(opts)
        self.crash_at = {}            # tile -> the step a crash happened on it
        self.erased_at = {}           # chord end (edge * HW + tile) -> the step its chord was erased
        self.hot_t, self.hot_ends = {}, {}  # tile -> the step of its last erase, and its hot chord ends then
        self.came = set()             # (tile, edge, rule) a wave came in by this step
        self.cooled = set()           # (chord end, step) already cooled

    def schedule(self, tip, t):
        if "phase" in self.opts and t % 2:
            t -= 1
        super().schedule(tip, t)
        if "early" in self.opts and t - 1 > self.now:
            self.due.setdefault(t - 1, []).append(tip)

    def moves(self, t, tips):
        early = "early" in self.opts
        wait = "wait" in self.opts
        seen = set()
        reg, ear = [], []
        for tp in tips:
            if tp in seen or not self.tp_alive[tp]:
                continue
            seen.add(tp)
            if self.tp_next[tp] == t:
                reg.append(tp)
            elif early and self.tp_next[tp] == t + 1:
                ear.append(tp)
        if not reg and not ear:
            return
        key = lambda tp: (self.ln_slot[self.tp_line[tp]], -self.tp_side[tp])
        reg.sort(key=key)
        ear.sort(key=key)
        hits, cands, aims, waiting = {}, {}, {}, []
        for tip in reg:
            nxt = self.peek(tip)
            if nxt is None:
                self.stop(tip, t, W_TAIL, TAIL)
            elif nxt[3] == "hit":
                hits.setdefault(nxt[0], []).append((tip, nxt[1], nxt[2]))
            elif nxt[3] == "absorb":
                self.stop(tip, t, W_ABSORBED, ABSORB)
            elif nxt[3] == "dying":
                if wait:
                    waiting.append(tip)
                else:
                    self.stop(tip, t, W_KILLED, KILL, -2)
            else:
                cands.setdefault(nxt[0], []).append((tip, nxt[1], nxt[2]))
        for tip in ear:  # only collisions: the rest is decided at its move
            nxt = self.peek(tip)
            if nxt is None:
                continue
            if nxt[3] == "hit":
                hits.setdefault(nxt[0], []).append((tip, nxt[1], nxt[2]))
            elif nxt[3] == "draw":
                aims.setdefault(nxt[0], []).append((tip, nxt[1], nxt[2]))
        for j in set(cands) | set(aims):
            group, aimed = cands.get(j, []), aims.get(j, [])
            codes = {self.ln_code[self.tp_line[tp]] for tp, _, _ in group + aimed}
            if len(codes) > 1:  # case 3 (the CA's crash), early aimers included
                hits.setdefault(j, []).extend(group + aimed)
                continue
            if j in hits:  # into a tile hit this step
                own = self.cell_code[j] if self.cell_code[j] >= 0 else self.dead_code[j] if self.cell_dead[j] > t else -1
                if wait and own in codes:  # its own rule's tile (drawn or hot): the CA's tip waits
                    waiting.extend(tp for tp, _, _ in group)
                else:
                    hits[j].extend(group)
                continue
            done = set()
            for tip, ein, eout in group:
                if ein in done:
                    self.stop(tip, t, W_ABSORBED, ABSORB)
                else:
                    done.update((ein, eout))
                    self.advance_tip(tip, j, ein, eout, t)
        if hits:
            self.hit(hits, t)
        for tip in waiting:
            if self.tp_alive[tip]:
                self.schedule(tip, t + self.period)

    def hit(self, hits, t):
        HW = self.HW
        now = []  # hitters of a tile with a rule on it (drawn or dying): their chord goes now, not by a wave back
        for j, group in hits.items():
            if any(self.edge_rec[e * HW + j] >= 0 for e in range(6)) or self.cell_dead[j] > t:
                if "hitter" in self.opts:
                    now += [(tip, j, self.tp_rec[tip], self.tp_out[tip]) for tip, _, _ in group if self.tp_alive[tip]]
            else:
                self.crash_at[j] = t  # a crash's tile (empty): its W pulse makes it hot for a tap on this step
        super().hit(hits, t)
        for tip, j, r, out in now:
            key = self.key_of(self.tp_line[tip])
            self.born.remove((j, OPP[out], key))  # the sim's wave back from the hit tile
            far = self.rc_b[r] if self.rc_a[r] == out else self.rc_a[r]
            self.erase(r, t)
            self.start_wave(self.rc_cell[r], far, key, t)
            self.start_wave(self.rc_cell[r], out, key, t)  # the CA's W out of the tip's end too, into the hit tile

    def step_front(self, front, t):
        cell, e, code = front
        HW = self.HW
        j = self.nb[e * HW + cell]
        if j < 0:
            return None
        k = OPP[e] * HW + j
        r = self.edge_rec[k]
        if "cross" in self.opts and r >= 0 and self.rc_on[r] == t:
            return None  # drawn this step: the CA's wave reads the step before, finds no chord here and is spent
        if "cold" not in self.opts:
            return super().step_front(front, t)
        self.came.add((j, OPP[e], code))  # waveIn: no W goes back out across this edge this step
        res = super().step_front(front, t)
        if res is not None or self.erased_at.get(k) == t:
            self.cool(j, k, t)  # erased this step with a wave in by this end: the CA sends no W out of it
        return res

    def cool(self, j, k, t):
        """End k of a chord on tile j, erased at t, is not hot: no W pulse out of it (and the tile is free at
        t + 1 if no end on it is hot and nothing is drawn there)."""
        if (k, t) in self.cooled:
            return
        self.cooled.add((k, t))
        self.edge_dead[k] = t
        self.hot_ends[j] -= 1
        if self.hot_ends[j] == 0 and self.cell_n[j] == 0:
            self.cell_dead[j] = t

    def advance_waves(self, t):
        self.came = set()
        super().advance_waves(t)
        if self.came:  # the CA sends no W out of an erased end a wave came in by (a hit's or a victim's included)
            self.fronts = [f for f in self.fronts if f not in self.came]

    def erase(self, r, t, cause_line=-1):
        if self.rc_off[r] >= INF:
            j = self.rc_cell[r]
            if self.hot_t.get(j) != t:
                self.hot_t[j], self.hot_ends[j] = t, 0
            self.hot_ends[j] += 2
            for x in (self.rc_a[r], self.rc_b[r]):
                self.erased_at[x * self.HW + j] = t
        return super().erase(r, t, cause_line)

    def refuse(self, i, t, why):
        p = self.taps[i]
        self.reason[i] = why
        self.events.append((t, REFUSE, -1, p.row, p.col, i))

    def tap(self, i, t):
        if "tap window" not in self.opts:
            return super().tap(i, t)
        p = self.taps[i]
        if 0 <= p.row < self.H and 0 <= p.col < self.W:
            cell = p.row * self.W + p.col
            if self.crash_at.get(cell) == t and self.cell_code[cell] < 0:
                return self.refuse(i, t, "a crash tile, hot")
            code = self.code_of[p.rule]
            if self.edge_rec[p.d1 * self.HW + cell] < 0 and self.edge_dead[p.d1 * self.HW + cell] > t \
                    and self.edge_dead[p.d1 * self.HW + cell] - self.linger == t:
                return self.refuse(i, t, "your own line, dying")  # the CA checks W on both ends; the sim, d0
            for d in (p.d0, p.d1):  # a wave of its rule about to cross into the tapped chord (LineCA's waveAt)
                m = self.nb[d * self.HW + cell]
                if m >= 0 and any(f[0] == m and f[1] == OPP[d] and f[2] == code for f in self.fronts):
                    return self.refuse(i, t, "a wave coming in")
        # the CA's hot step: what a wave erased at t (dying until t + linger) refuses a tap at t only, so what was
        # erased before t is hidden from the sim's tap for the call
        saved = [(arr, k, v) for arr in (self.cell_dead, self.edge_dead) for k, v in enumerate(arr)
                 if v > t and v - self.linger < t]
        for arr, k, _ in saved:
            arr[k] = 0
        try:
            return super().tap(i, t)
        finally:
            for arr, k, v in saved:
                arr[k] = v

    def run(self, horizon):
        early = "early" in self.opts
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
                if any(self.tp_alive[tp] and (self.tp_next[tp] == m or early and self.tp_next[tp] == m + 1)
                       for tp in self.due[m]):
                    nt = m
                    break
                del self.due[m]
            if ti < len(order):
                nt = min(nt, self.taps[order[ti]].t)
            if nt >= INF:
                break
            t = nt
        return self.result()


def phase_of(ca_eps, fx_eps):
    """LineCA's growth delay per tap parity: {t % 2: steps from a tap to its first grown chord}, from lone taps."""
    seen = defaultdict(Counter)
    for e, c in zip(fx_eps, ca_eps):
        if len(e["taps"]) != 1 or not c["taps"][0]["ok"]:
            continue
        t = e["taps"][0]["t"]
        later = [x[5] for x in c["chords"] if x[5] > t]
        if later:
            seen[t % 2][min(later) - t] += 1
    return {p: cnt.most_common(1)[0][0] for p, cnt in seen.items()}


def simulate_ca(tab, geo, taps, players, exits, opts=OPTIONS):
    """The sim's CA mode (wave 1, all tips at once) with the CA timing switches `opts`."""
    return _CATiming(tab, np.asarray(geo), taps, 2, 1, False, True, SLACK, players, exits, opts=opts).run(None)


def sim_runs(tab, geo, taps, players, exits):
    """The sim's views: its four modes, and the CA mode with all of the CA's timing ("CA timing")."""
    runs = {m: simulate(tab, geo, taps, players=players, exits=exits, **kw) for m, kw in SIM_MODES.items()}
    runs["CA timing"] = simulate_ca(tab, geo, taps, players, exits)
    return runs, {m: sim_view(r) for m, r in runs.items()}


def classify_vs_sim(cv, views, tab, geo, taps, players, exits):
    """Why LineCA differs from the sim's CA mode at rest: None (it doesn't), the CA timing switches it takes
    (the fewest; "timing: early + phase"), or "unexplained" (not even all six make them agree)."""
    if same(cv, views["CA"]):
        return None
    if not same(cv, views["CA timing"]):
        return "unexplained"
    for n in range(1, len(OPTIONS) + 1):
        for opts in combinations(OPTIONS, n):
            if n == len(OPTIONS) or same(cv, sim_view(simulate_ca(tab, geo, taps, players, exits, opts))):
                return "timing: " + " + ".join(opts)
    return "unexplained"


def sim_cause(views, eng):
    """sim_parity's cause of a sim CA-mode difference from the engine."""
    if same(views["wave0"], eng):
        return "waves"
    if same(views["seq"], eng):
        return "head timing"
    if same(views["game"], eng):
        return "waves + head timing"
    return "unexplained"


def first_diff(a, b, k=3):
    return sorted(a - b)[:k], sorted(b - a)[:k]


# ---------------------------------------------------------------- (b) the sim on draw_taps episodes

def busy_taps(rng, tab, geo, t_max=30):
    """A crowded episode (collide.json's recipe on the sim's side): 3-6 rules, 1-4 taps each at random times up to
    t_max; half the taps after the first land on a cell of an earlier tap's strand (their own: a join or a line
    beside; a rival's: a hit), the rest on a random chord of their rule."""
    rules = []
    while len(rules) < int(rng.integers(3, 7)):
        s, digits = tab.sample(rng, "train" if rng.random() < 0.8 else "heldout")
        r = (s, *[int(x) for x in digits])
        if r not in rules:
            rules.append(r)
    taps = []
    for r in rules:
        bits = tab.render_bits(r[0], np.asarray(r[1:], np.int64), geo)
        for _ in range(int(rng.integers(1, 5))):
            c = None
            if taps and rng.random() < 0.5:
                src = taps[int(rng.integers(len(taps)))]
                st = strand_of(tab, geo, src.rule, src.row, src.col, src.d0, src.d1)
                j = int(rng.integers(len(st)))
                rc = (int(st.rows[j]), int(st.cols[j]))
                v = int(bits[rc])
                pairs = [PAIRS[k] for k in range(15) if v >> k & 1]
                if pairs:
                    a, b = pairs[int(rng.integers(len(pairs)))]
                    c = (*rc, *((a, b) if rng.integers(2) else (b, a)))
            c = c or random_chord(rng, bits)
            if c is not None:
                taps.append(Tap(int(rng.integers(0, t_max + 1)), *c, r))
    return taps if len(taps) >= 2 else None


def draw_fixture(tab, bd, n, levels, seed):
    """n episodes of every kind on the levels' patches, in collide.json's format (no engine record)."""
    rng = np.random.default_rng(seed)
    boards, eps = {}, []
    groups = [f"L{lv}" for lv in levels]
    for kind in KINDS:
        k = 0
        while k < n:
            g = groups[int(rng.integers(len(groups)))]
            i = int(rng.integers(len(bd.hw[g])))
            geo = bd.board(g, i)
            taps = busy_taps(rng, tab, geo) if kind == "busy" else draw_taps(rng, tab, geo, kind)
            if taps is None:
                continue
            if kind == "single":  # both phases of the CA's go steps
                taps = [taps[0]._replace(t=int(rng.integers(0, 4)))]
            key = f"{g}/{i}"
            boards.setdefault(key, {"h": int(geo.shape[0]), "w": int(geo.shape[1]),
                                    "geo": [int(x) for x in geo.ravel()]})
            players = []
            for p in taps:
                if rule_key(p.rule) not in players:
                    players.append(rule_key(p.rule))
            eps.append({"kind": kind, "board": key, "players": [list(p) for p in players],
                        "taps": [{"t": int(p.t), "player": players.index(rule_key(p.rule)), "row": int(p.row),
                                  "col": int(p.col), "d0": int(p.d0), "d1": int(p.d1)} for p in taps]})
            k += 1
    return {"boards": boards, "episodes": eps}


def episode_taps(e, players):
    return [Tap(x["t"], x["row"], x["col"], x["d0"], x["d1"], players[x["player"]]) for x in e["taps"]]


def repro(e):
    """An episode in a line: board, players' rules, taps (t player row col d0 d1)."""
    return (f"board {e['board']} players {['{}·{}'.format(p[0], ''.join(map(str, p[1:]))) for p in e['players']]} taps "
            + " ".join(f"[t{x['t']} p{x['player']} ({x['row']},{x['col']}) {x['d0']}-{x['d1']}]" for x in e["taps"]))


def check_sim(tab, bd, fx, ca, show, out):
    """(b): LineCA against the sim's CA mode. Returns the unexplained count."""
    geos = {k: np.array(b["geo"], np.int16).reshape(b["h"], b["w"]) for k, b in fx["boards"].items()}
    delay = phase_of(ca["episodes"], fx["episodes"])
    out(f"LineCA's growth delay after a tap, by the tap step's parity: "
        + ", ".join(f"{'even' if p == 0 else 'odd'} t: {d}" for p, d in sorted(delay.items()))
        + " (the sim: 2 for both; the CA's go steps are even)")
    per = defaultdict(Counter)
    classes = defaultdict(Counter)
    don_all, doff_all = Counter(), Counter()
    shown = defaultdict(list)
    for e, c in zip(fx["episodes"], ca["episodes"]):
        kind = e["kind"]
        geo = geos[e["board"]]
        players = [tuple(p) for p in e["players"]]
        exits = {r: tab.exits(r[0], np.asarray(r[1:], np.int64), geo) for r in players}
        taps = episode_taps(e, players)
        runs, views = sim_runs(tab, geo, taps, players, exits)
        cv, sv = ca_view(c), views["CA"]
        st = per[kind]
        st["episodes"] += 1
        st["taps same"] += cv[2] == sv[2]
        st["end same"] += cv[0] == sv[0]
        st["same"] += same(cv, sv)
        tv = views["CA timing"]
        don2, doff2, count2 = timing(cv[1], tv[1])
        st["timed"] += len(don2)
        st["exact"] += sum(a == 0 and b == 0 for a, b in zip(don2, doff2))
        st["timed counts differ"] += count2
        st["timing end same"] += same(cv, tv)
        exact = len(cv[1]) == len(tv[1]) and cv[1] == tv[1]
        st["timing exact"] += exact
        if not exact and len(shown["not exact"]) < show:
            a, b = first_diff(set(cv[1]), set(tv[1]))
            shown["not exact"].append(f"{kind}: {repro(e)}; LineCA only {a}, CA timing only {b}")
        don, doff, count = timing(cv[1], sv[1])
        st["intervals"] += len(don)
        st["on within 1"] += sum(abs(d) <= 1 for d in don)
        st["off within 1"] += sum(abs(d) <= 1 for d in doff)
        st["interval counts differ"] += count
        don_all.update(max(-3, min(3, d)) for d in don)
        doff_all.update(max(-3, min(3, d)) for d in doff)
        cls = classify_vs_sim(cv, views, tab, geo, taps, players, exits)
        if cls:
            classes[kind][cls] += 1
            if len(shown[cls]) < show:
                a, b = first_diff(cv[0], sv[0])
                shown[cls].append(f"{kind}: {repro(e)}; LineCA only {a}, sim only {b}; taps LineCA {cv[2]} sim {sv[2]}")
        elif not ((cv[0] == sv[0]) and cv[2] == sv[2]):
            pass
        bad_timing = [d for d in don + doff if abs(d) > 1]
        if bad_timing and not cls and len(shown["timing"]) < show:
            shown["timing"].append(f"{kind}: {repro(e)}; deltas beyond 1: {sorted(Counter(bad_timing).items())}")
    unexplained = 0
    for kind in KINDS:
        st = per[kind]
        n = st["episodes"]
        out(f"  {kind:8s} {n} episodes: taps {st['taps same']}/{n}, end state {st['end same']}/{n}, both {st['same']}/{n}; "
            f"intervals both "
            f"drew {st['intervals']}: on within 1 step {st['on within 1']}, off within 1 {st['off within 1']}, "
            f"interval counts differ on {st['interval counts differ']} chords; differences: "
            + (", ".join(f"{k} {v}" for k, v in classes[kind].most_common()) or "none")
            + f"\n           against the sim with the CA's timing: end state and taps {st['timing end same']}/{n}, every "
              f"chord's on and off tick {st['timing exact']}/{n} ({st['exact']}/{st['timed']} intervals exact, interval "
              f"counts differ on {st['timed counts differ']} chords)")
        unexplained += classes[kind]["unexplained"]
    out("  on - sim: " + ", ".join(f"{d:+d}: {don_all[d]}" for d in sorted(don_all))
        + "; off - sim: " + ", ".join(f"{d:+d}: {doff_all[d]}" for d in sorted(doff_all)) + " (±3 = 3 or more)")
    for cls, msgs in shown.items():
        for m in msgs:
            out(f"    [{cls}] {m}")
    return unexplained


# ---------------------------------------------------------------- (c) the engine

def check_engine(tab, bd, fx, ca, show, out):
    """(c): LineCA against Spectacle's engine at rest on collide.json. Returns the unexplained count."""
    geos = {k: np.array(b["geo"], np.int16).reshape(b["h"], b["w"]) for k, b in fx["boards"].items()}
    st = Counter()
    classes = Counter()
    events = Counter()
    shown = defaultdict(list)
    per_episode = []
    for i, (e, c) in enumerate(zip(fx["episodes"], ca["episodes"])):
        geo = geos[e["board"]]
        players = [tuple(p) for p in e["players"]]
        exits = {r: tab.exits(r[0], np.asarray(r[1:], np.int64), geo) for r in players}
        eng = engine_view(e)
        cv = ca_view(c)
        st["episodes"] += 1
        st["taps same"] += cv[2] == eng[2]
        st["end same"] += cv[0] == eng[0]
        es = [ok and (x["player"], x["row"], x["col"], min(x["d0"], x["d1"]), max(x["d0"], x["d1"])) in eng[0]
              for x, ok in zip(e["taps"], eng[2])]
        cs = [ok and (x["player"], x["row"], x["col"], min(x["d0"], x["d1"]), max(x["d0"], x["d1"])) in cv[0]
              for x, ok in zip(e["taps"], cv[2])]
        st["lines"] += sum(eng[2])
        st["survive"] += sum(es)
        st["survive same"] += sum(a == b for a, b, ok in zip(es, cs, eng[2]) if ok)
        taps = episode_taps(e, players)
        runs, views = sim_runs(tab, geo, taps, players, exits)
        tv = views["CA timing"]
        exact = cv[1] == tv[1]
        st["timing exact"] += exact
        if not exact and len(shown["not exact"]) < show:
            a, b = first_diff(set(cv[1]), set(tv[1]))
            shown["not exact"].append(f"episode {i}: {repro(e)}; LineCA only {a}, CA timing only {b}")
        if same(cv, eng):
            per_episode.append("agree")
            continue
        vs = classify_vs_sim(cv, views, tab, geo, taps, players, exits)
        if vs is None:
            cls = "as the sim: " + sim_cause(views, eng)
            events.update({k: 1 for k, v in ca_events(runs["CA"]).items() if v})
        else:
            cls = vs
        classes[cls] += 1
        per_episode.append(cls)
        if len(shown[cls]) < show:
            a, b = first_diff(cv[0], eng[0])
            shown[cls].append(f"episode {i}: {repro(e)}; LineCA only {a}, engine only {b}; taps LineCA {cv[2]} "
                              f"engine {eng[2]}")
    n = st["episodes"]
    out(f"  {n} episodes: taps {st['taps same']}/{n}, end state {st['end same']}/{n}, both {per_episode.count('agree')}"
        f"/{n}; survival agrees for {st['survive same']}/{st['lines']} of the engine's lines ({st['survive']} survive)")
    out(f"  LineCA = the sim with the CA's timing, every chord's on and off tick: {st['timing exact']}/{n}")
    out("  differences: " + (", ".join(f"{k} {v}" for k, v in classes.most_common()) or "none")
        + ("; of the 'as the sim' ones, episodes whose sim CA run has (sim_parity.ca_events): "
           + ", ".join(f"{k} {v}" for k, v in events.most_common())
           if events else ""))
    for cls, msgs in shown.items():
        for m in msgs:
            out(f"    [{cls}] {m}")
    return sum(v for k, v in classes.items() if "unexplained" in k), per_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data/strand-v2")
    ap.add_argument("--collide", default=None, help="the engine fixture (default <data>/collide.json)")
    ap.add_argument("--n", type=int, default=200, help="(b): episodes per draw_taps kind")
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=6, help="examples to print per class")
    ap.add_argument("--classes", default=None, help="write (c)'s class per episode to this JSON file")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--skip-engine", action="store_true")
    args = ap.parse_args(argv)
    data = args.data or default_dir()
    tab, bd = RuleTable(data), Boards(data)
    bad = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        if not args.skip_sim:
            fx = draw_fixture(tab, bd, args.n, args.levels, args.seed)
            path = os.path.join(tmp, "draw.json")
            with open(path, "w") as f:
                json.dump(fx, f)
            ca = run_line_ca(path, os.path.join(tmp, "ca-draw.json"))
            print(f"(b) LineCA against sim.py's CA mode: {len(fx['episodes'])} draw_taps episodes on "
                  f"{', '.join(f'L{lv}' for lv in args.levels)} (seed {args.seed})")
            bad += check_sim(tab, bd, fx, ca, args.show, print)
        if not args.skip_engine:
            path = args.collide or os.path.join(data, "collide.json")
            with open(path) as f:
                fx = json.load(f)
            ca = run_line_ca(path, os.path.join(tmp, "ca-collide.json"))
            print(f"(c) LineCA against Spectacle's engine at rest: {path} (spectacle "
                  f"{fx.get('spectacle', {}).get('commit', '?')[:12]})")
            u, per_episode = check_engine(tab, bd, fx, ca, args.show, print)
            bad += u
            if args.classes:
                with open(args.classes, "w") as f:
                    json.dump({"what": "nca.strand.ca_parity's class per collide.json episode: LineCA against the "
                                       "engine at rest", "classes": per_episode}, f)
    print(f"{time.time() - t0:.1f} s; {bad} unexplained")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
