"""The multi-strand sim (nca/strand/sim.py) against Spectacle's engine (docs/spectacle-nca-taps.md §4.3, build item
2): `python -m nca.strand.sim --parity [--collide data/strand-v2/collide.json]`, the fixture from
`npx tsx scripts/strand-export.ts --collide` (Spectacle's engine with crossingMode 'tile', mutualCut,
overlapOwnLines, scripted taps of 2-4 players on the L2 / L3 patches; see its mainCollide).

Each episode is replayed four ways and compared with the engine, per player (= rule = code), chord by chord:
  game   wave=0, sequential=True: instant wipes and the engine's move order. Must equal the engine: the taps it
         accepts, the end state, and (reported) every chord's on and off tick
  CA     wave=1, sequential=False: the product. Compared at the settled end state (which chords each player
         holds at rest, which taps are accepted); its on ticks against the engine's are reported
  and the two halves (wave=0 alone, sequential=True alone), so every CA difference is put down to the waves
  (the game wipes a line in one tick; the CA's wave takes a tick a chord, and what it erases lingers for the
  slack window), to the engine's head timing (its two-way growth is a second head on the path: when the forward
  head stops at a tail the back head loses that tick's move, a join hands the absorbed path's far head the
  joiner's pace, and heads move one at a time in player order; the CA's tips keep their own pace, all at
  once), or to both. A difference none of them explains is a rule difference: it fails the check.
Per tap, a line SURVIVES if its tapped chord is still its player's at rest (lines of one player that joined
survive or go together).
The CA-only events behind the wave differences are counted from the CA run's timeline: a dying line's fleeing
tip hitting a third line ("worm hits"), tips hitting a line that was hit but the wave has not reached yet, or a
tile whose chords are dying ("dying-tile hits"), taps refused on such lines or tiles, tips meeting a wave of
their own rule head on, lines erased through a touching line of their rule ("contact").
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np

from .rules import Boards, RuleTable, default_dir
from .sim import CONTACT, HIT, INF, KILL, REFUSE, Tap, simulate

MODES = {"game": (0, True), "CA": (1, False), "wave0": (0, False), "seq": (1, True)}


def engine_view(e):
    """(end set, intervals list, accepted) of the engine's run: keys (player, row, col, a, b)."""
    iv = sorted((p, r, c, a, b, on, INF if off < 0 else off) for p, r, c, a, b, on, off in e["chords"])
    end = {x[:5] for x in iv if x[6] >= INF}
    return end, iv, [bool(x["ok"]) for x in e["taps"]]


def sim_view(ep):
    c = ep.chords[~ep.chords["maybe"] & (ep.chords["on"] < ep.chords["off"])]  # drawn and gone in one tick: unseen
    code = ep.lines["code"][c["line"]] if len(c) else np.zeros(0, np.int64)
    a, b = np.minimum(c["a"], c["b"]), np.maximum(c["a"], c["b"])
    iv = sorted(zip(code.tolist(), c["row"].tolist(), c["col"].tolist(), a.tolist(), b.tolist(), c["on"].tolist(),
                    c["off"].tolist()))
    merged = []  # gone and back in one tick (cut, then tapped again): one interval to a per-tick snapshot
    for x in iv:
        if merged and merged[-1][:5] == x[:5] and merged[-1][6] == x[5]:
            merged[-1] = merged[-1][:6] + (x[6],)
        else:
            merged.append(x)
    end = {x[:5] for x in merged if x[6] >= INF}
    return end, merged, ep.accepted.tolist()


def ca_events(ep):
    """Counts of the CA-only events in a CA run (each a place where the game, which wipes at once, goes on
    differently)."""
    ev, hit = ep.events, ep.lines["hit"]
    hits = ev[ev["kind"] == HIT]
    c = ep.chords[~ep.chords["maybe"]]
    dying_line_refusals = 0
    for x in ev[ev["kind"] == REFUSE]:
        why = ep.reason[x["other"]]
        if "dying" in why:
            dying_line_refusals += 1
        elif why in ("a rival's tile", "your own line"):  # the line there was hit (this tick or before): the game
            t = x["t"]                                   # has wiped it by the time the tap lands
            on = (c["row"] == x["row"]) & (c["col"] == x["col"]) & (c["on"] <= t) & (c["off"] > t)
            dying_line_refusals += bool(on.any() and (hit[c["line"][on]] <= t).all())
    return {"worm hits": int(sum(hit[line] < t for t, line in zip(hits["t"], hits["line"]))),
            "hits on a dying line": int(sum(v >= 0 and hit[v] < t for t, v in zip(hits["t"], hits["other"]))),
            "dying-tile hits": int((hits["other"] == -2).sum()),
            "taps refused on a dying line": dying_line_refusals,
            "waves met head on": int(((ev["kind"] == KILL) & (ev["other"] == -2)).sum()),
            "contact": int((ep.lines["role"] & CONTACT > 0).sum())}


def survivors(end, taps, accepted):
    return [bool(ok) and (x["player"], x["row"], x["col"], min(x["d0"], x["d1"]), max(x["d0"], x["d1"])) in end
            for x, ok in zip(taps, accepted)]


def first_diff(a, b):
    sa, sb = set(a), set(b)
    only_a, only_b = sorted(sa - sb), sorted(sb - sa)
    return (only_a[:3], only_b[:3])


def parity(data_dir=None, path=None, show=12, out=print) -> int:
    """Run the comparison and print the report. Returns the number of failures (rule differences, boards that
    don't match boards.npz)."""
    t0 = time.time()
    tab, bd = RuleTable(data_dir), Boards(data_dir)
    path = path or os.path.join(data_dir or default_dir(), "collide.json")
    with open(path) as f:
        fx = json.load(f)
    geos, board_ok = {}, 0
    for key, b in fx["boards"].items():
        geo = np.array(b["geo"], np.int16).reshape(b["h"], b["w"])
        g, ri = key.split("/")
        board_ok += np.array_equal(geo, bd.board(g, int(ri)))
        geos[key] = geo
    eps = fx["episodes"]
    n = len(eps)
    stats = Counter()
    causes = Counter()
    events = Counter()
    shown = {}
    on_same = on_total = 0
    deltas = Counter()

    def note(kind, msg):
        shown.setdefault(kind, [])
        if len(shown[kind]) < show:
            shown[kind].append(msg)

    for i, e in enumerate(eps):
        geo = geos[e["board"]]
        players = [tuple(p) for p in e["players"]]
        taps = [Tap(x["t"], x["row"], x["col"], x["d0"], x["d1"], players[x["player"]]) for x in e["taps"]]
        exits = {r: tab.exits(r[0], np.asarray(r[1:], np.int64), geo) for r in players}
        eng_end, eng_iv, eng_acc = engine_view(e)
        stats["taps"] += len(taps)
        stats["engine refused"] += sum(not x for x in eng_acc)
        stats["wipes"] += len(e["wipes"])
        stats[e["board"][:2]] += 1
        runs = {m: simulate(tab, geo, taps, wave=w, sequential=s, players=players, exits=exits)
                for m, (w, s) in MODES.items()}
        views = {m: sim_view(r) for m, r in runs.items()}
        same = {m: (v[0] == eng_end and v[2] == eng_acc) for m, v in views.items()}
        # game mode: the engine exactly
        g_end, g_iv, g_acc = views["game"]
        stats["game taps"] += g_acc == eng_acc
        stats["game end"] += g_end == eng_end
        stats["game timeline"] += g_iv == eng_iv
        if not same["game"]:
            a, b = first_diff(g_end, eng_end)
            note("game end", f"episode {i} ({e['board']}): sim only {a}, engine only {b}; taps sim {g_acc} engine {eng_acc}")
        elif g_iv != eng_iv:
            a, b = first_diff(g_iv, eng_iv)
            note("game timeline", f"episode {i}: sim only {a}, engine only {b}")
        # the CA: the end state, and its on ticks
        c_end, c_iv, c_acc = views["CA"]
        stats["CA taps"] += c_acc == eng_acc
        stats["CA end"] += same["CA"]
        eng_on = Counter(x[:6] for x in eng_iv)
        ca_on = Counter(x[:6] for x in c_iv)
        both = Counter(x[:5] for x in eng_iv) & Counter(x[:5] for x in c_iv)
        on_total += sum(both.values())
        on_same += sum((eng_on & ca_on).values())
        e_first = {}
        for x in eng_iv:
            e_first.setdefault(x[:5], x[5])
        c_first = {}
        for x in c_iv:
            c_first.setdefault(x[:5], x[5])
        for k in e_first.keys() & c_first.keys():
            deltas[max(-4, min(4, c_first[k] - e_first[k]))] += 1
        es, cs = survivors(eng_end, e["taps"], eng_acc), survivors(c_end, e["taps"], c_acc)
        stats["engine lines"] += sum(eng_acc)
        stats["survive"] += sum(es)
        stats["survive same"] += sum(a == b for a, b, ok in zip(es, cs, eng_acc) if ok)
        if not same["CA"]:
            if same["wave0"]:
                cause = "waves"
            elif same["seq"]:
                cause = "head timing"
            elif same["game"]:
                cause = "waves + head timing"
            else:
                cause = "unexplained"
            causes[cause] += 1
            ev = ca_events(runs["CA"])
            events.update({k: v for k, v in ev.items() if v})
            a, b = first_diff(c_end, eng_end)
            note(cause, f"episode {i} ({e['board']}): CA only {a}, engine only {b}; CA taps {c_acc} engine {eng_acc}; "
                        f"CA events {', '.join(f'{k} {v}' for k, v in ev.items() if v) or 'none'}")
    fails = (n - stats["game end"]) + (len(geos) - board_ok) + causes["unexplained"]
    out(f"parity: {n} episodes ({stats['L2']} on L2, {stats['L3']} on L3), {stats['taps']} taps "
        f"({stats['engine refused']} refused or not sent), {stats['wipes']} lines cut in collisions by the engine; "
        f"boards = boards.npz {board_ok}/{len(geos)}; spectacle {fx.get('spectacle', {}).get('commit', '?')[:12]}")
    out(f"game mode (instant wipes, the engine's move order) = the engine: taps {stats['game taps']}/{n}, end state "
        f"{stats['game end']}/{n}, every chord's on and off tick {stats['game timeline']}/{n}")
    out(f"CA (waves, all tips at once) = the engine at rest: taps {stats['CA taps']}/{n}, end state {stats['CA end']}/{n}, "
        f"survival agrees for {stats['survive same']}/{stats['engine lines']} of the engine's lines ({stats['survive']} survive); "
        f"on ticks of the chords both drew {on_same}/{on_total} ({100 * on_same / max(1, on_total):.1f} %); first on tick, "
        f"CA - engine: " + ", ".join(f"{'<=' if d == -4 else '>=' if d == 4 else ''}{d:+d}: {deltas[d]}"
                                      for d in sorted(deltas)))
    if causes:
        out("CA differences by cause: " + ", ".join(f"{k} {v}" for k, v in causes.most_common())
            + "; CA-only events in them: " + ", ".join(f"{k} {v}" for k, v in events.most_common()))
    for kind, msgs in shown.items():
        for m in msgs:
            out(f"  [{kind}] {m}")
    out(f"{time.time() - t0:.1f} s; {fails} failures (game-mode end states that differ from the engine, boards, "
        f"unexplained CA differences)")
    return fails
