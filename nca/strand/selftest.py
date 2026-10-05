"""Fast self-check of the M1 trainer (nca/strand/train.py): `python -m nca.strand.selftest` (about a minute on
the Pi; needs data/strand from scripts/strand-export.ts).

  1. shapes: a pool slot's consts [22,S,S] (mask, chords, tap), its planes [6,S,S]
  2. targets: the m1a target (every edge with a finite distance) equals the exported walkStrand walk's edges
     (loader.edge_planes) on 200 taps at levels 2-4, the tapped chord at distance 0; the m1b closed planes
     equal loader.pattern_sample's, and the strands are the exported ones (count and lengths)
  3. the quick check: the oracle injected as the state scores exact 1.0 (m1a: steps ratio 1, excess 0;
     m1b: board 1.0); an oracle one step late scores exact 1.0 with excess 1; edge 0 marked on every
     cell as well scores below 1 (m1a: exact < 0.5; m1b: board < 0.5)
  4. training: the collapse guard rolls back on a forced collapse (lr scale halved, weights = best.pt's), and
     with --max-rollbacks 0 the run ends {"stopped": "collapsed"}
  5. --resume: 3 + 3 iterations equal 6 in one go (weights, optimiser, pool states), bit for bit
  6. m1b runs from an m1a checkpoint (--init), and pool.npz reads through nca.dashboard.pool_to_json
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .loader import StrandFile, default_dir, edge_planes, file_paths, load_meta, pattern_sample
from . import train as T

FAILS = []


def check(msg, ok):
    print(("OK   " if ok else "FAIL ") + msg, file=sys.__stdout__, flush=True)
    if not ok:
        FAILS.append(msg)


def pad(x, S):
    return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, S - x.shape[-2]), (0, S - x.shape[-1])])


def targets():
    rng = np.random.default_rng(1)
    meta = load_meta()
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}
    bd = T.Boards(default_dir(), "train", 2, subset_of)
    s = T.new_slot(rng, "m1a", bd)
    S = bd.S
    check(f"m1a slot shapes: consts {s['consts'].shape}, a {s['a'].shape}, b {s['b'].shape} (S = {S})",
          s["consts"].shape == (22, S, S) and s["a"].shape == s["b"].shape == (6, S, S))
    s = T.new_slot(rng, "m1b", bd)
    check("m1b slot shapes, no tap planes", s["consts"].shape == (22, S, S) and s["consts"][16:].sum() == 0)

    bad = n = zero = 0
    for split, L in (("train", 2), ("train", 3), ("eval", 4)):
        ps = file_paths(None, split, (L,))
        for p in rng.choice(ps, 4, replace=False):
            f = StrandFile(p)
            Sx = max(f["mask"].shape[1:])
            for i in rng.choice(f.n_taps, 17, replace=False):
                b, row, col, d0, d1 = f.tap_args(int(i))
                h, w = f.hw(b)
                mask = pad(f.mask(b)[None], Sx)[0]
                ch = pad(f.chords(b), Sx)
                slot = T.m1a_slot(mask, ch, (row, col, d0, d1), (0, b), f.rule_id)
                want = pad(edge_planes(f.tap(int(i)), h, w), Sx) > 0
                bad += not np.array_equal(slot["a"] < T.INF, want)
                zero += not (slot["a"][d0, row, col] == 0 and slot["a"][d1, row, col] == 0)
                n += 1
    check(f"m1a target = the exported walkStrand strand's edges on {n} taps (levels 2-4), tap at distance 0",
          bad == 0 and zero == 0)

    bad = nb = 0
    for split, L in (("train", 2), ("train", 3)):
        for p in rng.choice(file_paths(None, split, (L,)), 2, replace=False):
            f = StrandFile(p)
            for b in range(f.n_boards):
                h, w = f.hw(b)
                pl = T.pattern_planes(f.mask(b), f.chords(b))
                _, want = pattern_sample(f, b)
                mine = f["strand_board"] == b
                bad += not (np.array_equal(pl["closed"] > 0, want > 0)
                            and sorted(pl["length"]) == sorted(f["strand_len"][mine])
                            and int(pl["closedS"].sum()) == int(f["strand_closed"][mine].sum()))
                nb += 1
    check(f"m1b closed planes = loader.pattern_sample, strands = the exported ones, on {nb} boards", bad == 0)


def oracles():
    meta = load_meta()
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}
    for task, n in (("m1a", 18), ("m1b", 6)):
        ev = T.EvalLevel(task, T.Boards(default_dir(), "eval", 2, subset_of), n, 8, 2000)
        a = torch.from_numpy(ev.a).float()
        out = T.OUT[task]

        def oracle(lag=0, extra=False):
            def step(st, walls, cs, t, sl):
                st = st.clone()
                st[:, out] = (a[sl] < t - lag).float() if task == "m1a" else a[sl]
                if extra:
                    st[:, out.start, :, :] = torch.maximum(st[:, out.start], walls[:, 0])  # edge 0 everywhere
                return st
            return step

        q = T.summarise(task, [ev], [T.evaluate(oracle(), ev, 32, torch.device("cpu"), 8)])
        ok = q["exact"] == 1.0 and q["balanced"] == 1.0 and q["iou"] == 1.0
        if task == "m1a":
            ok = ok and q["steps"]["ratio"] == 1.0 and q["steps"]["excess"] == 0.0
        else:
            ok = ok and q["board"] == 1.0
        check(f"{task}: the oracle as the state scores exact 1.0 ({json.dumps(q)[:160]}...)", ok)
        if task == "m1a":
            q = T.summarise(task, [ev], [T.evaluate(oracle(lag=1), ev, 32, torch.device("cpu"), 8)])
            check("m1a: an oracle one step late: exact 1.0, excess 1", q["exact"] == 1.0 and q["steps"]["excess"] == 1.0)
        q = T.summarise(task, [ev], [T.evaluate(oracle(extra=True), ev, 32, torch.device("cpu"), 8)])
        low = q["exact"] < 0.5 if task == "m1a" else q["exact"] < 1 and q["board"] < 0.5
        check(f"{task}: marking edge 0 on every cell as well scores below 1 (exact {q['exact']}"
              + (f", board {q['board']})" if task == "m1b" else ")"), low)


TINY = ["--levels", "2", "--eval-levels", "2", "--hidden", "8", "--channels", "16", "--batch", "4",
        "--pool-size", "8", "--eval-n", "6", "--eval-mult", "2", "--eval-cap", "30", "--steps-mult", "0.5", "1",
        "--bptt", "4", "--last-k", "2", "--threads", "1", "--snap-every", "0"]


def log_of(name):
    return [json.loads(x) for x in open(f"runs/{name}/log.jsonl")]


def runs():
    # 4. a forced collapse at the first check after iteration 0: rollback to best.pt (iteration 0)
    T.main(["--name", "rb", *TINY, "--iters", "50", "--eval-every", "50", "--force-collapse", "1"])
    lg = log_of("rb")
    rb = [x for x in lg if "rollback" in x]
    best = torch.load("runs/rb/best.pt", weights_only=False)
    ck = torch.load("runs/rb/ckpt.pt", weights_only=False)
    same = all(torch.equal(best["model"][k], ck["model"][k]) for k in best["model"])
    check(f"forced collapse: a rollback line (lrScale {rb[0]['lrScale'] if rb else '?'}, restored "
          f"{rb[0]['restored'] if rb else '?'}), ckpt.pt's weights = best.pt's",
          len(rb) == 1 and rb[0]["lrScale"] == 0.5 and rb[0]["restored"] == 0 and same and ck["rollbacks"] == 1)
    T.main(["--name", "rb0", *TINY, "--iters", "50", "--eval-every", "50", "--force-collapse", "1",
            "--max-rollbacks", "0"])
    check("...and with --max-rollbacks 0 the run ends {stopped: collapsed}", log_of("rb0")[-1].get("stopped") == "collapsed")

    # 5. resume: 3 + 3 = 6, bit for bit (flat lr: the decay depends on --iters)
    flat = ["--lr", "1e-3", "--lr-floor", "1e-3", "--warmup", "0", "--eval-every", "50"]
    T.main(["--name", "straight", *TINY, *flat, "--iters", "6"])
    T.main(["--name", "chunked", *TINY, *flat, "--iters", "3"])
    T.main(["--name", "chunked", "--resume", "--iters", "6", "--threads", "1", "--snap-every", "0"])
    a = torch.load("runs/straight/ckpt.pt", weights_only=False)
    b = torch.load("runs/chunked/ckpt.pt", weights_only=False)
    w = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    o = all(torch.equal(x, y) for x, y in zip(a["opt"]["state"][0].values(), b["opt"]["state"][0].values()))
    pa, pb = a["pool"][2], b["pool"][2]
    p = torch.equal(pa["state"], pb["state"]) and all(np.array_equal(pa[k], pb[k]) for k in ("consts", "a", "b", "age"))
    check(f"--resume: 3 + 3 iterations = 6 in one go (weights {w}, optimiser {o}, pool {p}), iteration "
          f"{b['iteration']}", w and o and p and b["iteration"] == 6)

    # 6. m1b from the m1a checkpoint; pool.npz through the dashboard's reader
    T.main(["--name", "m1b", "--task", "m1b", *TINY[:-1], "1", "--iters", "2", "--eval-every", "50",
            "--init", "runs/straight/best.pt"])
    lg = log_of("m1b")
    q = [x for x in lg if "q" in x]
    check(f"m1b --init from m1a runs: q has board/bySubset/byLen ({sorted(q[-1]['q']) if q else None})",
          lg[-1].get("stopped") == "done" and q and {"board", "bySubset", "byLen"} <= set(q[-1]["q"]))
    try:
        from ..dashboard import pool_to_json
        js = pool_to_json(Path("runs/m1b/pool.npz"))
        R = js["radii"][0]
        g = js["by_radius"][str(R)]
        check(f"pool.npz reads through nca.dashboard.pool_to_json (R {R}, S {g['S']}, n {g['n']}, C {g['C']})",
              g["S"] == 2 * R + 1 and g["n"] == 8 and g["C"] == 16)
    except Exception as e:  # noqa: BLE001
        check(f"pool.npz reads through nca.dashboard.pool_to_json ({e!r})", False)


def main():
    torch.set_num_threads(1)
    targets()
    oracles()
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            with open(os.devnull, "w") as null:
                out, sys.stdout = sys.stdout, null  # the trainer's log lines
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
