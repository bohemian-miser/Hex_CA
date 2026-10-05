"""Fast self-check of the rule-at-the-tap trainer (nca/strand/train2.py) and its data (nca/strand/rules.py):
`python -m nca.strand.selftest2` (a few minutes on the Pi; needs data/strand-v2 from
`npx tsx scripts/strand-export.ts --rule-table` and data/strand for the legacy set).

  1. rules: 300 parity.npz rule-boards (all 7 subsets) render and walk exactly as walkStrand; flipping the
     mirror sign or swapping two digits breaks it (negative controls); the split's held-out counts; sampled
     train rules are never held out (v2 or legacy)
  2. inputs, per --inputs arm: shapes; the static planes at a cell are its (type, rot, mirror) row and do not
     depend on the rule; the tap planes are on the tapped cell only and decode back to the rule and the chord;
     c-bc's broadcast code is on every board cell; nothing else in the consts changes with the rule
  3. targets: a slot's m1a target = the exported walkStrand strand (parity taps); m1b's open-news times
     (ends at their own distance, circuits = the draw time); edit damage re-types 1-3 cells, never the tap's
  4. the legacy set = train.py's quick-check taps (same boards, taps, targets), and its chords rendered from
     the table = the v1 chord planes
  5. the quick check: the oracle state scores exact 1.0 (steps ratio 1, excess 0) for m1a and m1b, one step
     late excess 1, an extra edge exact < 0.5; the code probe reads an injected code back exactly and a
     random state at chance
  6. training, every arm (a, c, c-bc, d, d-cs; one at --depth 2): the loss is finite and iteration 0's check
     runs; a forced collapse rolls back; --resume is bit-exact (3 + 3 = 6) and --ckpt-pool half resumes;
     m1b runs from an m1a checkpoint; pool.npz reads through nca.dashboard.pool_to_json
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import train as V1
from . import train2 as T
from .loader import load_meta
from .rules import CODE_BITS, Boards, RuleTable, Strand, chord_planes, default_dir, same_strand
from .walker import walk

FAILS = []
LEGACY = os.path.join(os.path.dirname(default_dir()), "strand")


def check(msg, ok):
    print(("OK   " if ok else "FAIL ") + msg, file=sys.__stdout__, flush=True)
    if not ok:
        FAILS.append(msg)


def parity_rows(tab, bd, n, seed=0):
    """(s, digits, geo, taps [(row, col, d0, d1, Strand)]) for n parity.npz samples."""
    with np.load(os.path.join(tab.dir, "parity.npz")) as z:
        p = {k: z[k] for k in z.files}
    rng = np.random.default_rng(seed)
    per = n // 7  # stratified: every subset, up to n / 7 rule-boards each
    pick = np.sort(np.concatenate([rng.choice(ix, min(per, len(ix)), replace=False)
                                   for ix in (np.nonzero(p["sample_subset"] == s)[0] for s in range(7))]))
    tb = p["tap_board"]
    out = []
    for j in pick:
        s, idx = int(p["sample_subset"][j]), int(p["sample_index"][j])
        geo = bd.board(("L2", "L3", "L4")[p["sample_group"][j]], int(p["sample_board"][j]))
        taps = []
        for i in np.nonzero(tb == j)[0][:4]:
            a, e = p["tap_ptr"][i], p["tap_ptr"][i + 1]
            g = lambda k: p[k][a:e].astype(np.int64)  # noqa: E731
            st = Strand(g("step_row"), g("step_col"), g("step_in"), g("step_out"), g("step_index"),
                        bool(p["tap_closed"][i]))
            taps.append((int(p["tap_row"][i]), int(p["tap_col"][i]), int(p["tap_d0"][i]), int(p["tap_d1"][i]), st))
        out.append((s, tab.digits_of(s, idx), geo, p["chord_bits"][j, :geo.shape[0], :geo.shape[1]], taps))
    return out


def rules_checks(tab, bd):
    rows = parity_rows(tab, bd, 300)
    bad = walks = 0
    subs = set()
    for s, digits, geo, bits, taps in rows:
        subs.add(s)
        bad += not np.array_equal(tab.render_bits(s, digits, geo), bits)
        ex = tab.exits(s, digits, geo)
        for r, c, d0, d1, st in taps:
            walks += 1
            bad += same_strand(walk(ex, geo >= 0, r, c, d0, d1), st) is not None
    check(f"rules: {len(rows)} parity rule-boards (stratified) over {len(subs)} subsets render and {walks} taps walk as walkStrand",
          bad == 0 and len(subs) == 7)
    flip = sum(not np.array_equal(tab.render_bits(s, d, np.where(geo >= 0, geo ^ 1, -1)), bits)
               for s, d, geo, bits, _ in rows if bits.any())
    swapped = 0
    n_sw = 0
    for s, d, geo, bits, _ in rows:
        if s != 6:
            continue
        d2 = d.copy()
        t = int(np.argmax(np.bincount(geo[geo >= 0] // 12, minlength=9)))  # the most common type on the board
        d2[t] = (d2[t] + 1) % 5
        n_sw += 1
        swapped += not np.array_equal(tab.render_bits(s, d2, geo), bits)
    n_on = sum(1 for r in rows if r[3].any())
    check(f"rules: negative controls -- the mirror bit flipped breaks {flip}/{n_on} renderings, the most common "
          f"type's digit changed breaks {swapped}/{n_sw}", flip == n_on and swapped == n_sw)
    held = [int(tab.held_out(s).sum()) for s in range(tab.n_sub)]
    check(f"split: held out per subset {held}", held == [1, 6, 13, 2, 3, 64, 390625])
    rng = np.random.default_rng(3)
    legacy = set(tab.legacy_heldout())
    draws = [tab.sample(rng) for _ in range(3000)]
    bad = sum(tab.held_out(s)[tab.index_of(s, d)] or (s, tab.index_of(s, d)) in legacy for s, d in draws)
    per = np.bincount([s for s, _ in draws], minlength=7)
    check(f"split: 3,000 sampled train rules, none held out (v2 or legacy); per subset {per.tolist()}",
          bad == 0 and per.min() > 300)


def input_checks(tab, bd):
    rng = np.random.default_rng(5)
    geo = T.pad_geo(bd.board("L3", 2), 37)
    s1, d1 = 6, np.array([0, 1, 2, 3, 4, 0, 1, 2, 3])
    s2, d2 = 2, tab.digits_of(2, 5)
    tap = random_tap = None
    for s, d in ((s1, d1), (s2, d2)):
        random_tap = T.random_chord(rng, tab.render_bits(s, d, geo))
    tap = random_tap
    codes = T.Codes(tab)
    for inputs in T.INPUTS:
        pl = T.Planes(tab, inputs, torch.device("cpu"))
        x1 = pl(geo[None], codes([[s1, *d1]]), [tap])[0].numpy()
        x2 = pl(geo[None], codes([[s2, *d2]]), [tap])[0].numpy()
        n_static = T.STATIC[T.static_of(inputs)]
        st = slice(1, 1 + n_static)
        ok = x1.shape == (T.n_inputs(inputs), 37, 37) and np.array_equal(x1[0], (geo >= 0).astype(np.float32))
        r, c = np.nonzero(geo >= 0)
        k = len(r) // 2
        ok &= np.allclose(x1[st, r[k], c[k]], tab.static[T.static_of(inputs)][geo[r[k], c[k]]])
        ok &= np.array_equal(x1[st], x2[st])  # the static planes don't see the rule
        tp = x1[pl.tap_at:]
        off = tp.copy()
        off[:, tap[0], tap[1]] = 0
        ok &= not off.any()
        cls, dig = tab.decode(tp[:CODE_BITS, tap[0], tap[1]])
        ok &= np.array_equal(cls, tab.code(s1, d1)[:8] > 0) and np.array_equal(dig, d1)
        ok &= set(np.nonzero(tp[CODE_BITS:, tap[0], tap[1]])[0].tolist()) == {tap[2], tap[3]}
        if inputs == "c-bc":
            bc = x1[1 + n_static:pl.tap_at]
            ok &= np.array_equal(bc[:, r, c], np.repeat(tab.code(s1, d1)[:, None], len(r), 1)) and \
                not bc[:, geo < 0].any()
            ok &= not np.array_equal(x1[1 + n_static:pl.tap_at], x2[1 + n_static:pl.tap_at])
        else:
            same = np.ones(len(x1), bool)
            same[pl.tap_at:pl.tap_at + CODE_BITS] = False
            ok &= np.array_equal(x1[same], x2[same])  # only the tap's code planes change with the rule
        check(f"inputs {inputs}: {x1.shape[0]} consts (mask, {n_static} static"
              + (", 53 broadcast" if inputs == "c-bc" else "") + ", tap 59); static planes = the cell's "
              "(type, rot, mirror) row and blind to the rule; the tap planes on the tapped cell only, decoding to "
              "the rule and the chord", ok)


def target_checks(tab, bd):
    rows = parity_rows(tab, bd, 120, seed=1)
    bad = zero = n = 0
    oc_bad = 0
    for s, digits, geo, _, taps in rows:
        S = max(geo.shape)
        g = T.pad_geo(geo, S)
        for r, c, d0, d1, st in taps:
            slot = T.make_slot(tab, g, np.concatenate([[s], digits]), (r, c, d0, d1), 0, "m1b")
            want = np.zeros((6, S, S), bool)
            want[st.ins, st.rows, st.cols] = want[st.outs, st.rows, st.cols] = True
            bad += not np.array_equal(slot["a"] < T.INF, want)
            zero += not (slot["a"][d0, r, c] == 0 and slot["a"][d1, r, c] == 0)
            on = slot["a"] < T.INF
            if st.closed:
                oc_bad += not np.array_equal(slot["oc"][on], slot["a"][on])
            else:
                first, last = 0, len(st) - 1  # the ends: the open news is there as soon as they are drawn
                for e in (first, last):
                    oc_bad += slot["oc"][st.outs[e] if e == last else st.ins[e], st.rows[e], st.cols[e]] != \
                        slot["a"][st.outs[e] if e == last else st.ins[e], st.rows[e], st.cols[e]]
                oc_bad += not (slot["oc"][on] >= slot["a"][on]).all()
                oc_bad += slot["ideal"] != int(slot["oc"][on].max()) + 1
            n += 1
    check(f"targets: {n} slots' m1a targets = the exported walkStrand strands, tapped chord at distance 0", bad == 0
          and zero == 0)
    check("targets: m1b's open news reaches a tail's ends as they are drawn and never precedes the draw; a "
          "circuit's closed planes are due as drawn; m1b's ideal = the latest news + 1", oc_bad == 0)
    # edit damage
    rng = np.random.default_rng(2)
    P = T.new_pool(rng, tab, bd, "L3", 37, 6, 16, "m1a", torch.device("cpu"))
    ok = True
    for i in range(6):
        g0, tap = P["geo"][i].copy(), tuple(P["tap"][i])
        applied, _ = T.damage(rng, tab, "m1a", P, i, "edit")
        diff = np.argwhere(P["geo"][i] != g0)
        ok &= applied and 1 <= len(diff) <= 3 and all((r, c) != tap[:2] for r, c in diff)
        ok &= all(P["geo"][i][r, c] % 2 == g0[r, c] % 2 and g0[r, c] >= 0 for r, c in diff)
    check("damage: an edit re-types 1-3 board cells (another type / rotation, same mirror), never the tapped cell", ok)


def legacy_checks(tab):
    meta = load_meta(LEGACY)
    subset_of = {r["id"]: r["subset"] for r in meta["rules"]}
    ok, n = True, 0
    for L in (2, 4):
        v1 = V1.EvalLevel("m1a", V1.Boards(LEGACY, "eval", L, subset_of), 30, 8, 2000)
        v2 = T.legacy_set("m1a", tab, LEGACY, L, 30, 8, 2000)
        ok &= np.array_equal(v1.a, v2.a) and np.array_equal(v1.items["bucket"], v2.items["bucket"])
        for i in range(len(v2.geo)):
            r, c, d0, d1 = v2.tap[i]
            ok &= v1.consts[i, 16 + d0, r, c] == 1 and v1.consts[i, 16 + d1, r, c] == 1
            ch = chord_planes(tab.render_bits(int(v2.rule[i, 0]), v2.rule[i, 1:], v2.geo[i]))
            ok &= np.array_equal(ch, v1.consts[i, 1:16])
            n += 1
    check(f"legacy set: {n} taps = train.py's quick-check taps and targets (levels 2, 4), the table's rendering "
          "of each rule = the v1 chord planes", ok)


def oracle_checks(tab, bd):
    for task in ("m1a", "m1b"):
        evs = [T.legacy_set(task, tab, LEGACY, 2, 18, 8, 2000), T.wide_set(task, tab, bd, 2, 18, 8, 2000)]
        pl = T.Planes(tab, "c", torch.device("cpu"))
        codes = T.Codes(tab)

        def oracle(ev, lag=0, extra=False):
            a, oc = torch.from_numpy(ev.a).float(), torch.from_numpy(ev.oc).float()
            closed = torch.from_numpy(ev.closed).view(-1, 1, 1, 1)

            def step(st, walls, cs, t, sl):
                st = st.clone()
                tt = t - lag
                st[:, 1:7] = (a[sl] < tt).float()
                if task == "m1b":  # closed until the open news arrives
                    on = a[sl] < tt
                    st[:, 7:13] = (on & (closed[sl] | (oc[sl] >= tt))).float()
                if extra:
                    st[:, 1] = torch.maximum(st[:, 1], walls[:, 0])
                return st
            return step

        def run(**kw):
            return T.summarise(task, evs, [T.evaluate(oracle(ev, **kw), pl, ev, 32, torch.device("cpu"), 8, codes)
                                           for ev in evs], tab)
        q = run()
        ok = q["exact"] == 1.0 and q["balanced"] == 1.0 and q["iou"] == 1.0 and q["steps"]["ratio"] == 1.0 and \
            q["steps"]["excess"] == 0.0 and set(q["bySet"]) == {"legacy", "wide"}
        if task == "m1b":
            ok &= q["parts"] == {"edges": 1.0, "closed": 1.0}
        check(f"{task}: the oracle state scores exact 1.0, steps ratio 1, excess 0 on both sets "
              f"({json.dumps(q)[:120]}...)", ok)
        q = run(lag=1)
        check(f"{task}: an oracle one step late: exact 1.0, excess 1", q["exact"] == 1.0 and q["steps"]["excess"] == 1.0)
        q = run(extra=True)
        check(f"{task}: marking edge 0 on every cell as well: exact {q['exact']} < 0.5", q["exact"] < 0.5)
    # the code probe
    rng = np.random.default_rng(4)
    P = T.new_pool(rng, tab, bd, "L2", 12, 200, 80, "m1a", torch.device("cpu"))  # enough codes to span the space
    P["age"][:] = 10 ** 4
    cd = torch.from_numpy(codes(P["rule"])).float()
    P["state"][:, T.N_FIXED:T.N_FIXED + CODE_BITS] = cd[:, :, None, None]
    W, mean = T.probe_fit({2: P}, codes)
    ev = T.wide_set("m1a", tab, bd, 2, 30, 8, 2000)
    x = np.zeros((0, 80 - T.N_FIXED), np.float32)
    ys, ds = [], []
    xs = []
    for i in range(len(ev.geo)):
        on = (ev.a[i] < T.INF).any(0)
        on[ev.tap[i, 0], ev.tap[i, 1]] = False
        n = int(on.sum())
        v = np.zeros((n, 80 - T.N_FIXED), np.float32)
        v[:, :CODE_BITS] = codes(ev.rule[i:i + 1])[0]
        xs.append(v)
        ys.append(np.repeat(codes(ev.rule[i:i + 1]), n, 0))
        ds.append(np.full(n, 3))
    cells = {"x": np.concatenate([x] + xs), "y": np.concatenate(ys), "dist": np.concatenate(ds)}
    sc = T.probe_score(W, mean, cells)
    rnd = dict(cells, x=np.random.default_rng(0).normal(size=cells["x"].shape).astype(np.float32))
    sc0 = T.probe_score(W, mean, rnd)
    check(f"code probe: an injected code reads back exactly (exact {sc['exact']}, n {sc['n']}); a random state "
          f"doesn't (exact {sc0['exact']}, digits {sc0['digits']})", sc["exact"] == 1.0 and sc0["exact"] < 0.05)


TINY = ["--levels", "2", "--eval-levels", "2", "--hidden", "8", "--channels", "16", "--batch", "4",
        "--pool-size", "8", "--eval-n", "6", "--eval-mult", "2", "--eval-cap", "30", "--steps-mult", "0.5", "1",
        "--bptt", "4", "--last-k", "2", "--threads", "1", "--snap-every", "0"]


def log_of(name):
    return [json.loads(x) for x in open(f"runs/{name}/log.jsonl")]


def runs():
    for inputs, extra in (("a", []), ("c", []), ("c-bc", []), ("d", []), ("d-cs", ["--depth", "2"])):
        name = f"arm-{inputs}"
        T.main(["--name", name, "--inputs", inputs, *TINY, *extra, "--iters", "3", "--eval-every", "50"])
        lg = log_of(name)
        st = lg[0]
        ok = st["config"]["nIn"] == T.n_inputs(inputs) and lg[-1].get("stopped") == "done" and \
            all(np.isfinite(x["loss"]) for x in lg if "loss" in x) and "code" in lg[1]["q"]
        check(f"run --inputs {inputs}{' --depth 2' if extra else ''}: {st['config']['nIn']} consts, "
              f"{st['params']} params, check + 3 iterations, finite loss", ok)

    T.main(["--name", "rb", "--inputs", "c", *TINY, "--iters", "50", "--eval-every", "50", "--force-collapse", "1"])
    rb = [x for x in log_of("rb") if "rollback" in x]
    best = torch.load("runs/rb/best.pt", weights_only=False)
    ck = torch.load("runs/rb/ckpt.pt", weights_only=False)
    same = all(torch.equal(best["model"][k], ck["model"][k]) for k in best["model"])
    check(f"forced collapse: a rollback (lrScale {rb[0]['lrScale'] if rb else '?'}), ckpt.pt's weights = best.pt's",
          len(rb) == 1 and rb[0]["lrScale"] == 0.5 and rb[0]["restored"] == 0 and same and ck["rollbacks"] == 1)

    flat = ["--inputs", "c", "--lr", "1e-3", "--lr-floor", "1e-3", "--warmup", "0", "--eval-every", "50"]
    T.main(["--name", "straight", *TINY, *flat, "--iters", "6"])
    T.main(["--name", "chunked", *TINY, *flat, "--iters", "3"])
    T.main(["--name", "chunked", "--resume", "--iters", "6", "--threads", "1", "--snap-every", "0"])
    a = torch.load("runs/straight/ckpt.pt", weights_only=False)
    b = torch.load("runs/chunked/ckpt.pt", weights_only=False)
    w = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    o = all(torch.equal(x, y) for x, y in zip(a["opt"]["state"][0].values(), b["opt"]["state"][0].values()))
    pa, pb = a["pool"][2], b["pool"][2]
    p = torch.equal(pa["state"], pb["state"]) and all(np.array_equal(pa[k], pb[k]) for k in ("geo", "rule", "a", "b", "age"))
    check(f"--resume: 3 + 3 iterations = 6 in one go (weights {w}, optimiser {o}, pool {p})", w and o and p)
    T.main(["--name", "half", *TINY, *flat, "--iters", "3", "--ckpt-pool", "half"])
    h = torch.load("runs/half/ckpt.pt", weights_only=False)
    T.main(["--name", "half", "--resume", "--iters", "6", "--threads", "1", "--snap-every", "0"])
    h2 = torch.load("runs/half/ckpt.pt", weights_only=False)
    check(f"--ckpt-pool half: pool states stored as {h['pool'][2]['state'].dtype}, the run resumes to iteration "
          f"{h2['iteration']}", h["pool"][2]["state"].dtype == torch.float16 and h2["iteration"] == 6)

    T.main(["--name", "m1b", "--task", "m1b", "--inputs", "c", *TINY[:-1], "1", "--iters", "2", "--eval-every", "50",
            "--init", "runs/straight/best.pt"])
    lg = log_of("m1b")
    q = [x for x in lg if "q" in x]
    check(f"m1b --init from m1a runs: q has parts/bySet/code ({sorted(q[-1]['q']) if q else None})",
          lg[-1].get("stopped") == "done" and q and {"parts", "bySet", "code", "bySubset"} <= set(q[-1]["q"]))
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
    tab, bd = RuleTable(), Boards()
    rules_checks(tab, bd)
    input_checks(tab, bd)
    target_checks(tab, bd)
    legacy_checks(tab)
    oracle_checks(tab, bd)
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
