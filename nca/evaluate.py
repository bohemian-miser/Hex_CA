"""Held-out metrics for a trained checkpoint. A board is exact if its fill equals ANY acceptable target.

    python -m nca.evaluate runs/p1/ckpt.pt [--n 200] [--n-big 100] [--radii 6 8 10 12 16 24] [--mults 8 16 32]
                           [--sets ...] [--edit]

At every radius R in --radii (default 6 8 10 12 16 24; --n boards per set at R <= 16, default 200, --n-big
above, default 100) it reads one rollout out at mult*R steps for each mult in --mults (default 8, 16 and 32:
the later ones show whether the answer holds or still moves), on these sets of fresh boards (seeds no
training run uses):

  mix            the training mix
  bridge         "bridge" boards with 2+ rim regions only
  page loops     the demo page's Random loop button (data.page_loops), which training never draws from
  ragged mix     the training mix on ragged masks (nca/masks.py: blobs, crops of Spectacle's hex fields)
  ragged bridge  bridge boards with 2+ rim regions on ragged masks
(the ragged sets need a model whose only const input is the mask).

Every row: per-cell accuracy and fill IoU (against the closest acceptable target), the exact-board rate with
+-2 standard errors (binomial, sqrt(p(1-p)/n)), and the trivial baseline (predict nothing filled). Rows of
boards with 2+ rim regions (the bridge sets; the others where they have any) add exact per area-ratio bin
(second-largest / largest rim region: [0,.25) [.25,.5) [.5,.75) [.75,1]) with +-2se and n, and the share of
the set's boards where every side filled ("both") or two or more stayed empty ("none") -- a side counts as
filled when most of its cells are. SETTLE TIME, per set and radius over the whole rollout (max(mults)*R steps): a
board's settle step is the first step after which its thresholded fill (ch1 > 0.5) never changes again (0: it
never changed from the fresh state); reported as the median, 90th percentile and max, in steps and in units
of R, and "late": the share of boards whose fill still changed in the last LATE (10%) of the rollout. For a
--pool checkpoint (or with --edit) the edit test runs on the hexagon sets at the radii up to EDIT_MAX_R (12):
settle a board for mults[0]*R steps, apply edit_walls, run as many more WITHOUT resetting, and score against
the new targets. Ends with one JSON line of everything.
"""

import argparse
import json

import numpy as np
import torch

from .data import _board, _labels, _rim_regions, edit_walls, pad_targets, page_loops, random_walls, targets
from .hexgrid import mask as hex_mask
from .masks import ragged_mask
from .model import HexNCA, const_stack, fresh_state

EVAL_SEED = 31337
PAGE_SEED = 27182    # page loops get a stream of their own
BRIDGE_SEED = 16180  # and so do bridge boards
RAGGED_SEED = 14142  # and the ragged sets
CHUNK = 100          # boards per forward pass
BIG = 16             # radii above this use --n-big boards
EDIT_MAX_R = 12      # the edit test runs at radii up to this
LATE = 0.1           # settle time: "late" = the fill still changed in the last LATE of the rollout
BINS = (0, 0.25, 0.5, 0.75, 1.0)  # area ratio bins; the last one includes 1 (exact ties)


def _bridge_board(rng, R, board=None):
    """A "bridge" board with 2+ rim regions (drawn again until it has)."""
    on = hex_mask(R) == 1 if board is None else board == 1
    while True:
        w = random_walls(rng, R, kind="bridge", board=board)
        if len(_rim_regions(_labels(on & (w == 0), R), R, board)[0]) >= 2:
            return w


def _ragged_mix(rng, R):
    """(walls, mask): the training mix on a ragged mask."""
    m = ragged_mask(rng, R)
    return random_walls(rng, R, board=m), m


def _ragged_bridge(rng, R):
    """(walls, mask): a bridge board with 2+ rim regions on a ragged mask (a new mask each try)."""
    while True:
        m = ragged_mask(rng, R)
        w = random_walls(rng, R, kind="bridge", board=m)
        if len(_rim_regions(_labels((m == 1) & (w == 0), R), R, m)[0]) >= 2:
            return w, m


def _ragged_any(rng, R):
    """(walls, mask): the quick check's ragged set, _ragged_bridge or _ragged_mix with probability 1/2 each."""
    return _ragged_bridge(rng, R) if rng.random() < 0.5 else _ragged_mix(rng, R)


SETS = {"mix": (random_walls, EVAL_SEED), "bridge": (_bridge_board, BRIDGE_SEED), "page loops": (page_loops, PAGE_SEED),
        "ragged mix": (_ragged_mix, RAGGED_SEED), "ragged bridge": (_ragged_bridge, RAGGED_SEED + 1)}
RAGGED = ("ragged mix", "ragged bridge")


def to_t(a):
    """numpy [B,S,S] -> float tensor [B,1,S,S]."""
    return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(1)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)  # a checkpoint from a GPU loads here too
    cfg = ck["config"]
    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"], cfg.get("nConsts", 1),
                   cfg.get("perception", "taps"))  # checkpoints from before v4 have no pool
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["iteration"]


def sides_of(w, R, board=None):
    """(region labels [S,S], rim region numbers, area ratio or nan) of one board, for the bridge metrics."""
    on, ragged = _board(R, board)
    board = board if ragged else None
    lab = _labels(on & (w == 0), R)
    ids, area, _ = _rim_regions(lab, R, board)
    return lab, ids, area[1] / area[0] if len(ids) >= 2 else np.nan


def boards(rng, R, m, gen):
    """(walls [m,S,S], fills [m,K,S,S] padded, sides, masks uint8 [m,S,S] or None) of m boards from
    gen(rng, R), which returns walls (on the hexagon: masks None) or (walls, mask) (a ragged set).

    sides[i] = sides_of(board i) for the bridge metrics.
    """
    got = [gen(rng, R) for _ in range(m)]
    if isinstance(got[0], tuple):
        w, ms = np.stack([g[0] for g in got]), np.stack([g[1] for g in got]).astype(np.uint8)
    else:
        w, ms = np.stack(got), None
    bs = [None] * m if ms is None else list(ms)
    sides = [sides_of(x, R, b) for x, b in zip(w, bs)]
    return w, pad_targets([targets(x, R, b)[0] for x, b in zip(w, bs)]), sides, ms


def score(pred, fills, mk, sides):
    """Counts for a chunk. pred [B,1,S,S] bool, fills [B,K,S,S] bool, mk [1,1,S,S].

    [exact boards, correct cells, on-board cells, intersection, union, boards where nothing filled is
    exact, boards where every side filled, boards with 2+ sides empty] + [exact, boards] per ratio bin.
    Cells, IoU and the trivial baseline are against each board's closest target.
    """
    on = mk > 0
    wrong = ((pred != fills) & on).flatten(2).sum(2)  # [B,K]
    best = wrong.argmin(1)
    t = fills[torch.arange(len(best), device=best.device), best].unsqueeze(1)  # [B,1,S,S], the closest target
    exact = wrong.min(1).values == 0
    n_on = on.expand(len(best), -1, -1, -1).sum()  # on-board cells of the chunk's boards (mk may be per board)
    c = np.zeros(8 + 2 * (len(BINS) - 1))
    c[0] = exact.sum().item()
    c[1] = (n_on - wrong.min(1).values.sum()).item()
    c[2] = n_on.item()
    c[3] = (pred & t).sum().item()
    c[4] = (pred | t).sum().item()
    c[5] = (fills.flatten(2).sum(2) == 0).any(1).sum().item()
    p = pred[:, 0].cpu().numpy()
    for b, (lab, ids, ratio) in enumerate(sides):
        if len(ids) < 2:
            continue
        empty = sum(p[b][lab == x].mean() <= 0.5 for x in ids)  # sides left (mostly) empty
        c[6] += empty == 0
        c[7] += empty >= 2
        k = min(int(np.searchsorted(BINS, ratio, side="right")) - 1, len(BINS) - 2)
        c[8 + 2 * k] += bool(exact[b])
        c[9 + 2 * k] += 1
    return c


def se(p, n):
    """Binomial standard error of a share p over n boards (None without boards)."""
    return None if not n or p is None else float(np.sqrt(p * (1 - p) / n))


def summarise(c, n, bridge=False):
    """Shares from score()'s counts over n boards, the exact share's standard error (exactSe); with bridge
    also both / none (shares of every board: the quick check's convention; on a bridge set every board has 2+
    rim regions) and exact per area-ratio bin (boards with 2+ rim regions) with standard errors (binSe)."""
    out = {"acc": c[1] / c[2], "iou": c[3] / c[4] if c[4] else 1.0, "exact": c[0] / n, "baseline": c[5] / n,
           "exactSe": se(c[0] / n, n)}
    if bridge:
        out.update({"both": c[6] / n, "none": c[7] / n, "bothOrNone": (c[6] + c[7]) / n,
                    "binExact": [c[8 + 2 * k] / c[9 + 2 * k] if c[9 + 2 * k] else None for k in range(len(BINS) - 1)],
                    "binN": [int(c[9 + 2 * k]) for k in range(len(BINS) - 1)]})
        out["binSe"] = [se(p, k) for p, k in zip(out["binExact"], out["binN"])]
    return out


@torch.no_grad()
def fresh_run(model, R, n, steps_list, board_set="mix"):
    """({steps: metrics}, settle) for n fresh boards, one rollout read out at every step count in steps_list.
    settle: the settle time (module docstring) over the rollout's max(steps_list) steps."""
    gen, seed = SETS[board_set]
    rng = np.random.default_rng(seed + R)
    cs0 = const_stack(R, model.n_consts)
    counts = {s: 0 for s in steps_list}
    horizon = max(steps_list)
    settle = []
    for i in range(0, n, CHUNK):
        w, f, sides, ms = boards(rng, R, min(CHUNK, n - i), gen)
        walls, fills = to_t(w), torch.from_numpy(f) > 0
        cs = cs0 if ms is None else to_t(ms)
        mk = cs[:, :1]
        state = fresh_state(walls, model.channels)
        prev = state[:, 1:2] > 0.5
        last = torch.zeros(len(w), dtype=torch.long)  # the last step at which the thresholded fill changed
        for t in range(1, horizon + 1):
            state = model.step(state, walls, cs)
            pred = state[:, 1:2] > 0.5
            last[((pred != prev) & (mk > 0)).flatten(1).any(1)] = t
            prev = pred
            if t in counts:
                counts[t] = counts[t] + score(pred, fills, mk, sides)
        settle += last.tolist()
    st = np.array(settle, dtype=float)
    settle_rec = {"steps": horizon, "median": float(np.median(st)), "p90": float(np.percentile(st, 90)),
                  "max": int(st.max()), "medianR": round(float(np.median(st)) / R, 2),
                  "p90R": round(float(np.percentile(st, 90)) / R, 2), "maxR": round(float(st.max()) / R, 2),
                  "late": float(np.mean(st > (1 - LATE) * horizon))}
    return {s: summarise(counts[s], n, True) for s in steps_list}, settle_rec


@torch.no_grad()
def edit_run(model, R, n, steps, board_set="mix"):
    """Settle for `steps`, edit the walls, run `steps` more without a reset; score against the new targets."""
    gen, seed = SETS[board_set]
    rng = np.random.default_rng(seed + 1000 + R)
    cs = const_stack(R, model.n_consts)
    c, changed = 0, 0
    for i in range(0, n, CHUNK):
        m = min(CHUNK, n - i)
        w, f, _, _ = boards(rng, R, m, gen)
        walls = to_t(w)
        state = model(fresh_state(walls, model.channels), walls, cs, steps)
        w2 = np.stack([edit_walls(rng, w[j], R) for j in range(m)])
        t2 = [targets(x, R)[0] for x in w2]
        changed += sum(bool((t2[j][0] != f[j, 0]).any()) for j in range(m))  # the primary changed
        sides = [sides_of(x, R) for x in w2]
        walls2 = to_t(w2)
        state = state.clone()
        state[:, 0:1] = walls2
        state = model(state, walls2, cs, steps)
        c = c + score(state[:, 1:2] > 0.5, torch.from_numpy(pad_targets(t2)) > 0, cs[:, :1], sides)
    out = summarise(c, n)
    out["fillChanged"] = changed / n
    return out


def _pm(p, s):
    """'0.833+-0.052' (2 standard errors), or '  -  ' without boards."""
    return "  -  " if p is None else f"{p:.3f}+-{2 * s:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=200, help=f"boards per set at radii up to {BIG}")
    p.add_argument("--n-big", type=int, default=100, help=f"boards per set at radii above {BIG}")
    p.add_argument("--radii", type=int, nargs="+", default=[6, 8, 10, 12, 16, 24])
    p.add_argument("--mults", type=int, nargs="+", default=[8, 16, 32], help="read out at mult*R steps")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--edit", action="store_true", help="run the edit test even for a phase 1 checkpoint")
    p.add_argument("--threads", type=int, default=3, help="torch threads (at most 3 on the Pi)")
    args = p.parse_args()
    torch.set_num_threads(min(3, args.threads))

    model, cfg, iters = load(args.ckpt)
    sets = [s for s in args.sets if s not in RAGGED or model.n_consts == 1]  # ragged boards: the mask alone
    if len(sets) < len(args.sets):
        print(f"(skipped {sorted(set(args.sets) - set(sets))}: the model takes {model.n_consts} consts, not the mask alone)")
    rows, edits, settles = [], [], []
    for R in args.radii:
        n = args.n if R <= BIG else args.n_big
        for board_set in sets:
            if board_set == "page loops" and R < 4:
                continue  # the page draws no loops below R=4
            res, settle = fresh_run(model, R, n, [k * R for k in args.mults], board_set)
            for s, m in res.items():
                rows.append({"set": board_set, "R": R, "steps": s, "n": n, **m})
            settles.append({"set": board_set, "R": R, "n": n, **settle})
        if (cfg.get("pool") or args.edit) and R <= EDIT_MAX_R:
            for board_set in sets:
                if board_set == "page loops" and R < 4 or board_set in RAGGED:
                    continue
                s = args.mults[0] * R
                edits.append({"set": board_set, "R": R, "steps": s, "n": n, **edit_run(model, R, n, s, board_set)})

    print(f"{'boards':<13} {'R':>3} {'steps':>5} {'n':>4} {'acc':>7} {'IoU':>6} {'exact +-2se':>12} {'trivial':>7}"
          f"   2+ rim regions: exact +-2se per area-ratio bin [0,.25) [.25,.5) [.5,.75) [.75,1] (n), both, none")
    for row in rows:
        extra = ""
        if "binExact" in row and sum(row["binN"]):
            bins = " ".join(_pm(x, e) for x, e in zip(row["binExact"], row["binSe"]))
            extra = f"   {bins} (n {'/'.join(map(str, row['binN']))}), both {row['both']:.3f}, none {row['none']:.3f}"
        print(f"{row['set']:<13} {row['R']:>3} {row['steps']:>5} {row['n']:>4} {row['acc']:7.4f} {row['iou']:6.3f} "
              f"{_pm(row['exact'], row['exactSe']):>12} {row['baseline']:7.3f}{extra}")
    print(f"\nsettle time (first step after which the thresholded fill no longer changes), over the whole rollout:")
    print(f"{'boards':<13} {'R':>3} {'steps':>5} {'median':>7} {'p90':>6} {'max':>5}   {'in R: median':>12} {'p90':>5} {'max':>5}   late")
    for x in settles:
        print(f"{x['set']:<13} {x['R']:>3} {x['steps']:>5} {x['median']:7.1f} {x['p90']:6.1f} {x['max']:5d}   "
              f"{x['medianR']:12.2f} {x['p90R']:5.2f} {x['maxR']:5.2f}   {x['late']:.3f}")
    for e in edits:
        print(f"edit test, {e['set']} (R={e['R']}, {e['steps']}+{e['steps']} steps, no reset): exact "
              f"{_pm(e['exact'], e['exactSe'])}, acc {e['acc']:.4f}, IoU {e['iou']:.3f}, trivial {e['baseline']:.3f}; "
              f"primary target changed on {e['fillChanged']:.2f} of boards")
    print(json.dumps({"ckpt": args.ckpt, "iteration": iters, "n": args.n, "nBig": args.n_big,
                      "mults": args.mults, "rows": rows, "settle": settles, "edits": edits}))


if __name__ == "__main__":
    main()
