"""Held-out metrics for a trained checkpoint. A board is exact if its fill equals ANY acceptable target.

    python -m nca.evaluate runs/p1/ckpt.pt [--n 500] [--n-big 50] [--radii 6 8 10 16 24] [--mults 8 16] [--edit]

At every radius R in --radii (default: the trained radii 6, 8, 10 and the
larger 16 and 24; --n boards each, --n-big for R > 12) it reads one rollout
out at mult*R steps for each mult in --mults (default 8 and 16, the second a
stability check), on three sets of fresh boards (seeds no training run uses):

  mix         the training mix
  bridge      "bridge" boards with 2+ rim regions only
  page loops  the demo page's Random loop button (data.page_loops), which
              training never draws from (an out-of-distribution check)

Every row: per-cell accuracy and fill IoU (against the closest acceptable
target), the exact-board rate, and the trivial baseline (predict nothing
filled). Bridge rows add exact per area-ratio bin (second-largest / largest
rim region: [0,.25) [.25,.5) [.5,.75) [.75,1]) and the share of boards where
every side filled ("both") or two or more stayed empty ("none") -- a side
counts as filled when most of its cells are. For a --pool checkpoint (or with
--edit) the edit test runs on each set at the radii up to 12: settle a board
for mults[0]*R steps, apply edit_walls, run as many more WITHOUT resetting,
and score against the new targets. Ends with one JSON line of everything.
"""

import argparse
import json

import numpy as np
import torch

from .data import _labels, _rim_regions, edit_walls, pad_targets, page_loops, random_walls, targets
from .hexgrid import mask as hex_mask
from .model import HexNCA, const_stack, fresh_state

EVAL_SEED = 31337
PAGE_SEED = 27182    # page loops get a stream of their own
BRIDGE_SEED = 16180  # and so do bridge boards
CHUNK = 100          # boards per forward pass
BIG = 12             # radii above this use --n-big boards and get no edit test
BINS = (0, 0.25, 0.5, 0.75, 1.0)  # area ratio bins; the last one includes 1 (exact ties)


def _bridge_board(rng, R):
    """A "bridge" board with 2+ rim regions (drawn again until it has)."""
    on = hex_mask(R) == 1
    while True:
        w = random_walls(rng, R, kind="bridge")
        if len(_rim_regions(_labels(on & (w == 0), R), R)[0]) >= 2:
            return w


SETS = {"mix": (random_walls, EVAL_SEED), "bridge": (_bridge_board, BRIDGE_SEED), "page loops": (page_loops, PAGE_SEED)}


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


def boards(rng, R, m, gen):
    """(walls [m,S,S], fills [m,K,S,S] padded, sides) of m boards from gen(rng, R).

    sides[i] = (region labels [S,S], rim region numbers, area ratio or nan) for the bridge metrics.
    """
    on = hex_mask(R) == 1
    w = np.stack([gen(rng, R) for _ in range(m)])
    sides = []
    for x in w:
        lab = _labels(on & (x == 0), R)
        ids, area, _ = _rim_regions(lab, R)
        sides.append((lab, ids, area[1] / area[0] if len(ids) >= 2 else np.nan))
    return w, pad_targets([targets(x, R)[0] for x in w]), sides


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
    c = np.zeros(8 + 2 * (len(BINS) - 1))
    c[0] = exact.sum().item()
    c[1] = (on.sum() * len(best) - wrong.min(1).values.sum()).item()
    c[2] = (on.sum() * len(best)).item()
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


def summarise(c, n, bridge=False):
    out = {"acc": c[1] / c[2], "iou": c[3] / c[4] if c[4] else 1.0, "exact": c[0] / n, "baseline": c[5] / n}
    if bridge:  # every board has 2+ rim regions
        out.update({"both": c[6] / n, "none": c[7] / n, "bothOrNone": (c[6] + c[7]) / n,
                    "binExact": [c[8 + 2 * k] / c[9 + 2 * k] if c[9 + 2 * k] else None for k in range(len(BINS) - 1)],
                    "binN": [int(c[9 + 2 * k]) for k in range(len(BINS) - 1)]})
    return out


@torch.no_grad()
def fresh_run(model, R, n, steps_list, board_set="mix"):
    """{steps: metrics} for n fresh boards, one rollout read out at every step count in steps_list."""
    gen, seed = SETS[board_set]
    rng = np.random.default_rng(seed + R)
    cs = const_stack(R, model.n_consts)
    mk = cs[:, :1]
    counts = {s: 0 for s in steps_list}
    for i in range(0, n, CHUNK):
        w, f, sides = boards(rng, R, min(CHUNK, n - i), gen)
        walls, fills = to_t(w), torch.from_numpy(f) > 0
        state, done = fresh_state(walls, model.channels), 0
        for s in sorted(steps_list):
            state = model(state, walls, cs, s - done)
            done = s
            counts[s] = counts[s] + score(state[:, 1:2] > 0.5, fills, mk, sides)
    return {s: summarise(counts[s], n, board_set == "bridge") for s in steps_list}


@torch.no_grad()
def edit_run(model, R, n, steps, board_set="mix"):
    """Settle for `steps`, edit the walls, run `steps` more without a reset; score against the new targets."""
    gen, seed = SETS[board_set]
    rng = np.random.default_rng(seed + 1000 + R)
    on = hex_mask(R) == 1
    cs = const_stack(R, model.n_consts)
    c, changed = 0, 0
    for i in range(0, n, CHUNK):
        m = min(CHUNK, n - i)
        w, f, _ = boards(rng, R, m, gen)
        walls = to_t(w)
        state = model(fresh_state(walls, model.channels), walls, cs, steps)
        w2 = np.stack([edit_walls(rng, w[j], R) for j in range(m)])
        t2 = [targets(x, R)[0] for x in w2]
        changed += sum(bool((t2[j][0] != f[j, 0]).any()) for j in range(m))  # the primary changed
        sides = []
        for x in w2:
            lab = _labels(on & (x == 0), R)
            ids, area, _ = _rim_regions(lab, R)
            sides.append((lab, ids, area[1] / area[0] if len(ids) >= 2 else np.nan))
        walls2 = to_t(w2)
        state = state.clone()
        state[:, 0:1] = walls2
        state = model(state, walls2, cs, steps)
        c = c + score(state[:, 1:2] > 0.5, torch.from_numpy(pad_targets(t2)) > 0, cs[:, :1], sides)
    out = summarise(c, n)
    out["fillChanged"] = changed / n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=500, help="boards per set at radii up to 12")
    p.add_argument("--n-big", type=int, default=50, help="boards per set at radii above 12")
    p.add_argument("--radii", type=int, nargs="+", default=[6, 8, 10, 16, 24])
    p.add_argument("--mults", type=int, nargs="+", default=[8, 16], help="read out at mult*R steps")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--edit", action="store_true", help="run the edit test even for a phase 1 checkpoint")
    p.add_argument("--threads", type=int, default=3, help="torch threads (at most 3 on the Pi)")
    args = p.parse_args()
    torch.set_num_threads(min(3, args.threads))

    model, cfg, iters = load(args.ckpt)
    rows, edits = [], []
    for R in args.radii:
        n = args.n if R <= BIG else args.n_big
        for board_set in args.sets:
            if board_set == "page loops" and R < 4:
                continue  # the page draws no loops below R=4
            for s, m in fresh_run(model, R, n, [k * R for k in args.mults], board_set).items():
                rows.append({"set": board_set, "R": R, "steps": s, "n": n, **m})
        if (cfg.get("pool") or args.edit) and R <= BIG:
            for board_set in args.sets:
                if board_set == "page loops" and R < 4:
                    continue
                s = args.mults[0] * R
                edits.append({"set": board_set, "R": R, "steps": s, "n": n, **edit_run(model, R, n, s, board_set)})

    print(f"{'boards':<11} {'R':>3} {'steps':>5} {'n':>4} {'acc':>7} {'IoU':>6} {'exact':>6} {'trivial':>7}"
          f"   bridge: exact per ratio bin [0,.25) [.25,.5) [.5,.75) [.75,1], both, none")
    for row in rows:
        extra = ""
        if row["set"] == "bridge":
            bins = " ".join("  -  " if x is None else f"{x:.3f}" for x in row["binExact"])
            extra = f"   {bins} (n {'/'.join(map(str, row['binN']))}), both {row['both']:.3f}, none {row['none']:.3f}"
        print(f"{row['set']:<11} {row['R']:>3} {row['steps']:>5} {row['n']:>4} {row['acc']:7.4f} {row['iou']:6.3f} "
              f"{row['exact']:6.3f} {row['baseline']:7.3f}{extra}")
    for e in edits:
        print(f"edit test, {e['set']} (R={e['R']}, {e['steps']}+{e['steps']} steps, no reset): exact {e['exact']:.3f}, "
              f"acc {e['acc']:.4f}, IoU {e['iou']:.3f}, trivial {e['baseline']:.3f}; "
              f"primary target changed on {e['fillChanged']:.2f} of boards")
    print(json.dumps({"ckpt": args.ckpt, "iteration": iters, "n": args.n, "nBig": args.n_big,
                      "mults": args.mults, "rows": rows, "edits": edits}))


if __name__ == "__main__":
    main()
