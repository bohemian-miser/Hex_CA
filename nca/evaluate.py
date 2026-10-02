"""Held-out metrics for a trained checkpoint.

    python -m nca.evaluate runs/p1/ckpt.pt [--n 500] [--R2 9] [--edit]

Runs --n fresh boards (a seed no training run uses) at the trained R and at a
larger R2, for T1 and 4*T1 steps, and prints per-cell accuracy, fill IoU,
exact-board rate and the trivial baseline (predict nothing filled). Then the
same at the trained R on "page loops" (data.page_loops: the demo page's Random
loop button), which training never draws from -- an out-of-distribution check.
For a --pool checkpoint (or with --edit) it also runs the edit test, on the
training mix and on page loops: settle a board for T1 steps, apply edit_walls,
run T1 more steps WITHOUT resetting, and score against the new target. Ends
with one JSON line of everything.
"""

import argparse
import json

import numpy as np
import torch

from .data import edit_walls, oracle, page_loops, random_walls
from .hexgrid import mask as hex_mask
from .model import HexNCA, fresh_state
from .train import to_t

EVAL_SEED = 31337
PAGE_SEED = 27182  # page loops get a stream of their own
CHUNK = 100  # boards per forward pass
SETS = {"mix": (random_walls, EVAL_SEED), "page loops": (page_loops, PAGE_SEED)}


def load(path):
    ck = torch.load(path, weights_only=False)
    cfg = ck["config"]
    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck["iteration"]


def score(pred, target, mk):
    """Counts for a chunk: (exact boards, correct cells, on-board cells, intersection, union, empty-target boards)."""
    ok = (pred == target) | (mk == 0)
    on = (mk > 0).expand_as(pred)
    return np.array([
        ok.flatten(1).all(1).sum().item(),
        (ok & on).sum().item(),
        on.sum().item(),
        (pred & target).sum().item(),
        (pred | target).sum().item(),
        (target.flatten(1).sum(1) == 0).sum().item(),
    ], dtype=np.float64)


def summarise(c, n):
    return {"acc": c[1] / c[2], "iou": c[3] / c[4] if c[4] else 1.0,
            "exact": c[0] / n, "baseline": c[5] / n}


def boards(rng, R, m, gen):
    """(walls [m,S,S], fill [m,S,S]) of m boards from gen(rng, R)."""
    w = np.stack([gen(rng, R) for _ in range(m)])
    return w, np.stack([oracle(x, R)[0] for x in w])


@torch.no_grad()
def fresh_run(model, R, n, steps_list, board_set="mix"):
    """{steps: metrics} for n fresh boards, one rollout read out at every step count in steps_list."""
    gen, seed = SETS[board_set]
    rng = np.random.default_rng(seed + R)
    mk = to_t(hex_mask(R)[None])
    counts = {s: np.zeros(6) for s in steps_list}
    for i in range(0, n, CHUNK):
        w, f = boards(rng, R, min(CHUNK, n - i), gen)
        walls, target = to_t(w), to_t(f) > 0.5
        state, done = fresh_state(walls, model.channels), 0
        for s in sorted(steps_list):
            state = model(state, walls, mk, s - done)
            done = s
            counts[s] += score(state[:, 1:2] > 0.5, target, mk)
    return {s: summarise(counts[s], n) for s in steps_list}


@torch.no_grad()
def edit_run(model, R, n, steps, board_set="mix"):
    """Settle for `steps`, edit the walls, run `steps` more without a reset; score against the new target."""
    gen, seed = SETS[board_set]
    rng = np.random.default_rng(seed + 1000 + R)
    mk = to_t(hex_mask(R)[None])
    c = np.zeros(6)
    changed = 0
    for i in range(0, n, CHUNK):
        m = min(CHUNK, n - i)
        w, f = boards(rng, R, m, gen)
        walls = to_t(w)
        state = model(fresh_state(walls, model.channels), walls, mk, steps)
        w2 = np.stack([edit_walls(rng, w[j], R) for j in range(m)])
        f2 = np.stack([oracle(w2[j], R)[0] for j in range(m)])
        changed += int(sum((f2[j] != f[j]).any() for j in range(m)))
        walls2 = to_t(w2)
        state = state.clone()
        state[:, 0:1] = walls2
        state = model(state, walls2, mk, steps)
        c += score(state[:, 1:2] > 0.5, to_t(f2) > 0.5, mk)
    out = summarise(c, n)
    out["fillChanged"] = changed / n
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--R2", type=int, default=None, help="larger radius for generalisation (default trained R + 3)")
    p.add_argument("--edit", action="store_true", help="run the edit test even for a phase 1 checkpoint")
    args = p.parse_args()
    torch.set_num_threads(3)

    model, cfg, iters = load(args.ckpt)
    R, T1 = cfg["R"], cfg["steps"][1]
    R2 = args.R2 or R + 3
    rows = []
    for board_set, r in (("mix", R), ("mix", R2), ("page loops", R)):
        if board_set == "page loops" and R < 4:
            continue  # the page draws no loops below R=4
        for s, m in fresh_run(model, r, args.n, [T1, 4 * T1], board_set).items():
            rows.append({"set": board_set, "R": r, "steps": s, **m})
    result = {"ckpt": args.ckpt, "iteration": iters, "n": args.n, "rows": rows}
    if cfg["pool"] or args.edit:
        result["edit"] = {"R": R, "steps": T1, **edit_run(model, R, args.n, T1)}
        if R >= 4:
            result["editPageLoops"] = {"R": R, "steps": T1, **edit_run(model, R, args.n, T1, "page loops")}

    print(f"{'boards':<11} {'R':>3} {'steps':>6} {'acc':>8} {'IoU':>7} {'exact':>7} {'trivial':>8}")
    for row in rows:
        print(f"{row['set']:<11} {row['R']:>3} {row['steps']:>6} {row['acc']:8.4f} {row['iou']:7.3f} "
              f"{row['exact']:7.3f} {row['baseline']:8.3f}")
    for key, board_set in (("edit", "mix"), ("editPageLoops", "page loops")):
        if key not in result:
            continue
        e = result[key]
        print(f"edit test, {board_set} (R={R}, {T1}+{T1} steps, no reset): exact {e['exact']:.3f}, acc {e['acc']:.4f}, "
              f"IoU {e['iou']:.3f}, trivial {e['baseline']:.3f}; fill changed on {e['fillChanged']:.2f} of boards")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
