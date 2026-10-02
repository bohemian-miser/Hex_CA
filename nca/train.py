"""Train the hex flood-fill NCA.

Phase 1 (fresh start every iteration):
    python -m nca.train --name p1 --R 6 --iters 4000
Phase 2 (persistent pool with wall edits), started from phase 1:
    python -m nca.train --name p2 --pool --init runs/p1/ckpt.pt --R 6 --iters 6000 --lr 1e-3
Carry on an interrupted run (same --name; --iters may be raised):
    python -m nca.train --name p1 --resume --iters 6000

Logs one JSON line per 50 iterations to runs/<name>/log.jsonl, checkpoints to
runs/<name>/ckpt.pt every 200 iterations and at the end (atomically).
"""

import argparse
import json
import os
import resource
import time

import numpy as np
import torch

from .data import batch, edit_walls, oracle, random_walls
from .hexgrid import mask as hex_mask, side
from .model import HexNCA, fresh_state

MARGIN = 4      # extra steps past the deepest outside cell, so the answer is always reachable
LAST_K = 4      # loss = mean over the last LAST_K steps (asks for a settled answer, not a lucky frame)
EVAL_N = 64     # fixed held-out batch for the quick log metric
EVAL_SEED = 999 # same held-out boards whatever --seed is


def default_steps(R: int):
    """[T0, T1] from the oracle depth at radius R: T0 = 99.9th percentile depth + MARGIN, T1 = 2*T0."""
    rng = np.random.default_rng(4242)
    depths = [int(oracle(random_walls(rng, R), R)[1].max()) for _ in range(2000)]
    t0 = int(np.percentile(depths, 99.9)) + MARGIN
    return t0, 2 * t0


def to_t(a):
    """numpy [B,S,S] -> float tensor [B,1,S,S]."""
    return torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(1)


def board_metrics(state, target, mk):
    """(exact-board rate, per-cell accuracy) of fill = state[:,1] > 0.5 against target, on-board cells."""
    pred = state[:, 1:2] > 0.5
    ok = (pred == (target > 0.5)) | (mk == 0)
    exact = ok.flatten(1).all(1).float().mean().item()
    acc = (ok.float() * mk).sum().item() / (mk.sum().item() * state.shape[0])
    return exact, acc


@torch.no_grad()
def quick_eval(model, heldout, mk, steps):
    walls, target = heldout
    state = model(fresh_state(walls, model.channels), walls, mk, steps)
    return board_metrics(state, target, mk)


def lr_at(it: int, iters: int, lr: float) -> float:
    """Step decay: full lr to 60%, x0.3 to 85%, x0.1 after."""
    f = it / max(1, iters)
    return lr * (1.0 if f < 0.6 else 0.3 if f < 0.85 else 0.1)


def save_ckpt(path, model, opt, it, cfg, rng, pool=None):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it,
                "config": cfg, "rng": rng.bit_generator.state, "pool": pool}, tmp)
    os.replace(tmp, path)


def per_sample_loss(fill, target, mk):
    """[B] mean squared error over on-board cells."""
    return (((fill - target) ** 2) * mk).flatten(1).sum(1) / mk.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--R", type=int, default=6)
    p.add_argument("--iters", type=int, default=None, help="total iterations, default 4000 (lr decay is relative to this)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", action="store_true", help="phase 2: persistent sample pool with wall edits")
    p.add_argument("--pool-size", type=int, default=256)
    p.add_argument("--edit-frac", type=float, default=0.5, help="phase 2: share of each batch given a wall edit")
    p.add_argument("--resume", action="store_true", help="carry on from runs/<name>/ckpt.pt")
    p.add_argument("--init", help="start from this checkpoint's weights (fresh optimiser, iteration 0)")
    p.add_argument("--clamp", type=float, nargs=2, default=[-1.0, 1.0], metavar=("LO", "HI"))
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--minutes", type=float, default=0, help="stop (with a checkpoint) after this long; 0 = no limit")
    args = p.parse_args()

    torch.set_num_threads(3)
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    start_it = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg = ckpt["config"]
        cfg["iters"] = args.iters or cfg["iters"]
        start_it = ckpt["iteration"]
    else:
        init = torch.load(args.init, weights_only=False) if args.init else None
        base = init["config"] if init else {}
        t0, t1 = base.get("steps") or default_steps(args.R)
        if init and base["R"] != args.R:
            t0, t1 = default_steps(args.R)
        cfg = {
            "R": args.R, "channels": 16, "hidden": 64,
            "clamp": base.get("clamp", None if args.no_clamp else list(args.clamp)),
            "fireRate": 1.0, "steps": [t0, t1], "margin": MARGIN, "lastK": LAST_K,
            "lr": args.lr, "batch": args.batch, "iters": args.iters or 4000, "seed": args.seed,
            "pool": args.pool, "poolSize": args.pool_size, "editFrac": args.edit_frac, "init": args.init,
            "loss": "MSE(ch1, fill) over on-board cells, mean of the last lastK steps",
            "prevIterations": (init["config"].get("prevIterations", 0) + init["iteration"]) if init else 0,
        }

    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"])
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        rng.bit_generator.state = ckpt["rng"]
    elif args.init:
        model.load_state_dict(init["model"])

    R, B = cfg["R"], cfg["batch"]
    T0, T1 = cfg["steps"]
    S = side(R)
    mk = to_t(hex_mask(R)[None])  # [1,1,S,S]
    hw, hf, _ = batch(np.random.default_rng(EVAL_SEED), R, EVAL_N)
    heldout = (to_t(hw), to_t(hf))
    print(json.dumps({"config": cfg, "startIteration": start_it}), flush=True)

    # Phase 2 pool, per sample: the board as first drawn (orig), its current walls, target,
    # depth, and state.
    pool = None
    if cfg["pool"]:
        n = cfg["poolSize"]
        if ckpt and ckpt.get("pool"):
            pool = ckpt["pool"]
        else:
            pw, pf, pd = batch(rng, R, n)
            pool = {"orig": pw.copy(), "walls": pw, "fill": pf, "depth": pd,
                    "state": fresh_state(to_t(pw), cfg["channels"])}
        po, pw, pf, pd, pool_state = pool["orig"], pool["walls"], pool["fill"], pool["depth"], pool["state"]

    def step_range(depth_max):
        lo = max(T0, int(depth_max) + MARGIN)
        return int(rng.integers(lo, max(T1, lo) + 1))

    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    t_start = time.time()
    t_win, losses = time.time(), []
    it = start_it
    while it < cfg["iters"]:
        if cfg["pool"]:
            idx = rng.choice(n, B, replace=False)
            tidx = torch.from_numpy(idx)
            orig_np, walls_np, fill_np, depth_np = po[idx], pw[idx], pf[idx], pd[idx]
            state = pool_state[tidx].clone()
            with torch.no_grad():
                cur = per_sample_loss(state[:, 1:2], to_t(fill_np), mk).numpy()
            order = np.argsort(-cur)  # worst first
            # The worst sample starts over from the fresh state on its own board as first drawn.
            # (Giving the worst a NEW board, as distill does, drifts the pool: boards with
            # something filled are the hard ones, so they leave and the pool ends up ~12% filled.)
            j = order[0]
            walls_np[j] = orig_np[j]
            fill_np[j], depth_np[j] = oracle(orig_np[j], R)
            state[j] = fresh_state(to_t(orig_np[j][None]), cfg["channels"])[0]
            # B//8 random others get brand-new boards (unbiased turnover).
            rest = rng.permutation(order[1:])
            n_new = max(1, B // 8)
            w, f, d = batch(rng, R, n_new)
            for k, j in enumerate(rest[:n_new]):
                orig_np[j], walls_np[j], fill_np[j], depth_np[j] = w[k], w[k], f[k], d[k]
                state[j] = fresh_state(to_t(w[k][None]), cfg["channels"])[0]
            # A random editFrac of the batch get a wall edit and carry on from their settled state
            # (a user drawing on a settled board). Edits never pile up: a board as first drawn gets
            # edit_walls of itself, an edited one goes back to how it was first drawn. Piled-up
            # edits drifted the pool to boards with nothing filled (edits mostly open loops), and
            # going back is the loop-closing case.
            n_edit = min(len(rest) - n_new, int(round(B * cfg.get("editFrac", 0.5))))
            for j in rest[n_new:n_new + n_edit]:
                if np.array_equal(walls_np[j], orig_np[j]):
                    walls_np[j] = edit_walls(rng, orig_np[j], R)
                else:
                    walls_np[j] = orig_np[j]
                fill_np[j], depth_np[j] = oracle(walls_np[j], R)
            walls, target = to_t(walls_np), to_t(fill_np)
            state[:, 0:1] = walls
        else:
            walls_np, fill_np, depth_np = batch(rng, R, B)
            walls, target = to_t(walls_np), to_t(fill_np)
            state = fresh_state(walls, cfg["channels"])

        T = step_range(depth_np.max())
        loss = 0.0
        for t in range(T):
            state = model.step(state, walls, mk)
            if t >= T - LAST_K:
                loss = loss + per_sample_loss(state[:, 1:2], target, mk).mean() / LAST_K

        opt.zero_grad()
        loss.backward()
        for prm in model.parameters():  # per-parameter gradient normalisation (distill notebook)
            if prm.grad is not None:
                prm.grad /= prm.grad.norm() + 1e-8
        for g in opt.param_groups:
            g["lr"] = lr_at(it, cfg["iters"], cfg["lr"])
        opt.step()
        it += 1
        losses.append(loss.item())

        if cfg["pool"]:
            pool_state[tidx] = state.detach()
            po[idx], pw[idx], pf[idx], pd[idx] = orig_np, walls_np, fill_np, depth_np

        if it % 50 == 0 or it == cfg["iters"]:
            exact, acc = quick_eval(model, heldout, mk, T1)
            rec = {"iteration": it, "loss": float(np.mean(losses)),
                   "secPerIter": (time.time() - t_win) / len(losses),
                   "exact": exact, "acc": acc, "lr": lr_at(it - 1, cfg["iters"], cfg["lr"]),
                   "maxRssMB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_win, losses = time.time(), []
        out_of_time = args.minutes and time.time() - t_start > 60 * args.minutes
        if it % 200 == 0 or it == cfg["iters"] or out_of_time:
            save_ckpt(ckpt_path, model, opt, it, cfg, rng, pool)
        if out_of_time:
            print(json.dumps({"stopped": "time", "iteration": it}), flush=True)
            break
    log.close()


if __name__ == "__main__":
    main()
