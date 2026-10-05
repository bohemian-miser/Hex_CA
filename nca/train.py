"""Train the hex fill NCA (spec v2: enclosed regions fill, and all rim regions but the largest;
spec v3: every cell also sees the constant inputs mask, theta1, theta2 -- its angle round the centre;
spec v4: and, per state channel, its max and min over the hex neighbourhood (--perception taps+pool,
the default), and "largest" is by area or by the angular span of a region's rim cells).

The spec v6 recipe is the default (--floods): 7 consts (mask, theta1, theta2 and the rim sources src1, src1c,
src2, src2c), channels 2..7 hand-written exact floods, frozen (model.install_floods), only the readout and the
free channels trained, on the fill loss alone (no aux, no teacher forcing, no pool): a fresh model, R 6 8 10,
steps [7R, 10R], last-k 8, batch 8, bridge-frac 0.25, lr 2e-3 with decay, clamp [-2, 2]:
    python -m nca.train --name v6f --iters 4000 --threads 2 --minutes 8.5 --eval-mults 8
The quick check then logs the comparative rule read off the (exact) channels once, at the start
(arcCeiling: the readout's ceiling), not at every check.
Older recipes need --no-floods (and their consts): v5 = --no-floods --n-consts 3 --aux --steps-mult 4 8 --teach 0 0;
v5 + teacher forcing = the same with --aux-w 20 and no --teach (on by default with dense aux, runs/experiments.md);
v4 = --no-floods --n-consts 3 --pool --aux --no-aux-dense --steps-mult 6 10 --last-k 16 --batch 16.
--init may name a checkpoint with fewer consts (a v1/v2 one has only the mask) or without the pool: its
w1 is copied and the new const columns and w1pool start at zero, so training starts exactly at the old
behaviour (checked and printed before the first iteration).
Carry on an interrupted run (same --name; --iters may be raised):
    python -m nca.train --name a2 --resume --iters 8000

Each iteration draws one radius R from --R and runs T ~ U[a*R, b*R] steps
(a, b = --steps-mult; never fewer than the batch's deepest rim depth + MARGIN).
A board can have several acceptable targets (data.targets); the loss is the
min over them of the per-board loss (MSE of ch1 over on-board cells, averaged
over the last --last-k steps).

--minutes stops before an iteration that would end past the limit, counting the
slowest iteration so far and, if one is due, the last quick check's time (kept
in the checkpoint), so a chunk keeps to the limit with its checks.

Logs one JSON line per 50 iterations to runs/<name>/log.jsonl: loss, s/iter,
peak RSS, and a quick held-out check at every trained radius read out at
mult*R steps for each --eval-mults: exact on the training mix ("mix"), on the
demo page's random loops ("page", never trained on), exact
on bridge boards with 2+ rim regions ("bridge", also per radius), and on those
the share where two or more sides stayed empty ("none") or every side filled
("both"). Exact = equals any acceptable target. Checkpoints to
runs/<name>/ckpt.pt every 200 iterations and at the end (atomically), and to
runs/<name>/best.pt whenever the quick check's (mix + bridge) / 2 at the first
--eval-mults readout is the best so far (training swings; the best is kept).

--aux (training only; nothing in the model or export changes): hidden channels
2..6 are also trained, weighted --aux-w, by an MSE over on-board cells towards
the floods behind the comparative rule (spec v5): max theta1, max (1 - theta1),
max theta2, max (1 - theta2) over the rim cells of the cell's region (ch 2..5),
and the largest span of any region on the board (ch 6, on every cell); span =
clip(min(ch2 + ch3 - 1, ch4 + ch5 - 1), 0, 1), and the rule: fill iff enclosed
(max(ch2, ch4) < 0.25) or span < ch6 - EPS (data.arc_fill). With --aux-dense
(the default) the target at step t of the rollout is data.aux_flood's F_t, the
flood one hop per step from 0, and the loss is the mean over EVERY step 1..T;
--no-aux-dense is v4's: the steady state (data.aux_targets) over the last
--last-k steps (the only aux loss a --pool run can use: its samples don't start
from 0). The quick check then also logs, per readout at m*R: auxDense (mean
over steps 1..m*R against F_t), auxLast (at step m*R against F_{m*R}), both on
mix and bridge boards (trivial all-zero values printed at the start), and the
rule read off the model's own channels 2..6, exact against any acceptable
target (arcMix, arcBridge). Channel 1 keeps its loss; channels 7.. stay free.

--teach P0 P1 (v6, default 1.0 0.2; needs --aux with --aux-dense): teacher forcing with scheduled sampling.
After each model step t of the rollout (its aux and fill losses taken on the model's own output first),
each board, with probability p, has channels 2..6 REPLACED by the exact reference F_t (constants: no
gradient flows back through them), and the rollout steps on from there. p falls linearly from P0 at
iteration 0 to P1 at --iters (logged as teachP); a --resume with a new --iters (or a new --teach) carries
on from the p it stopped at, down to the new P1 at the new --iters. --teach 0 0 is v5. The quick check adds
auxTeach (the same dense MSE, read with p = 1: the one-step rule's error, apart from free-running drift),
auxTeachCh (per channel 2..6), mixTeach / bridgeTeach (fill exact with the floods fed in), and the
trivial start line adds holdDense (a model that copies F_{t-1} through: the one-step baseline).
"""

import argparse
import json
import os
import resource
import time

import numpy as np
import torch

from .data import EPS, N_AUX, aux_flood, aux_targets, edit_walls, pad_targets, page_loops, random_walls, targets
from .evaluate import _bridge_board, boards, score, summarise
from .hexgrid import CONST_NAMES
from .model import PERCEPTIONS, HexNCA, const_stack, fresh_state, load_expanded

MARGIN = 4      # extra steps past the deepest rim-region cell
K_POOL = 4      # targets kept per pool board (K is 1 on ~95% of boards, 2 on ~4.5%)
EVAL_N = 32     # held-out boards per set per radius for the quick log metric
EVAL_SEED = 999 # same held-out boards whatever --seed is


def to_t(a):
    """numpy [B,S,S] -> float tensor [B,1,S,S]; [B,K,S,S] -> [B,K,S,S]."""
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    return t.unsqueeze(1) if t.dim() == 3 else t


def answers(walls, R):
    """(fills uint8 [K_POOL,S,S], depth, aux float32 [5,S,S]) of one board: its targets padded to a fixed K
    for the pool, and its aux targets."""
    f, d = targets(walls, R)
    return pad_targets([f], K_POOL)[0], d, aux_targets(walls, R)


def draw(rng, R, n, bridge_frac=0.0):
    """(walls [n,S,S], fills [n,K_POOL,S,S], depth [n,S,S], aux [n,5,S,S]) of n fresh boards from the training
    mix, each drawn as kind "bridge" instead with probability bridge_frac."""
    walls = np.stack([random_walls(rng, R, "bridge" if rng.random() < bridge_frac else None) for _ in range(n)])
    fills, depth = zip(*(targets(w, R) for w in walls))
    return walls, pad_targets(fills, K_POOL), np.stack(depth), np.stack([aux_targets(w, R) for w in walls])


def per_sample_loss(fill, fills, mk):
    """[B,K] mean squared error of fill [B,1,S,S] against each target fills [B,K,S,S], on-board cells."""
    return (((fill - fills) ** 2) * mk).flatten(2).sum(2) / mk.sum()


def aux_loss(state, aux, mk):
    """[B] mean squared error of channels 2..6 of state [B,C,S,S] against aux [B,5,S,S], on-board cells."""
    return (((state[:, 2:2 + N_AUX] - aux) ** 2) * mk).flatten(1).sum(1) / (N_AUX * mk.sum())


def aux_loss_ch(state, aux, mk):
    """[5] mean squared error of each of channels 2..6 (mean over the batch), on-board cells."""
    return (((state[:, 2:2 + N_AUX] - aux) ** 2) * mk).sum((0, 2, 3)) / (len(state) * mk.sum())


def teach(state, ref, coin):
    """state with channels 2..6 replaced by ref [B,5,S,S] (constants) on the boards where coin [B] is True."""
    c = coin.view(-1, 1, 1, 1)
    return torch.cat([state[:, :2], torch.where(c, ref, state[:, 2:2 + N_AUX]), state[:, 2 + N_AUX:]], 1)


def teach_p(it, cfg):
    """The teacher-forcing probability at iteration it: linear from teachFrom = [it0, p0] to teach[1] at
    cfg["iters"] (it0 = 0, p0 = teach[0] unless a --resume moved the end)."""
    if not cfg.get("teach"):
        return 0.0
    (it0, p0), p1 = cfg["teachFrom"], cfg["teach"][1]
    f = min(1.0, max(0.0, (it - it0) / max(1, cfg["iters"] - it0)))
    return p0 + (p1 - p0) * f


def arc_rule(state, walls, mk):
    """bool [B,1,S,S]: data.arc_fill read off the model's channels 2..6 -- an open cell fills iff its region
    reads enclosed (max(ch2, ch4) < 0.25) or span = clip(min(ch2 + ch3 - 1, ch4 + ch5 - 1), 0, 1) < ch6 - EPS."""
    c = state[:, 2:7]
    span = torch.minimum(c[:, 0:1] + c[:, 1:2] - 1, c[:, 2:3] + c[:, 3:4] - 1).clamp(0, 1)
    enclosed = torch.maximum(c[:, 0:1], c[:, 2:3]) < 0.25
    return (mk > 0) & (walls == 0) & (enclosed | (span < c[:, 4:5] - EPS))


def heldout(radii, consts):
    """{set: {R: (walls, fills, sides, consts)}}: fixed boards for the quick check, mix, bridge and page (the
    demo page's random loops, data.page_loops: never trained on). consts = {R: const stack [1,n,S,S]} (mask
    first). Their aux floods are recomputed per check (aux_flood)."""
    out = {"mix": {}, "bridge": {}, "page": {}}
    for R in radii:
        for name, gen, seed in (("mix", random_walls, EVAL_SEED), ("bridge", _bridge_board, EVAL_SEED + 50),
                                ("page", page_loops, EVAL_SEED + 100)):
            w, f, sides = boards(np.random.default_rng(seed + R), R, EVAL_N, gen)
            out[name][R] = (to_t(w), torch.from_numpy(f) > 0, sides, consts[R])
    return out


def trivial_aux(held, mults):
    """{mult: {auxDense, auxLast, blindDense, blindLast}}: the quick check's aux numbers for a model that keeps
    channels 2..6 at 0, and for one that floods as if there were no walls (the empty board's F_t on every
    board: right timing, nothing region-dependent), and holdDense: one that copies F_{t-1} through each
    step unchanged (the baseline of the one-step error auxTeach)."""
    res = {m: {"auxDense": [], "auxLast": [], "blindDense": [], "blindLast": [], "holdDense": [], "holdLast": []}
           for m in mults}
    for per_r in held.values():
        for R, (walls, _, _, cs) in per_r.items():
            w = walls[:, 0].numpy().astype(np.uint8)
            F = torch.from_numpy(aux_flood(w, R, max(mults) * R))
            blind = torch.from_numpy(aux_flood(np.zeros_like(w), R, max(mults) * R))
            pad = torch.zeros(len(w), 2, *w.shape[-2:])
            for key, guess in (("aux", lambda t: torch.zeros_like(F[t])), ("blind", lambda t: blind[t]),
                               ("hold", lambda t: F[t - 1] if t else torch.zeros_like(F[t]))):
                per_step = [float(aux_loss(torch.cat([pad, guess(t)], 1), F[t], cs[:, :1]).mean())
                            for t in range(len(F))]
                for m in mults:
                    res[m][key + "Dense"].append(np.mean(per_step[:m * R]))
                    res[m][key + "Last"].append(per_step[m * R - 1])
    return {m: {k: round(float(np.mean(v)), 4) for k, v in r.items()} for m, r in res.items()}


@torch.no_grad()
def quick_eval(model, held, mults, aux=False, arc=True, teach_check=False):
    """{mult: {mix, bridge, page, none, both, gap, bridgeByR, ...}} -- means over the radii, read out at mult*R
    steps; mix / bridge / page = exact on each held-out set.

    gap (bridge boards): mean ch1 on the rim regions the primary target fills minus mean ch1 on the one it
    leaves empty -- a soft trend that moves long before exact does. With aux: auxDense = channels 2..6 against
    the reference flood F_t (data.aux_flood), mean over steps 1..m*R; auxLast = the same at step m*R alone
    (trivial_aux gives both for an all-zero guess). With arc: arcMix / arcBridge / arcPage = the comparative
    rule read off channels 2..6 (arc_rule, = data.arc_fill), exact against any acceptable target (with the
    spec v6 floods these channels are exact, so that is the ceiling of the readout: logged once). With
    teach_check (spec v6 teacher forcing) a second rollout feeds F_t into channels 2..6 after every step
    (p = 1): auxTeach is its dense MSE (the one-step rule's error alone), auxTeachCh the same per channel
    2..6, mixTeach / bridgeTeach / pageTeach its fill exact (the readout given exact floods).
    """
    cap = lambda name: name[0].upper() + name[1:]
    res = {m: {"none": [], "both": [], "gap": [], "bridgeByR": {}, **{k: [] for k in held}} for m in mults}
    for name, per_r in held.items():
        for R, (walls, fills, sides, cs) in per_r.items():
            mk = cs[:, :1]
            if aux or teach_check:
                F = torch.from_numpy(aux_flood(walls[:, 0].numpy().astype(np.uint8), R, max(mults) * R))
            if teach_check:
                state, done, per_ch = fresh_state(walls, model.channels), 0, []
                for m in sorted(mults):
                    for t in range(done, m * R):
                        state = model.step(state, walls, cs)
                        per_ch.append(aux_loss_ch(state, F[t], mk))
                        state = teach(state, F[t], torch.ones(len(state), dtype=torch.bool))
                    done = m * R
                    ch = torch.stack(per_ch).mean(0)
                    res[m].setdefault("auxTeach", []).append(float(ch.mean()))
                    res[m].setdefault("auxTeachCh", []).append(ch.numpy())
                    s = summarise(score(state[:, 1:2] > 0.5, fills, mk, sides), len(sides))
                    res[m].setdefault(name + "Teach", []).append(s["exact"])
            state, done, per_step = fresh_state(walls, model.channels), 0, []
            for m in sorted(mults):
                for t in range(done, m * R):
                    state = model.step(state, walls, cs)
                    if aux:
                        per_step.append(float(aux_loss(state, F[t], mk).mean()))
                done = m * R
                s = summarise(score(state[:, 1:2] > 0.5, fills, mk, sides), len(sides), name == "bridge")
                res[m][name].append(s["exact"])
                if aux:
                    res[m].setdefault("auxDense", []).append(np.mean(per_step))
                    res[m].setdefault("auxLast", []).append(per_step[-1])
                if arc:
                    a = summarise(score(arc_rule(state, walls, mk), fills, mk, sides), len(sides))
                    res[m].setdefault("arc" + cap(name), []).append(a["exact"])
                if name == "bridge":
                    res[m]["none"].append(s["none"])
                    res[m]["both"].append(s["both"])
                    p, prim = state[:, 1].numpy(), fills[:, 0].numpy()
                    res[m]["gap"].append(np.mean([p[b][np.isin(lab, ids) & prim[b]].mean()
                                                  - p[b][np.isin(lab, ids) & ~prim[b]].mean()
                                                  for b, (lab, ids, _) in enumerate(sides)]))
                    res[m]["bridgeByR"][R] = round(float(s["exact"]), 3)
    out = {m: {k: (round(float(np.mean(v)), 4) if isinstance(v, list) and k != "auxTeachCh" else v)
               for k, v in r.items()} for m, r in res.items()}
    for r in out.values():
        if "auxTeachCh" in r:
            r["auxTeachCh"] = [round(float(x), 4) for x in np.mean(r["auxTeachCh"], 0)]
    return out


def lr_at(it: int, iters: int, lr: float) -> float:
    """Step decay: full lr to 60%, x0.3 to 85%, x0.1 after."""
    f = it / max(1, iters)
    return lr * (1.0 if f < 0.6 else 0.3 if f < 0.85 else 0.1)


def save_ckpt(path, model, opt, it, cfg, rng, pools=None, best=-1.0, eval_sec=0.0):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it,
                "config": cfg, "rng": rng.bit_generator.state, "pool": pools, "best": best,
                "evalSec": eval_sec}, tmp)
    os.replace(tmp, path)


def new_pool(rng, R, n, channels, bridge_frac):
    """A pool of n boards at radius R: the walls as first drawn (orig), current walls, targets, depth, state."""
    w, f, d, a = draw(rng, R, n, bridge_frac)
    return {"orig": w.copy(), "walls": w, "fill": f, "depth": d, "aux": a,
            "state": fresh_state(to_t(w), channels)}


@torch.no_grad()
def expansion_check(model, init, cs, tol=1e-6):
    """Assert that `model` (just loaded by load_expanded) computes what the --init checkpoint's own model
    does: one step from a random state, and a 4R-step rollout from the fresh state, on random walls.
    Returns the max abs differences."""
    ic = init["config"]
    n_old = init["model"]["w1"].shape[1] - ic["channels"]
    old = HexNCA(ic["channels"], ic["hidden"], ic["clamp"], ic["fireRate"], n_old, ic.get("perception", "taps"))
    old.load_state_dict(init["model"])
    S = cs.shape[-1]
    R = (S - 1) // 2
    g = torch.Generator().manual_seed(0)
    walls = (torch.rand(4, 1, S, S, generator=g) < 0.2).float() * cs[:, :1]
    state = (torch.rand(4, model.channels, S, S, generator=g) * 2 - 1) * cs[:, :1]
    state[:, 0:1] = walls
    one = float((model.step(state, walls, cs) - old.step(state, walls, cs[:, :n_old])).abs().max())
    fresh = fresh_state(walls, model.channels)
    roll = float((model(fresh, walls, cs, 4 * R) - old(fresh, walls, cs[:, :n_old], 4 * R)).abs().max())
    assert one <= tol and roll <= tol, f"--init expansion changed the model's output: {one:.2e} / {roll:.2e}"
    return {"from": n_old, "to": model.n_consts, "perception": [old.perception, model.perception], "R": R, "oneStepMaxDiff": one, f"rollout{4 * R}MaxDiff": roll,
            "tol": tol, "ok": True}


def main():
    t_start = time.time()  # --minutes counts from here: setup and quick checks included
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--R", type=int, nargs="+", default=[6, 8, 10], help="radii; each iteration uses one")
    p.add_argument("--steps-mult", type=float, nargs=2, default=[7, 10], metavar=("A", "B"),
                   help="steps per iteration ~ U[A*R, B*R]")
    p.add_argument("--last-k", type=int, default=8, help="fill loss = mean over the last K steps")
    p.add_argument("--iters", type=int, default=None, help="total iterations, default 4000 (lr decay is relative to this)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--n-consts", type=int, default=7, choices=(1, 3, 7),
                   help="constant inputs, the first n of mask, theta1, theta2, src1, src1c, src2, src2c "
                        "(1 = the v1/v2 model, 3 = v3-v5, 7 = v6: needed by --floods)")
    p.add_argument("--floods", action=argparse.BooleanOptionalAction, default=True,
                   help="spec v6 (default): channels 2..7 are the hand-written exact floods (model.install_floods), "
                        "frozen; only the rest is trained. --no-floods: everything is learned (v1-v5)")
    p.add_argument("--perception", default="taps+pool", choices=PERCEPTIONS,
                   help="taps (spec v1-v3) or taps+pool (spec v4: plus each state channel's hex max and min)")
    p.add_argument("--bridge-frac", type=float, default=0.25,
                   help="share of fresh boards drawn as kind \"bridge\" on top of the mix's own")
    p.add_argument("--pool", action=argparse.BooleanOptionalAction, default=False,
                   help="persistent sample pools (one per radius) with wall edits (v2-v4; v5 trains fresh starts only)")
    p.add_argument("--pool-size", type=int, default=256, help="boards per radius")
    p.add_argument("--edit-frac", type=float, default=0.5, help="pool: share of each batch given a wall edit")
    p.add_argument("--aux", action="store_true",
                   help="also train channels 2..6 towards data.aux_targets (the floods of the comparative rule)")
    p.add_argument("--aux-dense", action=argparse.BooleanOptionalAction, default=True,
                   help="--aux targets per step: the flood F_t at every step t (spec v5, default), or with "
                        "--no-aux-dense the steady state over the last-k steps (v4)")
    p.add_argument("--aux-w", type=float, default=1.0, help="weight of the --aux loss")
    p.add_argument("--teach", type=float, nargs=2, default=None, metavar=("P0", "P1"),
                   help="teacher forcing (spec v6; --aux --aux-dense only): after each step, with probability p "
                        "per board, channels 2..6 := the reference F_t; p linear from P0 at iteration 0 to P1 at "
                        "--iters (default 1.0 0.2; 0 0 = off). On --resume: anneal on from the current p to "
                        "the new P1")
    p.add_argument("--resume", action="store_true", help="carry on from runs/<name>/ckpt.pt")
    p.add_argument("--init", help="start from this checkpoint's weights (fresh optimiser, iteration 0); "
                   "it may have fewer consts: the new const columns start at zero")
    p.add_argument("--clamp", type=float, nargs=2, default=[-2.0, 2.0], metavar=("LO", "HI"),
                   help="state clamp of a fresh model; [-2, 2] since v4: the --aux targets of the max channels sit "
                        "at ~0.98, and under a clamp at 1 they stick there with no gradient (runs/experiments.md, v4)")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--eval-mults", type=int, nargs="+", default=[8, 16], help="quick check read out at mult*R steps")
    p.add_argument("--eval-every", type=int, default=50, help="quick check every this many iterations (a multiple of 50)")
    p.add_argument("--minutes", type=float, default=0, help="stop (with a checkpoint) after this long; 0 = no limit")
    p.add_argument("--threads", type=int, default=2, help="torch threads (at most 3 on the Pi)")
    args = p.parse_args()
    if args.floods and not args.resume and (args.n_consts != 7 or args.perception != "taps+pool" or args.aux
                                            or args.init):
        p.error("--floods (the default) needs --n-consts 7, the taps+pool perception, no --aux and no --init "
                "(channels 2..7 are written by hand); use --no-floods for the v1-v5 recipes")
    if args.pool and args.aux and args.aux_dense and not args.resume:
        p.error("--aux-dense floods from the fresh state, but --pool samples carry their state on: "
                "use --no-aux-dense with --pool")
    if args.teach and any(args.teach) and not args.resume and not (args.aux and args.aux_dense and not args.pool):
        p.error("--teach feeds in the reference flood F_t: it needs --aux --aux-dense and no --pool")

    torch.set_num_threads(min(3, args.threads))
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    start_it = 0
    ckpt = init = None
    if args.resume:
        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg = ckpt["config"]
        start_it = ckpt["iteration"]
        p_now = teach_p(start_it, cfg)  # where the anneal stopped, on the schedule it was saved with
        cfg["iters"] = args.iters or cfg["iters"]
        if cfg.get("teach") and args.teach:
            cfg["teach"] = [cfg["teach"][0], args.teach[1]]
        if cfg.get("teach") and abs(teach_p(start_it, cfg) - p_now) > 1e-12:  # new --iters / P1: go on from here
            cfg["teachFrom"] = [start_it, p_now]
    else:
        if args.init:
            init = torch.load(args.init, weights_only=False)
            ic = init["config"]
            if (ic["channels"], ic["hidden"]) != (args.channels, args.hidden):
                p.error(f"--init {args.init} has channels={ic['channels']} hidden={ic['hidden']}, but this run asks "
                        f"for --channels {args.channels} --hidden {args.hidden}; the sizes must match")
            init_consts = init["model"]["w1"].shape[1] - ic["channels"]
            if init_consts > args.n_consts:
                p.error(f"--init {args.init} has {init_consts} consts, more than --n-consts {args.n_consts}")
            if ic.get("perception", "taps") == "taps+pool" and args.perception == "taps":
                p.error(f"--init {args.init} pools; --perception taps would drop its w1pool")
        a, b = args.steps_mult
        cfg = {
            "R": sorted(args.R), "channels": args.channels, "hidden": args.hidden,
            "nConsts": args.n_consts, "consts": list(CONST_NAMES[:args.n_consts]), "perception": args.perception,
            "floods": args.floods,
            "clamp": init["config"]["clamp"] if init else (None if args.no_clamp else list(args.clamp)),
            "fireRate": 1.0, "stepsMult": [a, b],
            "steps": {R: [int(round(a * R)), int(round(b * R))] for R in sorted(args.R)},
            "margin": MARGIN, "lastK": args.last_k,
            "lr": args.lr, "batch": args.batch, "iters": args.iters or 4000, "seed": args.seed,
            "bridgeFrac": args.bridge_frac, "pool": args.pool, "poolSize": args.pool_size, "editFrac": args.edit_frac, "init": args.init,
            "loss": "min over acceptable targets of MSE(ch1, fill) over on-board cells, mean of the last lastK steps"
                    + ("" if not args.aux else
                       " + auxW * MSE(ch2..6, aux_flood F_t) over on-board cells, mean of every step t = 1..T"
                       if args.aux_dense else " + auxW * MSE(ch2..6, aux_targets) over on-board cells, last lastK steps"),
            "aux": args.aux, "auxDense": args.aux and args.aux_dense, "auxW": args.aux_w,
            "teach": None, "teachFrom": None,
            "prevIterations": (init["config"].get("prevIterations", 0) + init["iteration"]) if init else 0,
            "initConsts": init_consts if init else None,
        }
        if args.aux and args.aux_dense and not args.pool:
            p01 = args.teach or [1.0, 0.2]
            if any(p01):
                cfg["teach"], cfg["teachFrom"] = list(p01), [0, p01[0]]

    n_consts = cfg.get("nConsts", 1)  # checkpoints from before v3 have only the mask
    perception = cfg.setdefault("perception", "taps")  # and from before v4 no pool
    floods = cfg.get("floods", False)  # spec v6
    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"], n_consts, perception, floods)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"])
    radii, B, last_k, bf = cfg["R"], cfg["batch"], cfg["lastK"], cfg.get("bridgeFrac", 0.0)
    use_aux, aux_w, dense = cfg.get("aux", False), cfg.get("auxW", 1.0), cfg.get("auxDense", False)
    consts = {R: const_stack(R, n_consts) for R in radii}  # [1,n,S,S] each, built once per radius
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        rng.bit_generator.state = ckpt["rng"]
        assert not floods or all(torch.equal(getattr(model, k)[f], model.frozen_values[k][f])
                                 for k, f in model.frozen.items()), "the checkpoint's hand-written floods changed"
    elif init:
        load_expanded(model, init["model"])
        if init_consts < n_consts or "w1pool" not in init["model"] and perception == "taps+pool":
            print(json.dumps({"initExpansion": expansion_check(model, init, consts[radii[-1]])}), flush=True)

    held = heldout(radii, consts)
    start = {"config": cfg, "startIteration": start_it}
    if use_aux:
        start["trivialAux"] = {f"q{m}": q for m, q in trivial_aux(held, args.eval_mults).items()}
    if floods and not ckpt:  # the comparative rule read off the exact frozen floods: the readout's ceiling
        start["arcCeiling"] = {f"q{m}": {k: v for k, v in q.items() if k.startswith("arc")}
                               for m, q in quick_eval(model, held, args.eval_mults).items()}
    print(json.dumps(start), flush=True)

    pools = None
    if cfg["pool"]:
        pools = ckpt["pool"] if ckpt and ckpt.get("pool") else \
            {R: new_pool(rng, R, cfg["poolSize"], cfg["channels"], bf) for R in radii}
        for R, P in pools.items():  # pools saved before --aux existed, or with v3's 4 aux planes
            if "aux" not in P or P["aux"].shape[1] != N_AUX:
                P["aux"] = np.stack([aux_targets(w, R) for w in P["walls"]])

    def n_steps(R, depth_max):
        t0, t1 = cfg["steps"][R]
        lo = max(t0, int(depth_max) + MARGIN)
        return int(rng.integers(lo, max(t1, lo) + 1))

    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    t_win, losses, parts, taught = time.time(), [], [], []
    best = ckpt.get("best", -1.0) if ckpt else -1.0
    # --minutes: stop BEFORE an iteration that (with the slowest iteration seen so far, plus the last quick
    # check's time if one is due after it) would end past the limit -- so the quick check keeps to it too.
    eval_sec = ckpt.get("evalSec", 0.0) if ckpt else 0.0
    slowest = 0.0
    it = start_it
    while it < cfg["iters"]:
        due = (it + 1) % args.eval_every == 0 or it + 1 == cfg["iters"]
        if args.minutes and time.time() - t_start + slowest + (eval_sec if due else 0) > 60 * args.minutes:
            if it > start_it:
                save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec)
            print(json.dumps({"stopped": "time", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2),
                              "slowestIterSec": round(slowest, 2), "evalSec": round(eval_sec, 1)}), flush=True)
            break
        t_it = time.time()
        R = int(rng.choice(radii))
        cs = consts[R]
        mk = cs[:, :1]
        if cfg["pool"]:
            P = pools[R]
            n = len(P["orig"])
            idx = rng.choice(n, B, replace=False)
            tidx = torch.from_numpy(idx)
            orig_np, walls_np, fill_np, depth_np = P["orig"][idx], P["walls"][idx], P["fill"][idx], P["depth"][idx]
            aux_np = P["aux"][idx]
            state = P["state"][tidx].clone()
            with torch.no_grad():
                cur = per_sample_loss(state[:, 1:2], to_t(fill_np), mk).min(1).values.numpy()
            order = np.argsort(-cur)  # worst first
            # The worst sample starts over from the fresh state on its own board as first drawn
            # (a new board instead, as in distill, drifts the pool towards the easy boards).
            j = order[0]
            walls_np[j] = orig_np[j]
            fill_np[j], depth_np[j], aux_np[j] = answers(orig_np[j], R)
            state[j] = fresh_state(to_t(orig_np[j][None]), cfg["channels"])[0]
            # B//8 random others get brand-new boards (unbiased turnover).
            rest = rng.permutation(order[1:])
            n_new = max(1, B // 8)
            w, f, d, a = draw(rng, R, n_new, bf)
            for k, j in enumerate(rest[:n_new]):
                orig_np[j], walls_np[j], fill_np[j], depth_np[j], aux_np[j] = w[k], w[k], f[k], d[k], a[k]
                state[j] = fresh_state(to_t(w[k][None]), cfg["channels"])[0]
            # A random editFrac of the batch get a wall edit and carry on from their settled state
            # (a user drawing on a settled board). Edits never pile up: a board as first drawn gets
            # edit_walls of itself, an edited one goes back to how it was first drawn.
            n_edit = min(len(rest) - n_new, int(round(B * cfg.get("editFrac", 0.5))))
            for j in rest[n_new:n_new + n_edit]:
                if np.array_equal(walls_np[j], orig_np[j]):
                    walls_np[j] = edit_walls(rng, orig_np[j], R)
                else:
                    walls_np[j] = orig_np[j]
                fill_np[j], depth_np[j], aux_np[j] = answers(walls_np[j], R)
            walls, target = to_t(walls_np), to_t(fill_np)
            state[:, 0:1] = walls
        else:
            walls_np, fill_np, depth_np, aux_np = draw(rng, R, B, bf)
            walls, target = to_t(walls_np), to_t(fill_np)
            state = fresh_state(walls, cfg["channels"])

        T = n_steps(R, depth_np.max())
        if use_aux and dense:  # [T,B,5,S,S]: the reference flood at every step from the fresh state
            aux_t = torch.from_numpy(aux_flood(walls_np, R, T))
        elif use_aux:
            aux_t = torch.from_numpy(np.ascontiguousarray(aux_np))
        acc = acc_aux = 0.0
        p_teach = teach_p(it, cfg) if use_aux and dense and not cfg["pool"] else 0.0
        n_taught = 0
        for t in range(T):
            state = model.step(state, walls, cs)
            if use_aux and dense:
                acc_aux = acc_aux + aux_loss(state, aux_t[t], mk) / T
            if t >= T - last_k:
                acc = acc + per_sample_loss(state[:, 1:2], target, mk) / last_k
                if use_aux and not dense:
                    acc_aux = acc_aux + aux_loss(state, aux_t, mk) / last_k
            if p_teach > 0 and t < T - 1:  # teacher forcing: the losses above saw the model's own output
                coin = torch.from_numpy(rng.random(B) < p_teach)
                if coin.any():
                    state = teach(state, aux_t[t], coin)
                    n_taught += int(coin.sum())
        loss_fill = acc.min(1).values.mean()  # the closest acceptable target, per board
        loss = loss_fill + aux_w * acc_aux.mean() if use_aux else loss_fill

        opt.zero_grad()
        loss.backward()
        for prm in model.parameters():  # per-parameter gradient normalisation (distill notebook)
            if prm.grad is not None:
                prm.grad /= prm.grad.norm() + 1e-8
        for g in opt.param_groups:
            g["lr"] = lr_at(it, cfg["iters"], cfg["lr"])
        opt.step()
        model.restore_floods()  # spec v6: a no-op unless something besides the (masked) gradient moved them
        it += 1
        losses.append(loss.item())
        if use_aux:
            parts.append((loss_fill.item(), acc_aux.mean().item()))
        taught.append(n_taught / (B * max(1, T - 1)))
        slowest = max(slowest, time.time() - t_it)  # the iteration alone (no quick check, no checkpoint)

        if cfg["pool"]:
            P["state"][tidx] = state.detach()
            P["orig"][idx], P["walls"][idx], P["fill"][idx], P["depth"][idx] = orig_np, walls_np, fill_np, depth_np
            P["aux"][idx] = aux_np

        if it % 50 == 0 or it == cfg["iters"]:
            rec = {"iteration": it, "loss": float(np.mean(losses)),
                   "secPerIter": (time.time() - t_win) / len(losses),
                   "lr": lr_at(it - 1, cfg["iters"], cfg["lr"]),
                   "maxRssMB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
            if parts:
                rec["lossFill"], rec["lossAux"] = (float(x) for x in np.mean(parts, 0))
            if cfg.get("teach"):  # p at the window's last iteration, and the share of steps actually taught
                rec["teachP"], rec["taught"] = round(teach_p(it - 1, cfg), 4), round(float(np.mean(taught)), 4)
            if it % args.eval_every == 0 or it == cfg["iters"]:
                t_eval = time.time()
                rec.update({f"q{m}": q for m, q in
                            quick_eval(model, held, args.eval_mults, aux=use_aux, arc=not floods,
                                       teach_check=use_aux and dense and bool(cfg.get("teach"))).items()})
                eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
                q = rec[f"q{args.eval_mults[0]}"]
                if (q["mix"] + q["bridge"]) / 2 > best:  # training swings: keep the best quick check too
                    best = rec["best"] = (q["mix"] + q["bridge"]) / 2
                    save_ckpt(os.path.join(run_dir, "best.pt"), model, opt, it, cfg, rng, eval_sec=eval_sec)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_win, losses, parts, taught = time.time(), [], [], []
        if it % 200 == 0 or it == cfg["iters"]:
            save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec)
    log.close()


if __name__ == "__main__":
    main()
