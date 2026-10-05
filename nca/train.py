"""Train the hex fill NCA.

The default is the PURE recipe: a plain learned NCA trained the distill.pub "Growing NCA" way, with nothing
written by hand. 16 state channels (ch0 = walls, re-imposed every step; ch1 = fill; the rest hidden), ONE
const input (the mask: the board's shape, nothing else), perception = the 7 hex taps of a masked 3x3 conv
(no pooled max/min), --hidden 96 units; no hand-written floods, no aux targets, no teacher forcing. The
loss is the fill MSE alone.

It trains from a persistent pool per radius (--pool, --pool-size boards each). Each iteration draws one
radius R from --R and a batch of --batch boards from that pool, each with the state it was left in:
  - the worst of the batch (loss of its stored state) starts over from the fresh state on its board;
  - batch/8 others are replaced by brand-new boards (random_walls; --bridge-frac of them kind "bridge");
  - each of the rest, with probability --damage, gets ONE damage (kind drawn per board):
      edit    a small wall edit (data.edit_walls)
      burst   3-20 random wall additions and deletions, mixed
      erase   every wall inside a random disc of radius 1-3 opened
      stamp   a fresh closed loop or a rim-to-rim bridge ORed onto the board
      state   distill's damage: every channel but the walls zeroed inside a random disc of radius
              1..R/2 (walls and targets unchanged)
    after a wall damage the board's targets are recomputed. Damage piles up on the pool board: nothing
    toggles back to an original.
Then T ~ U[a*R, b*R] steps run from the stored states (a, b = --steps-mult; never fewer than the batch's
deepest rim depth + MARGIN), loss = min over each board's acceptable targets (data.targets; K_POOL kept,
padded) of the fill MSE over on-board cells, averaged over the last --last-k steps; Adam with per-parameter
gradient normalisation; states and boards go back into the pool (detached). A step whose loss or gradient
is not finite is skipped and its boards start over (logged as "skipped"; 20 in a row stop the run).

Every 200 iterations the log has each pool's statistics (pool_stats: share of boards with a filled cell,
with 2+ rim regions, mean wall density). While one leaves BAND the damage leans towards bringing it back and
new boards come twice as fast (steer, logged as "steer" with the reason).

    python -m nca.train --name pure-a --R 4 5 6 --iters 20000 --minutes 8.5     # stage 1, in chunks:
    python -m nca.train --name pure-a --resume --minutes 8.5                     # ... (exactly where it stopped)
    python -m nca.train --name pure-b --init runs/pure-a/best.pt --R 6 8 10 --iters 20000 --minutes 8.5
--init takes another checkpoint's weights (fresh optimiser, fresh pools, iteration 0, any --R): a stage of a
curriculum. --resume carries a run on exactly (weights, optimiser, pools, rng, steering; --iters may be
raised, which moves the lr decay).

The quick held-out check (every --eval-every iterations, and at iteration 0 of a fresh or --init run; fixed
boards per trained radius, means over the radii): exact (= equals any acceptable target) from the fresh
state at mult*R steps for each --eval-mults, on the training mix ("mix"), on bridge boards with 2+ rim regions
("bridge", also per radius, with "none" / "both": the share where two or more sides stayed empty / every
side filled, and "gap") and on the demo page's random loops ("page", never trained on); and the EDIT
SEQUENCE ("edit": what the page does): a board of the mix settled from the fresh state for eval_mults[0]*R
steps, then 3 successive wall damages (kinds edit, burst, erase, stamp), each followed by 6R steps WITHOUT a
reset, exact against the edited board's targets after each ([e1, e2, e3]; editHold at the start: the share
where the previous answer is still acceptable). best.pt keeps the best score = mean(mix, bridge, page,
mean(edit)) at eval_mults[0]*R. The log (runs/<name>/log.jsonl, a line per 50 iterations) also has loss,
s/iter, peak RSS and the damage counts.

--minutes stops before an iteration that would end past the limit, counting the slowest iteration so far
and, if one is due, the last quick check's time (kept in the checkpoint), so a chunk keeps to the limit with
its checks. Checkpoints (atomic) to runs/<name>/ckpt.pt every 200 iterations and at a stop.

The older HYBRID recipes stay available (off by default):
spec v6 (hand-written frozen floods, learned readout): --floods --n-consts 7 --perception taps+pool --no-pool
--hidden 64 --batch 8 --steps-mult 7 10 --R 6 8 10 (the 7 consts are mask, theta1, theta2 and the rim sources
src1, src1c, src2, src2c; channels 2..7 written by hand by model.install_floods, frozen).
v5 = --n-consts 3 --perception taps+pool --no-pool --hidden 64 --batch 8 --aux --steps-mult 4 8 --teach 0 0;
v5 + teacher forcing = the same with --aux-w 20 and no --teach; v4 = --n-consts 3 --perception taps+pool
--pool --aux --no-aux-dense --hidden 64 --steps-mult 6 10 --last-k 16 (its pool now damages as above).
--init may name a checkpoint with fewer consts or without the pool: its w1 is copied and the new const
columns and w1pool start at zero (checked and printed before the first iteration).

--aux (training only): hidden channels 2..6 are also trained, weighted --aux-w, by an MSE over on-board cells
towards the floods behind the comparative rule (spec v5): max theta1, max (1 - theta1), max theta2,
max (1 - theta2) over the rim cells of the cell's region (ch 2..5), and the largest span of any region on
the board (ch 6); with --aux-dense (its default) the target at step t is data.aux_flood's F_t, the mean
over EVERY step; --no-aux-dense: the steady state (data.aux_targets) over the last --last-k steps (the only
aux loss a pool can use). The quick check then also logs auxDense, auxLast and the rule read off channels
2..6 (arcMix, arcBridge, arcPage).
--teach P0 P1 (needs --aux --aux-dense --no-pool; default 1.0 0.2): teacher forcing with scheduled sampling,
channels 2..6 replaced by F_t with probability p after each step, p linear from P0 to P1 over --iters.
"""

import argparse
import json
import os
import resource
import time

import numpy as np
import torch

from .data import (EPS, N_AUX, WALL_DAMAGE, aux_flood, aux_targets, damage_walls, disc, pad_targets, page_loops,
                   random_walls, targets)
from .evaluate import _bridge_board, boards, score, summarise
from .hexgrid import CONST_NAMES, mask as hex_mask, rim
from .model import PERCEPTIONS, HexNCA, const_stack, fresh_state, load_expanded

MARGIN = 4      # extra steps past the deepest rim-region cell
K_POOL = 4      # targets kept per pool board (K is 1 on ~95% of boards, 2 on ~4.5%)
EVAL_N = 32     # held-out boards per set per radius for the quick log metric
EVAL_SEED = 999 # same held-out boards whatever --seed is
DAMAGE_KINDS = WALL_DAMAGE + ("state",)  # --damage kinds a-d (the walls) and e (the state)
BAND = {"fill": (0.35, 0.8), "multiRim": (0.15, 1.0), "density": (0.0, 0.45)}  # pool_stats kept in these
N_EDITS = 3     # the quick check's edit sequence: edits per board,
EDIT_MULT = 6   # and EDIT_MULT*R steps after each
STOP_SKIPS = 20 # non-finite steps in a row that stop the run
SNAP_N, SNAP_M = 48, 6  # pool.npz shows the first SNAP_N slots of each pool, the full state of the first SNAP_M


def to_t(a):
    """numpy [B,S,S] -> float tensor [B,1,S,S]; [B,K,S,S] -> [B,K,S,S]."""
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    return t.unsqueeze(1) if t.dim() == 3 else t


def answers(walls, R, aux=False):
    """(fills uint8 [K_POOL,S,S], depth, aux float32 [5,S,S] or None) of one board: its targets padded to a
    fixed K for the pool, and (only if asked) its aux targets."""
    f, d = targets(walls, R)
    return pad_targets([f], K_POOL)[0], d, aux_targets(walls, R) if aux else None


def draw(rng, R, n, bridge_frac=0.0, aux=False):
    """(walls [n,S,S], fills [n,K_POOL,S,S], depth [n,S,S], aux [n,5,S,S] or None) of n fresh boards from the
    training mix, each drawn as kind "bridge" instead with probability bridge_frac."""
    walls = np.stack([random_walls(rng, R, "bridge" if rng.random() < bridge_frac else None) for _ in range(n)])
    fills, depth = zip(*(targets(w, R) for w in walls))
    return (walls, pad_targets(fills, K_POOL), np.stack(depth),
            np.stack([aux_targets(w, R) for w in walls]) if aux else None)


def damage_state(rng, state, R):
    """distill's damage, in place on ONE board's state [C,S,S]: every channel but ch0 (the walls) zeroed inside
    a disc of radius 1..max(1, R//2) round a random on-board cell. Returns the disc (bool [S,S])."""
    cells = np.argwhere(hex_mask(R) == 1)
    row, col = cells[rng.integers(len(cells))]
    d = disc(R, row, col, int(rng.integers(1, max(1, R // 2) + 1)))
    state[1:, torch.from_numpy(d)] = 0
    return d


def pool_stats(walls, fills, R):
    """{fill, multiRim, density} of a pool (walls [n,S,S], fills [n,K,S,S]): the share of boards whose primary
    target fills a cell, the share with 2+ rim regions (= the primary target fills a rim cell: with one rim
    region no rim cell fills, with 2+ all but one rim region fill), the mean wall density on board."""
    prim = fills[:, 0] > 0
    return {"fill": round(float(prim.any((1, 2)).mean()), 3),
            "multiRim": round(float(prim[:, rim(R)].any(1).mean()), 3),
            "density": round(float(walls[:, hex_mask(R) == 1].mean()), 3)}


def steer(stats):
    """How a pool's damage leans, from its pool_stats: {p: probabilities over DAMAGE_KINDS, pBridge: share of
    stamps that are bridges, newMult: new boards per batch x this, why: the stats out of BAND}. In band:
    uniform, 0.5, 1. Out of band (each applies on top of the others):
      fill share low       stamps x3 (a stamped loop or bridge fills something), erase and burst x0.5
      fill share high      erase and burst x3, stamps x0.25
      2+ rim regions low   stamps x2, and 0.9 of them bridges
      density high         erase x3, no stamps
    and any of them doubles the new boards."""
    w = dict.fromkeys(DAMAGE_KINDS, 1.0)
    p_bridge, why = 0.5, []
    if stats["fill"] < BAND["fill"][0]:
        w["stamp"], w["erase"], w["burst"] = 3 * w["stamp"], 0.5 * w["erase"], 0.5 * w["burst"]
        why.append("fill low")
    if stats["fill"] > BAND["fill"][1]:
        w["stamp"], w["erase"], w["burst"] = 0.25 * w["stamp"], 3 * w["erase"], 3 * w["burst"]
        why.append("fill high")
    if stats["multiRim"] < BAND["multiRim"][0]:
        w["stamp"], p_bridge = 2 * w["stamp"], 0.9
        why.append("2+ rim low")
    if stats["density"] > BAND["density"][1]:
        w["stamp"], w["erase"] = 0.0, 3 * w["erase"]
        why.append("density high")
    p = np.array([w[k] for k in DAMAGE_KINDS])
    return {"p": (p / p.sum()).round(4).tolist(), "pBridge": p_bridge, "newMult": 2 if why else 1, "why": why}


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


def edit_sequence(R, cs, n=EVAL_N, seed=EVAL_SEED + 150):
    """The quick check's fixed edit-sequence boards at radius R: n boards of the training mix, each damaged
    N_EDITS times in succession (kinds edit, burst, erase, stamp, uniformly; data.damage_walls).
    (stages: N_EDITS + 1 walls [n,1,S,S] (the board, then after each edit), fills: N_EDITS bool [n,K,S,S]
    (each edited board's targets), cs, changed: N_EDITS shares where the primary target changed,
    hold: N_EDITS shares where the previous primary is still acceptable -- what a model that ignores the
    edit would score)."""
    rng = np.random.default_rng(seed + R)
    w = np.stack([random_walls(rng, R) for _ in range(n)])
    stages, fills, changed, hold = [to_t(w)], [], [], []
    prev = np.stack([targets(x, R)[0][0] for x in w])
    for _ in range(N_EDITS):
        w = np.stack([damage_walls(rng, x, R, WALL_DAMAGE[int(rng.integers(len(WALL_DAMAGE)))]) for x in w])
        f = pad_targets([targets(x, R)[0] for x in w])
        changed.append(float(np.mean([(f[i, 0] != prev[i]).any() for i in range(n)])))
        hold.append(float(np.mean([(f[i] == prev[i]).all((1, 2)).any() for i in range(n)])))
        stages.append(to_t(w))
        fills.append(torch.from_numpy(f) > 0)
        prev = f[:, 0]
    return stages, fills, cs, changed, hold


def exact_share(state, fills, mk):
    """Share of boards whose fill (ch1 > 0.5) equals one of their targets fills [B,K,S,S] bool on board."""
    wrong = (((state[:, 1:2] > 0.5) != fills) & (mk > 0)).flatten(2).sum(2)
    return float((wrong.min(1).values == 0).float().mean())


@torch.no_grad()
def edit_eval(model, eseq, settle_mult):
    """{edit: [exact after edit 1..N_EDITS] (mean over the radii), editByR: {R: [...]}}: each board of
    edit_sequence settled from the fresh state for settle_mult*R steps, then each edit applied WITHOUT a
    reset (only ch0 changes) and EDIT_MULT*R steps run, exact against the edited board's targets."""
    by_r = {}
    for R, (stages, fills, cs, _, _) in eseq.items():
        state = model(fresh_state(stages[0], model.channels), stages[0], cs, settle_mult * R)
        by_r[R] = []
        for walls, f in zip(stages[1:], fills):
            state = model(torch.cat([walls, state[:, 1:]], 1), walls, cs, EDIT_MULT * R)
            by_r[R].append(round(exact_share(state, f, cs[:, :1]), 3))
    return {"edit": [round(float(x), 4) for x in np.mean(list(by_r.values()), 0)], "editByR": by_r}


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


def save_ckpt(path, model, opt, it, cfg, rng, pools=None, best=-1.0, eval_sec=0.0, steering=None):
    """Atomically (a temp file, then a rename): weights, optimiser, iteration, config, rng, pools, best score,
    the last quick check's seconds and the pools' steering."""
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it,
                "config": cfg, "rng": rng.bit_generator.state, "pool": pools, "best": best,
                "evalSec": eval_sec, "steer": steering}, tmp)
    os.replace(tmp, path)


def new_pool(rng, R, n, channels, bridge_frac, aux=False):
    """A pool of n fresh boards at radius R: walls, targets, depth, aux targets (only with aux), state."""
    w, f, d, a = draw(rng, R, n, bridge_frac, aux)
    pool = {"walls": w, "fill": f, "depth": d, "state": fresh_state(to_t(w), channels)}
    if aux:
        pool["aux"] = a
    return bookkeeping(pool)


def bookkeeping(pool, it=0):
    """Add (where missing) a pool's per-sample bookkeeping, for pool.npz: loss (the sample's loss the last time
    it was in a batch, NaN before), born (the iteration it last started from the fresh state), edits (damages
    since then), last (kind of the last damage since then, an index into DAMAGE_KINDS; -1 none)."""
    n = len(pool["walls"])
    pool.setdefault("loss", np.full(n, np.nan, dtype=np.float32))
    pool.setdefault("born", np.full(n, it, dtype=np.int64))
    pool.setdefault("edits", np.zeros(n, dtype=np.int32))
    pool.setdefault("last", np.full(n, -1, dtype=np.int8))
    return pool


def write_snapshot(path, it, pools, last_R, last_idx):
    """runs/<name>/pool.npz for the live dashboard (nca/dashboard.py), atomically (a temp file, then a rename):
    iteration, radii, last_R and last_idx (the batch that just ran), and per radius R the first n =
    min(pool size, SNAP_N) slots of its pool: walls_R uint8 [n,S,S], fill_R float16 [n,S,S] (ch1 of the stored
    state), target_R uint8 [n,S,S] (the acceptable target closest to the thresholded fill; the primary on a
    tie), ntargets_R uint8 [n] (distinct acceptable targets), loss_R float32 [n] (NaN if never in a batch yet),
    age_R int32 [n] (iterations since its last fresh start), edits_R int32 [n] (damages since then), damage_R
    int8 [n] (the last one's kind: -1 none, then DAMAGE_KINDS' order: edit, burst, erase, stamp, state), and
    state_R float16 [m,C,S,S], the full state of the first m = min(n, SNAP_M)."""
    out = {"iteration": np.int64(it), "radii": np.array(sorted(pools), dtype=np.int64),
           "last_R": np.int64(last_R), "last_idx": np.asarray(last_idx, dtype=np.int64)}
    for R, P in pools.items():
        n = min(len(P["walls"]), SNAP_N)
        on = hex_mask(R) == 1
        fill = P["state"][:n, 1].numpy()
        fills = P["fill"][:n]  # [n,K,S,S]
        wrong = (((fill > 0.5) & on)[:, None] != (fills > 0)).sum((2, 3))  # [n,K]; off board both are 0
        out[f"walls_{R}"] = P["walls"][:n].astype(np.uint8)
        out[f"fill_{R}"] = fill.astype(np.float16)
        out[f"target_{R}"] = fills[np.arange(n), wrong.argmin(1)].astype(np.uint8)  # argmin: the first of a tie
        out[f"ntargets_{R}"] = np.array([len({t.tobytes() for t in f}) for f in fills], dtype=np.uint8)
        out[f"loss_{R}"] = P["loss"][:n].astype(np.float32)
        out[f"age_{R}"] = (it - P["born"][:n]).astype(np.int32)
        out[f"edits_{R}"] = P["edits"][:n].astype(np.int32)
        out[f"damage_{R}"] = P["last"][:n].astype(np.int8)
        out[f"state_{R}"] = P["state"][:min(n, SNAP_M)].numpy().astype(np.float16)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:  # a file object: savez would add ".npz" to a name
        np.savez_compressed(f, **out)
    os.replace(tmp, path)


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
    p.add_argument("--steps-mult", type=float, nargs=2, default=[6, 10], metavar=("A", "B"),
                   help="steps per iteration ~ U[A*R, B*R]")
    p.add_argument("--last-k", type=int, default=8, help="fill loss = mean over the last K steps")
    p.add_argument("--iters", type=int, default=None, help="total iterations, default 4000 (lr decay is relative to this)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--n-consts", type=int, default=1, choices=(1, 3, 7),
                   help="constant inputs, the first n of mask, theta1, theta2, src1, src1c, src2, src2c "
                        "(1 = the mask alone: the pure default; 3 = v3-v5, 7 = v6: needed by --floods)")
    p.add_argument("--floods", action=argparse.BooleanOptionalAction, default=False,
                   help="hybrid spec v6: channels 2..7 are the hand-written exact floods (model.install_floods), "
                        "frozen; only the rest is trained (needs --n-consts 7 --perception taps+pool). Off by default")
    p.add_argument("--perception", default="taps", choices=PERCEPTIONS,
                   help="taps (the default: a masked 3x3 conv) or taps+pool (hybrid spec v4: plus each state "
                        "channel's hex max and min)")
    p.add_argument("--bridge-frac", type=float, default=0.25,
                   help="share of fresh boards drawn as kind \"bridge\" on top of the mix's own")
    p.add_argument("--pool", action=argparse.BooleanOptionalAction, default=True,
                   help="persistent sample pools (one per radius) with damage (the default); --no-pool: fresh "
                        "starts only (v5, v6)")
    p.add_argument("--pool-size", type=int, default=256, help="boards per radius")
    p.add_argument("--damage", type=float, default=0.5,
                   help="pool: probability that a board of the batch (not the restarted or new ones) gets one "
                        "damage: edit, burst, erase, stamp or state")
    p.add_argument("--aux", action="store_true",
                   help="hybrid: also train channels 2..6 towards data.aux_targets (the floods of the comparative rule)")
    p.add_argument("--aux-dense", action=argparse.BooleanOptionalAction, default=True,
                   help="--aux targets per step: the flood F_t at every step t (spec v5, default), or with "
                        "--no-aux-dense the steady state over the last-k steps (v4)")
    p.add_argument("--aux-w", type=float, default=1.0, help="weight of the --aux loss")
    p.add_argument("--teach", type=float, nargs=2, default=None, metavar=("P0", "P1"),
                   help="teacher forcing (spec v6; --aux --aux-dense --no-pool only): after each step, with "
                        "probability p per board, channels 2..6 := the reference F_t; p linear from P0 at "
                        "iteration 0 to P1 at --iters (default 1.0 0.2; 0 0 = off). On --resume: anneal on from "
                        "the current p to the new P1")
    p.add_argument("--resume", action="store_true", help="carry on from runs/<name>/ckpt.pt, exactly")
    p.add_argument("--init", help="start from this checkpoint's weights (fresh optimiser, fresh pools, iteration 0; "
                   "--R and the rest from this command line: a curriculum stage); it may have fewer consts or "
                   "no pooled perception: the new weights start at zero")
    p.add_argument("--clamp", type=float, nargs=2, default=[-2.0, 2.0], metavar=("LO", "HI"),
                   help="state clamp of a fresh model ([-2, 2] since v4)")
    p.add_argument("--no-clamp", action="store_true")
    p.add_argument("--eval-mults", type=int, nargs="+", default=[8, 16], help="quick check read out at mult*R steps")
    p.add_argument("--eval-every", type=int, default=200, help="quick check every this many iterations (a multiple of 50)")
    p.add_argument("--minutes", type=float, default=0, help="stop (with a checkpoint) after this long; 0 = no limit")
    p.add_argument("--threads", type=int, default=2, help="torch threads (at most 3 on the Pi)")
    p.add_argument("--snap-every", type=int, default=25,
                   help="pool runs: write runs/<name>/pool.npz (write_snapshot, for nca/dashboard.py) every this "
                        "many iterations and at every checkpoint; 0 = never")
    args = p.parse_args()
    if args.eval_every % 50:
        p.error("--eval-every must be a multiple of 50 (the log's period)")
    if args.floods and not args.resume and (args.n_consts != 7 or args.perception != "taps+pool" or args.aux
                                            or args.init):
        p.error("--floods needs --n-consts 7, --perception taps+pool, no --aux and no --init "
                "(channels 2..7 are written by hand)")
    if args.pool and args.aux and args.aux_dense and not args.resume:
        p.error("--aux-dense floods from the fresh state, but --pool samples carry their state on: "
                "use --no-aux-dense with --pool, or --no-pool")
    if args.teach and any(args.teach) and not args.resume and not (args.aux and args.aux_dense and not args.pool):
        p.error("--teach feeds in the reference flood F_t: it needs --aux --aux-dense and --no-pool")

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
        if cfg.get("pool") and "damage" not in cfg:
            p.error(f"{ckpt_path} is from the old pool (edits toggled back to the board as first drawn); "
                    "start a new run, or --init from it")
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
            "bridgeFrac": args.bridge_frac, "pool": args.pool, "poolSize": args.pool_size,
            "damage": args.damage if args.pool else None, "damageKinds": list(DAMAGE_KINDS) if args.pool else None,
            "init": args.init,
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
    torch.manual_seed(cfg["seed"])  # the init of a fresh model (a loaded one overwrites it)
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
    eseq = {R: edit_sequence(R, consts[R]) for R in radii}
    m0 = args.eval_mults[0]

    def check():
        """The quick check: {q<m>: ..., edit: [...], editByR: {...}, score} (see the module docstring)."""
        rec = {f"q{m}": q for m, q in
               quick_eval(model, held, args.eval_mults, aux=use_aux, arc=use_aux,
                          teach_check=use_aux and dense and bool(cfg.get("teach"))).items()}
        rec.update(edit_eval(model, eseq, m0))
        q = rec[f"q{m0}"]
        rec["score"] = round((q["mix"] + q["bridge"] + q["page"] + float(np.mean(rec["edit"]))) / 4, 4)
        return rec

    start = {"config": cfg, "startIteration": start_it,
             "editChanged": [round(float(np.mean([e[3][k] for e in eseq.values()])), 3) for k in range(N_EDITS)],
             "editHold": [round(float(np.mean([e[4][k] for e in eseq.values()])), 3) for k in range(N_EDITS)]}
    if use_aux:
        start["trivialAux"] = {f"q{m}": q for m, q in trivial_aux(held, args.eval_mults).items()}
    if floods and not ckpt:  # the comparative rule read off the exact frozen floods: the readout's ceiling
        start["arcCeiling"] = {f"q{m}": {k: v for k, v in q.items() if k.startswith("arc")}
                               for m, q in quick_eval(model, held, args.eval_mults).items()}
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit(start)  # at every (re)start: the config; its "iters" is the target (a --resume may raise it)

    pools, steering = None, {}
    if cfg["pool"]:
        pools = ckpt["pool"] if ckpt and ckpt.get("pool") else \
            {R: new_pool(rng, R, cfg["poolSize"], cfg["channels"], bf, use_aux) for R in radii}
        for R, P in pools.items():  # aux targets for a pool that lacks them (or has v3's 4 planes)
            if use_aux and ("aux" not in P or P["aux"].shape[1] != N_AUX):
                P["aux"] = np.stack([aux_targets(w, R) for w in P["walls"]])
            bookkeeping(P, start_it)
        steering = (ckpt or {}).get("steer") or {R: steer(pool_stats(P["walls"], P["fill"], R))
                                                 for R, P in pools.items()}

    def n_steps(R, depth_max):
        t0, t1 = cfg["steps"][R]
        lo = max(t0, int(depth_max) + MARGIN)
        return int(rng.integers(lo, max(t1, lo) + 1))

    snap_path, snap_failed, last = os.path.join(run_dir, "pool.npz"), [], [None]

    def snapshot(it):
        """pool.npz after the batch last[0] = (R, idx) ran; a failure is logged once and never stops training."""
        if not pools or not args.snap_every or last[0] is None:
            return
        try:
            write_snapshot(snap_path, it, pools, *last[0])
        except Exception as e:  # noqa: BLE001 -- the dashboard is never worth a training run
            if not snap_failed:
                snap_failed.append(1)
                emit({"iteration": it, "snapshotError": repr(e)})

    best = ckpt.get("best", -1.0) if ckpt else -1.0
    # --minutes: stop BEFORE an iteration that (with the slowest iteration seen so far, plus the last quick
    # check's time if one is due after it) would end past the limit -- so the quick check keeps to it too.
    eval_sec = ckpt.get("evalSec", 0.0) if ckpt else 0.0
    if not ckpt:  # iteration 0: where this run (or stage) starts from; also times the quick check
        t_eval = time.time()
        rec = {"iteration": 0, **check()}
        eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
        if pools:
            rec["pool"] = {R: pool_stats(P["walls"], P["fill"], R) for R, P in pools.items()}
        best = rec["best"] = rec["score"]
        save_ckpt(os.path.join(run_dir, "best.pt"), model, opt, 0, cfg, rng, eval_sec=eval_sec)
        emit(rec)

    t_win, losses, parts, taught = time.time(), [], [], []
    kinds = dict.fromkeys(DAMAGE_KINDS, 0)
    skipped = skip_run = 0
    slowest = 0.0
    it = start_it
    while it < cfg["iters"]:
        due = (it + 1) % args.eval_every == 0 or it + 1 == cfg["iters"]
        if args.minutes and time.time() - t_start + slowest + (eval_sec if due else 0) > 60 * args.minutes:
            if it > start_it:
                save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, steering)
                snapshot(it)
            print(json.dumps({"stopped": "time", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2),
                              "slowestIterSec": round(slowest, 2), "evalSec": round(eval_sec, 1)}), flush=True)
            break
        t_it = time.time()
        R = int(rng.choice(radii))
        cs = consts[R]
        mk = cs[:, :1]
        if cfg["pool"]:
            P, st = pools[R], steering[R]
            idx = rng.choice(len(P["walls"]), B, replace=False)
            tidx = torch.from_numpy(idx)
            walls_np, fill_np, depth_np = P["walls"][idx], P["fill"][idx], P["depth"][idx]  # copies
            aux_np = P["aux"][idx] if use_aux else None
            state = P["state"][tidx].clone()
            with torch.no_grad():
                cur = per_sample_loss(state[:, 1:2], to_t(fill_np), mk).min(1).values.numpy()
            order = np.argsort(-cur)  # worst first
            # The worst sample starts over from the fresh state on its board (a new board instead, as in
            # distill, drifts the pool towards the easy boards).
            j = order[0]
            state[j] = fresh_state(to_t(walls_np[j][None]), cfg["channels"])[0]
            fresh = [j]
            # batch/8 random others get brand-new boards (twice as many while the pool is steered).
            rest = rng.permutation(order[1:])
            n_new = min(len(rest), max(1, B // 8) * st["newMult"])
            w, f, d, a = draw(rng, R, n_new, bf, use_aux)
            for k, j in enumerate(rest[:n_new]):
                walls_np[j], fill_np[j], depth_np[j] = w[k], f[k], d[k]
                if use_aux:
                    aux_np[j] = a[k]
                state[j] = fresh_state(to_t(w[k][None]), cfg["channels"])[0]
                fresh.append(j)
            P["born"][idx[fresh]], P["edits"][idx[fresh]], P["last"][idx[fresh]] = it, 0, -1
            # Each of the rest, with probability --damage, gets one damage and carries on from its state.
            for j in rest[n_new:]:
                if rng.random() >= cfg["damage"]:
                    continue
                k = int(rng.choice(len(DAMAGE_KINDS), p=st["p"]))
                kind = DAMAGE_KINDS[k]
                kinds[kind] += 1
                P["edits"][idx[j]] += 1
                P["last"][idx[j]] = k
                if kind == "state":
                    damage_state(rng, state[j], R)
                    continue
                walls_np[j] = damage_walls(rng, walls_np[j], R, kind, st["pBridge"])
                fill_np[j], depth_np[j], aux_j = answers(walls_np[j], R, use_aux)
                if use_aux:
                    aux_np[j] = aux_j
            walls, target = to_t(walls_np), to_t(fill_np)
            state[:, 0:1] = walls
        else:
            walls_np, fill_np, depth_np, aux_np = draw(rng, R, B, bf, use_aux)
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
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(prm.grad).all())
                                                    for prm in model.parameters() if prm.grad is not None)
        if finite:
            skip_run = 0
            for prm in model.parameters():  # per-parameter gradient normalisation (distill notebook)
                if prm.grad is not None:
                    prm.grad /= prm.grad.norm() + 1e-8
            for g in opt.param_groups:
                g["lr"] = lr_at(it, cfg["iters"], cfg["lr"])
            opt.step()
            model.restore_floods()  # spec v6: a no-op unless something besides the (masked) gradient moved them
            losses.append(loss.item())
            if use_aux:
                parts.append((loss_fill.item(), acc_aux.mean().item()))
        else:  # no step; the batch's boards start over from the fresh state
            skipped, skip_run = skipped + 1, skip_run + 1
            state = fresh_state(walls, cfg["channels"])
            if cfg["pool"]:
                P["born"][idx], P["edits"][idx], P["last"][idx] = it, 0, -1
        it += 1
        taught.append(n_taught / (B * max(1, T - 1)))
        slowest = max(slowest, time.time() - t_it)  # the iteration alone (no quick check, no checkpoint)

        if cfg["pool"]:
            P["state"][tidx] = state.detach()
            P["walls"][idx], P["fill"][idx], P["depth"][idx] = walls_np, fill_np, depth_np
            P["loss"][idx] = acc.detach().min(1).values.numpy()
            if use_aux:
                P["aux"][idx] = aux_np
            last[0] = (R, idx)
        if skip_run >= STOP_SKIPS:
            emit({"iteration": it, "stopped": f"{STOP_SKIPS} non-finite steps in a row", "skipped": skipped})
            raise SystemExit(f"{STOP_SKIPS} non-finite steps in a row; the last checkpoint is {ckpt_path}")

        if it % 50 == 0 or it == cfg["iters"]:
            rec = {"iteration": it, "loss": float(np.mean(losses)) if losses else None,
                   "secPerIter": (time.time() - t_win) / max(1, len(taught)),
                   "lr": lr_at(it - 1, cfg["iters"], cfg["lr"]),
                   "maxRssMB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
            if parts:
                rec["lossFill"], rec["lossAux"] = (float(x) for x in np.mean(parts, 0))
            if cfg.get("teach"):  # p at the window's last iteration, and the share of steps actually taught
                rec["teachP"], rec["taught"] = round(teach_p(it - 1, cfg), 4), round(float(np.mean(taught)), 4)
            if cfg["pool"]:
                rec["damage"] = kinds
            if skipped:
                rec["skipped"] = skipped
            if it % 200 == 0 and cfg["pool"]:  # the pools' statistics, and the steering they set
                rec["pool"] = {R: pool_stats(P["walls"], P["fill"], R) for R, P in pools.items()}
                steering = {R: steer(s) for R, s in rec["pool"].items()}
                if any(s["why"] for s in steering.values()):
                    rec["steer"] = {R: s for R, s in steering.items() if s["why"]}
            if it % args.eval_every == 0 or it == cfg["iters"]:
                t_eval = time.time()
                rec.update(check())
                eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
                if rec["score"] > best:  # training swings: keep the best quick check too
                    best = rec["best"] = rec["score"]
                    save_ckpt(os.path.join(run_dir, "best.pt"), model, opt, it, cfg, rng, eval_sec=eval_sec)
            emit(rec)
            t_win, losses, parts, taught = time.time(), [], [], []
            kinds = dict.fromkeys(DAMAGE_KINDS, 0)
        if it % 200 == 0 or it == cfg["iters"]:
            save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, steering)
            snapshot(it)
        elif args.snap_every and it % args.snap_every == 0:
            snapshot(it)
    log.close()


if __name__ == "__main__":
    main()
