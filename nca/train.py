"""Train the hex fill NCA.

The default is the PURE recipe: a plain learned NCA trained the distill.pub "Growing NCA" way, with nothing
written by hand. 16 state channels (ch0 = walls, re-imposed every step; ch1 = fill; the rest hidden), ONE
const input (the mask: the board's shape, nothing else), perception = the 7 hex taps of a masked 3x3 conv
(no pooled max/min), --hidden 96 units; no hand-written floods, no aux targets, no teacher forcing. The
loss is the fill MSE alone.

It trains from a persistent pool per radius (--pool, --pool-size boards each). Each iteration draws one
radius R from --R and a batch of --batch boards from that pool, each with the state it was left in:
  - batch/8 random ones are replaced by brand-new boards (random_walls; --bridge-frac of them kind "bridge",
    --near-tie of the bridge boards drawn near a tie; --mask-mix of them on a ragged board shape, nca/masks.py,
    the rest on the hexagon);
  - the worst of the batch (loss of its stored state) starts over from the fresh state on its (new) board;
  - each of the rest, with probability --damage, gets ONE damage (kind drawn per board):
      edit    a small wall edit (data.edit_walls)
      burst   3-20 random wall additions and deletions, mixed
      erase   every wall inside a random disc of radius 1-3 opened
      stamp   a fresh closed loop or a rim-to-rim bridge ORed onto the board
      state   distill's damage: every channel but the walls zeroed inside a random disc of radius
              1..R/2 (walls and targets unchanged)
    after a wall damage the board's targets are recomputed. Damage piles up on the pool board: nothing
    toggles back to an original.
All the board work (new boards, damage, targets, T) is made by a PRODUCER (nca/producer.py) from its own copy
of the pools' boards and its own rng (--producer: inline, or a thread / worker process running a few
iterations ahead, so a GPU need not wait for it; bit-identical whichever: selftest); only the choice of the
worst sample, which needs the model, stays on the training thread. Seeded CPU runs are reproducible (same
seed, same everything) with --schedule iters; the stream is not the one of the trainer before the producer.
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
mean(edit)) at eval_mults[0]*R. --eval-n boards per set per radius (default 32, logged as evalN; kept on
--resume unless given). The log (runs/<name>/log.jsonl, a line per 50 iterations) also has loss, s/iter, peak
RSS and the damage counts; every line has "time" (unix seconds) and "device"; a clean end adds a line
{"stopped": "time"} (the --minutes limit) or {"stopped": "done"} (--iters reached). nca.progress reads it.

COLLAPSE GUARD. Two GPU runs at a constant lr 2e-3 (runs/ga-r456, runs/gb-r456) peaked by iteration 2000-4000
and then collapsed (score 0.86 -> 0.24, later loss 2.2): the weights grew (w1 norm 17 -> 25..33) until 75-85% of
the hidden-channel cells sat at the clamp, where the gradient is zero. So at every log line (each 50 iterations)
the trainer checks: the window's loss non-finite (no finite step, or over 10% of its steps skipped) or above
--collapse-loss-x (4) times the median of the previous 10 windows since the last fresh optimiser; and at a quick
check, the best score so far >= 0.4 and the new score below --collapse-frac (0.6) of it. Either one is a
ROLLBACK: the weights reload from best.pt, a fresh optimiser (with its warm-up), the lr scale halves (it
multiplies the schedule; never below --lr-floor), every pool sample restarts from the fresh state on its current
board, a checkpoint is written, and the log gets {"iteration", "rollback": iteration, "lrScale", "rollbacks",
"reason", "restored": best.pt's iteration, "best"}. A collapse with --max-rollbacks (6) already made ends the
run cleanly: {"stopped": "collapsed"} (exit 0; nca.progress: COLLAPSED). best.pt only ever gets a model with a
higher score than every earlier one of the run (its score is kept in it, and a --resume takes the higher of
the checkpoint's and best.pt's), never one from a window whose loss tripped the guard. The lr scale, the
rollback count and the warm-up's start are kept in ckpt.pt (--resume restores them).

LR: --lr (default 5e-4: 2e-3 collapsed, see above) x 1 to 60% of --iters, x0.3 to 85%, x0.1 after, x the
rollback scale, never below --lr-floor (1e-5); and after every fresh optimiser (iteration 0 of a fresh or --init
run, a rollback) a linear warm-up over --warmup (100) iterations. The log's "lr" is the one the window used last.
--schedule time puts the two steps at 60% / 85% of the TIME BOX instead (the --minutes of the run's or stage's
first start, kept in the config; a --resume counts the minutes already used, elapsedSec in ckpt.pt): a box
that cannot reach --iters still decays (nca/cloud/startup.sh passes it). The start line's "schedule" says
which ({mode, decayAt, boxMin, usedMin} or {mode, decayAt, iters}); with time, every line has "schedFrac".
The decay then depends on the clock, so such a run is not bit-reproducible.

TIMING: every log line splits secPerIter into dataSec (the training thread's time on the board work: making
it inline, or waiting for the producer and applying its item) and modelSec (the rest), and gives prodSec, the
producer's own seconds per iteration wherever it ran.

--minutes stops before an iteration that would end past the limit, counting the slowest iteration so far
and, if one is due, the last quick check's time (kept in the checkpoint), so a chunk keeps to the limit with
its checks. Checkpoints (atomic) to runs/<name>/ckpt.pt every 200 iterations and at a stop.

--device auto (cuda if torch sees a GPU, else cpu): the model, consts, pool states and batches live there;
the board work (walls, targets, damage) stays numpy on the CPU. Checkpoints hold CPU tensors and every load
maps to the CPU, so a run moves between the Pi and a GPU with --resume or --init (it carries on correctly,
not bit-identically). On cuda: cudnn.benchmark on, TF32 off (float32 throughout). --threads defaults to
min(4, cores) (the Pi's runs so far used 2).

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
import math
import os
import resource
import time

import numpy as np
import torch

from .data import (EPS, N_AUX, WALL_DAMAGE, aux_flood, aux_targets, damage_walls, disc, pad_targets, page_loops,
                   random_walls, targets)
from .evaluate import _bridge_board, _ragged_any, boards, score, summarise
from .hexgrid import CONST_NAMES, mask as hex_mask, rim
from .model import PERCEPTIONS, HexNCA, const_stack, fresh_state, load_expanded
from .producer import (BAND, DAMAGE_KINDS, K_POOL, MARGIN, STATS_EVERY, Producer, Source, answers, data_rng,
                       draw_boards, pool_stats, steer)

EVAL_N = 32     # held-out boards per set per radius for the quick log metric
EVAL_SEED = 999 # same held-out boards whatever --seed is
DECAY_AT = (0.6, 0.85)  # the lr steps down (x0.3, then x0.1) at these shares of --iters, or of the time box
N_EDITS = 3     # the quick check's edit sequence: edits per board,
EDIT_MULT = 6   # and EDIT_MULT*R steps after each
STOP_SKIPS = 20 # non-finite steps in a row that stop the run
LR = 5e-4      # --lr default: 2e-3 collapsed on the GPU (module docstring)
COLLAPSE_MIN_BEST = 0.4  # the score rule of the collapse guard needs a best this good
COLLAPSE_HISTORY = 10    # the loss rule compares with the median of this many earlier windows
SNAP_N, SNAP_M = 48, 6  # pool.npz shows the first SNAP_N slots of each pool, the full state of the first SNAP_M


def to_t(a, device=None):
    """numpy [B,S,S] -> float tensor [B,1,S,S]; [B,K,S,S] -> [B,K,S,S]; on `device` if given (on the CPU, .to
    returns the tensor itself: no copy)."""
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    t = t.unsqueeze(1) if t.dim() == 3 else t
    return t if device is None else t.to(device)


def pick_device(name):
    """torch.device for --device: auto = cuda if torch sees a GPU, else cpu; cuda without one is an error."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: torch sees no CUDA device")
    return torch.device(name)


def to_cpu(x):
    """x with every tensor in it (nested dicts, lists, tuples) on the CPU: what checkpoints hold, so one saved
    on a GPU loads on the Pi and vice versa. On the CPU a tensor is returned as it is (no copy)."""
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, dict):
        y = type(x)((k, to_cpu(v)) for k, v in x.items())
        if hasattr(x, "_metadata"):  # a state_dict's module versions
            y._metadata = x._metadata
        return y
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def model_to(model, device):
    """model on `device`, with spec v6's frozen masks and values (plain attributes, not buffers) too."""
    model.to(device)
    if getattr(model, "frozen", None):
        model.frozen = {k: v.to(device) for k, v in model.frozen.items()}
        model.frozen_values = {k: v.to(device) for k, v in model.frozen_values.items()}
    return model


def draw(rng, R, n, bridge_frac=0.0, aux=False, mask_mix=0.0, near_tie=0.0):
    """(walls [n,S,S], fills [n,K_POOL,S,S], depth [n,S,S], aux [n,5,S,S] or None) of n fresh boards from the
    training mix (producer.draw_boards without the masks; with mask_mix > 0 use that for the masks too)."""
    return draw_boards(rng, R, n, bridge_frac, aux, mask_mix, near_tie)[:4]


def damage_state(rng, state, R, board=None):
    """distill's damage, in place on ONE board's state [C,S,S]: every channel but ch0 (the walls) zeroed inside
    a disc of radius 1..max(1, R//2) round a random on-board cell (board: its mask, None = the hexagon).
    Returns the disc (bool [S,S]). (A pool run gets these from the producer: zero_disc.)"""
    cells = np.argwhere((hex_mask(R) if board is None else board) == 1)
    row, col = cells[rng.integers(len(cells))]
    return zero_disc(state, R, row, col, int(rng.integers(1, max(1, R // 2) + 1)))


def zero_disc(state, R, row, col, rad):
    """Every channel but ch0 of ONE board's state [C,S,S] zeroed within hex distance rad of (row, col)."""
    d = disc(R, row, col, rad)
    state[1:, torch.from_numpy(d).to(state.device)] = 0
    return d


def per_sample_loss(fill, fills, mk):
    """[B,K] mean squared error of fill [B,1,S,S] against each target fills [B,K,S,S], on-board cells (mk:
    [1,1,S,S] the hexagon, or [B,1,S,S] a mask per board; each board's error is over its own cells)."""
    return (((fill - fills) ** 2) * mk).flatten(2).sum(2) / mk.flatten(1).sum(1, keepdim=True)


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


def heldout(radii, consts, n=EVAL_N, device=None, ragged=None):
    """{set: {R: (walls, fills, sides, consts)}}: fixed boards for the quick check, n per set per radius, mix,
    bridge and page (the demo page's random loops, data.page_loops: never trained on), and (ragged, default:
    when the consts are the mask alone) "ragged": boards on ragged masks (nca/masks.py), half bridge boards
    with 2+ rim regions, half the training mix (evaluate._ragged_any). consts = {R: const stack [1,n,S,S]}
    (mask first); the ragged set's consts are its masks, [n,1,S,S]. Their aux floods are recomputed per check
    (aux_flood). A larger n keeps the first EVAL_N boards of each set (the draws come in order) and adds more."""
    if ragged is None:
        ragged = all(c.shape[1] == 1 for c in consts.values())
    sets = [("mix", random_walls, EVAL_SEED), ("bridge", _bridge_board, EVAL_SEED + 50),
            ("page", page_loops, EVAL_SEED + 100)] + ([("ragged", _ragged_any, EVAL_SEED + 200)] if ragged else [])
    out = {name: {} for name, _, _ in sets}
    for R in radii:
        for name, gen, seed in sets:
            w, f, sides, ms = boards(np.random.default_rng(seed + R), R, n, gen)
            cs = consts[R] if ms is None else to_t(ms, device)
            out[name][R] = (to_t(w, device), (torch.from_numpy(f) > 0).to(device or "cpu"), sides, cs)
    return out


def edit_sequence(R, cs, n=EVAL_N, seed=EVAL_SEED + 150, device=None):
    """The quick check's fixed edit-sequence boards at radius R: n boards of the training mix, each damaged
    N_EDITS times in succession (kinds edit, burst, erase, stamp, uniformly; data.damage_walls).
    (stages: N_EDITS + 1 walls [n,1,S,S] (the board, then after each edit), fills: N_EDITS bool [n,K,S,S]
    (each edited board's targets), cs, changed: N_EDITS shares where the primary target changed,
    hold: N_EDITS shares where the previous primary is still acceptable -- what a model that ignores the
    edit would score)."""
    rng = np.random.default_rng(seed + R)
    w = np.stack([random_walls(rng, R) for _ in range(n)])
    stages, fills, changed, hold = [to_t(w, device)], [], [], []
    prev = np.stack([targets(x, R)[0][0] for x in w])
    for _ in range(N_EDITS):
        w = np.stack([damage_walls(rng, x, R, WALL_DAMAGE[int(rng.integers(len(WALL_DAMAGE)))]) for x in w])
        f = pad_targets([targets(x, R)[0] for x in w])
        changed.append(float(np.mean([(f[i, 0] != prev[i]).any() for i in range(n)])))
        hold.append(float(np.mean([(f[i] == prev[i]).all((1, 2)).any() for i in range(n)])))
        stages.append(to_t(w, device))
        fills.append((torch.from_numpy(f) > 0).to(device or "cpu"))
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
            w = walls[:, 0].cpu().numpy().astype(np.uint8)
            F = torch.from_numpy(aux_flood(w, R, max(mults) * R)).to(cs.device)
            blind = torch.from_numpy(aux_flood(np.zeros_like(w), R, max(mults) * R)).to(cs.device)
            pad = torch.zeros(len(w), 2, *w.shape[-2:], device=cs.device)
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
    if "ragged" in held:
        for r in res.values():
            r["raggedByR"] = {}
    for name, per_r in held.items():
        for R, (walls, fills, sides, cs) in per_r.items():
            mk = cs[:, :1]
            if aux or teach_check:
                F = torch.from_numpy(aux_flood(walls[:, 0].cpu().numpy().astype(np.uint8), R,
                                               max(mults) * R)).to(walls.device)
            if teach_check:
                state, done, per_ch = fresh_state(walls, model.channels), 0, []
                for m in sorted(mults):
                    for t in range(done, m * R):
                        state = model.step(state, walls, cs)
                        per_ch.append(aux_loss_ch(state, F[t], mk))
                        state = teach(state, F[t], torch.ones(len(state), dtype=torch.bool, device=state.device))
                    done = m * R
                    ch = torch.stack(per_ch).mean(0)
                    res[m].setdefault("auxTeach", []).append(float(ch.mean()))
                    res[m].setdefault("auxTeachCh", []).append(ch.cpu().numpy())
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
                    p, prim = state[:, 1].cpu().numpy(), fills[:, 0].cpu().numpy()
                    res[m]["gap"].append(np.mean([p[b][np.isin(lab, ids) & prim[b]].mean()
                                                  - p[b][np.isin(lab, ids) & ~prim[b]].mean()
                                                  for b, (lab, ids, _) in enumerate(sides)]))
                    res[m]["bridgeByR"][R] = round(float(s["exact"]), 3)
                if name == "ragged":
                    res[m]["raggedByR"][R] = round(float(s["exact"]), 3)
    out = {m: {k: (round(float(np.mean(v)), 4) if isinstance(v, list) and k != "auxTeachCh" else v)
               for k, v in r.items()} for m, r in res.items()}
    for r in out.values():
        if "auxTeachCh" in r:
            r["auxTeachCh"] = [round(float(x), 4) for x in np.mean(r["auxTeachCh"], 0)]
    return out


def lr_at(it: int, iters: int, lr: float, scale: float = 1.0, floor: float = 0.0, warm_from=None,
          warmup: int = 0, frac: float = None) -> float:
    """Step decay: full lr to 60%, x0.3 to 85%, x0.1 after (DECAY_AT) -- of it / iters, or of `frac` when given
    (--schedule time: the share of the time box used); times `scale` (the collapse guard halves it at each
    rollback), never below `floor`; then, `warmup` iterations from `warm_from` (a fresh optimiser; None = none),
    a linear warm-up: x (it - warm_from + 1) / warmup."""
    f = it / max(1, iters) if frac is None else frac
    x = max(floor, lr * scale * (1.0 if f < DECAY_AT[0] else 0.3 if f < DECAY_AT[1] else 0.1))
    if warm_from is not None and warmup > 0 and 0 <= it - warm_from < warmup:
        x *= (it - warm_from + 1) / warmup
    return x


def collapse_reason(score, best, frac, loss, history, skipped=0, steps=1, loss_x=4.0):
    """Why the run counts as collapsed (a short string), or None. score: this log line's quick-check score
    (None if there was none); best: the best score so far; loss: the window's mean loss over its finite steps
    (None if it had none); history: the earlier windows' losses since the last fresh optimiser; skipped / steps:
    the window's skipped (non-finite) and total steps."""
    if loss is None or not math.isfinite(loss) or skipped > 0.1 * steps:
        return f"loss not finite ({skipped} of {steps} steps skipped)"
    if len(history) >= COLLAPSE_HISTORY:
        med = float(np.median(history[-COLLAPSE_HISTORY:]))
        if loss > loss_x * med:
            return f"loss {loss:.4g} > {loss_x:g}x the median {med:.4g} of the previous {COLLAPSE_HISTORY} windows"
    if score is not None and best >= COLLAPSE_MIN_BEST and score < frac * best:
        return f"score {score:.4f} < {frac:g} x best {best:.4f}"
    return None


def best_score(run_dir):
    """(score, iteration) of the model in runs/<name>/best.pt, or None without a readable one. The score is the
    file's "best"; files from before the collapse guard have -1 there: then the log's last "best" at that
    iteration (None if the log has none)."""
    path = os.path.join(run_dir, "best.pt")
    try:
        b = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 -- missing or unreadable: no best to keep
        return None
    score, it = b.get("best", -1.0), b.get("iteration", 0)
    if score is None or score < 0:
        score = None
        try:
            with open(os.path.join(run_dir, "log.jsonl")) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("iteration") == it and isinstance(rec.get("best"), (int, float)):
                        score = float(rec["best"])
        except OSError:
            pass
    return None if score is None else (float(score), it)


def save_ckpt(path, model, opt, it, cfg, rng, pools=None, best=-1.0, eval_sec=0.0, steering=None, guard=None):
    """Atomically (a temp file, then a rename): weights, optimiser, iteration, config, rng, pools, best score
    (for best.pt: its own score), the last quick check's seconds, the pools' steering and the collapse guard's
    state (guard = {lrScale, rollbacks, warmFrom}). Every tensor goes in on the CPU (to_cpu), so the file loads
    on any device."""
    tmp = path + ".tmp"
    torch.save({"model": to_cpu(model.state_dict()), "opt": to_cpu(opt.state_dict()), "iteration": it,
                "config": cfg, "rng": rng.bit_generator.state, "pool": to_cpu(pools), "best": best,
                "evalSec": eval_sec, "steer": steering, **(guard or {})}, tmp)
    os.replace(tmp, path)


def new_pool(rng, R, n, channels, bridge_frac, aux=False, device=None, mask_mix=0.0, near_tie=0.0):
    """A pool of n fresh boards at radius R: walls, targets, depth, mask (uint8 [n,S,S]: each board's shape, the
    hexagon or, with probability mask_mix, a ragged one), aux targets (only with aux), state (a tensor on
    `device`; the rest numpy)."""
    w, f, d, a, m = draw_boards(rng, R, n, bridge_frac, aux, mask_mix, near_tie)
    pool = {"walls": w, "fill": f, "depth": d, "mask": m, "state": fresh_state(to_t(w, device), channels)}
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
    mask_R uint8 [n,S,S] (each board's shape: the hexagon or a ragged mask; pools from before masks: none),
    age_R int32 [n] (iterations since its last fresh start), edits_R int32 [n] (damages since then), damage_R
    int8 [n] (the last one's kind: -1 none, then DAMAGE_KINDS' order: edit, burst, erase, stamp, state), and
    state_R float16 [m,C,S,S], the full state of the first m = min(n, SNAP_M)."""
    out = {"iteration": np.int64(it), "radii": np.array(sorted(pools), dtype=np.int64),
           "last_R": np.int64(last_R), "last_idx": np.asarray(last_idx, dtype=np.int64)}
    for R, P in pools.items():
        n = min(len(P["walls"]), SNAP_N)
        on = P["mask"][:n] == 1 if "mask" in P else hex_mask(R)[None] == 1
        fill = P["state"][:n, 1].cpu().numpy()
        fills = P["fill"][:n]  # [n,K,S,S]
        wrong = (((fill > 0.5) & on)[:, None] != (fills > 0)).sum((2, 3))  # [n,K]; off board both are 0
        out[f"walls_{R}"] = P["walls"][:n].astype(np.uint8)
        if "mask" in P:
            out[f"mask_{R}"] = P["mask"][:n].astype(np.uint8)
        out[f"fill_{R}"] = fill.astype(np.float16)
        out[f"target_{R}"] = fills[np.arange(n), wrong.argmin(1)].astype(np.uint8)  # argmin: the first of a tie
        out[f"ntargets_{R}"] = np.array([len({t.tobytes() for t in f}) for f in fills], dtype=np.uint8)
        out[f"loss_{R}"] = P["loss"][:n].astype(np.float32)
        out[f"age_{R}"] = (it - P["born"][:n]).astype(np.int32)
        out[f"edits_{R}"] = P["edits"][:n].astype(np.int32)
        out[f"damage_{R}"] = P["last"][:n].astype(np.int8)
        out[f"state_{R}"] = P["state"][:min(n, SNAP_M)].cpu().numpy().astype(np.float16)
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
    p.add_argument("--lr", type=float, default=None,
                   help=f"peak learning rate (default {LR}; 2e-3 collapsed); with --resume it replaces the run's")
    p.add_argument("--lr-floor", type=float, default=1e-5, help="the lr never goes below this (after decay and "
                   "rollbacks; the warm-up still starts below it)")
    p.add_argument("--warmup", type=int, default=100,
                   help="linear lr warm-up over this many iterations after every fresh optimiser (0 = none)")
    p.add_argument("--collapse-frac", type=float, default=0.6,
                   help="collapse guard: roll back when a quick check scores below this share of the best so "
                        f"far (once the best is >= {COLLAPSE_MIN_BEST})")
    p.add_argument("--collapse-loss-x", type=float, default=4.0,
                   help="collapse guard: roll back when a log window's loss is above this many times the median "
                        f"of the {COLLAPSE_HISTORY} windows before it")
    p.add_argument("--max-rollbacks", type=int, default=6,
                   help="a collapse after this many rollbacks ends the run: {\"stopped\": \"collapsed\"}")
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
    p.add_argument("--mask-mix", type=float, default=None,
                   help="share of new pool boards drawn on a ragged mask (nca/masks.py: blobs, crops of Spectacle's "
                        "hex fields) instead of the hexagon (default 0.5; needs --n-consts 1; with --resume: the run's "
                        "unless given)")
    p.add_argument("--near-tie", type=float, default=None,
                   help="share of new bridge boards drawn near a tie: bridges near half way round, redrawn until the "
                        "two largest rim regions' area ratio is >= 0.75 (data.random_walls; default 0)")
    p.add_argument("--schedule", choices=("iters", "time"), default=None,
                   help="where the lr decays (x0.3 at 60%%, x0.1 at 85%%): of --iters (default), or of the time box "
                        "(time: --minutes at the start of the run or stage, kept by --resume, which counts the time "
                        "already used; not bit-reproducible, since the decay depends on the clock)")
    p.add_argument("--producer", choices=("auto", "inline", "thread", "process"), default="auto",
                   help="pool runs: who makes the boards, damage and targets (nca/producer.py): inline on the "
                        "training thread, a thread, or a worker process; auto = process on cuda, inline on cpu. "
                        "Bit-identical results whichever")
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
    p.add_argument("--threads", type=int, default=None,
                   help="torch threads (default min(4, cores); the Pi's runs used 2)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="auto = cuda if torch sees a GPU, else cpu. Checkpoints hold CPU tensors either way, so "
                        "a run moves between the Pi and a GPU with --resume / --init (not bit-identically)")
    p.add_argument("--eval-n", type=int, default=None,
                   help=f"held-out boards per set per radius in the quick check (default {EVAL_N}, or what the "
                        "resumed run used); a GPU run can afford more, which makes the check less noisy")
    p.add_argument("--ckpt-every", type=int, default=200,
                   help="write ckpt.pt every this many iterations (and at the end, and on a --minutes stop); "
                        "a fast GPU run wants a larger number, since a checkpoint carries the pools")
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
    if args.schedule == "time" and not args.minutes and not args.resume:
        p.error("--schedule time needs --minutes (the time box the lr decay is placed in)")

    torch.set_num_threads(args.threads or min(4, os.cpu_count() or 1))
    device = pick_device(args.device)
    if device.type == "cuda":  # fixed shapes per radius: let cudnn pick its kernels; true float32, no TF32
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    start_it = 0
    ckpt = init = None
    if args.resume:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        start_it = ckpt["iteration"]
        p_now = teach_p(start_it, cfg)  # where the anneal stopped, on the schedule it was saved with
        cfg["iters"] = args.iters or cfg["iters"]
        if cfg.get("teach") and args.teach:
            cfg["teach"] = [cfg["teach"][0], args.teach[1]]
        if cfg.get("teach") and abs(teach_p(start_it, cfg) - p_now) > 1e-12:  # new --iters / P1: go on from here
            cfg["teachFrom"] = [start_it, p_now]
        if args.lr:  # a new peak lr for the rest of the run (the schedule and the rollback scale still apply)
            cfg["lr"] = args.lr
        cfg.setdefault("maskMix", 0.0)  # runs from before ragged masks: hexagons only
        cfg.setdefault("nearTie", 0.0)
        cfg.setdefault("schedule", "iters")
        if args.mask_mix is not None:
            cfg["maskMix"] = args.mask_mix
        if args.near_tie is not None:
            cfg["nearTie"] = args.near_tie
        if args.schedule and args.schedule != cfg["schedule"]:  # a new schedule mode from here on
            cfg["schedule"] = args.schedule
            if args.schedule == "time" and not cfg.get("scheduleMin"):
                if not args.minutes:
                    p.error("--schedule time needs --minutes")
                cfg["scheduleMin"] = args.minutes
        for k, v in (("lrFloor", 0.0), ("warmup", 0), ("collapseFrac", args.collapse_frac),
                     ("collapseLossX", args.collapse_loss_x), ("maxRollbacks", args.max_rollbacks)):
            cfg.setdefault(k, v)  # older checkpoints: no floor, no warm-up (the optimiser carries on), the guard on
        if cfg.get("pool") and "damage" not in cfg:
            p.error(f"{ckpt_path} is from the old pool (edits toggled back to the board as first drawn); "
                    "start a new run, or --init from it")
    else:
        if args.init:
            init = torch.load(args.init, map_location="cpu", weights_only=False)
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
            "lr": args.lr or LR, "lrFloor": args.lr_floor, "warmup": args.warmup, "collapseFrac": args.collapse_frac,
            "collapseLossX": args.collapse_loss_x, "maxRollbacks": args.max_rollbacks, "batch": args.batch, "iters": args.iters or 4000, "seed": args.seed,
            "bridgeFrac": args.bridge_frac, "pool": args.pool, "poolSize": args.pool_size,
            "maskMix": 0.5 if args.mask_mix is None else args.mask_mix,
            "nearTie": args.near_tie or 0.0,
            "schedule": args.schedule or "iters", "scheduleMin": args.minutes if args.schedule == "time" else None,
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
    if cfg.get("maskMix", 0) > 0 and (n_consts != 1 or cfg.get("aux")):
        if args.mask_mix:  # asked for: refuse
            p.error("--mask-mix needs --n-consts 1 and no --aux (ragged boards have no theta/aux planes)")
        cfg["maskMix"] = 0.0  # the default: the hexagon only for these recipes
    perception = cfg.setdefault("perception", "taps")  # and from before v4 no pool
    floods = cfg.get("floods", False)  # spec v6
    eval_n = cfg["evalN"] = args.eval_n or cfg.get("evalN", EVAL_N)  # runs from before --eval-n used EVAL_N
    torch.manual_seed(cfg["seed"])  # the init of a fresh model, on the CPU whatever the device (a loaded one overwrites it)
    model = HexNCA(cfg["channels"], cfg["hidden"], cfg["clamp"], cfg["fireRate"], n_consts, perception, floods)
    rng = np.random.default_rng(cfg["seed"])
    radii, B, last_k, bf = cfg["R"], cfg["batch"], cfg["lastK"], cfg.get("bridgeFrac", 0.0)
    use_aux, aux_w, dense = cfg.get("aux", False), cfg.get("auxW", 1.0), cfg.get("auxDense", False)
    consts = {R: const_stack(R, n_consts) for R in radii}  # [1,n,S,S] each, built once per radius
    if ckpt:
        model.load_state_dict(ckpt["model"])
        rng.bit_generator.state = ckpt["rng"]
        assert not floods or all(torch.equal(getattr(model, k)[f], model.frozen_values[k][f])
                                 for k, f in model.frozen.items()), "the checkpoint's hand-written floods changed"
    elif init:
        load_expanded(model, init["model"])
        if init_consts < n_consts or "w1pool" not in init["model"] and perception == "taps+pool":
            print(json.dumps({"initExpansion": expansion_check(model, init, consts[radii[-1]])}), flush=True)
    # Everything the model touches lives on the device from here; the board work stays numpy on the CPU. The
    # optimiser comes after the move (Adam has no randomness), and loading its state moves that state too.
    model_to(model, device)
    consts = {R: c.to(device) for R, c in consts.items()}
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if ckpt:
        opt.load_state_dict(ckpt["opt"])
    # The collapse guard's state (module docstring): kept in ckpt.pt; a fresh or --init run warms up from 0.
    lr_scale = ckpt.get("lrScale", 1.0) if ckpt else 1.0
    rollbacks = ckpt.get("rollbacks", 0) if ckpt else 0
    warm_from = ckpt.get("warmFrom") if ckpt else 0
    guard = lambda: {"lrScale": lr_scale, "rollbacks": rollbacks, "warmFrom": warm_from, "elapsedSec": elapsed(),
                     "dataRng": data_state[0]}
    # --schedule time: the share of the time box used, counted across --resume (elapsedSec in ckpt.pt).
    prior_sec = ckpt.get("elapsedSec", 0.0) if ckpt else 0.0
    elapsed = lambda: round(prior_sec + time.time() - t_start, 1)
    sched_frac = lambda: (elapsed() / (60.0 * cfg["scheduleMin"]) if cfg.get("schedule") == "time"
                          and cfg.get("scheduleMin") else None)
    lr_of = lambda i: lr_at(i, cfg["iters"], cfg["lr"], lr_scale, cfg.get("lrFloor", 0.0), warm_from,
                            cfg.get("warmup", 0), sched_frac())

    held = heldout(radii, consts, eval_n, device, ragged=n_consts == 1 and not use_aux)
    eseq = {R: edit_sequence(R, consts[R], eval_n, device=device) for R in radii}
    m0 = args.eval_mults[0]

    def check():
        """The quick check: {q<m>: ..., edit: [...], editByR: {...}, score} (see the module docstring)."""
        rec = {f"q{m}": q for m, q in
               quick_eval(model, held, args.eval_mults, aux=use_aux, arc=use_aux,
                          teach_check=use_aux and dense and bool(cfg.get("teach"))).items()}
        rec.update(edit_eval(model, eseq, m0))
        q = rec[f"q{m0}"]
        parts = [q["mix"], q["bridge"], q["page"], float(np.mean(rec["edit"]))]
        if cfg.get("maskMix", 0) > 0 and "ragged" in q:  # a run that trains on ragged boards is judged on them too
            parts.append(q["ragged"])
        rec["score"] = round(sum(parts) / len(parts), 4)
        rec["scoreParts"] = len(parts)
        rec["evalN"] = eval_n  # boards per set per radius behind these shares (nca.progress's noise)
        return rec

    start = {"config": cfg, "startIteration": start_it,
             "editChanged": [round(float(np.mean([e[3][k] for e in eseq.values()])), 3) for k in range(N_EDITS)],
             "editHold": [round(float(np.mean([e[4][k] for e in eseq.values()])), 3) for k in range(N_EDITS)],
             "evalN": eval_n, "threads": torch.get_num_threads(), "torch": torch.__version__,
             "lrScale": lr_scale, "rollbacks": rollbacks,
             "schedule": {"mode": cfg.get("schedule", "iters"), "decayAt": list(DECAY_AT),
                          **({"boxMin": cfg["scheduleMin"], "usedMin": round(prior_sec / 60, 2)}
                             if cfg.get("schedule") == "time" else {"iters": cfg["iters"]})}}
    if device.type == "cuda":
        start["gpu"] = torch.cuda.get_device_name(device)
    if use_aux:
        start["trivialAux"] = {f"q{m}": q for m, q in trivial_aux(held, args.eval_mults).items()}
    if floods and not ckpt:  # the comparative rule read off the exact frozen floods: the readout's ceiling
        start["arcCeiling"] = {f"q{m}": {k: v for k, v in q.items() if k.startswith("arc")}
                               for m, q in quick_eval(model, held, args.eval_mults).items()}
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    def emit(rec):
        """A log line (and the same on stdout); every one says when (unix seconds) and on what device."""
        rec["time"], rec["device"] = round(time.time(), 2), device.type
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


    pools, steering, source = None, {}, None
    mask_mix, near_tie = cfg.get("maskMix", 0.0), cfg.get("nearTie", 0.0)
    if cfg["pool"]:
        pools = ckpt["pool"] if ckpt and ckpt.get("pool") else \
            {R: new_pool(rng, R, cfg["poolSize"], cfg["channels"], bf, use_aux, device, mask_mix, near_tie)
             for R in radii}
        for R, P in pools.items():  # aux targets for a pool that lacks them (or has v3's 4 planes)
            P["state"] = P["state"].to(device)
            if use_aux and ("aux" not in P or P["aux"].shape[1] != N_AUX):
                P["aux"] = np.stack([aux_targets(w, R) for w in P["walls"]])
            if "mask" not in P:  # a pool from before ragged masks: every board is the hexagon
                P["mask"] = np.repeat(hex_mask(R)[None], len(P["walls"]), 0).astype(np.uint8)
            bookkeeping(P, start_it)
        steering = (ckpt or {}).get("steer") or {R: steer(pool_stats(P["walls"], P["fill"], R, P["mask"]))
                                                 for R, P in pools.items()}
        # The board work runs in a producer (nca/producer.py) with its own rng, continued from the checkpoint.
        if ckpt and ckpt.get("dataRng"):
            d_state = ckpt["dataRng"]
        else:
            d_state = data_rng(cfg["seed"], start_it).bit_generator.state
        mode = args.producer if args.producer != "auto" else ("process" if device.type == "cuda" else "inline")
        spec = {"R": radii, "batch": B, "bridgeFrac": bf, "damage": cfg["damage"], "maskMix": mask_mix,
                "nearTie": near_tie, "aux": use_aux, "steps": cfg["steps"]}
        board_keys = ("walls", "fill", "depth", "mask") + (("aux",) if use_aux else ())
        source = Source(Producer(spec, {R: {k: P[k] for k in board_keys} for R, P in pools.items()}, d_state,
                                 steering, start_it), mode)
        start["producer"] = mode
    data_state = [d_state if cfg["pool"] else None]  # the producer's rng after the last item applied
    emit(start)  # at every (re)start: the config; its "iters" is the target (a --resume may raise it)

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
    best_path = os.path.join(run_dir, "best.pt")
    on_disk = best_score(run_dir) if ckpt else None
    if on_disk and on_disk[0] > best:  # best.pt can be newer than ckpt.pt: never overwrite it with a worse model
        best = on_disk[0]
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
        save_ckpt(best_path, model, opt, 0, cfg, rng, best=best, eval_sec=eval_sec)
        emit(rec)

    t_win, losses, parts, taught = time.time(), [], [], []
    t_data, t_prod = [], []  # per iteration: the training thread's seconds on data, the producer's own
    pool_rec = None  # the pools' statistics of the last item that carried them (logged at the next line)
    kinds = dict.fromkeys(DAMAGE_KINDS, 0)
    skipped = skip_run = 0
    win_skipped, win_steps, history = 0, 0, []  # the collapse guard's loss windows (since the last fresh optimiser)
    collapsed = False
    slowest = 0.0
    it = start_it
    while it < cfg["iters"]:
        due = (it + 1) % args.eval_every == 0 or it + 1 == cfg["iters"]
        if args.minutes and time.time() - t_start + slowest + (eval_sec if due else 0) > 60 * args.minutes:
            if it > start_it:
                save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, steering, guard())
                snapshot(it)
            # a clean end of this chunk (or time-boxed stage), in the log too: nca.progress reads it as FINISHED
            emit({"stopped": "time", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2),
                  "slowestIterSec": round(slowest, 2), "evalSec": round(eval_sec, 1)})
            break
        t_it = time.time()
        if cfg["pool"]:
            # The board work of this iteration, made by the producer (nca/producer.py): R, the batch's slots,
            # their boards after this iteration's new boards and damage, and T.
            item = source.get()
            R, idx = item["R"], item["idx"]
            P = pools[R]
            tidx = torch.from_numpy(idx).to(device)
            state = P["state"][tidx].clone()
            # The board shapes: per board on a ragged run (consts = the masks), else the radius's const stack.
            disc_R = hex_mask(R)[None]
            ragged_run = mask_mix > 0 or bool((P["mask"][idx] != disc_R).any() or (item["mask"] != disc_R).any())
            cs_old = to_t(P["mask"][idx], device) if ragged_run else consts[R]
            with torch.no_grad():  # each sample's loss on the board it was left on
                cur = per_sample_loss(state[:, 1:2], to_t(P["fill"][idx], device), cs_old[:, :1]).min(1).values
            # The worst sample starts over from the fresh state on its board (a new board instead, as in
            # distill, drifts the pool towards the easy boards); so do the slots given a brand-new board.
            j = int(cur.argmax())
            walls_np, fill_np, depth_np, mask_np = item["walls"], item["fill"], item["depth"], item["mask"]
            aux_np = item["aux"]
            walls = to_t(walls_np, device)
            fresh = sorted(set(item["new"]) | {j})
            for pos, k in item["damaged"]:  # one damage each; the state carries on (wall damage: new walls)
                if pos in fresh:
                    continue
                P["edits"][idx[pos]] += 1
                P["last"][idx[pos]] = k
            for pos, row, col, rad in item["stateDamage"]:
                if pos not in fresh:
                    zero_disc(state[pos], R, row, col, rad)
            for k, v in item["kinds"].items():
                kinds[k] += v
            fi = torch.tensor(fresh, device=device)
            state[fi] = fresh_state(walls[fi], cfg["channels"])
            P["born"][idx[fresh]], P["edits"][idx[fresh]], P["last"][idx[fresh]] = it, 0, -1
            target = to_t(fill_np, device)
            state[:, 0:1] = walls
            cs = to_t(mask_np, device) if ragged_run else consts[R]
            T = item["T"]
            t_prod.append(item["sec"])
            if "pool" in item:
                pool_rec, steering = item["pool"], item["steer"]
        else:
            R = int(rng.choice(radii))
            cs = consts[R]
            t_prod.append(0.0)
            walls_np, fill_np, depth_np, aux_np = draw(rng, R, B, bf, use_aux)
            walls, target = to_t(walls_np, device), to_t(fill_np, device)
            state = fresh_state(walls, cfg["channels"])
            T = n_steps(R, depth_np.max())
        mk = cs[:, :1]
        t_data.append(time.time() - t_it)
        if use_aux and dense:  # [T,B,5,S,S]: the reference flood at every step from the fresh state
            aux_t = torch.from_numpy(aux_flood(walls_np, R, T)).to(device)
        elif use_aux:
            aux_t = torch.from_numpy(np.ascontiguousarray(aux_np)).to(device)
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
                coin = torch.from_numpy(rng.random(B) < p_teach).to(device)
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
                g["lr"] = lr_of(it)
            opt.step()
            model.restore_floods()  # spec v6: a no-op unless something besides the (masked) gradient moved them
            losses.append(loss.item())
            if use_aux:
                parts.append((loss_fill.item(), acc_aux.mean().item()))
        else:  # no step; the batch's boards start over from the fresh state
            skipped, skip_run, win_skipped = skipped + 1, skip_run + 1, win_skipped + 1
            state = fresh_state(walls, cfg["channels"])
            if cfg["pool"]:
                P["born"][idx], P["edits"][idx], P["last"][idx] = it, 0, -1
        it += 1
        win_steps += 1
        taught.append(n_taught / (B * max(1, T - 1)))
        slowest = max(slowest, time.time() - t_it)  # the iteration alone (no quick check, no checkpoint)

        if cfg["pool"]:
            P["state"][tidx] = state.detach()
            P["walls"][idx], P["fill"][idx], P["depth"][idx], P["mask"][idx] = walls_np, fill_np, depth_np, mask_np
            P["loss"][idx] = acc.detach().min(1).values.cpu().numpy()
            if use_aux:
                P["aux"][idx] = aux_np
            last[0] = (R, idx)
            data_state[0] = item["rng"]
        if skip_run >= STOP_SKIPS:
            emit({"iteration": it, "stopped": f"{STOP_SKIPS} non-finite steps in a row", "skipped": skipped})
            raise SystemExit(f"{STOP_SKIPS} non-finite steps in a row; the last checkpoint is {ckpt_path}")

        if it % 50 == 0 or it == cfg["iters"]:
            spi = (time.time() - t_win) / max(1, len(taught))
            rec = {"iteration": it, "loss": float(np.mean(losses)) if losses else None,
                   "secPerIter": spi,
                   # secPerIter split: the training thread's seconds on data per iteration (inline: making
                   # it; thread/process: waiting for it, plus applying it), the rest (model: the rollout,
                   # backward, step, bookkeeping), and the producer's own seconds per item (wherever it ran)
                   "dataSec": round(float(np.mean(t_data)), 5) if t_data else None,
                   "modelSec": round(spi - float(np.mean(t_data)), 5) if t_data else None,
                   "prodSec": round(float(np.mean(t_prod)), 5) if t_prod else None,
                   "lr": lr_of(it - 1),
                   "maxRssMB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
            if parts:
                rec["lossFill"], rec["lossAux"] = (float(x) for x in np.mean(parts, 0))
            if cfg.get("schedule") == "time":
                rec["schedFrac"] = round(sched_frac(), 4)
            if cfg.get("teach"):  # p at the window's last iteration, and the share of steps actually taught
                rec["teachP"], rec["taught"] = round(teach_p(it - 1, cfg), 4), round(float(np.mean(taught)), 4)
            if cfg["pool"]:
                rec["damage"] = kinds
            if skipped:
                rec["skipped"] = skipped
            if pool_rec is not None:  # the pools' statistics (every STATS_EVERY), and the steering they set
                rec["pool"], pool_rec = pool_rec, None
                if any(s["why"] for s in steering.values()):
                    rec["steer"] = {R: s for R, s in steering.items() if s["why"]}
            # The collapse guard (module docstring): the loss rule first, so a window that trips it never
            # makes a best.pt; then the score rule at a quick check.
            why = collapse_reason(None, best, cfg["collapseFrac"], rec["loss"], history, win_skipped, win_steps,
                                  cfg["collapseLossX"])
            if it % args.eval_every == 0 or it == cfg["iters"]:
                t_eval = time.time()
                rec.update(check())
                eval_sec = rec["evalSec"] = round(time.time() - t_eval, 1)
                if rec["score"] > best and not why:  # training swings: keep the best quick check too
                    best = rec["best"] = rec["score"]
                    save_ckpt(best_path, model, opt, it, cfg, rng, best=best, eval_sec=eval_sec)
                why = why or collapse_reason(rec["score"], best, cfg["collapseFrac"], 0.0, [], 0, 1)
            emit(rec)
            if rec["loss"] is not None and math.isfinite(rec["loss"]):
                history.append(rec["loss"])
            t_win, losses, parts, taught = time.time(), [], [], []
            t_data, t_prod = [], []
            kinds = dict.fromkeys(DAMAGE_KINDS, 0)
            win_skipped = win_steps = 0
            if why:
                back = best_score(run_dir)
                if rollbacks >= cfg["maxRollbacks"] or back is None:
                    save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, steering, guard())
                    emit({"iteration": it, "stopped": "collapsed", "reason": why, "rollbacks": rollbacks,
                          "lrScale": lr_scale, "best": best,
                          **({} if back else {"note": "no readable best.pt to roll back to"})})
                    collapsed = True
                    break
                # Roll back: best.pt's weights, a fresh optimiser (warming up), half the lr, fresh pool states.
                b = torch.load(best_path, map_location="cpu", weights_only=False)
                model.load_state_dict(b["model"])
                model.restore_floods()
                opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
                lr_scale, rollbacks, warm_from, history = lr_scale / 2, rollbacks + 1, it, []
                if pools:
                    for P in pools.values():
                        P["state"] = fresh_state(to_t(P["walls"], device), cfg["channels"])
                        P["born"][:], P["edits"][:], P["last"][:], P["loss"][:] = it, 0, -1, np.nan
                emit({"iteration": it, "rollback": it, "lrScale": lr_scale, "rollbacks": rollbacks, "reason": why,
                      "restored": back[1], "best": best})
                save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, steering, guard())
                snapshot(it)
                continue
        if it % max(1, args.ckpt_every) == 0 or it == cfg["iters"]:
            save_ckpt(ckpt_path, model, opt, it, cfg, rng, pools, best, eval_sec, steering, guard())
            snapshot(it)
        elif args.snap_every and it % args.snap_every == 0:
            snapshot(it)
    if it >= cfg["iters"] and not collapsed:  # the run reached its target: a clean end, in the log (nca.progress: FINISHED)
        emit({"stopped": "done", "iteration": it, "minutes": round((time.time() - t_start) / 60, 2)})
    if source:
        source.close()
    log.close()


if __name__ == "__main__":
    main()
