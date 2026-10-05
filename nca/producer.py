"""The training pool's board work, off the training thread (review item 4/5): numpy only, no torch.

A pool run's iteration needs, before the model can step: a radius R, a batch of pool slots, brand-new boards
for some of them (walls, targets, depth, mask), one damage for some others (walls damaged and targets
recomputed, or a disc of the state to zero), and the rollout length T. None of that depends on the model, so a
Producer makes it, iteration after iteration, from its own copy of the pools' boards and its own rng:

    item = {"it", "R", "idx" [B] (pool slots), "walls" [B,S,S], "fill" [B,K,S,S], "depth" [B,S,S],
            "mask" [B,S,S] (each slot's board, after this iteration's changes), "aux" [B,5,S,S] or None,
            "new": [positions given a brand-new board], "damaged": [(position, damage kind index)],
            "stateDamage": [(position, row, col, radius)], "T", "kinds": {kind: count},
            "pool"/"steer" (every STATS_EVERY items: the pools' statistics and the steering they set),
            "rng": the producer's rng state after this item, "sec": seconds it took}

The trainer applies the items in order (train.py): its copy of the boards and the producer's stay equal, and
the worst sample of each batch (the one thing that needs the model) restarts from the fresh state on its
board after the item is applied. A checkpoint stores the boards and the "rng" of the last item applied, so a
--resume rebuilds the producer exactly where it stood. How the items are made does not depend on how they are
delivered, so these are bit-identical (selftest):
    inline   make() on the training thread (the CPU default)
    thread   a daemon thread fills a bounded queue (shares the GIL: little gain)
    process  a forked worker process fills a bounded multiprocessing queue (the GPU default)
"""

import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np

from .data import WALL_DAMAGE, aux_targets, damage_walls, edge, pad_targets, random_walls, targets
from .hexgrid import mask as hex_mask
from .masks import ragged_mask

K_POOL = 4      # targets kept per pool board (K is 1 on ~95% of boards, 2 on ~4.5%)
MARGIN = 4      # extra steps past the deepest rim-region cell
# --damage kinds a-d (the walls, data.WALL_DAMAGE), e (the state) and f (a stamped spiral, data._stamp_spiral):
# f is appended AFTER e so old pool.npz / log damage codes (0-4) keep their meaning.
DAMAGE_KINDS = WALL_DAMAGE + ("state", "spiral")
BAND = {"fill": (0.35, 0.8), "multiRim": (0.15, 1.0), "density": (0.0, 0.45)}  # pool_stats kept in these
STATS_EVERY = 200  # the pools' statistics (and the steering) every this many iterations
QUEUE = 4          # items a thread / process producer may run ahead


def answers(walls, R, aux=False, board=None):
    """(fills uint8 [K_POOL,S,S], depth, aux float32 [5,S,S] or None) of one board: its targets padded to a
    fixed K for the pool, and (only if asked; hexagon only) its aux targets."""
    f, d = targets(walls, R, board)
    return pad_targets([f], K_POOL)[0], d, aux_targets(walls, R) if aux else None


def draw_boards(rng, R, n, bridge_frac=0.0, aux=False, mask_mix=0.0, near_tie=0.0):
    """(walls [n,S,S], fills [n,K_POOL,S,S], depth [n,S,S], aux [n,5,S,S] or None, masks uint8 [n,S,S]) of n
    fresh boards from the training mix, each drawn as kind "bridge" instead with probability bridge_frac, on
    a ragged mask (masks.ragged_mask) with probability mask_mix (else the hexagon), its bridges near a tie
    with probability near_tie (data.random_walls). With mask_mix = near_tie = 0 the rng stream is the one of
    the hexagon-only code."""
    disc = hex_mask(R)
    walls, masks = [], []
    for _ in range(n):
        kind = "bridge" if rng.random() < bridge_frac else None
        m = ragged_mask(rng, R) if mask_mix > 0 and rng.random() < mask_mix else disc
        walls.append(random_walls(rng, R, kind, board=None if m is disc else m, near_tie=near_tie))
        masks.append(m)
    walls, masks = np.stack(walls), np.stack(masks).astype(np.uint8)
    fills, depth = zip(*(targets(w, R, m) for w, m in zip(walls, masks)))
    return (walls, pad_targets(fills, K_POOL), np.stack(depth),
            np.stack([aux_targets(w, R) for w in walls]) if aux else None, masks)


def pool_stats(walls, fills, R, masks=None):
    """{fill, multiRim, density} of a pool (walls [n,S,S], fills [n,K,S,S], masks [n,S,S] or None = the
    hexagon): the share of boards whose primary target fills a cell, the share with 2+ rim regions (= the
    primary target fills a rim cell: with one rim region no rim cell fills, with 2+ all but one rim region
    fill), the mean wall density on board."""
    prim = fills[:, 0] > 0
    if masks is None:
        on = hex_mask(R) == 1
        rim_cells = edge(on)
        multi = prim[:, rim_cells].any(1)
        dens = walls[:, on].mean()
    else:
        on = masks.astype(bool)
        multi = np.array([p[edge(m)].any() for p, m in zip(prim, on)])
        dens = (walls.astype(bool) & on).sum() / max(1, on.sum())
    return {"fill": round(float(prim.any((1, 2)).mean()), 3),
            "multiRim": round(float(multi.mean()), 3),
            "density": round(float(dens), 3)}


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


def data_rng(seed, start_it=0):
    """The producer's rng of a fresh run (its own stream, apart from the trainer's), or of a resumed checkpoint
    from before the producer existed (no stored state: seeded by the iteration it resumes at)."""
    return np.random.default_rng([int(seed), 1, int(start_it)])


class Producer:
    """Makes the items (module docstring) of iterations it, it+1, ... from its own copy of the pools' boards.

    spec: {R: radii, batch, bridgeFrac, damage, maskMix, nearTie, aux, steps: {R: [t0, t1]}}; boards: {R: {walls,
    fill, depth, mask[, aux]}} (copied); rng_state: a numpy Generator's bit_generator.state; steering: {R: steer
    output}."""

    def __init__(self, spec, boards, rng_state, steering, it):
        self.spec, self.it = spec, int(it)
        self.boards = {int(R): {k: np.array(v, copy=True) for k, v in P.items()} for R, P in boards.items()}
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = rng_state
        self.steering = {int(R): s for R, s in steering.items()}

    def make(self):
        t0 = time.perf_counter()
        sp, rng = self.spec, self.rng
        B, use_aux = sp["batch"], sp.get("aux", False)
        R = int(rng.choice(sp["R"]))
        P, st = self.boards[R], self.steering[R]
        idx = rng.choice(len(P["walls"]), B, replace=False)
        w, f, d, m = P["walls"][idx], P["fill"][idx], P["depth"][idx], P["mask"][idx]  # copies
        a = P["aux"][idx] if use_aux else None
        perm = rng.permutation(B)
        n_new = min(B, max(1, B // 8) * st["newMult"])
        new = perm[:n_new]
        nw, nf, nd, na, nm = draw_boards(rng, R, n_new, sp.get("bridgeFrac", 0.0), use_aux,
                                         sp.get("maskMix", 0.0), sp.get("nearTie", 0.0))
        w[new], f[new], d[new], m[new] = nw, nf, nd, nm
        if use_aux:
            a[new] = na
        damaged, state_damage, kinds = [], [], dict.fromkeys(DAMAGE_KINDS, 0)
        p_kind = np.asarray(st["p"], dtype=float)
        p_kind = p_kind / p_kind.sum()  # steer() rounds to 4 places: the sum can miss 1 by more than numpy allows
        for pos in perm[n_new:]:  # each of the rest, with probability --damage, gets one damage
            if rng.random() >= sp["damage"]:
                continue
            k = int(rng.choice(len(DAMAGE_KINDS), p=p_kind))
            kind = DAMAGE_KINDS[k]
            kinds[kind] += 1
            damaged.append((int(pos), k))
            if kind == "state":  # distill's damage: zero the state inside a disc round a random on-board cell
                cells = np.argwhere(m[pos] == 1)
                row, col = cells[rng.integers(len(cells))]
                state_damage.append((int(pos), int(row), int(col), int(rng.integers(1, max(1, R // 2) + 1))))
                continue
            w[pos] = damage_walls(rng, w[pos], R, kind, st["pBridge"], board=m[pos])
            f[pos], d[pos], aux_j = answers(w[pos], R, use_aux, m[pos])
            if use_aux:
                a[pos] = aux_j
        P["walls"][idx], P["fill"][idx], P["depth"][idx], P["mask"][idx] = w, f, d, m
        if use_aux:
            P["aux"][idx] = a
        t_lo, t_hi = sp["steps"][R]
        lo = max(t_lo, int(d.max()) + MARGIN)
        T = int(rng.integers(lo, max(t_hi, lo) + 1))
        item = {"it": self.it, "R": R, "idx": idx, "walls": w, "fill": f, "depth": d, "mask": m, "aux": a,
                "new": [int(x) for x in new], "damaged": damaged, "stateDamage": state_damage, "T": T,
                "kinds": kinds}
        self.it += 1
        if self.it % STATS_EVERY == 0:
            stats = {r: pool_stats(Q["walls"], Q["fill"], r, Q["mask"]) for r, Q in self.boards.items()}
            self.steering = {r: steer(s) for r, s in stats.items()}
            item["pool"], item["steer"] = stats, self.steering
        item["rng"] = rng.bit_generator.state
        item["sec"] = time.perf_counter() - t0
        return item


def _loop(producer, put, stop):
    """Make items forever (until stop is set), handing each to put; an exception is handed on as the item."""
    try:
        while not stop.is_set():
            put(producer.make())
    except BaseException as e:  # noqa: BLE001 -- the trainer re-raises it
        put({"error": repr(e)})


def _process_main(producer, q, stop, parent):
    q.cancel_join_thread()  # at exit, don't wait to flush items nobody will read

    def put(item):  # a full queue: wait, but never outlive a trainer that died without close() (a TERM)
        while not stop.is_set():
            try:
                return q.put(item, timeout=5)
            except queue.Full:
                if os.getppid() != parent:
                    stop.set()
    _loop(producer, put, stop)


class Source:
    """The trainer's end: get() -> the next item, made inline or by a thread / process (module docstring)."""

    def __init__(self, producer, mode="inline", depth=QUEUE):
        self.mode, self.producer, self.worker = mode, producer, None
        if mode == "inline":
            return
        if mode == "thread":
            self.stop, self.q = threading.Event(), queue.Queue(maxsize=depth)
            self.worker = threading.Thread(target=_loop, args=(producer, self.q.put, self.stop), daemon=True)
        elif mode == "process":
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
            self.stop, self.q = ctx.Event(), ctx.Queue(maxsize=depth)
            self.worker = ctx.Process(target=_process_main, args=(producer, self.q, self.stop, os.getpid()),
                                      daemon=True)
        else:
            raise ValueError(f"producer mode {mode!r}")
        self.worker.start()

    def get(self):
        item = self.producer.make() if self.mode == "inline" else self.q.get()
        if "error" in item:
            raise RuntimeError(f"the data producer failed: {item['error']}")
        return item

    def close(self):
        """Stop the worker (items it made ahead are dropped: a checkpoint holds the state before them)."""
        if self.worker is None:
            return
        self.stop.set()
        if self.mode == "process":
            self.worker.terminate()
            self.worker.join(timeout=5)
        else:
            try:  # unblock a put waiting on a full queue
                while True:
                    self.q.get_nowait()
            except queue.Empty:
                pass
        self.worker = None
