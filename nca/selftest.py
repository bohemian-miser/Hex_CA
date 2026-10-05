"""Plain-assert self-test for nca/hexgrid.py and nca/data.py. No pytest.

    python -m nca.selftest
"""

import tempfile
import time

import numpy as np

import math

from .hexgrid import CONST_NAMES, NEIGHBOURS, KERNEL_MASK, consts, mask, rim, side
from .data import (EPS, targets, oracle, random_walls, batch, edit_walls, page_loops, pad_targets, _KINDS, _MIX,
                   _axial_dist, _offset, _polyline, _poly_points, _labels, _rim_regions, _spans, _splitters)


# --------------------------------------------------------------------------
# An independent brute force of targets(), written straight from the spec, to
# check nca.data.targets against. Deliberately not sharing code with it:
# regions by propagating the smallest flat index to a fixpoint (not a BFS),
# depth by full-board rescans, candidates by plain Python over dicts, spans
# from the rim cells' angles collected cell by cell (the float32 consts as
# Python floats, so the arithmetic is the same doubles the spec asks for).
# --------------------------------------------------------------------------

def _shift(a: np.ndarray, dr: int, dc: int, fill):
    """out[row,col] = a[row+dr, col+dc], `fill` off the array."""
    S = a.shape[0]
    p = np.full((S + 2, S + 2), fill, dtype=a.dtype)
    p[1:-1, 1:-1] = a
    return p[1 + dr:1 + dr + S, 1 + dc:1 + dc + S]


def _bruteforce_targets(walls: np.ndarray, R: int):
    """(list of acceptable fills [S,S] uint8, primary first; depth int16 [S,S])."""
    S = side(R)
    on_board = mask(R) == 1
    is_rim = rim(R)
    open_cells = on_board & (walls == 0)

    # Regions: every open cell takes the smallest flat index among itself and its
    # open neighbours, until nothing changes; a region ends up named by its lowest cell.
    big = S * S
    lab = np.where(open_cells, np.arange(S * S).reshape(S, S), big)
    while True:
        new = lab.copy()
        for dr, dc in NEIGHBOURS:
            new = np.minimum(new, np.where(open_cells, _shift(lab, dr, dc, big), big))
        if np.array_equal(new, lab):
            break
        lab = new

    c = consts(R)
    area, t1s, t2s = {}, {}, {}
    for row in range(S):
        for col in range(S):
            if open_cells[row, col]:
                x = int(lab[row, col])
                area[x] = area.get(x, 0) + 1
                if is_rim[row, col]:
                    t1s.setdefault(x, []).append(float(c[1, row, col]))
                    t2s.setdefault(x, []).append(float(c[2, row, col]))
    rim_regions = [x for x in area if x in t1s]
    span = {x: min(max(t1s[x]) - min(t1s[x]), max(t2s[x]) - min(t2s[x])) for x in rim_regions}
    if not rim_regions:
        fills = [open_cells.astype(np.uint8)]
    else:
        amax = max(area[x] for x in rim_regions)
        smax = max(span[x] for x in rim_regions)
        cands = [x for x in rim_regions if area[x] == amax or span[x] >= smax - EPS]
        # Primary: max area, ties by max span, then by lowest cell index.
        primary = min((x for x in cands if area[x] == amax), key=lambda x: (-span[x], x))
        cands = [primary] + [x for x in cands if x != primary]
        fills = [(open_cells & (lab != x)).astype(np.uint8) for x in cands]

    # Depth: rescan the whole board until nothing changes (v1's brute force).
    outside = open_cells & is_rim
    depth = np.where(outside, 0, -1).astype(np.int16)
    d, changed = 0, True
    while changed:
        changed = False
        d += 1
        prev = outside.copy()
        for row in range(S):
            for col in range(S):
                if not open_cells[row, col] or outside[row, col]:
                    continue
                for dr, dc in NEIGHBOURS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < S and 0 <= nc < S and prev[nr, nc]:
                        outside[row, col] = True
                        depth[row, col] = d
                        changed = True
                        break
    return fills, depth


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"OK {msg}")


def main() -> None:
    rng = np.random.default_rng(12345)

    # -- mask cell count --------------------------------------------------
    for R in range(0, 9):
        expected = 3 * R * (R + 1) + 1
        got = int(mask(R).sum())
        check(got == expected, f"mask({R}) has {expected} cells")

    # -- kernel mask --------------------------------------------------------
    check(KERNEL_MASK.shape == (3, 3), "KERNEL_MASK is 3x3")
    check(KERNEL_MASK[0, 0] == 0.0 and KERNEL_MASK[2, 2] == 0.0, "KERNEL_MASK corners are zero")
    check(int(KERNEL_MASK.sum()) == 7, "KERNEL_MASK has 7 live weights")
    check(len(NEIGHBOURS) == 6, "six neighbour offsets")

    # -- targets vs. independent brute force ---------------------------------
    n_checked = n_multi = 0
    for R in (3, 5, 8):
        for _ in range(400 // 3 + (R == 8)):
            w = random_walls(rng, R)
            fills, depth = targets(w, R)
            bf_fills, bf_depth = _bruteforce_targets(w, R)
            if not np.array_equal(fills[0], bf_fills[0]):
                raise AssertionError(f"primary target disagrees with brute force (R={R})")
            if sorted(f.tobytes() for f in fills) != sorted(f.tobytes() for f in bf_fills):
                raise AssertionError(f"acceptable targets disagree with brute force (R={R})")
            if not np.array_equal(depth, bf_depth):
                raise AssertionError(f"depth disagrees with brute force (R={R})")
            if fills.dtype != np.uint8 or depth.dtype != np.int16:
                raise AssertionError("targets() dtypes")
            n_checked += 1
            n_multi += len(fills) > 1
    check(n_checked == 400, f"targets() matches an independent brute force on 400 random boards (R in {{3,5,8}}): "
          f"same primary, same set of targets ({n_multi} boards with 2+), same depth")

    # -- hand cases (R=6) -----------------------------------------------------
    R = 6
    S = side(R)
    on_board = mask(R) == 1
    q = (np.arange(S) - R).reshape(1, S) + np.zeros((S, 1), int)
    r = (np.arange(S) - R).reshape(S, 1) + np.zeros((1, S), int)
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))

    def walls_of(cells):
        return (cells & on_board).astype(np.uint8)

    def as_set(fills):
        return sorted(f.tobytes() for f in fills)

    def only(cells, w):
        """The target that fills exactly `cells` (open cells of the board)."""
        return (cells & on_board & (w == 0)).astype(np.uint8)

    empty = np.zeros((S, S), dtype=np.uint8)
    fills, _ = targets(empty, R)
    check(len(fills) == 1 and int(fills[0].sum()) == 0, "an empty board fills nothing (one target)")

    # A closed ring of radius 3 centred on the board: 19 interior cells (the radius-2 disk).
    ring = walls_of(dist == 3)
    fills, _ = targets(ring, R)
    check(len(fills) == 1 and np.array_equal(fills[0].astype(bool), dist < 3),
          "a radius-3 ring on R=6 fills exactly its 19 interior cells")

    # The same ring with one cell removed: the interior leaks out and fills nothing.
    leaky_ring = ring.copy()
    leaky_ring[tuple(np.argwhere(ring)[0])] = 0
    fills, _ = targets(leaky_ring, R)
    check(len(fills) == 1 and int(fills[0].sum()) == 0, "a radius-3 ring missing one cell fills nothing")

    # A wall ring exactly ON the rim: no rim region at all, the whole interior fills.
    on_rim = rim(R).astype(np.uint8)
    fills, _ = targets(on_rim, R)
    check(len(fills) == 1 and np.array_equal(fills[0].astype(bool), on_board & ~rim(R)),
          "a wall ring on the rim fills the whole interior")

    # ...with one rim cell left open: one rim region (the whole inside), nothing fills.
    gappy_rim = on_rim.copy()
    gappy_rim[tuple(np.argwhere(on_rim)[0])] = 0
    fills, _ = targets(gappy_rim, R)
    check(len(fills) == 1 and int(fills[0].sum()) == 0, "a rim-loop with one non-wall rim cell fills nothing")

    # A straight bridge across the board, 2 rows off the centre: the smaller side fills, only it.
    w = walls_of(r == 2)
    fills, _ = targets(w, R)
    check(len(fills) == 1 and np.array_equal(fills[0], only(r > 2, w)),
          "a straight wall across R=6 at offset 2 fills the smaller side only (34 of 116 cells)")

    # A bridge exactly through the centre: equal sides, two acceptable targets, one per side.
    w = walls_of(r == 0)
    fills, _ = targets(w, R)
    check(len(fills) == 2 and as_set(fills) == as_set([only(r > 0, w), only(r < 0, w)]),
          "a wall through the centre (equal areas) gives two targets, each filling exactly one side")

    # A corner cut off: walling the corner's three on-board neighbours fills the corner alone;
    # the arc at distance 2 round it fills the 4 cells within distance 1.
    corner_d = np.maximum(np.maximum(np.abs(q - R), np.abs(r)), np.abs(q - R + r))  # from (q, r) = (R, 0)
    w = walls_of(corner_d == 1)
    fills, _ = targets(w, R)
    check(len(fills) == 1 and np.array_equal(fills[0], only(corner_d == 0, w)) and int(fills[0].sum()) == 1,
          "a corner cut off by a wall fills that corner (one cell)")
    w = walls_of(corner_d == 2)
    fills, _ = targets(w, R)
    check(len(fills) == 1 and np.array_equal(fills[0], only(corner_d <= 1, w)) and int(fills[0].sum()) == 4,
          "...and a wider cut fills the 4 corner cells")

    # A bridge plus a closed loop on the larger side: the loop's inside AND the smaller side fill.
    loop_d = np.maximum(np.maximum(np.abs(q), np.abs(r + 3)), np.abs(q + r + 3))  # from (0, -3)
    w = walls_of((r == 2) | (loop_d == 1))
    fills, _ = targets(w, R)
    check(len(fills) == 1 and np.array_equal(fills[0], only((r > 2) | (loop_d == 0), w)),
          "a bridge plus a closed loop on the larger side fills the loop's inside and the smaller side")

    # Two bridges, three rim regions: rows r = 4 and columns q = 4 cut off 15 cells (9 rim cells)
    # each; the middle has 79 cells and 14 rim cells, the largest by both measures. (Two parallel
    # rows would not do: the middle band gets only two rim cells a row, so the measures disagree.)
    w = walls_of((r == 4) | (q == 4))
    fills, _ = targets(w, R)
    check(len(fills) == 1 and np.array_equal(fills[0], only((r > 4) | (q > 4), w)),
          "two bridges making three rim regions fill all but the largest")

    # The measures disagree: a bridge along row r = -1 whose north end carries on hugging the
    # rim (walling the rim cells) round the r >= 0 side, leaving only 3 of its rim cells open.
    # That side has more cells (54 vs 45) but the shorter open rim arc (3 vs 15 rim cells).
    hug = rim(R) & (r >= 0) & ~((q == -R) & (r <= 2))
    w = walls_of((r == -1) | hug)
    big, small = (r >= 0) & on_board & (w == 0), (r <= -2) & on_board
    area_b, area_s = int(big.sum()), int(small.sum())
    rim_b, rim_s = int((big & rim(R)).sum()), int((small & rim(R)).sum())
    check(_components(big) == 1 and _components(small) == 1 and area_b > area_s and rim_b < rim_s,
          f"hand-built bridge where the measures disagree: larger side {area_b} cells / {rim_b} rim cells, "
          f"smaller {area_s} / {rim_s}")
    fills, _ = targets(w, R)
    check(len(fills) == 2 and np.array_equal(fills[0], only(small, w))
          and as_set(fills) == as_set([only(small, w), only(big, w)]),
          "...gives two targets (either side may fill, exactly one), the primary filling the smaller-area side")

    span_cases(q, r, on_board, walls_of, only)

    # -- every target leaves exactly one rim region unfilled (when there is one) ---
    ok = 0
    for R in (2, 4, 7):
        on = mask(R) == 1
        for _ in range(60):
            w = random_walls(rng, R)
            fills, _ = targets(w, R)
            lab = _labels(on & (w == 0), R)
            ids = _rim_regions(lab, R)[0]
            for f in fills:
                unfilled = {int(x) for x in np.unique(lab[on & (w == 0) & (f == 0)])}
                if f[(w > 0) | ~on].any() or len(unfilled) != min(1, len(ids)) or not unfilled <= set(ids.tolist()):
                    raise AssertionError(f"a target that doesn't leave exactly one rim region unfilled (R={R})")
            if len({f.tobytes() for f in fills}) != len(fills):
                raise AssertionError("targets() repeated a target")
            ok += 1
    check(ok == 180, "every target leaves exactly one rim region unfilled, walls and off-board 0, no repeats")

    # -- _splitters is exact: walling the cell splits its region iff it says so ----
    n_cells = 0
    for R in (3, 5):
        on = mask(R) == 1
        for _ in range(30):
            w = random_walls(rng, R).astype(bool)
            sp = _splitters(w, R)
            before = int(_labels(on & ~w, R).max()) + 1
            for row, col in np.argwhere(on & ~w):
                w2 = w.copy()
                w2[row, col] = True
                if (int(_labels(on & ~w2, R).max()) + 1 > before) != bool(sp[row, col]):
                    raise AssertionError(f"_splitters wrong at ({row},{col}), R={R}")
                n_cells += 1
    check(n_cells > 0, f"_splitters agrees with relabelling on every open cell of 60 boards ({n_cells} cells)")

    # -- random_walls never puts walls off board -----------------------------
    off_board_checks = 0
    for R in (0, 1, 3, 6, 8):
        off = mask(R) == 0
        for kind in list(_KINDS.keys()) + [None]:
            for _ in range(10):
                w = random_walls(rng, R, kind=kind)
                if int(w[off].sum()) != 0:
                    raise AssertionError(f"random_walls kind={kind} R={R} put a wall off board")
                off_board_checks += 1
    check(off_board_checks > 0, f"random_walls ({off_board_checks} boards, every kind, R in {{0,1,3,6,8}}) stays on board")

    # -- edit_walls changes at least one cell --------------------------------
    edit_checks = 0
    for _ in range(20):
        R = int(rng.integers(2, 9))
        w = random_walls(rng, R)
        edited = edit_walls(rng, w, R)
        if np.array_equal(w, edited):
            raise AssertionError("edit_walls left the board unchanged")
        if int(edited[mask(R) == 0].sum()) != 0:
            raise AssertionError("edit_walls put a wall off board")
        edit_checks += 1
    check(edit_checks > 0, f"edit_walls ({edit_checks} boards) always changes >=1 cell and stays on board")

    # -- batch() shapes and padding ----------------------------------------------
    R = 5
    S = side(R)
    walls, fills, depth = batch(rng, R, 40)
    K = fills.shape[1]
    check(walls.shape == (40, S, S) and fills.shape == (40, K, S, S) and depth.shape == (40, S, S),
          f"batch() returns walls [n,S,S], fills [n,K,S,S] (K={K} here), depth [n,S,S]")
    check(walls.dtype == np.uint8 and fills.dtype == np.uint8 and depth.dtype == np.int16,
          "batch() returns the right dtypes")
    pad_ok = True
    for i in range(40):
        t, d = targets(walls[i], R)
        pad_ok &= np.array_equal(fills[i, :len(t)], t) and bool((fills[i, len(t):] == t[0]).all())
        pad_ok &= np.array_equal(depth[i], d)
    check(pad_ok, "batch(): each board's targets in order, padded by repeating its primary")
    check(pad_targets([t], 3).shape == (1, 3, S, S), "pad_targets pads to a given K")

    span_brute(rng)
    generator_checks(rng)
    board_stats(rng)
    edit_stats()
    distinctness()

    const_checks()
    aux_checks(rng)
    model_checks(rng)
    flood_checks(rng)
    damage_checks()
    pool_stat_checks()
    device_checks()
    snapshot_checks()
    collapse_checks()
    progress_checks()
    ragged_checks()
    near_tie_checks()
    schedule_checks()
    producer_checks()
    print("\nALL OK")


def damage_checks() -> None:
    """The training pool's damage kinds (data.damage_walls a-d, train.damage_state e)."""
    import torch

    from .data import WALL_DAMAGE, damage_walls, disc
    from .train import damage_state

    print("\n-- damage (train.py --damage) --")
    rng = np.random.default_rng(777)
    hexd = lambda R, a, b: max(abs(int(a[1]) - int(b[1])), abs(int(a[0]) - int(b[0])),
                               abs(int(a[1]) - int(b[1]) + int(a[0]) - int(b[0])))
    n_boards, on_ok, changed = 0, True, {k: 0 for k in WALL_DAMAGE}
    burst_ok, burst_add, burst_del, erase_ok, stamp_ok = True, 0, 0, True, True
    for _ in range(40):
        R = int(rng.integers(3, 9))
        on = mask(R) == 1
        w = random_walls(rng, R)
        for kind in WALL_DAMAGE:
            d = damage_walls(np.random.default_rng(int(rng.integers(1 << 30))), w, R, kind)
            on_ok &= d.dtype == np.uint8 and d.shape == w.shape and set(np.unique(d)) <= {0, 1} \
                and int(d[~on].sum()) == 0
            diff = d != w
            changed[kind] += bool(diff.any())
            if kind == "burst":
                burst_ok &= 1 <= int(diff.sum()) <= 20
                burst_add += int((diff & (d == 1)).sum())
                burst_del += int((diff & (d == 0)).sum())
            if kind == "erase":  # removed = exactly the walls inside a disc of radius 1-3 round one of the walls
                removed = diff & (w == 1)
                erase_ok &= not (diff & (d == 1)).any() and removed.any() == bool(w.any()) and (not w.any() or any(
                    np.array_equal(removed, (w == 1) & disc(R, r0, c0, rad))
                    for r0, c0 in np.argwhere(w == 1) for rad in (1, 2, 3)))
            if kind == "stamp":
                stamp_ok &= not (diff & (d == 0)).any()  # only adds walls
        n_boards += 1
    check(on_ok, f"damage_walls, every kind on {n_boards} boards (R 3-8): uint8 0/1, nothing off board")
    check(changed["edit"] == n_boards, "edit (edit_walls) always changes the board")
    check(burst_ok and burst_add > 0 and burst_del > 0,
          f"burst changes 1-20 cells, additions ({burst_add}) and deletions ({burst_del}) mixed")
    check(erase_ok and changed["erase"] >= n_boards - 2,
          "erase only removes walls: exactly every wall within distance 1-3 of a wall cell (the disc's centre)")
    check(stamp_ok and changed["stamp"] >= 0.9 * n_boards, f"stamp only adds walls (changed {changed['stamp']} of "
          f"{n_boards} boards)")

    # A stamp on an empty board: a bridge splits it into 2+ rim regions, a loop encloses something.
    R, nb, nl = 8, 0, 0
    on = mask(R) == 1
    empty = np.zeros((side(R), side(R)), dtype=np.uint8)
    for _ in range(40):
        b = damage_walls(rng, empty, R, "stamp", p_bridge=1.0)
        nb += len(_rim_regions(_labels(on & (b == 0), R), R)[0]) >= 2
        loop = damage_walls(rng, empty, R, "stamp", p_bridge=0.0)
        nl += bool(targets(loop, R)[0][0].any())
    check(nb >= 36 and nl >= 24, f"a stamped bridge splits an empty board into 2+ rim regions ({nb}/40), a stamped "
          f"loop fills something ({nl}/40; a blob run over the rim is open to the edge)")

    # Edits pile up: three in a row change the board further (nothing toggles back).
    w = random_walls(rng, 6)
    w1 = damage_walls(rng, w, 6, "stamp")
    w2 = damage_walls(rng, w1, 6, "stamp")
    check(not np.array_equal(w1, w2) and ((w1 == 1) <= (w2 == 1)).all(), "damage applies to the board as it is "
          "(a second stamp keeps the first)")

    # State damage (distill's): zeroes channels 1.. inside the disc, nothing else.
    ok, n_zeroed = True, 0
    for _ in range(30):
        R = int(rng.integers(3, 11))
        S, on = side(R), mask(R) == 1
        w = torch.from_numpy(random_walls(rng, R)).float()
        st = torch.from_numpy((rng.random((16, S, S)) * 2 - 1).astype(np.float32)) * torch.from_numpy(on)
        st[0] = w
        before = st.clone()
        dsc = damage_state(rng, st, R)
        rad = max(hexd(R, a, b) for a in np.argwhere(dsc & on) for b in np.argwhere(dsc & on)) / 2
        inside = torch.from_numpy(dsc)
        ok &= torch.equal(st[0], before[0]) and torch.equal(st[:, ~inside], before[:, ~inside]) \
            and bool((st[1:, inside] == 0).all()) and bool((dsc & on).any()) and rad <= max(1, R // 2)
        n_zeroed += int((inside & torch.from_numpy(on)).sum())
    check(ok, f"damage_state zeroes channels 1.. inside a disc of radius <= R/2 round an on-board cell, and only "
          f"there; walls (ch0) untouched (30 boards, {n_zeroed} cells zeroed)")


def pool_stat_checks() -> None:
    """train.pool_stats against a direct count, and train.steer."""
    from .train import BAND, DAMAGE_KINDS, K_POOL, draw, pool_stats, steer

    print("\n-- pool statistics --")
    rng = np.random.default_rng(778)
    R = 6
    on = mask(R) == 1
    w, f, _, _ = draw(rng, R, 120, 0.25)
    st = pool_stats(w, f, R)
    multi = np.mean([len(_rim_regions(_labels(on & (x == 0), R), R)[0]) >= 2 for x in w])
    fill = np.mean([targets(x, R)[0][0].any() for x in w])
    dens = np.mean([x[on].mean() for x in w])
    check(f.shape[1] == K_POOL and abs(st["multiRim"] - multi) < 1e-3 and abs(st["fill"] - fill) < 1e-3
          and abs(st["density"] - dens) < 1e-3,
          f"pool_stats = a direct count on 120 boards: fill {st['fill']}, 2+ rim regions {st['multiRim']} "
          f"(= the primary target fills a rim cell), density {st['density']}")
    ok_band = all(BAND[k][0] <= st[k] <= BAND[k][1] for k in BAND)
    calm = steer(st)
    check(ok_band and calm["why"] == [] and calm["newMult"] == 1 and calm["p"] == [0.2] * 5,
          "a fresh pool is in BAND, and steer leaves it alone (uniform damage, new boards as usual)")
    i = {k: n for n, k in enumerate(DAMAGE_KINDS)}
    lo = steer({"fill": 0.2, "multiRim": 0.1, "density": 0.2})
    dense = steer({"fill": 0.6, "multiRim": 0.4, "density": 0.6})
    high = steer({"fill": 0.9, "multiRim": 0.4, "density": 0.2})
    check(lo["p"][i["stamp"]] > 0.5 and lo["pBridge"] == 0.9 and lo["newMult"] == 2
          and dense["p"][i["stamp"]] == 0 and dense["p"][i["erase"]] > 0.4 and dense["newMult"] == 2
          and high["p"][i["stamp"]] < 0.05 and high["p"][i["erase"]] > 0.3,
          "steer: fill or 2+ rim regions low -> more stamps (bridges), density high -> erase and no stamps, fill "
          "high -> erase and burst; any of them doubles the new boards")


def snapshot_checks() -> None:
    """A 3-iteration pure training run (a subprocess, in a temp dir) writes runs/<name>/pool.npz to the
    dashboard contract, and log.jsonl starts with the config and its --iters. The same seed again, split into 2
    iterations and a --resume to 3 with --device cpu, ends bit-identical (weights, optimiser, rng, pools)."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    import torch

    from .train import DAMAGE_KINDS

    print("\n-- pool.npz snapshot (a 3-iteration run) --")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        base = [sys.executable, "-m", "nca.train", "--R", "3", "4", "--batch", "8", "--pool-size", "64",
                "--snap-every", "1", "--eval-mults", "1", "--hidden", "16", "--threads", "2"]

        def train(*extra):
            t = time.time()
            res = subprocess.run(base + list(extra), cwd=tmp, env={**os.environ, "PYTHONPATH": root},
                                 capture_output=True, text=True)
            check(res.returncode == 0, f"python -m nca.train {' '.join(extra)} runs ({time.time() - t:.1f} s)"
                  + ("" if res.returncode == 0 else ": " + res.stderr[-800:]))

        train("--name", "snap", "--iters", "3")
        with open(os.path.join(tmp, "runs", "snap", "log.jsonl")) as f:
            lines = [json.loads(x) for x in f]
        first = lines[0]
        check(first["config"]["iters"] == 3 and first["config"]["R"] == [3, 4],
              "log.jsonl's first line is the config, with the --iters target")
        checks = [x for x in lines if "score" in x]
        check(all(isinstance(x.get("time"), float) and abs(x["time"] - time.time()) < 3600 and x["device"] == "cpu"
                  for x in lines) and [x["iteration"] for x in checks] == [0, 3]
              and all(x["evalN"] == 32 for x in checks + [first]) and first["config"]["evalN"] == 32
              and lines[-1]["stopped"] == "done" and lines[-1]["iteration"] == 3,
              "every log line has time (unix s) and device; the checks and the config line evalN; the last line "
              "is the clean end {stopped: done}")

        train("--name", "split", "--iters", "2", "--device", "cpu")
        train("--name", "split", "--iters", "3", "--resume", "--device", "cpu")
        a, b = (torch.load(os.path.join(tmp, "runs", n, "ckpt.pt"), map_location="cpu", weights_only=False)
                for n in ("snap", "split"))

        def tensors(x):
            if torch.is_tensor(x):
                yield x
            elif isinstance(x, dict):
                for v in x.values():
                    yield from tensors(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    yield from tensors(v)

        same = all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]) \
            and all(torch.equal(x, y) for x, y in zip(tensors(a["opt"]["state"]), tensors(b["opt"]["state"]))) \
            and a["rng"] == b["rng"] and a["iteration"] == b["iteration"] == 3
        for R in (3, 4):
            same &= torch.equal(a["pool"][R]["state"], b["pool"][R]["state"]) \
                and all(np.array_equal(a["pool"][R][k], b["pool"][R][k], equal_nan=True)
                        for k in ("walls", "fill", "depth", "loss", "born", "edits", "last"))
        check(same, "seeded runs reproduce: 3 iterations straight = 2 iterations + --resume to 3 (--device cpu), "
              "bit-identical weights, optimiser state, rng and pools")
        check(all(t.device.type == "cpu" for t in tensors(a)) and len(list(tensors(a))) > 4,
              "checkpoints hold CPU tensors only (they load on any device)")
        z = np.load(os.path.join(tmp, "runs", "snap", "pool.npz"))
        B, n, m, C = 8, 48, 6, 16
        want = {"iteration": (np.int64, ()), "radii": (np.int64, (2,)), "last_R": (np.int64, ()),
                "last_idx": (np.int64, (B,))}
        for R in (3, 4):
            S = side(R)
            want.update({f"walls_{R}": (np.uint8, (n, S, S)), f"mask_{R}": (np.uint8, (n, S, S)),
                         f"fill_{R}": (np.float16, (n, S, S)),
                         f"target_{R}": (np.uint8, (n, S, S)), f"ntargets_{R}": (np.uint8, (n,)),
                         f"loss_{R}": (np.float32, (n,)), f"age_{R}": (np.int32, (n,)),
                         f"edits_{R}": (np.int32, (n,)), f"damage_{R}": (np.int8, (n,)),
                         f"state_{R}": (np.float16, (m, C, S, S))})
        bad = [k for k, (dt, sh) in want.items() if k not in z.files or z[k].dtype != dt or z[k].shape != sh]
        check(not bad and sorted(z.files) == sorted(want),
              f"pool.npz loads with np.load and has every key of the contract with its shape and dtype ({len(want)} "
              f"keys)" + (f"; wrong: {bad}" if bad else ""))
        ok = int(z["iteration"]) == 3 and list(z["radii"]) == [3, 4] and int(z["last_R"]) in (3, 4)
        for R in (3, 4):
            on = mask(R) == 1
            ok &= not z[f"walls_{R}"][:, ~on].any() and not z[f"fill_{R}"][:, ~on].any() \
                and not z[f"target_{R}"][:, ~on].any() and not z[f"state_{R}"][:, :, ~on].any()
            ok &= bool((z[f"ntargets_{R}"] >= 1).all()) and bool((z[f"age_{R}"] >= 0).all()) \
                and bool((z[f"age_{R}"] <= 3).all()) and bool((z[f"edits_{R}"] >= 0).all())
            ok &= bool(((z[f"damage_{R}"] >= -1) & (z[f"damage_{R}"] < len(DAMAGE_KINDS))).all())
            ok &= bool(((z[f"edits_{R}"] == 0) == (z[f"damage_{R}"] == -1)).all())
            ok &= np.array_equal(z[f"state_{R}"][:, 1], z[f"fill_{R}"][:m])
        lr = int(z["last_R"])
        ok &= bool(np.isfinite(z[f"loss_{lr}"][z["last_idx"][z["last_idx"] < n]]).all())
        check(ok, "...and its values make sense: off-board 0, ages 0..3, edits 0 iff no damage kind, the batch "
              "that just ran has a loss, state_R[:, 1] = fill_R")


def device_checks() -> None:
    """The trainer's tensor code on torch's "meta" device, standing in for a GPU the Pi doesn't have: like CUDA,
    meta raises on an elementwise op, cat or where that mixes it with a CPU tensor (and allows CPU indices and
    0-dim scalars), so a tensor left on the CPU in a training step, the pools, the held-out sets or the edit
    sequences fails here. Meta has no data, so .item() / .numpy() paths (the quick check's scoring) are not
    covered; those all go through .cpu() first."""
    import torch

    from . import train as T
    from .data import damage_walls
    from .model import HexNCA, const_stack, fresh_state

    print("\n-- device placement (meta device as a stand-in GPU) --")
    dev = torch.device("meta")
    rng = np.random.default_rng(0)
    for perception, floods, n_consts in (("taps", False, 1), ("taps+pool", True, 7)):
        model = HexNCA(16, 40, (-2.0, 2.0), 1.0, n_consts, perception, floods)
        T.model_to(model, dev)
        cs = {R: const_stack(R, n_consts).to(dev) for R in (3, 4)}
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        R, B = 4, 8
        mk = cs[R][:, :1]
        pool = T.new_pool(rng, R, 16, 16, 0.25, False, dev)
        idx = rng.choice(16, B, replace=False)
        tidx = torch.from_numpy(idx).to(dev)
        state = pool["state"][tidx].clone()
        walls_np, fill_np = pool["walls"][idx], pool["fill"][idx]
        T.per_sample_loss(state[:, 1:2], T.to_t(fill_np, dev), mk).min(1)
        state[0] = fresh_state(T.to_t(walls_np[0][None], dev), 16)[0]
        T.damage_state(rng, state[1], R)
        walls_np[2] = damage_walls(rng, walls_np[2], R, "stamp", 0.5)
        walls, target = T.to_t(walls_np, dev), T.to_t(fill_np, dev)
        state[:, 0:1] = walls
        acc = 0.0
        for t in range(6):
            state = model.step(state, walls, cs[R])
            if t >= 2:
                acc = acc + T.per_sample_loss(state[:, 1:2], target, mk) / 4
        T.teach(state, torch.zeros(B, 5, *state.shape[-2:], device=dev), torch.from_numpy(rng.random(B) < 0.5).to(dev))
        T.arc_rule(state, walls, mk)
        opt.zero_grad()
        acc.min(1).values.mean().backward()  # through the spec v6 gradient hooks too
        for prm in model.parameters():
            prm.grad /= prm.grad.norm() + 1e-8
        opt.step()
        pool["state"][tidx] = state.detach()
        held = T.heldout([3, 4], cs, 4, dev)
        eseq = {r: T.edit_sequence(r, cs[r], 4, device=dev) for r in (3, 4)}
        w, f, _, c = held["bridge"][3]
        st = model(fresh_state(w, 16), w, c, 6)
        ((st[:, 1:2] > 0.5) != f) & (c[:, :1] > 0)
        stages, fills, cs3, _, _ = eseq[3]
        st = model(torch.cat([stages[1], st[:, 1:]], 1), stages[1], cs3, 3)
        ((st[:, 1:2] > 0.5) != fills[0]) & (cs3[:, :1] > 0)
        on_dev = [*model.parameters(), *model.buffers(), pool["state"], w, f, stages[0], fills[0],
                  *(t for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.dim()),
                  *getattr(model, "frozen", {}).values(), *getattr(model, "frozen_values", {}).values()]
        check(all(t.device == dev for t in on_dev) and all(p.grad.device == dev for p in model.parameters()),
              f"{perception}{' + floods' if floods else ''}: a pool training step (fresh, damage, rollout, loss, "
              f"backward, Adam), the held-out sets and the edit sequence run on the device with nothing on the CPU")
    seen = []
    orig = torch.Tensor.cpu
    torch.Tensor.cpu = lambda t: seen.append(t.device.type) or torch.zeros(t.shape)  # meta: nothing to copy
    try:
        sd = T.to_cpu({"model": model.state_dict(), "opt": opt.state_dict(), "pool": {R: pool}})
    finally:
        torch.Tensor.cpu = orig
    n = len(model.state_dict()) + 1 + sum(torch.is_tensor(v) for s in opt.state_dict()["state"].values()
                                          for v in s.values())
    check(len(seen) == n and type(sd["model"]) is type(model.state_dict()) and hasattr(sd["model"], "_metadata"),
          f"to_cpu (what save_ckpt writes) moves every tensor of the weights, optimiser and pools ({n})")


def collapse_checks() -> None:
    """The collapse guard and the lr schedule (nca.train). collapse_reason and lr_at on their own, then a forced
    collapse: a tiny pool run in-process with the quick check's scores scripted (quick_eval / edit_eval patched)
    0.7, 0.8, 0.3 at iterations 0, 50, 100 (--iters 100): a rollback to best.pt of 50, lr scale 0.5, ckpt.pt =
    best.pt's weights; then --resume to 300 with 0.75, 0.2, 0.1: the scale comes back, a second rollback, then
    {stopped: collapsed} (--max-rollbacks 2). best.pt stays the one of iteration 50 throughout."""
    import contextlib
    import hashlib
    import io
    import json
    import os
    import sys
    import tempfile

    import torch

    from . import train
    from .progress import analyse
    from .train import collapse_reason, lr_at

    print("\n-- the collapse guard --")
    hist = [0.03] * 10
    check(collapse_reason(None, 0.8, 0.6, 0.05, hist) is None and collapse_reason(0.5, 0.8, 0.6, 0.03, hist) is None,
          "collapse_reason: loss 1.7x the median, or a score 0.62 of the best -> no collapse")
    check("4x the median" in (collapse_reason(None, 0.8, 0.6, 0.13, hist) or "")
          and collapse_reason(None, 0.8, 0.6, 0.13, hist[:9]) is None,
          "...a loss 4.3x the median of the 10 windows before -> collapse (with 9 windows: not yet)")
    check("not finite" in (collapse_reason(None, 0.8, 0.6, float("nan"), []) or "")
          and "not finite" in (collapse_reason(None, 0.8, 0.6, None, []) or "")
          and "not finite" in (collapse_reason(None, 0.8, 0.6, 0.03, [], skipped=6, steps=50) or ""),
          "...a non-finite window loss, none at all, or over 10% of its steps skipped -> collapse")
    check("score" in (collapse_reason(0.47, 0.8, 0.6, 0.03, hist) or "")
          and collapse_reason(0.1, 0.39, 0.6, 0.03, hist) is None,
          "...a score below 0.6 x best -> collapse, but only once the best is >= 0.4")
    check(lr_at(0, 1000, 1e-3) == 1e-3 and abs(lr_at(700, 1000, 1e-3) - 3e-4) < 1e-12
          and abs(lr_at(900, 1000, 1e-3) - 1e-4) < 1e-12,
          "lr_at: the step decay as before (x1, x0.3 from 60%, x0.1 from 85%)")
    check(abs(lr_at(0, 1000, 1e-3, warm_from=0, warmup=100) - 1e-5) < 1e-15
          and abs(lr_at(149, 1000, 1e-3, 0.5, warm_from=100, warmup=100) - 2.5e-4) < 1e-12
          and lr_at(200, 1000, 1e-3, 0.5, warm_from=100, warmup=100) == 5e-4
          and lr_at(900, 1000, 1e-3, 1 / 64, floor=1e-5) == 1e-5,
          "...x the rollback scale, a linear warm-up from a fresh optimiser, never below the floor")

    scores = iter([0.7, 0.8, 0.3, 0.75, 0.2, 0.1])
    seen = {}

    def fake_quick(model, held, mults, **kw):
        s = next(scores)
        seen.setdefault("calls", []).append(s)
        if s == 0.3:  # the check that collapses: best.pt as it is now
            with open("runs/col/best.pt", "rb") as f:
                seen["bestHash"] = hashlib.sha256(f.read()).hexdigest()
        return {m: {"mix": s, "bridge": s, "page": s, "none": 0.0, "both": 0.0, "gap": 0.0, "bridgeByR": {}}
                for m in mults}

    def fake_edit(model, eseq, settle_mult):
        s = seen["calls"][-1]
        return {"edit": [s, s, s], "editByR": {}}

    base = ["nca.train", "--name", "col", "--R", "3", "--batch", "4", "--pool-size", "16", "--hidden", "16",
            "--eval-mults", "1", "--eval-every", "50", "--snap-every", "0", "--threads", "1", "--device", "cpu",
            "--max-rollbacks", "2"]
    real = train.quick_eval, train.edit_eval, os.getcwd(), sys.argv, torch.get_num_threads()
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            train.quick_eval, train.edit_eval = fake_quick, fake_edit
            os.chdir(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                sys.argv = base + ["--iters", "100"]
                train.main()
                a_ckpt, a_best = load("runs/col/ckpt.pt"), load("runs/col/best.pt")
                sys.argv = base + ["--iters", "300", "--resume"]
                train.main()
            with open("runs/col/log.jsonl") as f:
                lines = [json.loads(x) for x in f]
            with open("runs/col/best.pt", "rb") as f:
                end_hash = hashlib.sha256(f.read()).hexdigest()
            end_best = load("runs/col/best.pt")
            info = analyse("runs/col")
        finally:
            train.quick_eval, train.edit_eval = real[0], real[1]
            os.chdir(real[2])
            sys.argv = real[3]
            torch.set_num_threads(real[4])
    rb = [x for x in lines if "rollback" in x]
    check(len(rb) == 2 and [x["iteration"] for x in rb] == [100, 200] and [x["lrScale"] for x in rb] == [0.5, 0.25]
          and all(x["restored"] == 50 and "score" in x["reason"] for x in rb),
          f"forced collapse: rollbacks at 100 and 200 (score 0.3, 0.2 < 0.6 x best 0.8), each to best.pt of 50, the "
          f"lr scale 0.5 then 0.25 ({rb[0]['reason'] if rb else 'none'})")
    check(a_ckpt["lrScale"] == 0.5 and a_ckpt["rollbacks"] == 1 and a_ckpt["warmFrom"] == 100
          and all(torch.equal(a_ckpt["model"][k], a_best["model"][k]) for k in a_best["model"]),
          "...the checkpoint written at the rollback has best.pt's weights, lrScale 0.5, rollbacks 1, warmFrom 100")
    starts = [x for x in lines if "config" in x]
    w150 = next(x for x in lines if x.get("iteration") == 150 and "loss" in x)
    check(len(starts) == 2 and starts[1]["lrScale"] == 0.5 and starts[1]["rollbacks"] == 1
          and abs(w150["lr"] - 5e-4 * 0.5 * 0.5) < 1e-12,
          "...--resume keeps the scale: its start line says lrScale 0.5, rollbacks 1, and iteration 149 runs at "
          f"5e-4 x 0.5 x warm-up 0.5 = {w150['lr']:.3g}")
    check(lines[-1].get("stopped") == "collapsed" and lines[-1]["iteration"] == 250 and lines[-1]["rollbacks"] == 2
          and not any(x.get("stopped") == "done" and x["iteration"] > 100 for x in lines),
          "...a third collapse with --max-rollbacks 2 ends the run cleanly: {stopped: collapsed} at 250")
    check(end_best["iteration"] == 50 and end_best["best"] == 0.8 and end_hash == seen.get("bestHash"),
          "...best.pt is the one of iteration 50 (score 0.8 kept in it), byte-identical from before the first "
          "rollback to the end")
    check(info["verdict"] == "COLLAPSED" and info["exitCode"] == 5 and info["rollbacks"] == 2
          and info["lrScale"] == 0.25,
          f"...nca.progress: COLLAPSED (exit 5), rollbacks 2, lr scale 0.25 ({info['reason']})")


def progress_checks() -> None:
    """nca.progress's verdicts on synthetic logs (one per verdict)."""
    from .progress import selftest as progress_selftest

    print("\n-- nca.progress --")
    progress_selftest()


def _components(cells: np.ndarray) -> int:
    """Number of 6-connected components of the True cells."""
    seen = np.zeros_like(cells, dtype=bool)
    S0, S1 = cells.shape
    count = 0
    for start in map(tuple, np.argwhere(cells)):
        if seen[start]:
            continue
        count += 1
        seen[start] = True
        stack = [start]
        while stack:
            row, col = stack.pop()
            for dr, dc in NEIGHBOURS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < S0 and 0 <= nc < S1 and cells[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def span_cases(q, r, on_board, walls_of, only) -> None:
    """The second measure by hand (spec v4 §2, R=6): span = min(max1 - min1, max2 - min2) over a region's
    rim cells. Angles by hand: (q, r) = (3, 3) is at 30 degrees, (-6, 3) at 150, (3, -6) at 270."""
    R = 6
    from .data import arc_fill, aux_span, aux_targets

    def spans_of(w):
        lab = _labels(on_board & (w == 0), R)
        return lab, _spans(lab, R)

    # Row r = 2: the small side's rim runs from (3, 3) at 30 degrees to (-6, 3) at 150: a third of the circle,
    # in theta1 and in theta2 alike (it straddles neither 0 nor 0.5).
    w = walls_of(r == 2)
    lab, sp = spans_of(w)
    small, large = sp[lab[R + 3, R]], sp[lab[R - 3, R]]
    check(abs(small - 1 / 3) < 1e-6 and large > 0.9,
          f"span by hand: row r=2 cuts off a side whose rim runs 30..150 degrees, span {small:.6f} = 1/3; "
          f"the other side {large:.3f}")

    # Column q = 2: the east side's rim runs from (3, -6) at 270 degrees through 0 to (3, 3) at 30. Its theta1
    # values straddle 0 (span ~1), theta2 = theta1 + 0.5 doesn't: span = theta2's 0.5833 - 0.25 = 1/3.
    w = walls_of(q == 2)
    lab, sp = spans_of(w)
    x = lab[R, R + 4]
    c = consts(R)
    east = (lab == x) & rim(R)
    s1 = float(c[1][east].max()) - float(c[1][east].min())
    s2 = float(c[2][east].max()) - float(c[2][east].min())
    check(s1 > 0.9 and abs(s2 - 1 / 3) < 1e-6 and abs(sp[x] - 1 / 3) < 1e-6,
          f"span by hand, wrap-around: column q=2 cuts off 270..30 degrees through 0: theta1 span {s1:.3f}, "
          f"theta2 span {s2:.6f}, span {sp[x]:.6f} = 1/3")
    fills, _ = targets(w, R)
    check(len(fills) == 1 and np.array_equal(fills[0], only(q > 2, w)), "...and the east side is the one that fills")

    # Rim walls eat the large side's angle: row r = 2 again, but the large side keeps only its rim cells
    # from 199 to 341 degrees open (span 0.394). Both sides now span less than half the rim, so v3's
    # threshold rule (fill iff span < 0.5) fills both; the comparative rule (span < board max - EPS) fills
    # the small side only, which is the one target (the large side is the larger by area and by span).
    deg = np.degrees(np.arctan2(1.5 * r, math.sqrt(3) * (q + r / 2))) % 360
    eaten = rim(R) & (r < 2) & ~((deg > 195) & (deg < 345))
    w = walls_of((r == 2) | eaten)
    lab, sp = spans_of(w)
    s_small, s_large = sp[lab[R + 3, R]], sp[lab[R - 3, R]]
    a = aux_targets(w, R)
    span_map = aux_span(a)
    threshold = on_board & (w == 0) & (span_map < 0.5)
    fills, _ = targets(w, R)
    check(s_small < 0.5 and s_large < 0.5 and s_large > s_small + EPS and len(fills) == 1
          and np.array_equal(fills[0], only(r > 2, w)) and threshold[R - 3, R]
          and np.array_equal(arc_fill(a, w, R), fills[0] > 0)
          and np.allclose(a[4][on_board], s_large, rtol=0, atol=1e-6),
          f"rim walls eat angle: spans {s_small:.3f} (small side) and {s_large:.3f} (large side), both < 0.5, so "
          f"'span < 0.5' would fill both; arc_fill (span < ch6 - EPS, ch6 = {a[4][R, R]:.3f} on every cell) "
          f"fills the small side only = the target")

    # Two sides within EPS: both are candidates whatever their areas.
    w = walls_of(r == 0)
    lab, sp = spans_of(w)
    fills, _ = targets(w, R)
    check(abs(sp[lab[R + 3, R]] - sp[lab[R - 3, R]]) < EPS and len(fills) == 2,
          f"a bridge through the centre: spans {sp[lab[R + 3, R]]:.4f} / {sp[lab[R - 3, R]]:.4f}, within EPS: "
          f"two targets")


def span_brute(rng) -> None:
    """_spans and targets() against the brute force on bridge-heavy boards (where the span decides)."""
    n = n_multi = n_span = 0
    for R in (4, 6, 8):
        on = mask(R) == 1
        for _ in range(60):
            w = random_walls(rng, R, "bridge")
            fills, depth = targets(w, R)
            bf_fills, _ = _bruteforce_targets(w, R)
            if not np.array_equal(fills[0], bf_fills[0]) or \
                    sorted(f.tobytes() for f in fills) != sorted(f.tobytes() for f in bf_fills):
                raise AssertionError(f"targets() on a bridge board disagrees with brute force (R={R})")
            ids, area, span = _rim_regions(_labels(on & (w == 0), R), R)
            n += 1
            n_multi += len(ids) >= 2
            n_span += len(ids) >= 2 and int(np.argmax(span)) != 0  # max span is not the max-area region
    check(n == 180, f"targets() matches the brute force on 180 bridge boards (R 4/6/8; {n_multi} with 2+ rim "
          f"regions, {n_span} where the max-span region is not the max-area one)")


def generator_checks(rng) -> None:
    """The line drawer, the page-loop port, and determinism."""
    R = 8
    on_board = mask(R) == 1
    corners = np.array([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)], float)
    exact = all(np.array_equal(_polyline(R, corners * r0 + (cq, cr), True) & on_board,
                               (_axial_dist(R, cq, cr) == r0) & on_board)
                for cq, cr, r0 in [(0, 0, 1), (0, 0, 8), (2, -3, 4), (-5, 1, 5), (3, 3, 2)])
    check(exact, "_polyline through a hexagon's corners draws exactly the hex ring")

    connected = 0
    for _ in range(200):
        cq, cr = _offset(rng, 2)  # with s <= 4 the points stay within hex distance 7: on the board
        line = _polyline(R, _poly_points(rng, cq, cr, rng.uniform(1, 4)), closed=True)
        connected += _components(line & on_board) == 1
    check(connected == 200, "200 random closed polylines on the board are each one 6-connected line")

    n_page = 0
    for R in range(4, 10):
        for _ in range(30):
            w = page_loops(rng, R)
            if int(w[rim(R)].sum()) != 0:
                raise AssertionError(f"page_loops put a wall on the rim (R={R})")
            if int(oracle(w, R)[0].sum()) == 0:
                raise AssertionError(f"page_loops drew a board that fills nothing (R={R})")
            n_page += 1
    check(n_page == 180, "page_loops (R 4..9, 180 boards): never on the rim, always fills something")
    check(int(page_loops(rng, 3).sum()) == 0, "page_loops draws nothing below R=4, like the page")

    def draw(seed):
        g = np.random.default_rng(seed)
        out = []
        for R in (3, 6, 8):
            w = random_walls(g, R)
            out += [w, edit_walls(g, w, R), page_loops(g, R), random_walls(g, R, kind="bridge")]
        return out
    check(all(np.array_equal(a, b) for a, b in zip(draw(7), draw(7))),
          "random_walls (incl. bridges), edit_walls and page_loops are deterministic given the Generator")


BINS = (0, 0.25, 0.5, 0.75, 1.0)  # area ratio (second-largest / largest rim region) bins, last one closed


def _info(w, R):
    """(primary filled cells, number of targets K, number of rim regions, area ratio or nan, max depth)."""
    fills, depth = targets(w, R)
    lab = _labels((mask(R) == 1) & (w == 0), R)
    _, area, _ = _rim_regions(lab, R)
    ratio = area[1] / area[0] if len(area) >= 2 else np.nan
    return int(fills[0].sum()), len(fills), len(area), ratio, int(depth.max())


def _hist(ratios):
    """Shares of the (non-nan) ratios in the four BINS, and the share of exact ties."""
    r = ratios[~np.isnan(ratios)]
    h = np.histogram(r, bins=BINS)[0] / max(1, len(r))
    return h, float((r == 1).mean()) if len(r) else 0.0


def _stats(boards, R):
    """Per-board arrays: filled cells, K, rim regions, area ratio, max depth."""
    return [np.array(x) for x in zip(*(_info(w, R) for w in boards))]


def board_stats(rng) -> None:
    """Per kind and for the mix at R=8: fill share, rim regions, targets, oracle depths, speed."""
    R = 8
    cells = int(mask(R).sum())
    n_per_kind = 300
    print(f"\n-- R={R}, {n_per_kind} boards per kind: share with a filled cell, mean filled fraction, "
          f"share with 2+ rim regions, share with K>=2 targets, max depth p50/p99 --")

    def row(name, boards, note=""):
        filled, K, rims, ratio, depth = _stats(boards, R)
        print(f"  {name:6s}: {(filled > 0).mean():.2f}  {filled.mean() / cells:.3f}  {(rims >= 2).mean():.2f}  "
              f"{(K >= 2).mean():.3f}  {np.percentile(depth, 50):3.0f} {np.percentile(depth, 99):3.0f}{note}")
        return filled, K, rims, ratio, depth

    for kind in _KINDS:
        row(kind, [random_walls(rng, R, kind=kind) for _ in range(n_per_kind)])
    row("page", [page_loops(rng, R) for _ in range(n_per_kind)], "   (held out: evaluation only)")

    n_mix = 4000
    t0 = time.perf_counter()
    boards = [random_walls(rng, R) for _ in range(n_mix)]
    for w in boards:
        targets(w, R)
    ms = 1000 * (time.perf_counter() - t0) / n_mix
    filled, K, rims, ratio, depth = row("mix", boards, f"   (n={n_mix})")
    share, multi = float((filled > 0).mean()), float((rims >= 2).mean())
    sizes = np.percentile(filled[filled > 0], [10, 50, 90, 100])
    deep = np.percentile(depth, [50, 90, 99, 99.9])
    hist, ties = _hist(ratio)
    b_hist, b_ties = _hist(_stats([random_walls(rng, R, kind="bridge") for _ in range(600)], R)[3])
    fmt = lambda h: " ".join(f"{x:.2f}" for x in h)
    print(f"    weights: " + ", ".join(f"{k} {v}" for k, v in _MIX.items()))
    print(f"    filled cells, boards with any (of {cells}): p10 {sizes[0]:.0f}, p50 {sizes[1]:.0f}, "
          f"p90 {sizes[2]:.0f}, max {sizes[3]:.0f}")
    print(f"    2+ rim regions: {multi:.3f} ({(rims >= 3).mean():.3f} with 3+); K>=2 targets: {(K >= 2).mean():.3f}, "
          f"max K {K.max()}")
    print(f"    area ratio of boards with 2+ rim regions, bins [0,.25) [.25,.5) [.5,.75) [.75,1]: "
          f"mix {fmt(hist)} (ties {ties:.2f}); bridge kind {fmt(b_hist)} (ties {b_ties:.2f})")
    print(f"    max oracle depth: p50 {deep[0]:.0f}, p90 {deep[1]:.0f}, p99 {deep[2]:.0f}, p99.9 {deep[3]:.0f}, "
          f"max {depth.max()}")
    print(f"    {ms:.3f} ms/board (random_walls + targets, R={R})")
    check(0.45 <= share <= 0.7, f"overall share with a filled cell in [0.45, 0.7] (got {share:.3f})")
    check(0.25 <= multi <= 0.3, f"overall share with 2+ rim regions in [0.25, 0.3] (got {multi:.3f})")
    check(b_hist.min() >= 0.1 and b_ties >= 0.05,
          "bridge boards spread over all four area-ratio bins (each >= 10%), with exact ties (>= 5%)")
    check(sizes[0] <= 4 and sizes[2] >= 40, "filled regions run from a few cells to a large part of the board")
    check(deep[2] < 40, f"99th percentile of max oracle depth at R={R} under 40 (got {deep[2]:.0f})")
    check(ms < 3, f"under 3 ms a board (got {ms:.2f})")


def edit_stats() -> None:
    """How often an edit changes the primary target (the pool's "user edits a settled board" signal)."""
    R, n = 8, 1000
    print(f"\n-- edits, R={R}, {n} boards --")
    for kind in (None, "bridge"):
        g = np.random.default_rng(2024)
        changed = beyond = 0
        t0 = time.perf_counter()
        for _ in range(n):
            w = random_walls(g, R, kind=kind)
            e = edit_walls(g, w, R)
            diff = oracle(e, R)[0] != oracle(w, R)[0]
            changed += bool(diff.any())
            beyond += bool((diff & (e == w)).any())
        ms = 1000 * (time.perf_counter() - t0) / n
        print(f"  {kind or 'mix':6s}: the primary target changed on {changed / n:.3f} of edits ({beyond / n:.3f} beyond "
              f"the edited cells themselves); {ms:.2f} ms per board + edit")
        if kind is None:
            share = changed / n
    check(0.6 <= share <= 0.7, f"60-70% of edits change the primary target (got {share:.3f})")


def distinctness() -> None:
    """Exact repeats within one seed, and held-out boards that also turn up in training."""
    R = 6
    g = np.random.default_rng(0)
    seen, dups, empty_dups = set(), 0, 0
    for _ in range(2000):
        key = random_walls(g, R).tobytes()
        if key in seen:
            dups += 1
            empty_dups += not any(key)
        seen.add(key)
    g = np.random.default_rng(0)
    train = {random_walls(g, R).tobytes() for _ in range(20000)}
    g = np.random.default_rng(31337)
    again = sum(random_walls(g, R).tobytes() in train for _ in range(500)) / 500
    print(f"\n-- distinctness, R={R} --")
    print(f"  2000 boards from one seed: {dups} repeat an earlier one ({empty_dups} of them empty); "
          f"{len(train)} distinct of 20000")
    print(f"  500 boards from another seed: {again:.3f} also among 20000 training-seed boards")
    check(dups - empty_dups <= 20, f"at most 1% of 2000 boards repeat (got {dups})")
    check(again <= 0.03, f"at most 3% of held-out boards occur in training (got {again:.3f})")


def const_checks() -> None:
    """hexgrid.consts (spec v3 + v6 §1): mask, theta1, theta2, src1, src1c, src2, src2c."""
    import math

    from .hexgrid import CONST_NAMES, consts

    print("\n-- constant inputs --")
    check(CONST_NAMES == ("mask", "theta1", "theta2", "src1", "src1c", "src2", "src2c"),
          "consts are (mask, theta1, theta2, src1, src1c, src2, src2c), in that order")
    ok_shape = ok_mask = ok_range = ok_off = ok_t2 = ok_centre = ok_src = True
    worst_opp, n_opp = 0.0, 0
    for R in range(0, 17):
        S = side(R)
        c = consts(R)
        on = mask(R) == 1
        ok_shape &= c.shape == (7, S, S) and c.dtype == np.float32
        ok_mask &= np.array_equal(c[0], mask(R).astype(np.float32))
        ok_range &= bool(((c[1:3] >= 0) & (c[1:3] < 1)).all())
        # The rim sources, from the angles in double: rim * theta, rim * (1 - theta), stored float32.
        th1 = np.mod(np.arctan2(1.5 * (np.arange(S) - R)[:, None],
                                math.sqrt(3) * ((np.arange(S) - R)[None, :] + (np.arange(S) - R)[:, None] / 2))
                     / (2 * math.pi) + 1, 1)
        th2 = np.mod(th1 + 0.5, 1)
        e = rim(R)
        want = np.stack([e * th1, e * (1 - th1), e * th2, e * (1 - th2)]).astype(np.float32)
        ok_src &= np.array_equal(c[3:], want)
        ok_off &= bool((c[:, ~on] == 0).all())
        t1, t2 = c[1].astype(np.float64), c[2].astype(np.float64)
        d = np.abs(np.mod(t1 + 0.5, 1.0) - t2)[on]
        ok_t2 &= bool((np.minimum(d, 1 - d) < 1e-6).all())  # mod 1: 0.9999999 and 0 are neighbours
        ok_centre &= c[1, R, R] == 0 and c[2, R, R] == np.float32(0.5)
        # (q, r) and (-q, -r) are opposite (the array flipped both ways): theta1 differs by 0.5 (mod 1).
        for row, col in np.argwhere(rim(R)) if R > 0 else ():  # R = 0: the centre is its own opposite
            diff = (float(c[1, row, col]) - float(c[1, S - 1 - row, S - 1 - col])) % 1.0
            worst_opp = max(worst_opp, abs(diff - 0.5))
            n_opp += 1
    check(ok_shape and ok_mask, "consts(R) is float32 [7,S,S] and plane 0 is mask(R) (R = 0..16)")
    check(ok_src, "src1, src1c, src2, src2c = rim * theta1, rim * (1 - theta1), rim * theta2, rim * (1 - theta2), "
          "computed in double and stored float32, 0 off the rim")
    check(ok_range, "theta1 and theta2 are in [0, 1)")
    check(ok_off, "every const plane is 0 off board")
    check(ok_t2, "theta2 = (theta1 + 0.5) mod 1 on every on-board cell")
    check(ok_centre, "the centre cell has theta1 = 0 (atan2(0, 0) = 0), so theta2 = 0.5")
    check(n_opp > 0 and worst_opp < 1e-6,
          f"opposite rim cells differ by 0.5 in theta1 ({n_opp} rim cells, worst |diff - 0.5| {worst_opp:.1e})")
    # A few cells by hand (x = sqrt(3)(q + r/2), y = 1.5 r; row = r+R, col = q+R).
    R = 4
    c = consts(R)
    hand = {(1, 0): 0.0, (0, 1): 1 / 6, (-1, 1): 1 / 3, (-1, 0): 0.5, (0, -1): 2 / 3, (1, -1): 5 / 6,
            (1, 1): 1 / 12, (R, -R): 5 / 6, (-2, 1): math.atan2(1.5, -1.5 * math.sqrt(3)) / (2 * math.pi)}
    worst = max(abs(float(c[1, rr + R, qq + R]) - v) for (qq, rr), v in hand.items())
    check(worst < 1e-6, f"theta1 by hand on 9 cells of R=4 (the six directions at 0, 1/6, .., 5/6) (max err {worst:.1e})")


def _flood_brute(w: np.ndarray, R: int, T: int) -> np.ndarray:
    """float32 [T,5,S,S]: spec v5 §2's flood written out from distances instead of iterated (an independent
    check of data.aux_flood). F_t on plane c (0..3) at an open cell x is the max of the rim source over the
    open rim cells s with geodesic distance d(s, x) <= t - 1 through open cells (own BFS), 0 if none -- so it
    can only come from x's own region; plane 4 at an on-board y is the max over k <= t and on-board x with
    hex distance |x - y| <= t - k of span_k(x) (ungated: the hexagon is convex, so that is the on-board path
    length too)."""
    S = side(R)
    on = mask(R) == 1
    opn = on & (w == 0)
    src = consts(R)[3:7]  # spec v6 §1: theta1, 1 - theta1, theta2, 1 - theta2 on the rim (read at rim cells only)
    cells = [tuple(x) for x in np.argwhere(on)]
    out = np.zeros((T, 5, S, S), dtype=np.float32)
    for s in map(tuple, np.argwhere(opn & rim(R))):
        d = {s: 0}
        todo = [s]
        for a in todo:
            for dr, dc in NEIGHBOURS:
                b = (a[0] + dr, a[1] + dc)
                if 0 <= b[0] < S and 0 <= b[1] < S and opn[b] and b not in d:
                    d[b] = d[a] + 1
                    todo.append(b)
        for x, k in d.items():
            for c in range(4):
                out[k:, c, x[0], x[1]] = np.maximum(out[k:, c, x[0], x[1]], src[c][s])
    p = out[:, :4]
    span = np.clip(np.minimum(p[:, 0] + p[:, 1] - 1, p[:, 2] + p[:, 3] - 1), 0, 1)  # [T,S,S]
    qr = np.array([(c - R, r - R) for r, c in cells])
    dq, dr_ = qr[:, None, 0] - qr[None, :, 0], qr[:, None, 1] - qr[None, :, 1]
    D = np.maximum(np.maximum(np.abs(dq), np.abs(dr_)), np.abs(dq + dr_))  # [N,N] hex distance
    sp = np.stack([span[t][on] for t in range(T)])  # [T,N], cells in np.argwhere order = boolean order
    for t in range(T):
        best = np.zeros(len(cells), dtype=np.float32)
        for k in range(t + 1):
            best = np.maximum(best, (sp[k][:, None] * (D <= t - k)).max(0))
        out[t, 4][on] = best
    return out


def aux_checks(rng) -> None:
    """Spec v5: data.aux_flood (the reference flood F_1..F_T of channels 2..6: max-floods from 0, one hop per
    step) and its steady state data.aux_targets; how long the flood takes to settle; and how often the
    comparative rule read off the steady state (arc_fill: fill iff enclosed or span < ch6 - EPS) is an
    acceptable target."""
    from .data import N_AUX, arc_fill, aux_flood, aux_span, aux_targets

    print("\n-- aux floods (spec v5: rim-theta max-floods from 0, largest span) --")
    R = 6
    S = side(R)
    on = mask(R) == 1
    q = (np.arange(S) - R).reshape(1, S) + np.zeros((S, 1), int)
    r = (np.arange(S) - R).reshape(S, 1) + np.zeros((1, S), int)
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))
    rm = rim(R)
    whole = [v[rm].max() for v in consts(R)[3:7]]  # the rim sources src1, src1c, src2, src2c
    whole_span = float(np.clip(min(whole[0] + whole[1] - 1, whole[2] + whole[3] - 1), 0, 1))
    T = 12 * R

    w = np.zeros((S, S), np.uint8)
    F, a = aux_flood(w, R, T), aux_targets(w, R)
    centre = F[:, 0, R, R]
    check(N_AUX == 5 and a.shape == (5, S, S) and a.dtype == np.float32 and F.shape == (T, 5, S, S)
          and F.dtype == np.float32, "aux_targets is float32 [5,S,S], aux_flood(walls, R, T) float32 [T,5,S,S]")
    check(all(np.all(a[c][on] == whole[c]) for c in range(4)) and np.all(a[4][on] == np.float32(whole_span))
          and not a[:, ~on].any() and np.array_equal(F[-1], a),
          f"empty board: every cell settles to the whole rim's max theta1, max 1-theta1, max theta2, max "
          f"1-theta2; span = ch6 = {whole_span:.3f}; F_T = aux_targets")
    check(not centre[:R].any() and centre[R] > 0 and np.all(F[0, :4][:, ~rm] == 0),
          f"one hop per step: F_1 is the rim sources alone, the centre (R={R} from the rim) is 0 up to F_{R} and "
          f"reached at F_{R + 1}")

    # A straight bridge (row r = 2): two rim regions, the south side (r > 2) the smaller.
    w = ((r == 2) & on).astype(np.uint8)
    F, a = aux_flood(w, R, T), aux_targets(w, R)
    small, large = on & (r > 2), on & (r < 2)
    sp = aux_span(a)
    check(np.array_equal(F[-1], a) and sp[small].max() == sp[small].min() and sp[large].max() == sp[large].min()
          and sp[small].max() < sp[large].min() - EPS and np.all(F[-1, 4][on] == sp[large].min())
          and not F[:, :4][:, :, w > 0].any() and np.array_equal(arc_fill(a, w, R), targets(w, R)[0][0] > 0),
          f"straight bridge (row r=2): each side settles to one span, small {sp[small].max():.3f} < large "
          f"{sp[large].min():.3f} - EPS; ch6 = the large side's span on every cell (walls too); the wall row "
          f"is 0 in ch2..5 at every step; arc_fill of the steady state is the target")

    # A closed ring (dist 3): nothing reaches inside in ch2..5, ever; ch6 does (through the wall).
    w = ((dist == 3) & on).astype(np.uint8)
    F, a = aux_flood(w, R, T), aux_targets(w, R)
    check(not F[:, :4][:, :, dist <= 3].any() and np.array_equal(F[-1], a)
          and np.all(F[-1, 4][on] == np.float32(whole_span)),
          "closed ring: ch2..5 stay 0 inside and on the ring at every step; ch6 crosses the ring and settles to "
          "the whole rim's span on every cell")

    w = rm.astype(np.uint8)
    F, a = aux_flood(w, R, T), aux_targets(w, R)
    check(not F.any() and not a.any() and arc_fill(a, w, R).sum() == int((on & ~rm).sum()),
          "rim fully walled (no rim region): every plane 0 at every step, and arc_fill fills every open cell")

    w = rm.copy()
    w[R, 2 * R] = False  # a single open rim cell: one rim region of span 0
    w = (w | ((dist == 3) & on)).astype(np.uint8)
    a = aux_targets(w, R)
    check(np.array_equal(arc_fill(a, w, R), targets(w, R)[0][0] > 0) and a[4].max() == 0
          and np.array_equal(aux_flood(w, R, T)[-1], a),
          "one open rim cell (span 0, ch6 0) plus a closed ring: arc_fill fills the ring's inside only = the target")

    # Brute force from distances (_flood_brute) on random boards, every step: exact.
    n = 0
    for R in (3, 4, 5):
        for i in range(12):
            w = random_walls(rng, R, "bridge" if i % 3 == 0 else None)
            T = 6 * R + 4
            if not np.array_equal(aux_flood(w, R, T), _flood_brute(w, R, T)):
                raise AssertionError(f"aux_flood disagrees with the brute force (R={R}, board {i})")
            n += 1
    check(True, f"aux_flood = an independent brute force from distances at every step on {n} random boards "
          f"(R 3/4/5, a third of kind bridge): ch2..5 only from the cell's own region's rim cells within t-1 "
          f"steps (never across a wall), ch6 the spans seen within the ungated hex distance")

    # 100 random boards (a quarter of kind bridge), R 6/8/10: F_T = aux_targets, ch6 one value on the board,
    # and how many steps until F_t stops changing (planes 0..3, then plane 4).
    settle, settle4, per_r = [], [], []
    same = flat = 0
    for i in range(100):
        R = (6, 8, 10)[i % 3]
        w = random_walls(rng, R, "bridge" if i % 4 == 0 else None)
        on = mask(R) == 1
        T = 12 * R
        F, a = aux_flood(w, R, T), aux_targets(w, R)
        same += np.array_equal(F[-1], a)
        flat += F[-1, 4][on].min() == F[-1, 4][on].max()
        diff = [not np.array_equal(F[t], a) for t in range(T)]
        diff4 = [not np.array_equal(F[t, :4], a[:4]) for t in range(T)]
        settle.append(T - diff[::-1].index(True) + 1 if any(diff) else 1)
        settle4.append(T - diff4[::-1].index(True) + 1 if any(diff4) else 1)
        per_r.append(R)
    settle, settle4, per_r = np.array(settle), np.array(settle4), np.array(per_r)
    check(same == 100, "aux_targets = aux_flood's F_T (T = 12R) exactly on 100 random boards (R 6/8/10)")
    check(flat == 100, "...and once settled, ch6 holds one value on every on-board cell of every board")
    print(f"  steps to settle (F_t = the steady state from t on), 100 boards: all planes max {settle.max()} "
          f"(= {(settle / per_r).max():.2f} R), mean {np.mean(settle / per_r):.2f} R, 95th pct "
          f"{np.percentile(settle / per_r, 95):.2f} R; ch2..5 alone max {(settle4 / per_r).max():.2f} R, "
          f"mean {np.mean(settle4 / per_r):.2f} R; by R: " + ", ".join(
              f"R {R}: max {settle[per_r == R].max()}" for R in (6, 8, 10)))
    check((settle / per_r).max() <= 6, f"the flood settles within 6R steps on all 100 (max "
          f"{(settle / per_r).max():.2f} R)")

    # Random boards with 2+ rim regions (half of them drawn as kind "bridge"), R in 6, 8, 10: is the comparative
    # rule, read off exact aux targets, an acceptable target? Near-tie = the two largest spans within EPS.
    rows = []
    while len(rows) < 300:
        R = int(rng.choice([6, 8, 10]))
        w = random_walls(rng, R, "bridge" if rng.random() < 0.5 else None)
        on = mask(R) == 1
        lab = _labels(on & (w == 0), R)
        ids, _, sp = _rim_regions(lab, R)
        if len(ids) < 2:
            continue
        f = arc_fill(aux_targets(w, R), w, R)
        ok = any(np.array_equal(f, t > 0) for t in targets(w, R)[0])
        top = np.sort(sp)[::-1]
        empty = sum(not f[lab == x].any() for x in ids)
        rows.append((ok, top[0] - top[1] < EPS, top[0] == top[1], len(ids), empty))
    ok, near, tie, n_reg, empty = (np.array(x) for x in zip(*rows))
    print(f"  300 boards with 2+ rim regions: arc_fill (exact floods) = an acceptable target on {ok.mean():.3f}; "
          f"not a near-tie ({(~near).mean():.2f} of boards): {ok[~near].mean():.3f}; near-ties: {ok[near].mean():.3f} "
          f"({near.sum()} boards, {tie.sum()} exact ties -- the point-symmetric bridge boards; there the rule "
          f"leaves both sides empty)")
    for k, sel in (("2", n_reg == 2), ("3", n_reg == 3), ("4+", n_reg >= 4)):
        print(f"    {k:2s} rim regions: n {sel.sum():3d}, exact {ok[sel].mean():.3f} (not near-tie "
              f"{ok[sel & ~near].mean() if (sel & ~near).any() else float('nan'):.3f}); when wrong: every "
              f"region filled {((empty == 0) & ~ok & sel).sum()}, 2+ left empty {((empty >= 2) & ~ok & sel).sum()}")
    check(ok[~near].mean() >= 0.99 and not (~ok & (empty == 0)).any(),
          f"the comparative rule with exact floods is an acceptable target on {ok[~near].mean():.3f} of the boards "
          f"that aren't near-ties (>= 0.99), and never fills every side")


def model_checks(rng) -> None:
    """HexNCA invariants, the export round trip (versions 1 and 2), and --init expansion."""
    import json
    import os
    import tempfile

    import torch

    from .export import export, model_from_json, run_parity, parity_walls, self_check, weights_json
    from .model import HexNCA, const_stack, fresh_state, load_expanded

    print("\n-- model --")
    torch.manual_seed(0)
    R = 5
    S = side(R)
    cs = const_stack(R)
    check(tuple(cs.shape) == (1, 3, S, S), "const_stack(R) is [1, 3, S, S]")
    w = np.stack([random_walls(rng, R) for _ in range(4)])
    walls = torch.from_numpy(w).float().unsqueeze(1)

    pool_checks()

    # One optimiser step, the way train.py takes it (grad normalisation + Adam).
    model = HexNCA()
    check(model.n_consts == 3 and tuple(model.w1.shape) == (64, 19, 3, 3) and model.perception == "taps+pool"
          and tuple(model.w1pool.shape) == (64, 32, 1, 1),
          "HexNCA() takes 3 consts and pools: w1 [64, 19, 3, 3], w1pool [64, 32, 1, 1]")
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):  # twice, so the second step runs with a non-zero w2
        out = model(fresh_state(walls), walls, cs, 8)
        loss = (out[:, 1] - 1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        for prm in model.parameters():
            prm.grad /= prm.grad.norm() + 1e-8
        opt.step()
    check(float(model.w1[:, :, 0, 0].abs().max()) == 0.0 and float(model.w1[:, :, 2, 2].abs().max()) == 0.0,
          "corner taps (k=0, k=8) are exactly zero after optimiser steps")
    check(float(model.w2.abs().max()) > 0 and float(model.w1pool.grad.abs().max()) > 0,
          "...and the optimiser did move w2, and w1pool gets a gradient")

    # Off-board cells stay exactly zero, and ch0 is the wall picture, after every step.
    with torch.no_grad():
        model.w2.normal_(0, 0.3)  # a lively model, so these aren't trivially true
        model.b2.normal_(0, 0.3)
        state = fresh_state(walls)
        off_ok, wall_ok = True, True
        for _ in range(30):
            state = model.step(state, walls, cs)
            off_ok &= bool((state * (1 - cs[:, :1])).abs().max() == 0)
            wall_ok &= bool(torch.equal(state[:, 0:1], walls))
    check(off_ok, "off-board cells stay exactly zero for 30 steps")
    check(wall_ok, "the wall channel equals the wall picture after every step")

    # One step = the spec v4 formula written out: w1pool[h, j] times the max of channel j plus w1pool[h, C+j]
    # times its min (hex_pool, checked against a brute force above), added to the taps' pre-activation.
    import torch.nn.functional as F

    from .model import hex_pool
    with torch.no_grad():
        mk = cs[:, :1]
        mx, mn = hex_pool(state, mk, -1.0, 1.0)
        x = torch.cat([state, cs.expand(len(state), -1, -1, -1)], 1)
        pre = F.conv2d(x, model.w1 * model.kmask, model.b1, padding=1) + F.conv2d(torch.cat([mx, mn], 1), model.w1pool)
        want = (state + F.conv2d(F.relu(pre), model.w2, model.b2)).clamp(-1, 1)
        want = torch.cat([walls, want[:, 1:]], 1) * mk
        err = float((model.step(state, walls, cs) - want).abs().max())
    check(err < 1e-5, f"one step = relu(taps . w1 + max . w1pool[:, :C] + min . w1pool[:, C:] + b1) . w2 + b2, "
          f"clamped, as written in the spec (max {err:.1e})")

    # The theta columns matter: zeroing them changes the output of a lively model.
    with torch.no_grad():
        blind = HexNCA()
        blind.load_state_dict(model.state_dict())
        blind.w1[:, 17:] = 0
        d = float((model(fresh_state(walls), walls, cs, 8) - blind(fresh_state(walls), walls, cs, 8)).abs().max())
        blind.load_state_dict(model.state_dict())
        blind.w1pool.zero_()
        dp = float((model(fresh_state(walls), walls, cs, 8) - blind(fresh_state(walls), walls, cs, 8)).abs().max())
    check(d > 1e-3 and dp > 1e-3, f"the theta inputs and the pools reach the output (zeroing their weights moves "
          f"it by {d:.2f} / {dp:.2f})")

    # Export -> reload round trip (rounded to 6 significant digits), format version 3.
    with torch.no_grad():
        model.w2.normal_(0, 0.05)
        model.b2.normal_(0, 0.05)
    pw = parity_walls()
    with tempfile.TemporaryDirectory() as tmp:
        wp, fp = os.path.join(tmp, "w.json"), os.path.join(tmp, "f.json")
        wj, fix = export(model, {"trainedR": [R], "steps": [0, 0], "iterations": 0, "note": "selftest"}, wp, fp)
        check(wj["version"] == 3 and wj["consts"] == ["mask", "theta1", "theta2"] and wj["perception"] == "taps+pool",
              "export writes version 3 with consts [mask, theta1, theta2] and perception taps+pool")
        check(len(wj["w1"]) == 64 * 19 * 9 and len(wj["w2"]) == 16 * 64 and len(wj["b1"]) == 64
              and len(wj["b2"]) == 16 and len(wj["w1pool"]) == 64 * 32,
              "export has the v4 array sizes (w1: H * (C+3) * 9, w1pool: H * 2C)")
        h, j = 7, 16 + 3  # the min of channel 3
        check(wj["w1pool"][h * 32 + j] == float(f"{float(model.w1pool[h, j, 0, 0]):.6g}"),
              "w1pool index h*2C + j: j >= C is the min of channel j - C")
        check(all(wj["w1"][i] == 0 for i in range(0, len(wj["w1"]), 9))
              and all(wj["w1"][i] == 0 for i in range(8, len(wj["w1"]), 9)), "exported corner taps are zero")
        # w1 layout: ((h*(C+3)) + c)*9 + k, the consts after the state channels.
        h, c, k = 5, 17, 4  # theta1, centre tap
        check(wj["w1"][((h * 19) + c) * 9 + k] == float(f"{float(model.w1[h, c, 1, 1]):.6g}"),
              "w1 index ((h*(C+3))+c)*9+k: c = C+1 is theta1")
        diff = np.abs(run_parity(model, pw, 24) - run_parity(model_from_json(wj), pw, 24)).max()
        check(diff < 1e-4, f"export -> reload round trip matches the original model to 1e-4 (max {diff:.1e})")
        err = self_check(wp, fp)
        check(err < 1e-5, f"the parity fixture reproduces from the files (max {err:.1e})")
        with open(wp) as f:
            check(json.load(f)["version"] == 3, "...the weights file on disk is version 3")
        fixture_bridge = targets(np.array(fix["walls"], dtype=np.uint8).reshape(side(fix["R"]), -1), fix["R"])
        lab = _labels((mask(fix["R"]) == 1) & (pw == 0), fix["R"])
        check(fix["R"] == 5 and fix["steps"] == 24 and len(_rim_regions(lab, fix["R"])[0]) == 2
              and int(fixture_bridge[0].sum()) > 0,
              "the parity picture (R=5, 24 steps) has a rim-to-rim bridge: 2 rim regions, something fills")

    # Version 2 files (3 consts, no pool) still read back, as a taps model.
    with torch.no_grad():
        v2 = HexNCA(perception="taps")
        v2.w2.normal_(0, 0.05)
        v2.b2.normal_(0, 0.05)
    j2 = json.loads(json.dumps(weights_json(v2, {"note": "selftest"})))
    m2 = model_from_json(j2)
    diff = np.abs(run_parity(v2, pw, 24) - run_parity(m2, pw, 24)).max()
    check(j2["version"] == 2 and "perception" not in j2 and "w1pool" not in j2 and m2.perception == "taps"
          and diff < 1e-4, f"a 3-const taps model exports as version 2 and reads back (max {diff:.1e})")

    # Version 1 files (mask only) still read back, as a 1-const model.
    with torch.no_grad():
        v1 = HexNCA(n_consts=1, perception="taps")
        v1.w2.normal_(0, 0.05)
        v1.b2.normal_(0, 0.05)
    j1 = json.loads(json.dumps(weights_json(v1, {"note": "selftest"})))
    m1 = model_from_json(j1)
    diff = np.abs(run_parity(v1, pw, 24) - run_parity(m1, pw, 24)).max()
    check(j1["version"] == 1 and "consts" not in j1 and len(j1["w1"]) == 64 * 17 * 9 and m1.n_consts == 1
          and diff < 1e-4, f"a 1-const model exports as version 1 and reads back (max {diff:.1e})")

    # --init expansion: a 1-const model's weights in a 3-const model compute exactly the same thing.
    with torch.no_grad():
        v1.w2.normal_(0, 0.3)
        v1.b2.normal_(0, 0.3)
        big = HexNCA(n_consts=3)
        big.w1.normal_(0, 1)  # junk that load_expanded must overwrite
        big.w1pool.normal_(0, 1)
        old_n = load_expanded(big, v1.state_dict())
        R = 8
        S = side(R)
        cs8 = const_stack(R)
        g = torch.Generator().manual_seed(1)
        walls8 = (torch.rand(4, 1, S, S, generator=g) < 0.2).float() * cs8[:, :1]
        state = (torch.rand(4, 16, S, S, generator=g) * 2 - 1) * cs8[:, :1]
        state[:, 0:1] = walls8
        one = float((big.step(state, walls8, cs8) - v1.step(state, walls8, cs8[:, :1])).abs().max())
        fresh = fresh_state(walls8)
        roll = float((big(fresh, walls8, cs8, 64) - v1(fresh, walls8, cs8[:, :1], 64)).abs().max())
    check(old_n == 1 and float(big.w1[:, 17:].abs().max()) == 0 and torch.equal(big.w1[:, :17], v1.w1)
          and float(big.w1pool.abs().max()) == 0,
          "load_expanded copies the old w1 columns and zeroes the new const columns and w1pool")
    check(one <= 1e-6 and roll <= 1e-6,
          f"the expanded model reproduces the old one: one step from a random state {one:.1e}, "
          f"64-step rollout {roll:.1e} (tol 1e-6)")
    try:
        load_expanded(HexNCA(n_consts=1), big.state_dict())
        shrank = True
    except ValueError:
        shrank = False
    try:
        load_expanded(HexNCA(perception="taps"), big.state_dict())
        unpooled = True
    except ValueError:
        unpooled = False
    check(not shrank and not unpooled, "load_expanded refuses to drop consts or the pool")


def flood_checks(rng) -> None:
    """Spec v6 §2: the hand-written floods (model.install_floods), checked by direct computation on a model
    whose free parameters are random (so the free channels are busy): channels 2..5 = data.aux_flood's F_t at
    every step, ch7 = the span of the previous step's channels 2..5, ch6 = the ungated max-flood of ch7 (=
    F_{t-2} plane 4), settled after 6R + 4 steps to the largest region span; the frozen parameters bit-identical
    after optimiser steps; the export round trip with the 7 consts."""
    import os
    import tempfile

    import torch

    from .data import aux_flood, aux_targets
    from .export import export, model_from_json, parity_walls, run_parity, self_check
    from .model import FLOOD_UNITS, HexNCA, const_stack, fresh_state, hex_pool

    print("\n-- hand-written floods (spec v6) --")
    torch.manual_seed(6)
    model = HexNCA(n_consts=7, clamp=(-2.0, 2.0), floods=True)
    with torch.no_grad():  # liven up everything that is free: it must not reach channels 2..7
        model.w2.normal_(0, 0.3)
        model.b2.normal_(0, 0.3)
        model.b1.normal_(0, 0.3)
        model.restore_floods()
    check(len(model.frozen) == 5 and int(model.frozen["b1"].sum()) == FLOOD_UNITS == 26
          and int(model.frozen["w2"].sum()) == 6 * 64 and int(model.frozen["b2"].sum()) == 6,
          "26 frozen hidden units (w1, w1pool, b1 rows) and the output rows of channels 2..7 (w2, b2)")

    worst = {"ch2..5 vs F_t": 0.0, "ch7 vs span(prev)": 0.0, "ch7 vs span(F_t-1)": 0.0,
             "ch6 vs maxflood(prev)": 0.0, "ch6 vs F_t-2 plane 4": 0.0, "ch6 settled vs largest span": 0.0}
    n_boards, flat, walls_zero, busy = 0, True, True, 0.0
    for R in (4, 6, 8):
        cs = const_stack(R, 7)
        mk = cs[:, :1]
        T = 6 * R + 4
        w = np.stack([random_walls(rng, R, "bridge" if i % 3 == 0 else None) for i in range(17)])
        walls = torch.from_numpy(w).float()[:, None]
        F = torch.from_numpy(aux_flood(w, R, T))  # [T,B,5,S,S]
        state = fresh_state(walls)
        with torch.no_grad():
            for t in range(T):
                prev, state = state, model.step(state, walls, cs)
                d = lambda a, b: float(((a - b) * mk).abs().max())
                worst["ch2..5 vs F_t"] = max(worst["ch2..5 vs F_t"], d(state[:, 2:6], F[t, :, :4]))
                span = torch.minimum(prev[:, 2] + prev[:, 3] - 1, prev[:, 4] + prev[:, 5] - 1)[:, None] * mk
                worst["ch7 vs span(prev)"] = max(worst["ch7 vs span(prev)"], d(state[:, 7:8], span))
                if t >= 1:
                    f = F[t - 1]
                    sf = torch.minimum(f[:, 0] + f[:, 1] - 1, f[:, 2] + f[:, 3] - 1)[:, None]
                    worst["ch7 vs span(F_t-1)"] = max(worst["ch7 vs span(F_t-1)"], d(state[:, 7:8], sf))
                mx6 = hex_pool(prev[:, 6:7], mk, -2.0, 2.0)[0]
                worst["ch6 vs maxflood(prev)"] = max(worst["ch6 vs maxflood(prev)"],
                                                     d(state[:, 6:7], torch.maximum(mx6, prev[:, 7:8])))
                if t >= 2:  # state after t+1 steps vs F_{t-1} (F[t-2], 0-based)
                    worst["ch6 vs F_t-2 plane 4"] = max(worst["ch6 vs F_t-2 plane 4"], d(state[:, 6], F[t - 2, :, 4]))
                walls_zero &= bool((state[:, 2:6] * walls).abs().max() <= 1e-5)
            busy = max(busy, float(state[:, 8:].abs().max()))
        on = mk[0, 0] > 0
        for b in range(len(w)):
            v = state[b, 6][on]
            flat &= float(v.max() - v.min()) <= 1e-5
            worst["ch6 settled vs largest span"] = max(worst["ch6 settled vs largest span"],
                                                       abs(float(v.mean()) - float(aux_targets(w[b], R)[4].max())))
            n_boards += 1
    for k, v in worst.items():
        print(f"  max |diff| {k}: {v:.1e}")
    check(n_boards >= 50 and worst["ch2..5 vs F_t"] <= 1e-5 and walls_zero,
          f"channels 2..5 = aux_flood's F_t at every step on {n_boards} random boards (R 4/6/8, a third bridges; "
          f"max {worst['ch2..5 vs F_t']:.1e}), 0 on walls, with the free channels busy (|ch8..| up to {busy:.2f})")
    check(worst["ch7 vs span(prev)"] <= 1e-5 and worst["ch7 vs span(F_t-1)"] <= 1e-5,
          f"ch7 = min(ch2 + ch3 - 1, ch4 + ch5 - 1) of the step before, = the span of F_t-1 "
          f"(max {worst['ch7 vs span(prev)']:.1e} / {worst['ch7 vs span(F_t-1)']:.1e})")
    check(worst["ch6 vs maxflood(prev)"] <= 1e-5 and worst["ch6 vs F_t-2 plane 4"] <= 1e-5,
          f"ch6 = max(hex max of ch6, ch7) of the step before (the running board max, through walls), = F_t-2 "
          f"plane 4 (max {worst['ch6 vs maxflood(prev)']:.1e} / {worst['ch6 vs F_t-2 plane 4']:.1e})")
    check(flat and worst["ch6 settled vs largest span"] <= 1e-5,
          f"after 6R + 4 steps ch6 is one value on every on-board cell, = aux_targets' largest span "
          f"(max {worst['ch6 settled vs largest span']:.1e})")

    # Frozen: 20 optimiser steps on random losses over every channel. The hooks zero the frozen gradient
    # entries, so the train.py optimiser (grad normalisation + Adam) leaves them bit-identical with no help;
    # Adam with weight decay would move them (it adds wd * param after the hooks): restore_floods undoes that.
    R = 5
    cs = const_stack(R, 7)
    w = np.stack([random_walls(rng, R) for _ in range(4)])
    walls = torch.from_numpy(w).float()[:, None]
    for wd in (0.0, 0.1):
        m = HexNCA(n_consts=7, clamp=(-2.0, 2.0), floods=True)
        before = {k: getattr(m, k).detach().clone() for k in m.frozen}
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=wd)
        g = torch.Generator().manual_seed(int(wd * 10))
        zero_grads = moved_raw = True
        for _ in range(20):
            out = m(fresh_state(walls), walls, cs, 6)
            loss = (out * torch.randn(out.shape, generator=g)).sum()
            opt.zero_grad()
            loss.backward()
            zero_grads &= all(float(getattr(m, k).grad[f].abs().max()) == 0 for k, f in m.frozen.items())
            for prm in m.parameters():
                prm.grad /= prm.grad.norm() + 1e-8
            opt.step()
            moved_raw &= any(not torch.equal(getattr(m, k)[f], before[k][f]) for k, f in m.frozen.items())
            m.restore_floods()
        same = all(torch.equal(getattr(m, k)[f], before[k][f]) for k, f in m.frozen.items())
        free_moved = all(not torch.equal(getattr(m, k)[~f], before[k][~f]) for k, f in m.frozen.items())
        if wd == 0:
            check(zero_grads and not moved_raw and same and free_moved,
                  "Adam + grad normalisation, 20 steps on random losses: the frozen entries get exactly zero "
                  "gradient and stay bit-identical (no restore needed); every free parameter tensor moved")
        else:
            check(zero_grads and moved_raw and same,
                  "Adam with weight decay 0.1 moves frozen entries in its step (it adds wd * param after the "
                  "hooks); restore_floods() puts them back bit-identical, every step")

    # Export: version 3 with the 7 consts; the rounded hand weights are exact; the fixture reproduces.
    with tempfile.TemporaryDirectory() as tmp:
        wp, fp = os.path.join(tmp, "w.json"), os.path.join(tmp, "f.json")
        wj, fix = export(model, {"note": "selftest floods", "pool": False}, wp, fp)
        m2 = model_from_json(wj)
        check(wj["version"] == 3 and wj["consts"] == list(CONST_NAMES) and len(wj["w1"]) == 64 * 23 * 9
              and all(torch.equal(getattr(m2, k)[f], model.frozen_values[k][f]) for k, f in model.frozen.items()),
              "export: version 3, consts = the 7 names, w1 H * (C+7) * 9; the frozen weights survive the "
              "6-digit rounding exactly")
        pw = parity_walls()
        diff = np.abs(run_parity(model, pw, 24) - run_parity(m2, pw, 24)).max()
        err = self_check(wp, fp)
        check(diff < 1e-3 and err < 1e-5, f"export -> reload of a floods model matches it (max {diff:.1e}) and the "
              f"parity fixture reproduces from the files (max {err:.1e})")


def pool_checks() -> None:
    """model.hex_pool (spec v4 §1): max and min of each state channel over the on-board taps among the 7."""
    import torch

    from .model import hex_pool

    R = 3
    S = side(R)
    on = mask(R) == 1
    g = np.random.default_rng(3)
    st = (g.random((2, 4, S, S)) * 1.6 - 0.8) * on  # values in (-0.8, 0.8), 0 off board
    mk = torch.from_numpy(on.astype(np.float32))[None, None]
    mx, mn = (t.numpy() for t in hex_pool(torch.from_numpy(st).float(), mk, -1.0, 1.0))
    # Brute force: for every on-board cell, the 7 taps that exist and are on board (no corner taps).
    ok, n_rim_taps = True, 0
    for row, col in np.argwhere(on):
        taps = [(row, col)] + [(row + dr, col + dc) for dr, dc in NEIGHBOURS
                               if 0 <= row + dr < S and 0 <= col + dc < S and on[row + dr, col + dc]]
        n_rim_taps += len(taps) < 7
        vals = np.stack([st[:, :, a, b] for a, b in taps])
        ok &= np.allclose(mx[:, :, row, col], vals.max(0), atol=1e-6) and np.allclose(mn[:, :, row, col], vals.min(0), atol=1e-6)
    check(ok and n_rim_taps == 6 * R and not mx[:, :, ~on].any() and not mn[:, :, ~on].any(),
          f"hex_pool = max/min over the on-board taps among the 7, on every cell of R={R} ({n_rim_taps} rim "
          f"cells with taps off board; off-board cells 0)")

    # By hand: a negative board (every value -0.5) -- an off-board tap read as 0 would win the max on the rim;
    # a large value on a corner tap (k=0, (drow,dcol) = (-1,-1)) must not reach the cell.
    st = np.where(on, -0.5, 0.0)[None, None].astype(np.float32)
    row, col = R, R  # the centre (q, r) = (0, 0)
    st[0, 0, row - 1, col - 1] = 0.9   # corner tap of the centre: (q, r) = (-1, -1), on board, not a neighbour
    st[0, 0, row - 1, col + 1] = 0.7   # (q, r) = (1, -1): a neighbour
    st[0, 0, row + 1, col] = -0.9      # (q, r) = (0, 1): a neighbour
    mx, mn = (t.numpy()[0, 0] for t in hex_pool(torch.from_numpy(st), mk, -1.0, 1.0))
    corner_cell = (R, 2 * R)  # (q, r) = (R, 0): a corner of the board, 3 of its taps off board
    check(mx[row, col] == np.float32(0.7) and mn[row, col] == np.float32(-0.9)
          and mx[corner_cell] == np.float32(-0.5) and mn[corner_cell] == np.float32(-0.5),
          "hex_pool by hand: the centre's max is its neighbour's 0.7, not the corner tap's 0.9, its min -0.9; "
          "a board corner on a -0.5 board reads max -0.5 (its off-board taps don't count as 0)")


def _area_brute(walls, board, R):
    """Independent of data.targets: (fills set, depth) on any board by plain BFS over (q, r) dicts, the rim =
    board cells with a neighbour off the board (or off the array); area only (+ every max-area region)."""
    S = side(R)
    cells = {(int(c), int(r)) for r, c in np.argwhere(board == 1)}
    free = {x for x in cells if not walls[x[1], x[0]]}
    nbrs = lambda x: [(x[0] + dc, x[1] + dr) for dr, dc in NEIGHBOURS]
    rim_c = {x for x in cells if any(y not in cells for y in nbrs(x))}
    seen, regions = set(), []
    for x in sorted(free):
        if x in seen:
            continue
        comp, stack = {x}, [x]
        seen.add(x)
        while stack:
            for y in nbrs(stack.pop()):
                if y in free and y not in seen:
                    seen.add(y)
                    comp.add(y)
                    stack.append(y)
        regions.append(comp)
    rims = [g for g in regions if g & rim_c]
    big = max((len(g) for g in rims), default=0)
    out = []
    for keep in [g for g in rims if len(g) == big] or [set()]:
        f = np.zeros((S, S), np.uint8)
        for g in regions:
            if g is not keep:
                for q, r in g:
                    f[r, q] = 1
        out.append(f.tobytes())
    depth = np.full((S, S), -1, np.int16)
    front = sorted(free & rim_c)
    for q, r in front:
        depth[r, q] = 0
    for x in front:
        for y in nbrs(x):
            if y in free and depth[y[1], y[0]] < 0:
                depth[y[1], y[0]] = depth[x[1], x[0]] + 1
                front.append(y)
    return set(out), depth


def ragged_checks() -> None:
    """Ragged board masks (nca/masks.py) and the targets on them (data.targets with board=)."""
    from .data import _axial_dist, area_ratio, damage_walls, edge
    from .masks import blob_mask, field, field_mask, ragged_mask

    print("\n-- ragged boards (nca/masks.py; review item 1b) --")
    check(all(np.array_equal(edge(mask(R) == 1), rim(R)) for R in range(0, 25)),
          "data.edge (on-board cells with an off-board neighbour) of the hexagon is its rim(R), R 0..24")
    q3, d3 = field(3)
    q4, d4 = field(4)
    check(len(q3) == 496 and len(q4) == 3905 and len({tuple(x) for x in q4}) == 3905 and d4.min() == 1,
          f"Spectacle's hex fields (nca/fields): level 3 = 496 tiles, level 4 = 3,905 distinct axial cells "
          f"({int((d4 == 1).sum())} on its outline)")
    rng = np.random.default_rng(4242)
    ok, shares, concave, src = True, [], 0, {"blob": 0, "field": 0}
    for i in range(60):
        R = int(rng.integers(3, 17))
        m = (blob_mask(rng, R) if i % 2 else field_mask(rng, R, 3 + i % 4 // 2)) == 1
        src["blob" if i % 2 else "field"] += 1
        lab = _labels(m, R)
        off = _labels((mask(R) == 1) & ~m, R)
        holes = [k for k in range(int(off.max()) + 1) if not ((off == k) & rim(R)).any()]
        ok &= bool(m.any()) and not (m & (mask(R) == 0)).any() and lab.max() == 0 and not holes
        shares.append(m.sum() / mask(R).sum())
        # concave: a board cell outside it with 4+ of its 6 neighbours on board (a notch / bay)
        nb_on = sum(np.roll(np.roll(m, dr, 0), dc, 1) for dr, dc in NEIGHBOURS)
        concave += bool((~m & (mask(R) == 1) & (nb_on >= 4)).any())
    check(ok and min(shares) >= 0.3, f"60 ragged masks (blobs and level-3/4 field crops, R 3..16): inside the "
          f"hexagon, one component, no holes, {min(shares):.2f}..{max(shares):.2f} of its cells; {concave} have a "
          f"concave bay (an off-board cell with 4+ board neighbours)")
    n_ok, n_multi, dmg_ok, n = 0, 0, True, 0
    for i in range(80):
        R = int(rng.integers(3, 9))
        m = ragged_mask(rng, R)
        w = random_walls(rng, R, "bridge" if i % 2 else None, board=m)
        fills, depth = targets(w, R, m)
        want, bdepth = _area_brute(w, m, R)
        got = {f.tobytes() for f in fills}
        n_ok += want <= got and np.array_equal(depth, bdepth) and not w[m == 0].any() and not fills[:, m == 0].any()
        n_multi += len(want) > 0 and fills[0][edge(m == 1)].any()
        for kind in ("edit", "burst", "erase", "stamp"):
            d = damage_walls(rng, w, R, kind, board=m)
            dmg_ok &= not d[m == 0].any()
        n += 1
    check(n_ok == n and dmg_ok, f"targets on {n} random ragged boards (R 3..8): every max-area answer of an "
          f"independent brute force (rim = board cells with an off-board neighbour) is acceptable, depth equal; "
          f"walls, targets and damage stay on the board ({n_multi} boards fill a rim cell: 2+ rim regions)")
    R = 6
    S = side(R)
    m = np.zeros((S, S), np.uint8)  # two lobes, (q, r) within 2 of (-4, 0) and of (3, 0), and a neck (-1, 0), (0, 0)
    m[_axial_dist(R, -4, 0) <= 2] = 1
    m[_axial_dist(R, 3, 0) <= 2] = 1
    m[R, R - 1:R + 1] = 1
    w = np.zeros((S, S), np.uint8)
    w[R, R] = 1  # wall (0, 0): the left lobe and (-1, 0) = 20 cells, the right lobe 19
    f, _ = targets(w, R, m)
    check(f[0].sum() == 19 and not f[0][:, :R + 1].any() and area_ratio(w, R, m) == 19 / 20,
          f"a wall across a one-cell isthmus cuts a ragged board in two: the primary target fills the smaller side "
          f"(19 cells) and leaves the larger (20) empty ({len(f)} acceptable: the two spans round the centroid are "
          f"within EPS, so the other side is acceptable too)")


def near_tie_checks() -> None:
    """--near-tie: data.random_walls(near_tie=) biases bridge boards towards area ratio >= NEAR_RATIO."""
    from .data import NEAR_RATIO, area_ratio

    print("\n-- near-tie bridges (--near-tie; review item 2) --")
    out = {}
    for nt in (0.0, 0.5, 1.0):
        rng = np.random.default_rng(99)
        rs = np.array([area_ratio(random_walls(rng, 8, "bridge", near_tie=nt), 8) for _ in range(150)])
        rs = rs[rs == rs]
        out[nt] = float((rs >= NEAR_RATIO).mean())
    a = np.random.default_rng(3)
    b = np.random.default_rng(3)
    same = all(np.array_equal(random_walls(a, 6), random_walls(b, 6, near_tie=0.0)) for _ in range(20))
    check(out[0.0] < 0.4 and out[1.0] > 0.9 and out[0.0] < out[0.5] < out[1.0] and same,
          f"share of bridge boards (2+ rim regions, R 8) with area ratio >= {NEAR_RATIO}: {out[0.0]:.2f} at "
          f"near_tie 0, {out[0.5]:.2f} at 0.5, {out[1.0]:.2f} at 1; near_tie 0 draws the stream as before")


def schedule_checks() -> None:
    """--schedule time: lr_at with frac (the share of the time box used) instead of it / iters."""
    from .train import lr_at

    print("\n-- lr schedule by time (--schedule time) --")
    check(lr_at(10, 10 ** 6, 1e-3, frac=0.59) == 1e-3 and abs(lr_at(10, 10 ** 6, 1e-3, frac=0.6) - 3e-4) < 1e-12
          and abs(lr_at(10, 10 ** 6, 1e-3, frac=0.9) - 1e-4) < 1e-12
          and abs(lr_at(10 ** 6 - 1, 10 ** 6, 1e-3, frac=0.1) - 1e-3) < 1e-12,
          "lr_at(frac=): x1 before 60% of the time box, x0.3 from 60%, x0.1 from 85%, whatever it / iters is")
    with tempfile.TemporaryDirectory() as tmp:
        lines = _train_tiny(tmp, "sch", ["--iters", "100000", "--schedule", "time", "--minutes", "0.6",
                                         "--warmup", "0", "--lr", "1e-3", "--lr-floor", "0"])
    st = lines[0]["schedule"]
    lrs = [(x["schedFrac"], x["lr"]) for x in lines if "schedFrac" in x]
    stop = lines[-1]
    good = all(abs(lr - 1e-3 * (1 if f < 0.6 else 0.3 if f < 0.85 else 0.1)) < 1e-9 or abs(f - 0.6) < 0.05
               or abs(f - 0.85) < 0.05 for f, lr in lrs)
    check(st["mode"] == "time" and st["boxMin"] == 0.6 and stop.get("stopped") == "time" and good
          and any(lr < 1e-3 for _, lr in lrs),
          f"a 0.6-minute run with --iters 100000 --schedule time: the start line says {st}; its lr decays on time "
          f"({len(lrs)} lines, the last at schedFrac {lrs[-1][0] if lrs else None} with lr {lrs[-1][1] if lrs else None})")


def _train_tiny(tmp, name, extra):
    """Run nca.train in-process (cwd tmp) on a tiny config; returns its log lines."""
    import contextlib
    import io
    import json
    import os
    import sys

    import torch

    from . import train
    base = ["nca.train", "--name", name, "--R", "3", "4", "--batch", "4", "--pool-size", "16", "--hidden", "16",
            "--eval-mults", "1", "--eval-every", "50", "--eval-n", "4", "--snap-every", "0", "--threads", "1",
            "--device", "cpu"]
    cwd, argv, nt = os.getcwd(), sys.argv, torch.get_num_threads()
    try:
        os.chdir(tmp)
        sys.argv = base + extra
        with contextlib.redirect_stdout(io.StringIO()):
            train.main()
        with open(f"runs/{name}/log.jsonl") as f:
            return [json.loads(x) for x in f]
    finally:
        os.chdir(cwd)
        sys.argv = argv
        torch.set_num_threads(nt)


def producer_checks() -> None:
    """The data producer (nca/producer.py): inline, thread and process give the same run, bit for bit, and a
    --resume rebuilds it exactly (weights, pool states, boards, bookkeeping)."""
    import torch

    print("\n-- the data producer (--producer; review item 5) --")
    extra = ["--warmup", "0", "--lr", "5e-4", "--lr-floor", "5e-4", "--mask-mix", "0.5", "--near-tie", "0.5"]
    load = lambda tmp, n: torch.load(f"{tmp}/runs/{n}/ckpt.pt", map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as tmp:
        lines = {}
        for mode in ("inline", "thread", "process"):
            lines[mode] = _train_tiny(tmp, mode, extra + ["--iters", "60", "--producer", mode])
        _train_tiny(tmp, "res", extra + ["--iters", "30", "--producer", "process"])
        _train_tiny(tmp, "res", extra + ["--iters", "60", "--producer", "inline", "--resume"])
        a = load(tmp, "inline")
        same = {}
        for n in ("thread", "process", "res"):
            b = load(tmp, n)
            same[n] = (all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
                       and all(torch.equal(a["pool"][R]["state"], b["pool"][R]["state"]) for R in a["pool"])
                       and all(np.array_equal(a["pool"][R][k], b["pool"][R][k]) for R in a["pool"]
                               for k in ("walls", "fill", "depth", "mask", "born", "edits", "last"))
                       and a["dataRng"] == b["dataRng"])
        timed = [x for x in lines["process"] if "dataSec" in x]
        ragged = any((a["pool"][R]["mask"] != mask(int(R))[None]).any() for R in a["pool"])
    check(all(same.values()) and ragged,
          f"60 iterations with --producer inline, thread, process, and process to 30 then --resume inline to 60: "
          f"identical weights, pool states, boards (with ragged masks), bookkeeping and producer rng ({same})")
    check(all(x["dataSec"] is not None and x["modelSec"] is not None and x["prodSec"] > 0 for x in timed),
          f"every log line splits secPerIter into dataSec + modelSec, and has prodSec (e.g. {timed[-1]['dataSec']:.4f} "
          f"+ {timed[-1]['modelSec']:.4f}, producer {timed[-1]['prodSec']:.4f} s)")


if __name__ == "__main__":
    main()
