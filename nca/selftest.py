"""Plain-assert self-test for nca/hexgrid.py and nca/data.py. No pytest.

    python -m nca.selftest
"""

import time

import numpy as np

from .hexgrid import NEIGHBOURS, KERNEL_MASK, mask, rim, side
from .data import oracle, random_walls, batch, edit_walls, page_loops, _KINDS, _axial_dist, _offset, _polyline, _poly_points


# --------------------------------------------------------------------------
# An independent brute-force oracle, to check nca.data.oracle against.
# Deliberately not sharing code with it: full-board rescans to a fixpoint,
# not a BFS queue.
# --------------------------------------------------------------------------

def _bruteforce_oracle(walls: np.ndarray, R: int):
    S = side(R)
    on_board = mask(R) == 1
    is_rim = rim(R)
    open_cells = on_board & (walls == 0)

    outside = open_cells & is_rim
    depth = np.where(outside, 0, -1).astype(np.int16)

    d = 0
    changed = True
    while changed:
        changed = False
        d += 1
        prev = outside.copy()  # this pass only sees last pass's outside
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

    fill = (open_cells & ~outside).astype(np.uint8)
    return fill, depth


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

    # -- oracle vs. independent brute force --------------------------------
    n_checked = 0
    for R in (3, 5, 8):
        for _ in range(100):
            w = random_walls(rng, R)
            fill, depth = oracle(w, R)
            bf_fill, bf_depth = _bruteforce_oracle(w, R)
            if not np.array_equal(fill, bf_fill) or not np.array_equal(depth, bf_depth):
                raise AssertionError(f"oracle disagrees with brute force (R={R})")
            n_checked += 1
    check(n_checked == 300, "oracle matches an independent brute force on 300 random boards (R in {3,5,8})")

    # -- hand cases ---------------------------------------------------------
    R = 6
    S = side(R)
    empty = np.zeros((S, S), dtype=np.uint8)
    fill, _ = oracle(empty, R)
    check(int(fill.sum()) == 0, "an empty board fills nothing")

    # A closed ring of radius 3 centred on the board: 19 interior cells
    # (3*2*3+1, the radius-2 disk).
    q = (np.arange(S) - R).reshape(1, S)
    r = (np.arange(S) - R).reshape(S, 1)
    dist = np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))
    ring = (dist == 3).astype(np.uint8)
    fill, _ = oracle(ring, R)
    check(int(fill.sum()) == 19, "a radius-3 ring on R=6 fills its 19 interior cells")
    check(np.array_equal(fill.astype(bool), dist < 3), "...exactly the radius-2 disk")

    # The same ring with one cell removed: the interior leaks out and fills
    # nothing.
    leaky_ring = ring.copy()
    gap = np.argwhere(ring)[0]
    leaky_ring[gap[0], gap[1]] = 0
    fill, _ = oracle(leaky_ring, R)
    check(int(fill.sum()) == 0, "a radius-3 ring missing one cell fills nothing")

    # A wall ring exactly ON the rim fills the whole interior (no non-wall
    # rim cell left for the outside to seed from).
    on_rim = rim(R).astype(np.uint8)
    fill, _ = oracle(on_rim, R)
    expected_interior = int(mask(R).sum()) - int(rim(R).sum())
    check(int(fill.sum()) == expected_interior, "a wall ring on the rim fills the whole interior")
    check(np.array_equal(fill.astype(bool), (mask(R) == 1) & ~rim(R)), "...every non-rim cell")

    # The same rim-ring, but one rim cell is left non-wall: that cell is
    # itself "outside" by definition, and the gap leaks the whole interior
    # back out, so nothing fills.
    gappy_rim = on_rim.copy()
    gap = np.argwhere(on_rim)[0]
    gappy_rim[gap[0], gap[1]] = 0
    fill, _ = oracle(gappy_rim, R)
    check(int(fill.sum()) == 0, "a rim-loop with one non-wall rim cell fills nothing there")

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

    # -- batch() shapes -------------------------------------------------------
    R = 5
    walls, fill, depth = batch(rng, R, 7)
    S = side(R)
    check(walls.shape == (7, S, S) and fill.shape == (7, S, S) and depth.shape == (7, S, S),
          "batch() returns the right shapes")
    check(walls.dtype == np.uint8 and fill.dtype == np.uint8 and depth.dtype == np.int16,
          "batch() returns the right dtypes")

    generator_checks(rng)
    board_stats(rng)
    edit_stats()
    distinctness()

    model_checks(rng)
    print("\nALL OK")


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
            out += [w, edit_walls(g, w, R), page_loops(g, R)]
        return out
    check(all(np.array_equal(a, b) for a, b in zip(draw(7), draw(7))),
          "random_walls, edit_walls and page_loops are deterministic given the Generator")


def _stats(boards, R):
    """(share with a filled cell, mean filled fraction, filled cell counts, max depths) of a list of boards."""
    cells = int(mask(R).sum())
    filled, depth = [], []
    for w in boards:
        f, d = oracle(w, R)
        filled.append(int(f.sum()))
        depth.append(int(d.max()))
    filled, depth = np.array(filled), np.array(depth)
    return float((filled > 0).mean()), float(filled.mean() / cells), filled, depth


def board_stats(rng) -> None:
    """Per kind and for the mix at R=8: fill share, filled sizes, oracle depths, speed."""
    R = 8
    n_per_kind = 300
    print(f"\n-- R={R}, {n_per_kind} boards per kind: share with a filled cell, mean filled fraction, "
          f"max oracle depth p50/p99 --")
    for kind in _KINDS:
        share, frac, _, depth = _stats([random_walls(rng, R, kind=kind) for _ in range(n_per_kind)], R)
        print(f"  {kind:6s}: {share:.2f}  {frac:.3f}  {np.percentile(depth, 50):3.0f} {np.percentile(depth, 99):3.0f}")
    share, frac, _, depth = _stats([page_loops(rng, R) for _ in range(n_per_kind)], R)
    print(f"  {'page':6s}: {share:.2f}  {frac:.3f}  {np.percentile(depth, 50):3.0f} {np.percentile(depth, 99):3.0f}"
          f"   (held out: evaluation only)")

    n_mix = 2000
    t0 = time.perf_counter()
    boards = [random_walls(rng, R) for _ in range(n_mix)]
    share, frac, filled, depth = _stats(boards, R)
    ms = 1000 * (time.perf_counter() - t0) / n_mix
    sizes = np.percentile(filled[filled > 0], [10, 50, 90, 100])
    deep = np.percentile(depth, [50, 90, 99, 99.9])
    print(f"  mix   : {share:.3f}  {frac:.3f}  (n={n_mix})")
    print(f"    filled cells, boards with any (of {int(mask(R).sum())}): p10 {sizes[0]:.0f}, p50 {sizes[1]:.0f}, "
          f"p90 {sizes[2]:.0f}, max {sizes[3]:.0f}")
    print(f"    max oracle depth: p50 {deep[0]:.0f}, p90 {deep[1]:.0f}, p99 {deep[2]:.0f}, p99.9 {deep[3]:.0f}, "
          f"max {depth.max()}")
    print(f"    {ms:.3f} ms/board (random_walls + oracle, R={R})")
    check(0.35 <= share <= 0.7, f"overall share with a filled cell in [0.35, 0.7] (got {share:.3f})")
    check(sizes[0] <= 4 and sizes[2] >= 40, "filled regions run from a few cells to a large part of the board")
    check(deep[2] < 40, f"99th percentile of max oracle depth at R={R} under 40 (got {deep[2]:.0f})")
    check(ms < 3, f"under 3 ms a board (got {ms:.2f})")


def edit_stats() -> None:
    """How often an edit changes the answer (the pool's "user edits a settled board" signal)."""
    R, n = 8, 1000
    g = np.random.default_rng(2024)
    changed = beyond = 0
    t0 = time.perf_counter()
    for _ in range(n):
        w = random_walls(g, R)
        e = edit_walls(g, w, R)
        diff = oracle(e, R)[0] != oracle(w, R)[0]
        changed += bool(diff.any())
        beyond += bool((diff & (e == w)).any())
    ms = 1000 * (time.perf_counter() - t0) / n
    print(f"\n-- edits, R={R}, {n} boards --")
    print(f"  the answer changed on {changed / n:.3f} of edits ({beyond / n:.3f} beyond the edited cells "
          f"themselves); {ms:.2f} ms per board + edit")
    check(0.6 <= changed / n <= 0.75, f"60-75% of edits change the answer (got {changed / n:.3f})")


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


def model_checks(rng) -> None:
    """HexNCA invariants and the export round trip."""
    import os
    import tempfile

    import torch

    from .export import export, model_from_json, run_parity, parity_walls, self_check
    from .model import HexNCA, fresh_state

    print("\n-- model --")
    torch.manual_seed(0)
    R = 5
    S = side(R)
    mk = torch.from_numpy(mask(R)).float().view(1, 1, S, S)
    w = np.stack([random_walls(rng, R) for _ in range(4)])
    walls = torch.from_numpy(w).float().unsqueeze(1)

    # One optimiser step, the way train.py takes it (grad normalisation + Adam).
    model = HexNCA()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(2):  # twice, so the second step runs with a non-zero w2
        out = model(fresh_state(walls), walls, mk, 8)
        loss = (out[:, 1] - 1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        for prm in model.parameters():
            prm.grad /= prm.grad.norm() + 1e-8
        opt.step()
    check(float(model.w1[:, :, 0, 0].abs().max()) == 0.0 and float(model.w1[:, :, 2, 2].abs().max()) == 0.0,
          "corner taps (k=0, k=8) are exactly zero after optimiser steps")
    check(float(model.w2.abs().max()) > 0, "...and the optimiser did move w2")

    # Off-board cells stay exactly zero, and ch0 is the wall picture, after every step.
    with torch.no_grad():
        model.w2.normal_(0, 0.3)  # a lively model, so these aren't trivially true
        model.b2.normal_(0, 0.3)
        state = fresh_state(walls)
        off_ok, wall_ok = True, True
        for _ in range(30):
            state = model.step(state, walls, mk)
            off_ok &= bool((state * (1 - mk)).abs().max() == 0)
            wall_ok &= bool(torch.equal(state[:, 0:1], walls))
    check(off_ok, "off-board cells stay exactly zero for 30 steps")
    check(wall_ok, "the wall channel equals the wall picture after every step")

    # Export -> reload round trip (rounded to 6 significant digits).
    with torch.no_grad():
        model.w2.normal_(0, 0.05)
        model.b2.normal_(0, 0.05)
    with tempfile.TemporaryDirectory() as d:
        wp, fp = os.path.join(d, "w.json"), os.path.join(d, "f.json")
        wj, _ = export(model, {"trainedR": R, "steps": [0, 0], "iterations": 0, "note": "selftest"}, wp, fp)
        check(len(wj["w1"]) == 64 * 17 * 9 and len(wj["w2"]) == 16 * 64 and len(wj["b1"]) == 64
              and len(wj["b2"]) == 16, "export has the spec's array sizes")
        check(all(wj["w1"][i] == 0 for i in range(0, len(wj["w1"]), 9))
              and all(wj["w1"][i] == 0 for i in range(8, len(wj["w1"]), 9)), "exported corner taps are zero")
        pw = parity_walls()
        diff = np.abs(run_parity(model, pw, 24) - run_parity(model_from_json(wj), pw, 24)).max()
        check(diff < 1e-4, f"export -> reload round trip matches the original model to 1e-4 (max {diff:.1e})")
        err = self_check(wp, fp)
        check(err < 1e-5, f"the parity fixture reproduces from the files (max {err:.1e})")


if __name__ == "__main__":
    main()
