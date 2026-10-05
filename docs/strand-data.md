# Strand data for the M1 CA (2026-10-06)

The data pipeline of `spectacle-nca-plan.md` §7 for milestones M1a (a strand from a tap) and M1b (closed
or tail): Spectacle's hex fields on hex_ca's axial lattice, every clean hex rule without edge class 0
rendered as 15 chord bits per cell, and strands walked by Spectacle's own `walkStrand`. Nothing here
trains anything.

```bash
npx tsx scripts/strand-export.ts            # -> data/strand/ (git-ignored), ~40 s on the Pi
python -m nca.strand.parity                 # 2,000 random taps; --all for every tap and board
python -m nca.strand.stats                  # the tables below
```

Exporter flags: `--out`, `--taps` (per board, 32), `--crops` (level-4 crops per rule, 16), `--eval-taps`
(64), `--eval-crops` (16), `--crop-radius` (8-14), `--seed` (1; eval uses its own fixed 20261006),
`--rules` (id range, 0-99), `--spectacle` (or `SPECTACLE_DIR`, default `../Spectacle`; imported
read-only by path, so `npm run typecheck` and CI do not need it).

## Conventions

- **Cells.** Axial (q, r); array `row = r - board_r0`, `col = q - board_q0` (`nca/hexgrid.py`'s radius-R
  boards are the case r0 = q0 = -R). Direction d is `src/hex.ts` `DIRS[d]` as (dq, dr): 0 (1,0),
  1 (1,-1), 2 (0,-1), 3 (-1,0), 4 (-1,1), 5 (0,1); as (drow, dcol) 0 (0,1), 1 (-1,1), 2 (-1,0),
  3 (0,-1), 4 (1,-1), 5 (1,0), which are `hexgrid.NEIGHBOURS`, so the masked 3x3 kernel sees all six.
  Opposite = d + 3.
- **Chord bits.** Plane p in 0..14 is the unordered direction pair `PAIRS[p]` = (0,1), (0,2), ..,
  (4,5): the rule joins the cell's edges a and b. The tap planes (6) mark the tapped chord's two edges.
- **From Spectacle's world.** With (x, y) a tile centre minus tile 0's, the base axial is
  q = (x/1.5 + 2y/√3)/2, r = (x/1.5 - 2y/√3)/2; Spectacle's hexagons are flat-top with edge 1, so the
  neighbour at world angle 30° + 60°·d is direction d. Each board then takes, of the lattice's 12
  symmetries, the one with the smallest array box (`board_orient` = m·6 + k: swap q and r if m, then k
  times (q, r) → (-r, q + r)); every plane, direction and step of that board is in that frame. That is
  what gives level 3 a 30 × 37 box, not 37 × 38.
- **Type and rotation** (for the "purer" control arm): `tile_type` (index in `HEX_LEAF_ORDER`) and
  `tile_rot`: local edge k (Spectacle's `HEX_PTS` edge k) faces direction (`board_mirror`·k + `tile_rot`)
  mod 6. `board_mirror` is the patch's mirror sign in the board frame (levels alternate).
- **Steps.** A strand is a list of chords (a cell can hold two chords of one strand), oriented from the
  tap's `d0` end (Spectacle's chord end 0) to its `d1` end; `step_index` is signed from the tap (0 at the
  tap, + ahead through d1, - behind through d0; a circuit is walked ahead only, 0..n-1).

## Rules, boards, files

**Rules**: the 100 clean hex rules without class 0, as the plan counts them: subsets 15 (4), 128 (32),
258 (64); ids in that order, then matching vectors in leaf order. **Split**: within each subset the
round(0.2·n) rules with the smallest fmix32(FNV-1a(`strand-split-v1|` + ruleKey)) are HELD-OUT: 1 + 6 + 13
= 20. (Bare FNV-1a put rules that differ only in the Gamma digit next to each other, so held-out came in
pairs; the finaliser fixes that.) HELD-OUT ids: 0 `15·000000000`; 4 `128·000000000`, 9, 19, 21, 22,
**24 (the infinite-line rule; its matching is deliberately not written here)**; 44, 46, 50, 60, 62, 67, 71, 73, 77, 80, 81, 88, 98 (258).
`meta.json` lists every rule with its key, combo string, split and per-type local chords.

**Boards**: levels 2 and 3 are the nine root-type patches (all of them; 55-63 and 433-496 tiles).
Level 4 is crops: a union of 1-3 hex discs (radius 8-14; the others centred inside the first), cut by
one of the nine level-4 patches, 108-852 tiles, diameter 16-40.

| dir | rules | L2 | L3 | L4 | taps per board |
|---|---|---|---|---|---|
| `train/L{2,3,4}/rNNN.npz` | the 80 TRAIN | 9 patches | 9 patches | 16 crops per rule (seed 1) | 32 |
| `eval/L{2,3,4}/rNNN.npz` | the 20 HELD-OUT | 9 patches | 9 patches | the full Delta patch (3,905 tiles) + 16 crops (seed 20261006) | 64 |

Taps are uniform over a board's chords without replacement (so length-weighted). One file per
(split, level, rule); B boards padded to the largest (mask 0 outside), T taps, N steps, S strands:

| array | shape | |
|---|---|---|
| `mask`, `chords` | [B,H,W] u1, [B,15,H,W] u1 | the M1a inputs |
| `tile_index`, `tile_type`, `tile_rot` | [B,H,W] | Spectacle tile index in its field, leaf type, rotation; -1 off board |
| `board_root/kind/orient/mirror/q0/r0/h/w/tiles` | [B] | kind 0 = full patch, 1 = crop |
| `chord_strand` | [B,15,H,W] i4 | every chord's strand (M1b "whole pattern"), -1 = no chord |
| `strand_len/closed/board` | [S] | per strand, from walkStrand both ways |
| `tap_board/row/col/d0/d1/closed/strand`, `tap_ptr` | [T], [T+1] | tap i's strand is steps `tap_ptr[i]:tap_ptr[i+1]` |
| `step_row/col/in/out/index` | [N] | walkStrand's walk: cell, entry and exit direction, signed index |

**Loader** (`nca/strand/loader.py`): `M1aBatcher(paths, fresh=True).batch(n)` gives `inputs [n,22,H,W]`
(chords 0-14, tap 15-20, mask 21: the plan's 22 planes), targets `edges [n,6,H,W]` and `closed [n,1,H,W]`
(1 on the strand iff a circuit), `on [n,1,H,W]`, and per sample `length`, `grow`, `rule`, `level`;
`fresh` draws new taps and walks them in Python (64 level-3 samples ≈ 0.36 s on the Pi), else it uses
the exported walks. `pattern_sample` is M1b without a tap, with per-edge closed targets (see below).

## Parity: the 15 chord bits specify the patterns

- **Lattice** (all 27 fields, levels 2-4 × 9 roots, asserted by the exporter): every tile centre on the
  axial lattice (residual < 1e-6), one tile per cell; for every tile the lattice neighbours present are
  exactly Spectacle's vertex-sharing neighbours; each tile's six edges face six distinct directions that
  are a rotation of its local order (mirror sign = sign of the transform, uniform per patch, alternating
  with level); facing edges of neighbours always have the same edge class (0 mismatches).
- **Walks**: the Python walker (`nca/strand/walker.py`, ~40 lines of walking) from the chord planes alone
  equals walkStrand exactly (cells, entry and exit edges, order, signed index, closed flag) on
  **2,000 / 2,000** random taps (L2 513, L3 525, L4 962) and on **all 131,840** exported taps.
- **Boards**: on all 3,420 boards the walker's decomposition into strands is the same partition as
  walkStrand's (332,159 strands, same lengths and closed flags), and the chord planes equal the rule's
  per-type chords rotated by `tile_rot`, so type + rotation + rule determine the planes.
- No chord used an edge twice, and no walk ended at a junction or a limit (524,700 walks: 139,618
  closed, 385,082 dead ends).
- Negative controls: one chord bit cleared on a strand, or the directions mirrored in the walker, and
  the comparison fails (115 and 260 of 288 taps on one level-3 file).

So the plan's representational claim holds with no fix: without class 0, a hex tile's chords join edge
midpoints, each edge carries at most one chord end, the tile across edge d is the lattice neighbour d
and it is entered across d + 3, so "the chord at the entry edge" is a table lookup on the 15 bits.

## Statistics

Lengths in steps (chords). Per strand: every strand of every board. Per tap: as sampled.

| split | L | rules | boards | tiles/board | strands/board | strand len p50/p90/p99/max | closed (strands / chords) | circuit len p50/p99/max | tail len p50/p99/max |
|---|---|---|---|---|---|---|---|---|---|
| train | 2 | 80 | 720 | 55-63 | 28 (1-46) | 2 / 8 / 19 / 83 | 0.16 / 0.30 | 4 / 28 / 41 | 2 / 17 / 83 |
| train | 3 | 80 | 720 | 433-496 | 154 (1-298) | 3 / 10 / 30 / 667 | 0.37 / 0.53 | 4 / 33 / 282 | 2 / 27 / 667 |
| train | 4 | 80 | 1280 | 108-852 | 93 (9-308) | 3 / 11 / 37 / 398 | 0.49 / 0.57 | 4 / 30 / 168 | 2 / 59 / 398 |
| eval | 2 | 20 | 180 | 55-63 | 28 (1-44) | 2 / 7 / 22 / 83 | 0.14 / 0.26 | 7 / 28 / 41 | 2 / 19 / 83 |
| eval | 3 | 20 | 180 | 433-496 | 155 (1-280) | 2 / 9 / 29 / 667 | 0.35 / 0.49 | 4 / 35 / 85 | 2 / 24 / 667 |
| eval | 4 | 20 | 340 | 111-3905 | 145 (4-1937) | 3 / 11 / 35 / 3689 | 0.52 / 0.60 | 4 / 35 / 168 | 2 / 48 / 3689 |

"Two-way grow" = CA steps for a line growing one chord per step both ways from the tap (a circuit:
n/2); "one-way" = through d1 only, like a Spectacle head. "Circuit+tail cells" = share of cells with
chords that hold chords of a circuit and of a tail at once; "twice-visited" = share holding two chords of
one strand.

| split | L | taps | tap strand len p50/p90/p99/max | closed | two-way grow p50/p90/p99/max | one-way p99/max | circuit+tail cells | twice-visited cells | MB |
|---|---|---|---|---|---|---|---|---|---|
| train | 2 | 23040 | 6 / 25 / 80 / 83 | 0.31 | 3 / 16 / 58 / 82 | 47 / 82 | 0.086 | 0.100 | 1.1 |
| train | 3 | 23040 | 9 / 77 / 643 / 667 | 0.55 | 5 / 58 / 453 / 664 | 361 / 664 | 0.080 | 0.140 | 3.3 |
| train | 4 | 40960 | 9 / 52 / 182 / 398 | 0.54 | 5 / 36 / 136 / 393 | 107 / 373 | 0.099 | 0.144 | 5.4 |
| eval | 2 | 11520 | 7 / 37 / 83 / 83 | 0.27 | 3 / 20 / 70 / 82 | 59 / 82 | 0.084 | 0.105 | 0.4 |
| eval | 3 | 11520 | 8 / 184 / 667 / 667 | 0.50 | 4 / 115 / 569 / 666 | 469 / 666 | 0.089 | 0.142 | 1.0 |
| eval | 4 | 21760 | 8 / 53 / 263 / 3689 | 0.53 | 5 / 38 / 221 / 3613 | 177 / 3339 | 0.085 | 0.154 | 3.0 |

Per subset (levels 3 and 4): subset 128 is the whole long tail.

| split | L | subset | rules | strand len p50/p90/p99/max | closed strands | tap len p99 | two-way grow p99/max |
|---|---|---|---|---|---|---|---|
| train | 3 | 128 | 26 | 8 / 17 / 118 / 667 | 0.94 | 667 | 586 / 664 |
| train | 3 | 15 | 3 | 2 / 5 / 9 / 9 | 0.38 | 9 | 4 / 4 |
| train | 3 | 258 | 51 | 2 / 7 / 23 / 104 | 0.30 | 98 | 77 / 103 |
| train | 4 | 128 | 26 | 8 / 14 / 76 / 306 | 0.54 | 206 | 168 / 305 |
| train | 4 | 15 | 3 | 3 / 6 / 9 / 9 | 0.64 | 9 | 4 / 7 |
| train | 4 | 258 | 51 | 3 / 10 / 34 / 398 | 0.47 | 157 | 109 / 393 |
| eval | 3 | 128 | 6 | 8 / 19 / 205 / 667 | 0.93 | 667 | 632 / 666 |
| eval | 3 | 15 | 1 | 2 / 3 / 3 / 3 | 0.49 | 3 | 1 / 1 |
| eval | 3 | 258 | 13 | 2 / 8 / 24 / 56 | 0.28 | 52 | 40 / 55 |
| eval | 4 | 128 | 6 | 7 / 18 / 93 / 3689 | 0.66 | 2639 | 1519 / 3613 |
| eval | 4 | 15 | 1 | 3 / 3 / 3 / 3 | 0.70 | 3 | 1 / 1 |
| eval | 4 | 258 | 13 | 3 / 10 / 32 / 168 | 0.48 | 79 | 51 / 131 |

Dataset: 300 files, 14.1 MB (train 9.9, eval 4.3), 3,420 boards, 131,840 taps. Generation 36 s on the
Pi (fields and lattices 1 s); parity 20 s (2,000) or 31 s (`--all`); stats 7 s. Board diameters (hex
distance): level 2 12, level 3 37, level 4 108, level-4 crops 16-40.

## What the plan got wrong or left open

1. **Patch sizes.** Level 2 is 55-63 tiles (the plan: "≤ 71", "55-559" for levels 2-3), level 3 433-496,
   level 4 3,409-3,905. The eight non-Gamma roots give the *same shape* at each level (only the types
   differ), so "held-out patches" by root are two masks per level; a held-out board has to be a crop or
   another level. Level 4's best box is 93 × 101, not 101 × 109.
2. **Long strands are tails, not loops.** The FASS rule never closes on a finite patch: 1-5 tail strands
   per level-3 patch, the longest 259-667 steps by root, and 4 tails (longest 2,639) on the full level-4
   patch. 667 steps is more than the 496 tiles because ~14 % of cells carry two chords of one strand. The
   longest level-3 circuit is 282; the longest strand in the data is a 3,689-step tail
   (`128·010010000`, full level-4 patch). M1b's hard case is therefore "open" reaching the middle of a
   667-step tail (~333 steps), not a 496-tile loop closing.
3. **Horizons.** Two-way growth from a tap needs 453 steps at p99 on level 3 (664 max) and 136 on
   level-4 crops: about 12 × the level-3 diameter. 2.8 % of TRAIN level-3 taps (5.3 % of held-out) need
   more than the 8 × diameter readout (296 steps), 3.8 % / 7.0 % more than 6 × diameter, all of them
   subset 128. Either the persistent pool carries settling across visits, as the plan intends, or the
   level-3 readout needs ~700 steps; report exactness per subset and per length bin, or the 128 tail will
   hide in the mean. Medians are tiny (2-3 per strand,
   6-9 per tap), so the sampling weight (per tap vs per strand) changes what the model sees.
4. **M1b's per-cell closed plane is ill-defined without a tap.** 8-10 % of cells with chords hold a
   circuit's chord and a tail's at once. With a tap it is fine (only the tapped strand is drawn). The
   smallest fix: closed per edge (6 planes, "the line across edge d is a circuit"), which is the same on
   both sides of an edge; `pattern_sample` returns that.
5. **The split.** Held-out rules reuse per-type matchings seen in TRAIN (only the combinations are new),
   subset 15's single held-out rule (`15·000000000`) draws strands of ≤ 3 steps, and the FASS rule fell
   in HELD-OUT. Changing `SPLIT_SALT` re-draws the split, if the owner wants FASS in TRAIN.
6. **Layout.** The plan's single `nca/strands.py` is the `nca/strand/` package; the plan's 2,000-walk
   JSON fixture became the walkStrand walks stored for every tap (parity samples 2,000 of them).
