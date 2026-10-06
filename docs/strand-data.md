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
**24 `128·010100000` (the infinite-line rule)**; 44, 46, 50, 60, 62, 67, 71, 73, 77, 80, 81, 88, 98 (258).
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

## v2: the whole kernel, the rule at the tap (2026-10-06)

The data and trainer for launch 6 (`spectacle-nca-options.md` §6.3, launch-6 scope), under the owner's input
rule: each cell gets only static board facts (on the board, which hex type, which way round) and, on the tapped
cell only, the tap (the rule's code and the tapped chord). No chord is ever an input; the CA has to work out
every tile's chords and carry the rule along the line itself, for rules it never saw.

```bash
npx tsx scripts/strand-export.ts --rule-table        # -> data/strand-v2 (git-ignored), ~30 s on an idle Pi
python -m nca.strand.rules --split --parity --bench  # the split vs the exporter, rendering + walks vs walkStrand, speed
python -m nca.strand.probe --hidden 128 1024         # the per-cell lookup probe (§3), A / C / D / D-cs / D-fourier / E
python -m nca.strand.train2 --name c-l2 --inputs c --levels 2 --minutes 25
python -m nca.strand.selftest2                       # a few minutes; the v1 selftest is unchanged
```

**Files** (`data/strand-v2`; `strand-v2.tgz` in the private bucket, `tar -C data -czf strand-v2.tgz strand-v2`):

| file | what |
|---|---|
| `rules-hex.json` | per leaf type its six local edges' classes; per kernel subset (all 7, class 0 included) and type the non-crossing matchings in Spectacle's order (`nonCrossingForTile`) and each one's local chords (`localChords`); the split's definition, per-subset check values and 448 keyed sample rules; Spectacle's commit |
| `boards.npz` | geometry only: per group `<g>_type/rot/tile` [B,H,W] (-1 off board), `<g>_mirror/root/orient/h/w/tiles`; groups `L2`, `L3` (the nine patches each), `L4` (2,048 training crops, box 42 x 38), `L4eval` (64 crops, seed 20261006), `L4full` (the Delta patch) |
| `parity.npz` | 2,000 rule-boards (every rule of the six small subsets + 1,556 of the fully packed one, each on an L2 / L3 / L4 board in turn): Spectacle's own rendering as 15 bits per cell, its whole-board strand decomposition, and 16 taps per board walked by `walkStrand` both ways |

**Rules.** A rule is `(s, digits)`: `s` one of the 7 subsets, `digits[t]` the position of type `t`'s matching
among its non-crossing ones (0-4; 0 for a type with one choice or none). Its index is the digits in mixed radix,
Delta most significant; its Spectacle `ruleKey` is `hex|<subset>|<matching indices>`. **Split v2**, stratified by
subset (the owner's decision): in the v1 subsets `15`, `128`, `258` the held-out rules are exactly v1's 20 legacy
held-out rules (so the legacy numbers stay comparable); in the four class-0 subsets, the round(0.2 n) rules with
the smallest `(fmix32(FNV-1a("strand-split-v2|" + key)), index)`. (`rules-hex.json`'s per-subset check values are
that hash ranking for all 7 subsets; `rules --split` checks it against the exporter.)

| subset | `15` | `128` | `258` | `01346` | `03456` | `023468` | `01234568` |
|---|---|---|---|---|---|---|---|
| rules | 4 | 32 | 64 | 8 | 16 | 320 | 1,953,125 |
| held out | 1 (legacy) | 6 (legacy) | 13 (legacy) | 2 | 3 | 64 | 390,625 |
| train | 3 | 26 | 51 | 6 | 13 | 256 | 1,562,500 |

Every held-out rule of the six small subsets uses only (type, digit) pairs some train rule of its subset has:
a held-out rule is a new combination, never a new per-tile matching.

Training draws the subset uniformly (1/7 each), then a train rule uniformly within it (rejection within the
subset), then a board of the level and a uniformly random chord as the tap.

**The code** (T1, 53 bits, on the tapped cell only, held): 8 bits "class m carries a line" for m in
(0, 1, 2, 3, 4, 5, 6, 8), then per leaf type a one-hot of its digit (9 x 5). With it, the tapped chord's two
edge directions (6). No owner slot (M1 is one player; owner's decision). Tap = 59 planes.

**Static planes** (`rules.RuleTable.static`, from the cell's type, rotation `rot` = the direction local edge 0
faces, and the mirror sign; theta = 2 pi rot / 6):

| option | planes | what |
|---|---|---|
| A | 16 | type one-hot 9, rotation one-hot 6, mirror |
| C | 55 | per direction the class (one-hot 8) of the edge facing it (6 x 8), the anchor (rotation one-hot 6), mirror |
| D | 11 | type one-hot 9, rot / 6 as one number, mirror |
| D-cs | 12 | type one-hot 9, cos theta, sin theta, mirror |
| D-fourier | 15 | type one-hot 9, cos / sin k theta for k = 1, 2, 3 without sin 3 theta (0 at every 60 degrees), mirror: an invertible linear map of A's rotation one-hot (probe only) |
| E | 9 | type one-hot only; the rotation and mirror are the cell's frame (nets.FrameNCA), not inputs |

`train2 --inputs` takes a, c, c-bc (C plus the code broadcast to every cell: the diagnostic ceiling), d, d-cs,
e and e-bc (E plus the broadcast code, E's diagnostic). Consts = mask + static + tap: 76 / 115 / 168 / 71 / 72 /
69 / 122 (E: + the frame plane, which picks the cell's permutation and is never a feature).

**Parity** (`rules --parity`): of 2,000 rule-boards over all 7 subsets, the table's rendering equals Spectacle's
on every cell, all 32,000 taps walk identically (cells, entry / exit edges, order, signed index, closed), and
every whole-board decomposition (270,430 strands) has the same lengths and closed flags. No walk stopped at a
junction or a limit, class 0 included. The split's per-subset counts and checksums, and 448 keys and hashes,
equal the exporter's; the v1 held-out set recomputed from salt v1 equals `data/strand/meta.json`'s, and every
legacy rule's local chords equal the table's.

**Throughput** (`rules --bench`, one Pi core, idle machine): a fresh training sample (rule + rendering + tap +
walk) 1,471 / s at level 2, 842 / s at level 3, 1,041 / s on level-4 crops (mean strand 10 / 39 / 33 chords).
A training iteration needs about batch / 8 + damage ~ 12 of them: ~10 ms.

**The per-cell lookup probe** (`nca/strand/probe.py`, §3's cheap experiment, the owner's arms added): a
1-hidden-layer MLP from one cell's static planes + the code to its 15 chord bits, fresh draws of (type,
rotation, mirror) uniform over the 108 and TRAIN rules (subsets uniform), Adam 3e-3 (x0.3 at 60 %, x0.1 at 85 %),
batch 512, 3 M samples per run; scored on 20,000 fixed draws with v2 HELD-OUT rules ("held-out") and 20,000
with train rules ("train": is the lookup learned at all). E's arm is the local frame (type only -> the chords
in the tile's own frame). Cell-exact = all 15 bits right. One Pi core each, under heavy load (wall times are
only comparable within a width):

| arm | width | held-out @ 1 M | held-out @ 3 M | train @ 3 M | bit | `15` | `01234568` | 0.9 reached | s |
|---|---|---|---|---|---|---|---|---|---|
| A | 128 | 0.384 | 0.556 | 0.679 | 0.9375 | 0.43 | 0.125 | - | 393 |
| C | 128 | 0.615 | 0.727 | 0.825 | 0.9587 | 0.77 | 0.198 | - | 432 |
| D | 128 | 0.040 | 0.116 | 0.148 | 0.8913 | 0.15 | 0.000 | - | 386 |
| D-cs | 128 | 0.278 | 0.491 | 0.588 | 0.9258 | 0.54 | 0.011 | - | 386 |
| D-fourier | 128 | 0.455 | 0.558 | 0.678 | 0.9364 | 0.46 | 0.088 | - | 329 |
| E (local) | 128 | 0.955 | 0.952 | 1.000 | 0.9938 | 0.67 | 0.997 | 0.8 M | 251 |
| A | 1024 | 0.822 | 0.923 | 0.996 | 0.9913 | 0.54 | 0.968 | 1.8 M | 860 |
| C | 1024 | 0.927 | 0.982 | 0.999 | 0.9957 | 0.88 | 0.996 | 1.0 M | 909 |
| D | 1024 | 0.327 | 0.591 | 0.719 | 0.9447 | 0.45 | 0.044 | - | 501 |
| D-cs | 1024 | 0.695 | 0.825 | 0.908 | 0.9746 | 0.71 | 0.423 | - | 443 |
| D-fourier | 1024 | 0.864 | 0.939 | 0.998 | 0.9927 | 0.59 | 0.977 | 1.6 M | 464 |
| E (local) | 1024 | 0.985 | 0.985 | 1.000 | 0.9959 | 0.89 | 1.000 | 0.2 M | 444 |

Reading it: E's local frame is near-trivial (the doc expected so); of the grid-frame inputs C learns fastest
and most exactly at both widths, A catches up only when wide, and D (rot / 6 as one number) is far behind --
8x the width lifts it from 0.12 to 0.59 held-out but does not close the gap (cos / sin closes part of it).
D-fourier tracks A at both widths (0.558 / 0.556, 0.939 / 0.923; train 0.678 / 0.679, 0.998 / 0.996): it is an
invertible linear map of A's one-hot, as expected, so it needs no trainer arm. What every arm misses on
held-out rules is mostly subset `15` (measured with the first split, where its held-out rule needed a digit
no `15` train rule had: departure 1 below); the fully packed subset is where narrow grid-frame nets fail.

**Trainer** (`nca/strand/train2.py`; its docstring has every detail): the v1 trainer's pool, ages, damage, light
cones, truncated backprop, schedule, collapse guard, resume and log, with the v2 inputs; `--channels 96 --hidden
128` by default, `--depth` hidden layers in the per-cell update (`nets.StrandNCA`; 1 = HexNCA), and `--inputs e`
on `nets.FrameNCA`. Damage "edit" re-types 1-3 cells with chords (never the tapped one); "move" moves the tap.
**M1b** keeps the tap and adds six planes, "the chord through edge d is on a circuit", for the tapped strand:
a circuit's edge must say 1 once drawn; a tail's edge must say 0 once the open news could have got there (the
front reaches an end at its distance from the tap, the news comes back a chord a step: chord i of n with the
tap at p by min(p + i, 2(n-1) - p - i)), don't-care before; after an edit the new strand's closed planes are
don't-care until its settle time. **Quick check**: the legacy set (train.py's very taps on the 20 v1 held-out
rules, inputs built from those boards' types and rotations) and the wide set (v2 held-out rules, up to 40 a
subset, every subset; 120 taps a level stratified by length, round-robin over the subsets), `bySet`,
`bySubset` (7), the score = length-balanced exact averaged over (set, level); and the **code-fidelity probe**
`q.code`: a ridge probe from drawn strand cells' hidden channels to the 53 code bits, fitted on the training
pools, scored on the quick check's held-out read-outs, overall and by distance from the tap.

**Departures from `spectacle-nca-options.md`** (what the build chose where the doc was open or the owner changed it):

1. The split is rank-based per subset (exactly round(0.2 n) held out in each), not `hash < 0.2`, which could
   hold out none of `15`'s four rules; and in `15`, `128`, `258` it is the legacy rules alone (owner's decision).
   A first version held out a v2 20 % there on top of the legacy rules: `15` kept 2 train rules, both with Pi's
   digit 1, so its held-out rules needed a (type, digit) never trained in that subset. The probe table above was
   measured with that version (its `15` column).
2. No owner slot (the owner's decision for launch 6): the tap is 59 planes, not 63.
3. The rule table is `data/strand-v2/rules-hex.json`, with the boards' geometry and the parity sample beside
   it, not `data/strand/rules-hex.json`; the legacy data stays `data/strand` (launch 5's `strand.tgz`).
4. The wide set holds every held-out rule of a small subset (1 / 6 / 13 / 2 / 3), not "all 4 of `15`".
5. M1b's light cone with a tap, and its eval (exact = the edge planes and the closed planes both right;
   `q.parts` gives each), are this build's definitions.
6. Added at the owner's request: options D and D-cs (and D-fourier in the probe), `--depth`, option E in the
   trainer with an equivariance test. The perception is the 7 taps only (no `taps+pool`).
7. `--ckpt-pool half` (float16 pool states in ckpt.pt; plan 6 uses it) and pool.npz's state limited to 32
   channels, for the bucket and the dashboard.

**CPU smoke** (`train2`, level 2, real width 96 / 128, batch 4, pool 64, steps U[12, 24], backprop 12, eval at
level 2 with 30 taps a set; one thread each, the Pi at load 10-25). Loss per 50 iterations; edge IoU on held-out
taps at the checks (exact stays 0 this early); code bits = the fidelity probe's per-bit accuracy at the last check:

| arm | consts | params | loss, every 50 iterations | IoU at 0 / 150 / 300 | code bits | s / iteration |
|---|---|---|---|---|---|---|
| a | 76 | 210,656 | 0.0435 0.0444 0.0349 0.0336 0.0332 0.0325 | 0 / 0.21 / 0.25 | 0.76 | 3.7 |
| c | 115 | 255,584 | 0.0442 0.0454 0.0425 0.0362 0.0360 0.0425 | 0 / 0.20 / 0.23 | 0.82 | 4.7 |
| d | 71 | 204,896 | 0.0436 0.0417 0.0371 0.0309 0.0283 0.0304 | 0 / 0.21 / 0.25 | 0.73 | 3.6 |
| c-bc | 168 | 316,640 | 0.0425 0.0440 0.0417 0.0345 0.0347 0.0335 | 0 / 0.16 / 0.27 | 0.84 | 5.4 |
| e | 69 | 160,352 | 0.0433 0.0509 0.0357 0.0336 (200 iterations) | 0 / 0.18 / - | 0.81 | 11.1 |
| big (c, 128 / 256 / depth 2) | 115 | 658,816 | 0.0441 0.0347 0.0388 (150 iterations) | 0 / 0.17 / - | 0.81 | 6.8 |

**CPU cost per iteration** (one process, the nets interleaved, best of 3; level 3, batch 8, 24 steps without
gradient + 12 with), relative to the default C at 96 / 128 / depth 1: A 0.86, D 0.78, E 2.21 (its per-frame
weight selection and the recomputed 7-tap rows; not measured on a GPU), big (C at 128 / 256 / depth 2) 2.35, for
2.6x the parameters.

**Memory per arm** (`python -m nca.strand.memprobe`: a fresh process's peak RSS over one gradient window on
level-4 crops, S 42, at two window lengths): per backprop step at batch 8, E 19.4 MB (+325 MB transient, its
checkpointed 7-tap recompute), C 35.6 MB (+43), big 75.9 MB (+33); autograd's saved tensors are ~70 % of that.
At the trainer's window of 48 and batch 16: E 2.5 GB, C 3.5 GB, big 7.35 GB. `nca/cloud/plan-7.txt` sizes its
batches from these (all three arms at 16, ~15.6 GB with pools and CUDA contexts).

**E, rewritten as one gather** (after launch 7, whose tap-e ran 2.33 s / iteration on the L4 against C's 0.09 and
was stopped). The first `FrameNCA` permuted the shared weights to each of the 12 frames and ran the frames one
after another: per step a `nonzero` per frame (a host sync on a GPU), a row gather, two matmuls and a full-size
`index_copy` per frame, the 12 permuted weight copies rebuilt, and the whole update checkpointed, so all of it
ran again in backward: ~320 dispatched ops and 23 host syncs per gradient step against C's 27 and none. Now every
cell's 7 taps, its directional channels in its own frame, are gathered from the zero-padded board by one
per-cell index (cached per consts tensor, so built once a rollout) straight into the rows of the shared per-cell
MLP: one matmul, 51 ops a step, no syncs, no frame loop, no per-cell weights; backward keeps only the board rows
(about the input's size) and gathers again. Parameters and state_dict are unchanged (launch 7's tap-e
checkpoints load). selftest2 keeps the first version as the reference: outputs and every gradient agree to
5e-15 (float64, all 12 frames, depth 1 and 2, e and e-bc); the equivariance check stays exact (0.0).
CPU per iteration relative to C at plan 7's settings (level 2, batch 16, 8 + 48 steps, forward + backward +
Adam; the Pi loaded, medians of repeated runs): E 0.66-0.69 at one thread (the first version 1.93-2.36),
0.86-0.89 at three (3.1-3.3), 0.67-0.75 at level 3 with three threads (1.92-2.33); E-bc 0.84, E at hidden
512 1.98. Memory per backprop step at batch 8 (memprobe, level-4 crops, windows 4 / 20): E 47-50 MB (saved
tensors 28), E-bc 52 (31), E at hidden 512 64.5 (50), C 39 (24.5) the same day; `nca/cloud/plan-8.txt` sizes
its three E arms from these.

**Launch 8's plateau** (read on the Pi, 2026-10-07; nothing touched on the VM). Every arm held out at exact ~0.15
(short ~0.3, medium and long 0), E-bc included. Not a bug: train2's loop overfits one rule (e-bc and c-bc, 4 taps,
level 2) to exact 1.0 in 150 iterations, 350 with damage, and a minimal full-backprop loop does it in 75-100.
The models draw the tapped strand and almost nothing else (E-bc at step 60: 70 % of its edges on, 1 % of other
strands', none elsewhere); the lines stop because the update picks the wrong exit at cells with 2-3 chords. The
**right-exit rate** per hop from the tap (fresh taps, level 2): E-bc 0.82 (1-chord cells 0.96, 2-chord 0.81,
3-chord 0.41-0.57; held-out rules 0.83, so not generalisation), E 0.66, C 0.55, big-C 0.59; at 0.82 a line dies
~5 chords out. It is the update net's routing: `probe.py --arms e-route` (type + code + the entry edge -> the
edges drawn, one cell) at 1 hidden layer x 128 reaches 0.82 after 1 M samples and 0.95 after 3 M, at 2 layers
0.95 and 0.99; at the NCA's lr 5e-4 (2 M samples) 0.66 and 0.84 -- and l3 ran at 2e-4, where E-bc's rate did not
move in 3,000 iterations. The quick check now logs it as `q.exit` (rate, c1 / c2 / c3, n, tap; train2's
docstring), a default dashboard series. Also: with the loss on the last 8 steps of a 48-step window, no line
younger than 29 steps (level 2; 41 at level 3) was ever scored, and E-bc first drew the tapped chord at step ~9;
`--last-k 48` scores the whole window (one rule with damage: exact 1.0 at 300 iterations instead of 350, 0.75
steps late instead of 15-18). `nca/cloud/plan-9.txt`: three E arms at depth 2, lr 5e-4 through l2 and l3,
`--last-k 48`. Memory per backprop step at batch 8 (memprobe, level-4 crops): E depth 2 51 MB (saved tensors 35),
E-bc 46-49 (38), E at hidden 256 68-73 (49); `--last-k 48` adds ~1 % to the saved tensors.
