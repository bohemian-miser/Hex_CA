# Spectacle NCA under the strict input rule: literature, options, plan (2026-10-05, ~22:15 UTC)

Supersedes the input design of `spectacle-nca-plan.md` (§1, §4, §6): the rule is no longer rendered
onto every cell. Each cell gets only static board facts (on the board, which hex, which way round) and
tap events (this cell was tapped by player P with pattern X). The CA must carry the pattern along the
strand, work out each tile's chords from its own type and orientation, grow the line, and tell circuits
from tails — for every player and for rules it never saw. Builds on `nca-review.md` §4 (literature) and
`strand-data.md`; does not repeat them. **Measured** = read from code, logs or a probe run today;
**inference** = my judgement. Nothing here is implemented.

## 0. Decisions for the owner

**Owner update (2026-10-05, after this was written): Spectre tiles come for free.** Spectacle's
Spectre view is purely client-side: a hexagon arena is drawn as Spectres through the isomorphism
(`'spectre-iso'` labels: every seam of a hexagon is a seam of its Spectre with the same class;
Spectacle `client/src/spectre-view.ts`), and the server, engine and wire only ever run hexagons (solo is
hexagons only). So a CA that plays the hex game plays the Spectre game too. The plan's M4 (a graph NCA on
Spectre tiles) is dropped, and every "Spectre later" argument below (§3's row, the judgement, §7's
transfer risk) no longer counts: options A, C and E are to be ranked on the hex task alone.

**The principle behind every option below, in the owner's words: "you can't prerender the rules, it
changes when a player takes over another player's line."** In Spectacle a line carries its own rule
(`path.rule` / `path.table`, `CLAUDE.md` "Captured patterns"): a captured line changes owner and keeps its
pattern (`takeEnclosed` / `take`); in normal mode a rival line inside your circuit is wiped and *your*
pattern sprouts on its tiles (`convertPath`). One board holds lines of many rules, and which rule a tile's
line follows is a fact about the line — not the player, not the board. So the pattern must live in the
line's own cell state, carried from the tap, where a later owner change or replacement is a change of
state along the line and never a new per-cell input (§4). A player switching their own rule stays out of
scope.

1. **Rule space: train on the whole hex kernel, 1,953,569 clean rules, class 0 included**, split by hash
   (20 % held out), rules sampled uniformly per subset. Measured: hex class 0 makes no junctions (§2), so
   every rule fits the existing 15-bit chord layout and walker. Recommend yes. The current 100-rule set
   stays as the "legacy" held-out set so launch 6 compares with the overnight runs.
2. **Static input per cell: option C — the edge class facing each of the six directions, plus which
   direction local edge 0 faces, plus the patch's mirror sign** (55 binary planes). Baseline arm A =
   one-hot type + rotation + mirror (16 planes). Recommend C as primary, A alongside. Option E (local-frame
   perception, the equivariant design) is launch 7's arm (§3).
3. **Tap encoding: the rule as the game itself states it** — 8 bits "which edge classes carry a line"
   + 9 × 5 one-hot "matching digit per tile type" — plus a 4-way player slot and the tapped chord's two
   edge directions, on the tapped cell only, held (63 planes there, 0 elsewhere). Recommend yes (§4).
4. **State width 32 → 96 channels, hidden 128.** The line has to carry a 53-bit code plus its working
   state; hidden 128 matched 256 on the overnight task (measured, §5.5). Recommend yes.
5. **M1b (closed vs tail) becomes "the tapped strand says whether it is closed"**: same tap, six more
   output planes, trained from the M1a checkpoint. "The whole pattern with no tap" no longer exists —
   with no tap there is no rule. Recommend yes.
6. **Let the current launch finish its level-4 stage (~23:30) and cut its M1b stage** — ~$1.1 saved,
   and its only transferable result (closed-flag consensus vs loop length) comes from launch 6's own
   M1b stage on the real inputs (§6).
7. **Launch 6 = three arms side by side, ~8 h, ~$3.5**: A, C, and C with the rule code broadcast to
   every cell (a diagnostic ceiling that separates "cannot compute chords from type + orientation + code"
   from "cannot carry the code along the line"). Then launch 7: E, C without the anchor, and a width
   ablation; launch 8: several players. Recommend yes, serially, with the owner's review between them.
8. **Several players: a 4-way owner slot carried in state; output = 6 edge planes + 4 owner planes per
   cell**, not 6 planes per owner. Capture is then a relabel of the owner planes along the line (a
   wave), conversion a wipe plus a sprout with the capturer's code — both state changes. Launch 8.

## 1. Literature review and prior art

What `nca-review.md` §4 already covers (Growing NCA, MNIST, Textures, IsoNCA, Steerable NCA, Graph CA,
Pathfinding NCA, Reasoning NCA, DiffLogic, E(n)-GNCA, MeshNCA, Med-NCA, ViTCA, NoiseNCA) is taken as
read; here is what the stricter input rule adds, grouped as the owner asked. One line per source: what
it tells us to do or avoid.

### 1.1 Conditioning and goal-directed NCAs — where the "program" enters

| Source | What it does | Tells us |
|---|---|---|
| [NCA Manifold](https://arxiv.org/abs/2006.12155) (Hernandez et al., CVPR 2021) | one NCA, a vector placed at the *seed* decides which of thousands of images grows; the vector space generalises to unseen codes | **The pattern can enter at one cell and still shape the whole growth.** Closest precedent for "the rule enters only at the tap". Their codes are learned embeddings; ours is the game's own description of the rule, which is what lets held-out rules work. |
| [GoalNCA](https://arxiv.org/abs/2205.06806) (Sudhakaran, Risi 2022) | a goal vector fed to *every* cell at *every* step steers behaviour live | The easy way (broadcast) works and is exactly what the owner excludes; we use it only as the diagnostic ceiling arm (§6). |
| [HyperNCA](https://arxiv.org/abs/2204.11674) (Najarro et al. 2022) | an NCA grows the *weights* of a policy | A hypernetwork keyed on the rule would be a per-rule model, not one CA; rejected. |
| [EngramNCA](https://arxiv.org/abs/2504.11855) (2025) | private per-cell "gene" channels; a code written into a few cells spreads and switches the colony's behaviour | **Codes can be copied cell to cell and read out locally**; the nearest published thing to carrying a rule along a line. Keep code channels distinct from working channels (we give them room, we do not reserve them). |
| [Adversarial reprogramming of NCA](https://distill.pub/selforg/2021/adversarial/) | a perturbation on a few cells changes the collective's answer | Same message: local injection, global effect, learned not engineered. |
| [Self-classifying MNIST](https://distill.pub/2020/selforg/mnist/) | consensus along a connected 1-D-ish shape | Consensus along a strand (the closed flag, the carried code) is a known-learnable thing; read out in a window, L2 to a bounded target, noise only if it flickers. |

### 1.2 NCAs / GNNs that execute algorithms; length generalisation

| Source | Tells us |
|---|---|
| [Neural GPU](https://arxiv.org/abs/1511.08228) (Kaiser, Sutskever 2015) | a CA-like recurrent net learns binary add/multiply from length ≤ 20 and runs at 2,000; needs a length curriculum, parameter sharing across steps and noise; generalisation is brittle to seeds. **Curriculum over size, one shared step, many seeds.** |
| [Neural execution of graph algorithms](https://arxiv.org/abs/1910.10593), [CLRS-30](https://arxiv.org/abs/2205.15659), [no-hint NAR](https://arxiv.org/abs/2306.13411) | supervising intermediate states ("hints") helps algorithm execution and size generalisation; later work finds no-hint competitive. Our per-step light-cone targets (every chord within t steps must be drawn) are a mild hint consistent with both: they say *what* must be there by when, not *how*. Keep. |
| [Deep thinking](https://arxiv.org/abs/2106.04537), [Logical extrapolation without overthinking](https://arxiv.org/abs/2202.05826) (Schwarzschild, Bansal et al.) | recurrent nets trained on small mazes solve far larger ones **if** (a) the input is re-presented every iteration ("recall") and (b) training starts from random intermediate states ("progressive loss"). Our consts are concatenated every step (a) and the pool with ages is (b). Keep both; never drop the static planes into the state. |
| [Pathfinding NCA](https://arxiv.org/abs/2301.06820) (Earle et al. 2023) | **hand-coded BFS and DFS NCAs exist** — the DFS one carries a stack along a path, the one published mechanism closest to "carry a program along a path"; learned ones generalise best with weight-tied steps, more channels / smaller hidden, and adversarially evolved hard mazes. Max-pool for global quantities is their engineered hint — not for us. |
| [Reasoning with NCA](https://arxiv.org/abs/2609.36126) (2026) | chunked BPTT with a replay pool: 9×9 → 201×201. Our `--bptt 48` with persistent pool states is this recipe. Keep. |

### 1.3 Graph and mesh NCAs

[Graph CA](https://arxiv.org/abs/2110.14237) (universal for finite-state rules on bounded-degree graphs
with edge attributes), [MeshNCA](https://arxiv.org/abs/2311.02820), [E(n)-GNCA](https://arxiv.org/abs/2301.10497).
Tells us: a message function that takes **per-edge attributes** (here: which seam class, which local edge
index, the neighbour's facing edge) is the standard way a graph NCA knows its geometry. **Option C below
is the grid-shaped special case of exactly that** — per-direction edge labels are edge attributes laid out
in the 7-tap kernel — which is why it transfers to Spectre tiles and option A does not.

### 1.4 Rotation: isotropic, steerable, equivariant

[IsoNCA](https://arxiv.org/abs/2205.01681): rotation invariance by construction (Laplacian / gradient
magnitude) loses the ability to break symmetry; [Steerable NCA](https://arxiv.org/abs/2302.10197): **each
cell carries an angle and rotates its perception by it, so the weights never see the global frame** —
that is option E; [HexaConv](https://arxiv.org/abs/1803.02108) (ICLR 2018): group convolutions on the
hexagonal lattice, the weight-tying arithmetic for the six rotations (p6). Tells us: feeding orientation
as a *local frame the perception rotates by* is standard, shares weights 6× (12× with the mirror), and is
an architectural symmetry like convolution itself — not a hand-set weight or an engineered channel.
Feeding orientation as a *number* (cos, sin) is what the steerable paper's cells *output*, not an input
convention anyone recommends for a discrete 6-fold grid.

### 1.5 Cellular automata on aperiodic tilings

[Owens & Stepney](https://www-users.york.ac.uk/~ss44/bib/ss/nonstd/jca10.pdf) (Life on Penrose kite/dart
and rhomb, 2008–2010), a [universal semi-totalistic CA on Penrose tilings](https://arxiv.org/abs/1208.2771),
and — new, Sept 2026 — [Gliders on Aperiodic Monotilings: CA on the Hat and Spectre](https://arxiv.org/abs/2609.22579)
(Life-like rules on the tiles' adjacency graph, gliders found by evolutionary search; the paper says no
CA dynamics on these tilings had been reported before). Tells us: every CA on these tilings is a CA on
the **adjacency graph** with per-tile neighbour counts of 4–7, none is learned, none feeds per-tile
orientation. Our M4 (graph NCA on Spectre) has no precedent to copy and nothing to contradict it.

### 1.6 Loop models and Truchet-style matchings — the maths of the strands

Truchet tiles ([Smith 1987](https://www.jstor.org/stable/1578535); [generalised Truchet](https://cp4space.hatsya.com/2013/01/09/generalised-truchet-tiles/))
are exactly "a non-crossing matching of edge midpoints per tile"; random Truchet tilings are critical
percolation hulls, loop sizes power-law with fractal dimension 7/4 ([worked example](https://blog.schawe.me/en/truchet-kacheln.html)).
The [fully packed loop model on the honeycomb lattice](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.72.1372)
(Batchelor et al. 1994) has its sites at edge midpoints, i.e. on the kagome lattice — our hex rules with
all six classes selected (subset `01234568`, 99.98 % of the rules) are a *deterministic, tiling-driven*
fully packed loop configuration on that kagome lattice. Tells us: heavy-tailed strand lengths are
structural, not a data accident (measured at level 3, §2); **evaluate per length bucket and never size the
step budget to the maximum**.

### 1.7 "A cell-conditioned NCA that carries a program along a path"

Searched (NCA + program/code + path/route, 2020–2026): **nothing published does this.** The pieces exist
separately — a seed-borne code (NCA Manifold), code transfer between cells (EngramNCA), a hand-coded
stack carried along a path (Pathfinding NCA's DFS), consensus along a shape (MNIST), tile-local geometry
as edge attributes (Graph CA). Expect to be the first to put them together, and budget for it (inference).

### 1.8 Our recipe against the literature

| Ingredient | Verdict |
|---|---|
| Pool with ages, worst-slot restart, batch/8 fresh | Growing NCA + "progressive loss" (deep thinking): keep |
| Damage (state disc, tap move, cell edit) | Growing NCA; "edit" must now re-type a cell (chords are no longer an input) |
| Min-over-targets loss | not needed: strands are deterministic; the light-cone don't-cares stay |
| Truncated BPTT, persistent states | Reasoning NCA: keep |
| Hard clamp [-2, 2] | the review's open item; overflow loss is the literature's answer; an A/B for launch 7 |
| Level curriculum 2 → 3 → 4 | Neural GPU, deep thinking: keep; cut level 2 short (it saturates fast, measured) |
| Static planes re-fed every step | "recall": keep — the code enters once at the tap, the geometry every step |
| Fire rate 1.0 | fine for exactness; unchanged |

## 2. Rule space (measured with Spectacle's code, `shared/tiles`, `shared/game`)

- **Hex family**: 9 leaf types (`HEX_LEAF_ORDER`: Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi, Gamma),
  6 physical edges each, edge classes ("majors") {0, 1, 2, 3, 4, 5, 6, 8}. Every tile is the same
  regular hexagon placed by a rigid transform: 6 rotations, and a mirror sign that is uniform within a
  patch and alternates with level (`strand-data.md`). The kernel over GF(2) has 7 non-empty subsets.
- **Clean rules** = a kernel subset × one non-crossing matching per leaf type (`validEdgeSubsets` ×
  `nonCrossingForTile`). A type with 2 / 4 / 6 connection ends has 1 / 2 / 5 non-crossing matchings
  (Catalan), 0 ends → 1 (draws nothing):

| subset | `15` | `128` | `258` | `01346` | `03456` | `023468` | `01234568` | total |
|---|---|---|---|---|---|---|---|---|
| rules | 4 | 32 | 64 | 8 | 16 | 320 | 5⁹ = 1,953,125 | **1,953,569** |

- **The "600k+" is the Spectre family**: the same count there is **625,270** (8 subsets, 10 types;
  `01235678` alone 625,000). The hex family the CA trains on first is three times bigger. Both from the
  same script run today.
- **Class 0 makes no junctions on hexagons** (measured: 120 random clean rules over all 7 subsets,
  level-3 Delta patch, every chord walked by `walkStrand`: 0 `junction`, 0 `limit`). In the hex family the
  class-0 contract is the edge midpoint (`DESIGN.md` §3: a vertex only in the spectre family), so every
  edge carries at most one chord end and the 15-bit chord layout plus `nca/strand/walker.py` cover the
  whole 1.95 M. The old plan's exclusion of class 0 was unnecessary for hex.
- **Strand lengths at level 3 by subset** (same probe; chords, p50 / p90 / p99 / max): `15` 2/4/9/9;
  `128` 8/25/184/465; `258` 2/9/28/104; `01346` 6/21/81/179; `03456` 2/7/34/180; `023468` 4/24/82/210;
  `01234568` 3/10/72/390. Subset `128` (the infinite-line family) stays the longest tail; the fully packed
  subset is mostly short loops with a long tail.
- **A cell's chords as a function of (type, rotation, rule)** — this is what the CA must learn per cell:
  1. the tile's six local edges carry majors `m₀..m₅` (`HEX_EDGE_LABELS`, e.g. Delta `3 2 5 1 3 6`, signs
     and A/B variants do not matter for chords);
  2. the rule's subset `S` selects the ends: local edge `k` has a connection point iff `mₖ ∈ S`;
  3. the rule's digit for this type picks a non-crossing pairing of those ends **in cyclic order starting
     at local edge 0** (`enumerateMatchings` order; for 6 ends the non-crossing ones are indices 0, 2, 6,
     12, 14 = `01·23·45`, `01·25·34`, `03·12·45`, `05·12·34`, `05·13·24`; for 4 ends indices 0, 2);
  4. local edge `k` faces grid direction `(mirror · k + rot) mod 6`.
  So the rotation matters twice: where each edge faces, and where the digit's cyclic order starts.
  Option C feeds the model step 1 and the anchor of step 3 directly; option A feeds `(type, rot)` and
  leaves all four steps to be learned per combination.

## 3. Options for giving each cell its tile type and orientation

Common to all: the board mask stays (1 plane). "Bitter-lesson standing" is the owner's line — static
board facts and the tap are task specification; anything that pre-computes part of the answer is not.
"Spectre" = how it carries to 10 chiral types, 14 edges in ~6 seams, 30° rotations, 4–7 neighbours, on a
graph NCA. Costs are input planes into the first (7-tap) layer only; parameter cost ≈ planes × 7 × hidden.

| | A. one-hot type + rotation (+ mirror) | B. joint (type, rotation, mirror) code | C. per-direction edge classes + anchor + mirror | D. type one-hot + angle (cos, sin) | E. local-frame (equivariant) perception | F. learned embedding table |
|---|---|---|---|---|---|---|
| Planes | 9 + 6 + 1 = 16 | 108 one-hot | 6 dirs × 8 majors + 6 anchor + 1 = 55 | 9 + 2 + 1 = 12 | 9 type (+ nothing for rotation: the frame is the rotation) | d (8–16) per cell from a 108-row table |
| What the model must learn | the full lookup (type, rot, mirror, S, digit) → 15 bits: 54 × 2 geometries, no sharing across rotations; the walk in the grid frame | as A, each combination its own feature: fastest to memorise, least shared | "end at direction d iff major(d) ∈ S" (a bilinear AND of two 8-bit vectors), then "pair the ends cyclically from the anchor by the digit"; the walk in the grid frame | as A plus decoding a continuous angle to one of six | only (type, S, digit) → local chords (9 × 7 × 5 = 315 cases) and the walk in the cell's own frame; rotation never seen | as B through a bottleneck |
| Bitter-lesson standing | pure facts; the model does everything | pure facts | pure facts in the game's own vocabulary (the rule editor shows edge classes); the anchor is a fact (where edge 0 points); nothing about chords is precomputed | pure facts, awkwardly encoded for a 6-fold grid | an architectural symmetry (weight tying under the tile's frame), like convolution; no hand-set weight, no channel | learned end to end; same standing as A/B |
| Spectre later | does not carry: 10 types × 12 rotations × chirality, and a graph has no "rotation" plane | same | carries directly: per-edge attributes on the message function are the graph NCA norm (§1.3); the 14 edges' classes and seam membership become edge features | poorly: angle is meaningless on a graph | carries best: a graph NCA whose messages are ordered by the receiving tile's local edge index *is* this design | carries (an embedding per Spectre type) but says nothing about edges |
| Cost | ~16 × 7 × 128 = 14k params; trivial compute | ~97k params on the first layer | ~49k params; trivial compute | trivial | 7-tap gather per cell with a per-cell permutation (12 frames), then a 1×1: ~7× activation memory, ≈ same FLOPs; ~2 h to build + a parity test | trivial |
| Cheap experiment to rank it | **per-cell supervised lookup, no CA**: a 1-hidden-layer MLP (same width) from (static planes, rule code) → 15 chord bits, on 1.9 M random (type, rot, rule) draws; held-out rules; minutes on the Pi's CPU. Ranks A/B/C/D by how fast and how exactly the lookup is learnable | same | same | same | same probe in the local frame (target = local chords) — expect near-trivial | same |

**Judgement** (written before the owner's update in §0; the Spectre reasons no longer count). C is the recommendation: it is still only board facts, it is the representation a graph
NCA on Spectre tiles would use anyway, and it makes the per-cell lookup two bilinear steps instead of a
108-way memorisation. A is the baseline the owner can compare it against for the cost of one more run.
E is the strongest design for the long run (6–12× weight sharing, the Spectre form) but needs a model
change and its own parity test; it is launch 7's arm, so launch 6 keeps one variable (the input planes). B and F add nothing over A's first layer
(which already is a learned embedding of the one-hot). D is not recommended for a discrete 6-fold grid.
The one risk specific to C: the model could read "which directions have ends" without learning *why*
(the AND with the subset bits) and fail on rules whose subset it never saw — the held-out split holds
out digits, not subsets (there are only 7), so this is covered by the legacy and wide sets jointly.

## 4. Options for how a tap supplies the pattern

The tap must identify one of 1.95 M rules, the line must carry it, and unseen rules must work. The size
matters because every cell on the strand ends up holding a copy in its hidden state.

| | T1. the rule as the game states it | T2. the same, digits in binary | T3. a learned rule embedding | T4. broadcast to every cell | T5. only the tapped tile's chords |
|---|---|---|---|---|---|
| Planes on the tapped cell | 8 class bits + 9 × 5 digit one-hot = 53 | 8 + 9 × 3 = 35 | d (e.g. 16) from a table or hypernetwork | 53 on every cell | 6 (today's tap) and nothing else |
| Generalises to unseen rules? | yes if the model learns the composition (class bits × per-direction classes; digit × type) | yes, harder decoding | **no** — an embedding table has no row for an unseen rule; a hypernetwork from the code is T1 plus a layer the model already has | yes, and trivially easy | no: the line would have to *infer* the rule from what it sees — impossible for types it has not visited |
| Standing | task specification: the tap says which pattern | same | engineering | the old control arm; excluded by the owner for the product, kept as a diagnostic | purest, but ill-posed |
| Verdict | **recommended** | fallback if 53 channels prove costly | no | diagnostic arm only (§6) | no |

Alongside the code, the tap carries **the tapped chord's two edge directions** (6 bits, as today): in
Spectacle the pointer picks the nearest chord of the tile (`nearestChord`), so which chord was tapped is a
fact about the tap, not a hint — and the model still has to compute that tile's other chords and every
tile beyond. Alternative for a stricter reading: give the *nearest edge* only (6-way) and let the cell
pick the chord through it; same information once the cell's chords are known. Both exist in the data
(`tap_d0/d1`). I recommend the two edges for launch 6 (it is what the exported taps are).

**Several players.** A 4-way **owner slot** one-hot on the tap (`K = 4`, Spectacle's rooms hold ten but
collisions are per pair of lines; four at once per board is enough for M3). The line carries the slot in
state like the code. **Output per cell = 6 edge planes + K owner planes** (a tile holds one owner's line
— a rival's line owns its whole tile, `CLAUDE.md`), rather than 6 planes per owner: the output does not
grow with K, and "which player's line is this" is a readout, not a separate channel bank. Patterns are
then per *tap*, not per player: a player with several patterns is several taps with different codes, and
the engine's flip/burn between a player's own patterns stays outside (as the plan already said). A cell
with two chords of one strand (14 % of strand cells, measured in `strand-data.md`) holds one owner; a cell
where two owners' chords would meet is a collision and is M3's business, not M1's.

**Why the pattern must live in the line, and what that fixes in the design.** The owner's reason for
ruling out pre-rendering is the game itself: a line carries its own rule (`path.rule` / `path.table`),
and lines change hands. **Capture** (`takeEnclosed` → `take` event): the line's owner changes, its pattern
stays. **Conversion** (normal mode, `convertPath`): a rival line wholly inside your circuit is wiped and
your pattern sprouts on its tiles. A per-cell rule input would have to be rewritten by the host on every
capture and conversion, for every tile of every affected line — the "prerender" the owner rejects. With
the code and the owner slot both in the line's cell state (T1 + the slot), the M3 mechanics are state
changes the CA can make locally: a capture is the owner planes relabelling along the line, a wave from
where the enclosing circuit's cells (which hold the capturer's slot) touch it; a conversion is the old
line's wipe (the damage recipe already trains erasure waves) followed by a sprout whose code is the
capturer's — present locally, because the capturer's circuit cells carry it and the territory fill (M2)
can carry it inward. Nothing in M1 depends on these; what M1 must get right is that **a strand cell's
state holds its line's code and slot, readable by its neighbours** — the code-fidelity probe (§5.4)
measures exactly that. A player switching their own rule (regrow) stays out of scope.

**Capacity.** 53 + 4 bits of code plus the walker's working state in `channels − 13` hidden channels:
32 is too few; 96 gives 83. Hidden 128. (Inference; the overnight runs say 128 ≈ 256 on the old task.)

## 5. Task and training plan

### 5.1 Milestones, restated

| | M1a: a strand from a tap | M1b: the strand knows if it is closed | M1c: several players |
|---|---|---|---|
| Input per cell | mask; static planes (A or C); on tapped cells only: rule code 53 + owner 4 + chord edges 6, held | same | same, several taps with different codes and slots |
| Output | 6 planes: edge d crossed | + 6 planes: the chord through edge d is on a circuit | + 4 owner planes |
| Fixed point | the whole strand through the tapped chord, both ways; nothing beyond a tail | every strand cell agrees | every tap's strand, each cell's owner right |
| Target | `walker.walk` from the rendered chords of (board, rule) — exactly today's targets with light-cone don't-cares | `closed` from the walk, per edge, with the "open news" light cone (today's m1b targets, tap kept) | union of walks; taps drawn so the strands are disjoint (collisions are M3) |
| Bar | held-out-rule exact ≥ 0.95 at level 3 on the legacy set (the overnight bar), and the wide set reported per subset and length bucket | agreement ≥ 0.95 | exact ≥ 0.9 with 2–4 taps |

Rule changes, regrow, flips, collisions, captures: out, as the owner said.

### 5.2 Targets and data

Everything the targets need already exists: the exported boards carry `tile_type`, `tile_rot`,
`board_mirror` and `mask` per cell; `walker.render_chords(type, rot, mirror, local_pairs)` renders any
rule's chords from them and `walker.walk` gives the strand (both verified against `walkStrand` on
2,000 + 131,840 taps, `strand-data.md`). What is missing is `local_pairs` for rules outside the 100: a
**rule table** from Spectacle — per (subset, type, digit) the local chord pairs, 315 entries — exported
once (`scripts/strand-export.ts` gains a `--rule-table` output), plus a Python `rules.py` that samples a
rule (uniform over the 7 subsets, then uniform within, held-out by `fmix32(FNV-1a("strand-split-v2|" +
key)) < 0.2`), encodes it as the 53-plane code, renders it onto a board and walks a tap. Rules are drawn
*on the fly* per pool slot: with 1.95 M rules and ~10⁶ samples a night, nearly every training sample is a
rule the model has never seen, which is the generalisation pressure we want. The hash split still
guarantees the eval rules were never drawn.

### 5.3 Split, curriculum, budget, size

- **Held-out**: the legacy 20 rules of the 100 (for comparison with the overnight numbers, same taps,
  same `EVAL_SEED`) and a **wide** set: 40 held-out rules per subset (all 4 of `15`, etc.), 120 taps per
  level stratified by length bucket as now. Report both, per subset.
- **Curriculum**: level 2 (25 min) → levels 2 + 3 (240) → 3 + level-4 crops (150) → M1b (60), per
  arm, `--init` chaining as in `plan-5.txt`. Later stages start at `--lr 2e-4`, not 5e-4: the overnight
  level-4 stage restarted at the peak lr and its long-strand exact fell from 0.66 to 0.36–0.69 for an hour
  (measured, `runs/m128-l4`).
- **Step budget vs strand length**: unchanged — one chord per step both ways; rollouts U[3, 6] × S with
  persistent pool states so long strands complete across visits; readout at max(8 S, longest ideal + 8).
  Settle ratio (steps to exact / ideal) stays a reported metric: the code copy must not slow the walk.
- **Model**: `HexNCA`, perception `taps`, channels 96, hidden 128, clamp [-2, 2]; batch 32, pool 256
  per level, damage 0.3 with "edit" = re-type 1–3 cells (type and rotation), never the tapped one.

### 5.4 Metrics

`exact` (every edge of every strand cell right, nothing else drawn), **`balanced`** (mean of exact over
short / medium / long buckets — the headline, as now; best.pt follows it), `iou`, `steps` (settle ratio
and excess), `byLevel`, `bySubset` (7), `byLen`, and `bySet` (legacy / wide). For M1b: agreement per
strand and per board. Add one new number: **code fidelity** — a linear probe from the hidden state of a
strand cell to the 53 code bits, trained on the fly from the pool (minutes); it says whether the line
carries the rule at all, independently of whether it draws right (the single most diagnostic number for
this design; inference).

### 5.5 What carries over

- From `nca/strand`: the walker, the loader's fresh-tap path, the pool/damage/light-cone targets, the
  quick check and its length buckets, `selftest`, plan/launch tooling. The exporter's boards.
- From the overnight runs (measured, end of the level-3 stage, held-out rules, levels 3 + 4):

| hidden | exact | balanced | long | subset `128` | level 3 | level 4 | s / iteration (3 runs sharing) |
|---|---|---|---|---|---|---|---|
| 64 | 0.855 | 0.688 | 0.29 | 0.65 | 0.90 | 0.81 | 0.32 |
| 128 | 0.946 | 0.867 | 0.66 | 0.85 | 0.97 | 0.93 | 0.36 |
| 256 | 0.925 | 0.842 | 0.66 | 0.85 | 0.98 | 0.86 | 0.87 (8k iterations only) |

  Lessons: the local walk is easy for any width; long strands are the whole difficulty and 128 is as good
  as 256 there at a third of the cost; `dataSec` is already 25–30 % of an iteration (0.09 of 0.36 s), and
  rendering rules on the fly adds to it — the review's "data path off the training thread" is worth doing
  before launch 7. The weights do not carry over (new input layout).

## 6. Compute plan for the remaining GPU budget

**The budget is money, not time.** $20 in all; ~$4.70 spent so far (11.5 VM-hours, launches 1–5); the
current launch (plan-5: rendered-chord M1a/M1b at hidden 64/128/256) costs ~$1.70 more if it runs to its
end at ~02:15 UTC. That leaves **~$13.5 ≈ 32 Spot L4 hours at ~$0.41/h** (`g2-standard-4`; Spot prices
move), to spend over whatever time it takes. GPU quota is 1 per project, so launches are serial; runs
*within* a launch share the one GPU, which costs nothing extra while the GPU stays busy (three runs kept
it saturated tonight, measured) — so A/B arms are free in dollars and only split the iterations. An idle
day costs nothing; a launch costs ~5 min of setup. Now ~22:30 UTC 10-05.

### 6.1 The current launch: keep the level-4 stage, cut the M1b stage

| Stage (UTC) | What it gives | Cost to run out | Verdict |
|---|---|---|---|
| level 4, until ~23:30 | long-strand exact on level-4 crops for the superseded input — the reference every new arm is judged against (the only measured long-strand numbers we have: 0.29 / 0.66 / 0.66 at the end of level 3) | ~$0.4 | **keep** |
| M1b with pre-rendered chords, ~23:30 → 02:15 | closed-vs-tail consensus on *every strand of a board at once* with no tap — a task that no longer exists: with no tap there is no rule, and under the new design M1b is "the tapped strand says whether it is closed", which launch 6 trains anyway on the real inputs. The one transferable fact (how consensus cost grows with loop length) is read off launch 6's M1b stage for free | ~$1.1 (2.75 h) | **cut at ~23:30** (`down.sh` after the level-4 stage's last checkpoint); 8 % of what is left, better spent on an extra arm or a longer stage |

Cutting also frees the GPU: with serial quota, launch 6 cannot start while plan-5 runs, and the build
(§6.3) is expected to pass selftest around 01:30 if two agents start now.

### 6.2 Launches, in order, with the owner's review between them

| Launch | Arms (share one GPU) | Per arm | GPU-h | ~$ | Decides |
|---|---|---|---|---|---|
| **6** (as soon as the build passes selftest) | **A** one-hot type + rot + mirror (16 planes) · **C** per-direction classes + anchor + mirror (55) · **C-bc** C with the 53-bit code on every cell (diagnostic ceiling). All: rule-at-tap T1, channels 96, hidden 128, batch 32 | l2 25 → l3 240 → l4 150 → m1b 60 min = 475 min | ~8.3 | ~3.5 | A vs C on the legacy and wide sets; whether the line carries the code (C vs C-bc); long-strand exact against the overnight 0.66 |
| **7** | **E** local-frame perception · **C′** C without the anchor · width: the launch-6 winner at channels 64 / 128 (96 is in launch 6) | same chain | ~8.3 | ~3.5 | the representation for the Spectre graph model; the smallest width that works |
| **8** | **M1c** 2–4 taps per board, owner planes; M1b from the launch-7 winner | l3 240 → l4 150 min | ~6.5 | ~2.7 | several players; the owner's three milestones done |
| reserve | Spot preemption restarts (a resumed launch pays setup again), `evaluate` at n ≥ 200 on the VM after each launch (~10 min each), one longer run of the final winner if the wide set is still short of 0.95 | | ~9 | ~3.8 | |
| **Total** | | | **~32** | **~$13.5** | |

Gates (readable from the dashboard, no one has to be awake): at +90 min into launch 6's l3 stage an arm
with held-out exact ≥ 0.6 and rising is alive; C-bc far ahead of A and C says "the line does not carry
the code" (fix before launch 7: channels, the code-fidelity probe, EngramNCA-style noise on the code
channels); all three flat under 0.3 says "the per-cell lookup is not learned" (run the §3 supervised
lookup probe on the Pi before spending more). Launch 7 is not written until launch 6 is read: if C-bc
and C tie, the code-carrying question is closed and launch 7 can spend its arms on E and width alone.

What the owner should see after launch 6: A vs C ranked on the same held-out taps as the overnight table,
the diagnostic ceiling, the legacy-set comparison with the rendered-chord model, and the per-subset wide
numbers — enough to pick the representation and size launch 7.

### 6.3 What must be built first (coding-agent estimates; two agents in parallel ≈ 3 h)

| # | Item | Where | Est. |
|---|---|---|---|
| 1 | Rule table export: per (subset, type, digit) local chord pairs + per-type majors, from `chordTableFor`/`localChords`; parity check on 200 random rules incl. class 0 against `walkStrand` | `scripts/strand-export.ts` (`--rule-table`), `data/strand/rules-hex.json` | 45 min |
| 2 | `nca/strand/rules.py`: hash split v2 over the whole kernel, stratified sampling, the 53-plane code, `render_chords` + `walk` for any rule on any exported board | new, ~120 lines | 45 min |
| 3 | Static planes per option (`--inputs a|c|c-bc`): A from `tile_type/tile_rot/board_mirror`; C from the majors table; tap planes = code + owner + chord edges on the tapped cell; `N_IN` per option; `consts_of`; "edit" damage = re-type a cell | `nca/strand/train.py`, `loader.py` | 1.5 h |
| 4 | Eval sets: legacy-20 (unchanged files) + wide (40 held-out rules per subset, rendered on the fly, fixed seed); `bySet`; the code-fidelity probe | `train.py` | 45 min |
| 5 | `selftest` (3 min on the Pi per option), `plan-6.txt`, tgz + bucket upload (lead) | `nca/strand/selftest.py`, `nca/cloud/` | 30 min |
| 6 | M1b with the tap held (today's m1b targets + today's m1a tap/targets, 12 output planes) | `train.py` | 30 min (last stage of the chain; can land after launch 6 starts) |
| 7 | Before launch 7: option E (per-cell frame gather + 1×1; parity: outputs identical under rotating a board), the data path off the training thread (`dataSec` is 25–30 % now and on-the-fly rendering adds to it) | `nca/model.py`, `train.py` | 2 h + 2–3 h |
| 8 | Before launch 8: multi-tap loader and targets, owner planes, disjoint-strand tap sampling | `loader.py`, `train.py` | 2 h |

Critical path for launch 6: 1 → 2 → 3/4 → 5. If item 1 slips, items 2–5 run on the 100-rule set (the
legacy split; `local_pairs` already in `meta.json`) and the wide set waits for launch 7 — a weaker test of
"unseen rules among 1.95 M" but the same A / C / C-bc ranking. Nothing here is urgent in dollars; the
order is set by what each launch needs to be readable.

## 7. Risks, each with its cheapest test

| Risk | Cheapest test | Cost |
|---|---|---|
| The per-cell lookup (static facts + code → chords) is not learnable at this width or in this encoding | the §3 supervised MLP probe on 1.9 M random draws, A vs C, held-out rules — before the launch if time allows, else alongside it | 15 min CPU |
| The line does not carry the code (copies drift along hundreds of cells) | C-bc arm vs C; the code-fidelity probe vs distance from the tap | in launch 6 |
| Long strands get worse, not better, with the code to carry | `byLen.long` on the legacy set against the overnight 0.66 | in launch 6 |
| Sampling uniformly over subsets starves `01234568` (99.98 % of rules) of its own variety, or uniform over rules starves everything else | report `bySubset` on the wide set; if the fully packed subset lags, weight it 50 % | 0 |
| Rule-space statistics differ from the 100-rule set (fully packed boards: 3 chords on every tile) | `nca.strand.stats` over the wide set once item 2 lands (lengths, closed share, twice-visited cells) | 10 min |
| Mirror sign: levels 2 and 4 share one sign, level 3 the other; a model trained on 2 + 3 sees both, but a stage on 3 + 4 could tilt | `bySet` × level in the quick check; the eval boards cover both | 0 |
| The lr restart per stage knocks a stage back (measured tonight) | `--lr 2e-4` on later stages; or `--init` with a warm lr floor | 0 |
| `dataSec` grows with on-the-fly rendering and the GPU idles | `dataSec` in the log; if > 40 % of an iteration, pre-render per-slot rules in a worker (review item 5) | 2–3 h eng, before launch 7 |
| A 53-bit tap plane set is a strong per-cell signal that could be "read" from neighbours by position rather than copied — fine on one tap, wrong with several | M1c with 2–4 taps of different codes: owner planes and exact per tap | launch 8 |
| Option C's anchor plane is "where local edge 0 points" — a fact, but one the owner might read as a hint | drop it (C′): the label cycle of every hex type has no rotational symmetry (measured: all six labels distinct per type), so the anchor is recoverable from the labels; run C′ as an arm in launch 7 | 1 run |
| Spectre transfer: A does not carry; C/E do | build the graph NCA on the hex adjacency graph first and match M1's numbers (the plan's M4 order) | 1 day eng |
| Server/client float disagreement (unchanged from the plan) | 1,000-step rollouts in `src/nca.ts` vs PyTorch once a model is worth porting | 20 min |

What is deliberately not in this document: implementation, the browser port, Spectre tiles beyond the
transfer column, rule changes and everything the plan's §10 already left for later.
