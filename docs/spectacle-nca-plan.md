# Spectacle as a cellular automaton: plan for the next stage (2026-10-05)

Companion to `nca-review.md` (the state of the flood-fill NCA) and `nca-journey.md` (the history).
"Measured" = read from code, logs or a read-only probe today; "inference" = my judgement. Nothing here
is a settled decision; §2 and §9 list the ones the owner has to make.

## Summary

- **Two phrases, two bars.** *Recreate the patterns* = given a tiling and a player's rule, the CA draws
  what Spectre's explorer draws: every strand, each known to be a circuit or a tail. *Replace
  Spectacle* = the CA is the physics: lines grow from taps one tile per step by the rule, stop at
  tails, close as circuits, claim the smaller side, collide; the server stays the referee (taps,
  heads, speeds, scores, captures' bookkeeping, the wire).
- **Hexagons first, Spectre tiles later, by a wide margin.** Spectacle's hex field is an exact axial
  lattice (measured: 0 off-lattice tile centres at levels 3-6), a ragged mask in a 30×37 (level 3) or
  101×109 (level 4) box; every tile has the same six edge directions; the current `HexNCA`, the
  trainer and `src/nca.ts` apply with new input planes and no new geometry. Spectre tiles (14 edges,
  ~6 seams, 4-7 neighbours, chiral) want a graph CA; the recipe transfers, the weights don't.
- **But the flood model does not yet work on that ragged field** (measured today: with the level-3
  patch as the mask, bridges 0.21 exact, "none" 0.79; on a disc of the same radius 0.79 / 0.13). It
  has only seen convex rims. Ragged masks in the training pool are the first task of any game-facing
  stage, and M1 trains on ragged patches from day one.
- **The rule goes in as the board the player sees**: per cell, 15 bits saying which pairs of the six
  edge directions are joined by a chord under the player's rule (the rule rendered onto the tile,
  exactly what the rule editor's thumbnails and the patch preview show). Tile type and rotation are
  then not needed as inputs. This is task specification, not an algorithmic hint; the "purer" variant
  (type + rotation + the rule as a vector, 69 planes) is kept as a control arm if the owner wants it.
- **Milestones**: M1a a strand from a tap (the local walk) → M1b every strand knows whether it is
  closed (the patterns) → M2 one player's game (taps, growth, circuits, claims = the flood model
  reused) → M3 two players (collisions, captures) → M4 Spectre tiles (graph CA) → M5 the referee.
  M1a/M1b are the next 24 h; the rest is scoped but not started.
- **The "smaller side" rule**: Spectacle compares polygon **areas** per claim (`field.ts:618`,
  `polygonArea(a) <= polygonArea(b) ? a : b`), each claim independent of every other line. The flood
  model mostly learned area already (4:1 over the rim-arc rule, measured). For the game stage: area
  only, per claim, ties either side. The flood model needs a fine-tune, not a restart (§2).
- **Compute**: launch 3 (running, plateaued) → launch 4 at ~14:00 UTC: the flood queue from the
  review (3.5 h) → launch 5 overnight: M1a/M1b with a width ablation (≤ 10 h). ≈ $8-9 of the $20;
  the single L4 and the clock are the limits, not money. Keep `g2-standard-4`.

## 1. What the two phrases mean as a CA

| | Input (per cell, per step) | Output (per cell) | Fixed point |
|---|---|---|---|
| Recreate the patterns (M1) | board mask; the rule rendered as chords; optionally a tap | which of the 6 edges a line crosses; whether that line is closed | Spectre's `analyze()` picture for the rule: every strand, circuits vs tails |
| Replace Spectacle (M2-M5) | the above, plus tap events with an owner, and the referee's control planes (who may advance this step) | per owner: chords occupied; territory (the fill); head position | the engine's state after the lines have grown, closed, claimed and collided |

The current flood-fill model is M2's territory half: "closed line → inside fills; edge-to-edge line →
smaller side fills" is exactly its task, with lines standing where walls stand now.

## 2. Which "smaller side" rule (owner decision)

Three rules exist today (journey doc, "What is being computed"); Spectacle's is now read from the code:

| Rule | Measure | Exact tie | Several edge-to-edge claims |
|---|---|---|---|
| Hand-written CA (`DESIGN.md` §4) | region area; exterior = dead cells + rim-connected walls | nothing fills | region rule: every exterior-touching region smaller than the largest fills |
| Trained NCA (`nca/data.py targets()`) | area **or** rim-arc span (EPS 0.02), either acceptable | one side, either | region rule: all rim regions but one largest fill |
| Spectacle (`shared/game/field.ts boundaryRegion`, line 580-618; `engine.ts closeCircuit` 1495) | **polygon area** of (line + outline arc), the two arcs compared | `<=` picks the "forward" arc: deterministic but arbitrary | **per claim, independent**: each line's region is its own polygon; other lines are ignored; regions may nest or overlap; a region is kept on the path, used for captures (`captureEnclosed`: every step midpoint of a rival line inside) and tap refusal (`insideRivalCircuit`); **not scored** (score = distinct tiles your lines are on) |

Where they differ in practice (inference):

- Measure: the model already follows area on boards where area and arc disagree (20/24 at R 8,
  15/24 at R 10; arc 2 and 5; neither 2 and 4). Switching the training target to area only is a
  fine-tune and makes the metric stricter.
- Ties: measure-zero in the game (polygon areas are real-valued); for training, "either side" stays
  the only learnable target. Spectacle's forward-arc pick is an artefact; the owner may want to make
  it explicit, nothing in the CA depends on it.
- Several claims: identical for one claim; different for two or more (ends 40 / 40, middle 20: the
  per-claim rule fills both ends, the region rule fills one end and the middle). Multiplayer needs
  the per-claim form anyway, because territory must have an owner, and in the engine the only lines
  that can lie inside a claim are wholly inside (lines cannot cross a line without both dying), so
  "other lines transparent" and "other lines are walls" differ only on nested claims.

**Recommendation.** Learn Spectacle's rule: **area, per claim, ties either side.** In the CA that is:
when a line closes (loop or edge to edge), the territory it claims is the smaller of the two sides it
cuts, other lines transparent, latched into a per-cell *territory owner* channel. The flood model
needs retraining either way, because its inputs change (lines as chords instead of walls; ragged
masks; per-owner fill) — as a fine-tune from the current weights through a widening loader, with
area-only targets. M2a (one line on the board at a time) is exactly the current task, so the transfer
is direct; M2b adds the per-claim latch, which needs the same "this line is closing" wave along the
line that M1b's closed flag needs. Measure first how often nested claims occur in bot games (10 min
with the engine: count claims whose polygon contains another claim's line) — if rare, M2 can treat
other lines as walls and only M2b pays for transparency.

## 3. Representation: hexagons first, then Spectre tiles as a graph

**Hex (measured, Spectacle `buildField`)**: every hex leaf is the same regular hexagon; the 9 types
differ only by their edge labels; within a patch every tile has the same mirror sign, so a cell's
state is (type ∈ 9, rotation ∈ 6) and only rotation varies the geometry; edge k faces direction
d0 + 60°k; the neighbour across edge k is the lattice neighbour in that direction and its facing edge
is k + 3. Patch sizes: level 3 = 496 tiles (30×37 axial box, 45 % used), level 4 = 3,905 (101×109,
35 %), level 5 = 30,744 (243×298), level 6 = 242,047 (802×865). The level-n hex patch is the
supertile graph of the level-n Spectre patch (`docs/FASS_THEOREM.md` §1: every supertile is a
combinatorial hexagon with the hex family's labels), so a hex CA is also the Spectre tiling's coarse
structure.

**Spectre**: 10 leaf types, 14 unit edges in ~6 seams, edge-to-edge but a seam can straddle two
neighbours, 4-7 neighbours per interior tile, chiral, rotations in 30° steps. No regular neighbourhood,
so no convolution. The fit is a graph NCA (Grattarola et al., NeurIPS 2021): per-node state, messages
from neighbours with per-edge attributes (our edge index, the seam's class and sign, the neighbour's
facing edge), sum or attention aggregation; their universality result says any finite-state rule on a
bounded-degree graph — `stepForward` is one — is representable exactly. No published NCA on an
aperiodic tiling exists (searched); Voronoi/mesh NCAs are the closest precedent.

**Order and why**: M1-M3 on hex, M4 on Spectre. (1) Everything exists for hex: trainer, pool, mask,
browser port, parity fixture. (2) The rendered-chord input makes rotation irrelevant to the model,
so there is no isotropy problem to solve first. (3) The graph CA should be built and checked on the
hex graph (6-regular, where the conv model gives the answer) before the Spectre graph — M4's first
test is "same numbers as M1 on the same data". (4) Strand lengths and board diameters are the cost
drivers, and they are the same on both tilings at a given level, so what is learned about horizons
on hex carries over.

## 4. Milestones

Smallest to full. Each row: what the CA sees, what it must output, the oracle and how data reaches
PyTorch, the bar, and what is deliberately left out.

| | M1a: a strand from a tap | M1b: closed or tail (the patterns) | M2: one player's game | M3: two players | M4: Spectre tiles | M5: the referee |
|---|---|---|---|---|---|---|
| Input per cell | mask (1); chords: 15 bits, which of the C(6,2) edge pairs are joined under the rule, grid frame; tap: the 2 edge bits of the tapped chord on the tapped tile, held | same | same, but the tap is a one-step pulse with an owner slot; a per-owner "go" pulse (host-controlled speed) | + owner slots (≤ 4) on taps; referee planes: tap refused / allowed | per node: the seam pairing of the tile's rule (per-edge attributes), mask, tap | as M3 |
| State / output | 6 planes: edge k crossed (> 0.5); hidden | + 1 plane: this line is a circuit | per owner: 6 edge planes; territory (1 per owner); head flag; hidden | as M2 ×owners | as M1/M2 on nodes | as M3 |
| Fixed point | the whole strand through the tapped chord, both ways: a circuit or tail-to-tail | every strand cell agrees on closed/open (no tap: the whole pattern) | after each tap: the line grown to its end, closed loops and edge-to-edge claims filled (smaller side, area) | lines that meet a rival's chord both die (mutual cut); a rival line wholly inside a new circuit changes owner (conquest) or converts (normal) | same as M1/M2 | the engine's state |
| Oracle | `walkStrand` from both exit ends (Spectacle `shared/game/strand.ts:202`); Python port of the walk over exported chord tables, parity fixture from Node | `walkStrand(...).closed` | Spectacle `Engine` with one player: `tap`, `tick`, `snapshot()`; territory from `tilesEnclosed` / `boundaryRegion` | `Engine` with two players (`Bots` or scripted taps), `tapOntoOthers` default | Spectre `automaton.ts chordPairing`, `analyze`; `Engine` with family `spectre` | the engine |
| Data to Python | Node: field geometry (axial a,b, type, rotation per tile) + per-rule chord tables + 2,000 random walks as a fixture → JSON/`.npz`; then targets are computed in Python on the fly (a table-lookup walk) | same | Node runs scripted games on level-2/3 patches and writes per-tick `.npz` (inputs, chord occupancy, territory); the trainer's pool holds (state, target) pairs | same | Node exports the adjacency with edge attributes | — |
| Boards | random supertile patches (9 types × levels 2-3: 55-559 tiles) + random masks/crops of the level-4 patch | same | level-2/3 patches | level-2/3 | level 2-3 Spectre | level 4-5 |
| Rules | the 100 clean hex rules without class 0 (subsets 15, 128, 258: 4 + 32 + 64), 80 train / 20 held out; later class-0 subsets (8 + 16 + 320 + 1.95 M) | same | same | same | Spectre's kernel (`1278` etc.) | — |
| Success | exact strand (all 6 bits on every cell) ≥ 0.95 on held-out rules and patches at level 3; nothing drawn beyond a tail; steps to completion ≈ strand length / 2 | agreement (every strand cell right) ≥ 0.95; loops up to level-3 size | per-tick exact match of chords and territory after K CA steps per engine step, over 3-tap sequences ≥ 0.9 | collisions resolved exactly ≥ 0.9; captures (M3b) | M1a/M1b numbers on Spectre ≥ hex numbers − 0.05 | plays |
| Out of scope | class-0 junctions (engine's `junctionPolicy` 'random' is nondeterministic), several taps, closed flag, colours | circuit length classes (the explorer's colouring) | rivals, speeds, scores, flips/burns between own patterns, rule changes | > 2 owners, speeds, scores, regrow | the game on Spectre | — |

Notes on M1: with the tap held constant the strand is a fixed point (like a seed in Growing NCA);
"the whole pattern" (M1b without a tap) is the chord input itself plus the closed flag, which is why
the closed flag is the content of "recreate the patterns". Counting (measured with Spectacle's
`validEdgeSubsets` / `nonCrossingForTile`): hex clean rules per subset 15: 4, 128: 32, 258: 64,
01346: 8, 03456: 16, 023468: 320, 01234568: 1,953,125. The infinite-line rule `128·010100000` is one
of the 100 and is a legitimate training case (the owner's call stands: finding it should pay).

## 5. Mechanics: local, regional, global

From the engine (`shared/game/engine.ts`, read today); "carry" = how a CA would do it.

| Mechanic | Class | Carry in the CA | Or keep outside |
|---|---|---|---|
| Strand step (entry edge → exit edge via the tile's chord; neighbour's facing edge = k + 3) | local | learned (M1a); the rendered chords make it a lookup | — |
| Tail (no chord at the entry edge; board edge) | local | learned | — |
| Loop closing vs joining your own loose end | regional (needs "this is my start") | an identity along the line: the tap seeds a random vector the line copies along (MNIST-style agreement on a 1-D manifold); closed = the chain is a cycle (M1b) | — |
| Edge-to-edge claim, smaller side by area | global geometry | the flood model (M0) with lines as walls; per-claim latch (M2b) | — |
| Inside a circuit (captures, tap refusal) | regional | the flood model; `tilesEnclosed` is the ready-made oracle | — |
| Collision (rival chord conflicts on the entered tile) | local detect, 1-D regional effect | detect from per-owner edge planes; the wipe is a death wave along both lines (un-flooding, which the damage pool already trains) | — |
| Capture / convert (owner relabel of a wholly-inside line) | regional + along the line | owner change propagates along the line inside the flood (M3b) | the referee could do it from the CA's territory channel |
| Flip / burn / sprout between a player's own patterns (waves = tap order) | local + bookkeeping | — | outside; one pattern per player in M2/M3 |
| Join / foldInto | local | set semantics: two chains on the same chord are one chain | ids and points stay outside |
| Head limit, respawn delay, tap throttle, tap refusal | bookkeeping | — | the referee; it writes the tap plane only when allowed |
| Speed (5.5 + 0.0075 × score tiles/s) | global per player | a per-owner "go" pulse plane: heads advance only on steps where it is 1 (the host seeds it; a fire-rate per owner) | the host computes the rate |
| Score (distinct tiles your lines are on) | global count | — | a reduction over the per-owner edge planes |
| Rule change / regrow (`planRegrow`) | global | — | outside: the host rewrites the chord planes and re-seeds |
| Determinism | — | the server runs the CA; clients draw. Float NCAs are not bit-reproducible across devices (IsoNCA broke symmetry on float non-associativity alone) | matches Spectacle's settled "server is authoritative" |

Timing (inference): a line step per CA step and fills that must settle between line steps means
K ≈ board diameter CA steps per line step (~40 at level 3, ~100 at level 4) unless the "go" pulse
slows the lines; at Spectacle's 5.5 tiles/s that is 200-500 CA steps per second per room.

## 6. Architecture and recipe per milestone

Inherits `nca/model.py`'s `HexNCA` (masked 3×3 = 7 hex taps, ReLU hidden, zero-init residual update,
clamp) and `nca/train.py`'s pool with damage, random rollout length, min-over-targets loss, gradient
normalisation, quick checks and the rollback guard. What changes:

| | M1a / M1b | M2 | M3 | M4 |
|---|---|---|---|---|
| Per-board input planes (like `walls` today) | chords 15, tap 6, mask 1 (= 22) | + owner "go" 1; tap pulse 6 | owner slots on the tap (6 × 4), referee 2 | per-edge attributes in the message function |
| State | 24: 6 edge outputs (+1 closed) + hidden | 6 × owners + territory × owners + hidden ≈ 32-40 | same | same per node |
| Hidden width | 256 to start; **ablate 64 / 128 / 256** (the browser needs the answer) | the width M1 picked | same | 256 messages (GNCA default) |
| Loss | MSE on the 6 edge planes (+ the closed plane), last 8 steps; L2 to {0,1} is the MNIST lesson for a vote channel | + territory MSE per owner; min over acceptable targets for ties | + collision outcomes | as M1 |
| Pool & damage | pool per patch size; damage: zero the state in a disc (regrow), move the tap to another chord (the old strand must vanish), flip a few tiles' chord bits (rule edit → re-route), new board; the worst sample restarts | pool of engine episodes; damage: a new tap, a cut line, a wall of chords | + rival taps | as M1 |
| Rollouts | T ~ U[3, 6] × diameter with persistent pool states (launch 3's lesson: short rollouts, settling spans visits) | same | same | same |
| Readout / eval | 8 × diameter; exact strand; closed agreement; steps-to-complete vs strand length | per engine step: K CA steps then compare | same | same |
| Init | from scratch (new input layout); 51k params at 256 | fill channels from the flood model via a zero-padded widening loader (review item 4); area-only targets | from M2 | from scratch |
| Curriculum | level-2 patches (≤ 71 tiles) → level 3 → crops of level 4 | level 2 → 3 | level 2 | hex graph → Spectre level 2 → 3 |
| Fire rate | 1.0 (synchronous) for exactness; a seeded-mask A/B once M1a works | the owner "go" pulse plays the fire-rate role | same | same |

Expected difficulty (inference): M1a is a chain flood — easier than region fill with bridges; tens of
minutes on the L4. M1b is a consensus task whose cost grows with strand length (a level-3 FASS strand
is 496 tiles: the closed flag needs ≥ 250 steps of propagation — the persistent pool is what makes
that trainable in 3-6 × diameter chunks). M2's territory is the known-hard part and is already solved
to R 10; its per-claim latch (M2b) is new.

## 7. The data pipeline

1. **Node exporter** (Spectacle, read-only use of `buildField`, `chordTableFor`, `walkStrand`,
   `validEdgeSubsets`, `nonCrossingForTile`): for a family/level/root, per tile: axial (a, b), type,
   rotation (from `xforms`), the 6 neighbours (`acrossEdge`) — plus, per clean rule, the chord table as
   edge pairs per type, and 2,000 random (rule, tile, chord) walks with their full tile sequence and
   `closed`. Written as JSON + flat binary (`.npz` via a tiny writer, or JSON → `np.savez` on the Pi).
2. **Python walker** (`nca/strands.py`, ~40 lines): the axial lattice, edge k ↔ neighbour, chord lookup
   from the 15-bit rendering; walks a strand both ways; computes targets for any (board, rule, tap) on
   the fly, so the pool needs no Node in the loop. **Parity test**: the 2,000 exported walks match tile
   for tile; the lattice neighbour of every tile and edge matches `acrossEdge` (this is the same
   pattern as `tests/nca.test.ts`'s parity fixture, in the other direction).
3. **Trainer generalisation**: per-board input planes beyond `walls` (a list of named planes), a mask
   per board (today it is per radius), targets from the walker, new quick-check sets (held-out rules,
   held-out patches, the FASS rule alone), the width ablation via plan lines.
4. **M2+** keeps the engine as the oracle: a Node script plays scripted or bot games and writes per-tick
   snapshots; the trainer samples (state, target-after-settling) pairs into the pool. Porting the
   engine to Python is not worth it.

## 8. Compute plan: the next 24 h

Now ≈ 12:35 UTC 2026-10-05; the owner's window ends ≈ 12:00 UTC 10-06. Spot L4 ≈ $0.41/h; GPU quota
1; `MAX_HOURS` 12 per launch, ledger cap 45 VM-hours (≈ $18.5). Spent so far: 0.93 VM-h (launches 1-2)
+ launch 3 (1.2 h at writing, ≤ 5.5 h). Money is not binding — the clock and the one GPU are.

| When (UTC) | What | GPU-h | $ | Go / no-go |
|---|---|---|---|---|
| now → ~13:30 | launch 3 runs on. Both plateaued at 12:13 and will not reach their lr decay. Owner's call: `down.sh` once a further hour shows no gain (watch the per-radius bridge rows, not the mean). `hb-big/best.pt` (iteration ≥ 39k, quick score 0.982) is the new base | ≤ 1 | ≤ 0.4 | — |
| ~14:00 → ~17:30 | **launch 4, flood queue** (review §6): slot A R 6-16 curriculum with ragged masks (if the per-board-mask refactor lands by 14:00; else masks go into launch 5's plan), then R 8-24; slot B the near-tie arm, then clamp/overflow and fire-rate A/Bs; `evaluate` n = 200 incl. a ragged set at the end | 3.5 | 1.5 | at +90 min: R 16 bridge quick ≥ 0.9 and R 8 holds → go to R 24; ragged bridges ≥ 0.7 → masks are "just data"; else stop and schedule the width ablation |
| 14:00 → ~21:00 (Pi, agents) | M1 engineering: exporter, walker + parity test, trainer planes, eval sets; `selftest.sh` green; push | 0 | 0 | parity fixture 2,000/2,000; a 10-minute Pi smoke run on level-2 patches learns something (strand IoU rising) |
| ~21:30 → ~08:00 10-06 | **launch 5, M1**: three runs, hidden 64 / 128 / 256, R-mix = level-2 patches then level 3 then level-4 crops; stage 2 = M1b (closed flag) from each run's best | ≤ 10 | ≤ 4.1 | at +60 min: held-out-rule exact strand ≥ 0.8 and rising in at least one width; at +180 min ≥ 0.95 at level 3 → start M1b; if 64 ties 256, the browser question is answered |
| ~08:30 → 12:00 | `down.sh`, evaluate, write up; dashboards already public | 0 | 0 | — |
| **Total** | | **≈ 15-20 VM-h** | **≈ $8-9** | leaves ~$10 for a relaunch after a preemption or a second M1 night |

Machine type: keep `g2-standard-4`. The GPU was at 65 % with two runs because the Python board work
sits on the training thread; `g2-standard-8` (same L4, ~20 % more per hour — estimate, check) only
pays if that work is parallelised or Node generation runs on the VM. Do the software fix (review
item 5) first; pre-generate M1 data on the Pi.

What the owner should see at 12:00 UTC on 10-06: (a) a flood checkpoint with per-bin tables at
R 6-24, ≥ 0.9 at R 16 and ≥ 0.8 at R 24, or the clean negative (near ties do not move with data and
curriculum, so capacity is next) — plus ragged-mask numbers; (b) an M1a model that traces strands on
Spectacle's level-3 hex patch for rules it never saw, with the width that suffices known; (c) M1b
started, with a steps-vs-strand-length curve; (d) the ledger under $10. Not in the 24 h: M2 and
beyond, any browser GPU port, Spectre tiles.

## 9. Risks and unknowns, with the cheapest experiment for each

| Risk / unknown | Cheapest experiment | Cost |
|---|---|---|
| Near-tie precision does not improve with curriculum and near-tie data | launch 4 as planned; then the width ablation (channels and hidden separately) | 3.5 GPU-h + 3 × 1 h |
| The flood model fails on ragged masks — **confirmed** (review §2.4: bridges 0.21, none 0.79 on the level-3 patch) | masks per board in the pool (crops, supertile patches, concave blobs) in launch 4's slot A; a ragged held-out set; if ragged bridges stay < 0.7 after 90 min while disc numbers hold, it is a model question (receptive field at concave corners) | 2 h eng; 0 extra GPU |
| The owner rejects rendered chords as "engineering" | ask before launch 5; the control arm (type 9 + rotation 6 + rule vector: subset 9 + per-type non-crossing one-hot ≤ 45) costs one more run | 1 GPU-h |
| Long strands (FASS, 496 tiles at level 3) need long settling | M1a on rule `128` alone: steps-to-complete vs strand length (expect ≈ length/2 + c); if it is much more, the chain flood is slower than 1 tile/step and the pool must carry settling across visits | 20 min CPU after M1a |
| Closed-flag consensus on long loops is slow or flickers | M1b on level-2 patches first (loops ≤ 63 tiles); read agreement in a window (MNIST: it peaks then erodes); small update noise σ 0.02 if it flickers | in launch 5 |
| Class-0 junctions (3 tiles meet at a dot; engine policy 'random') | exclude class 0 in M1; later run the engine with `junctionPolicy: 'stop'` or feed the host's random pick as an input plane | 0 |
| Server/client disagreement of a float NCA | roll `src/nca.ts` and PyTorch 1,000 steps on 20 boards, count thresholded differences (the parity fixture only pins one step at 1e-3); the design answer is "server runs it, clients draw" | 20 min |
| Browser cost at game size (~0.7 s/step in JS at level 4) | the width ablation; then a WebGL2/WebGPU step kernel (the hand CA's `DESIGN.md` MRT layout is a template) | 1-2 owner-days |
| Collisions / death waves (M3) are not learnable in this form | a toy: level-2 patch, 2 scripted players, collisions only | 1 GPU-h |
| Graph CA on Spectre tiles | build the GNCA on the hex graph and match M1's numbers before touching Spectre | 1 day eng + 2 GPU-h |
| Nested-claim semantics gap (§2) | count, in bot games with the engine, claims whose polygon contains another claim's line | 10 min |
| Rule-space coverage: only 100 rules without class 0 | hold out matchings, not subsets; report per-subset; add class-0 subsets with 'stop' as M1c | 0 |

## 10. Deliberately left for later

Multiple patterns per player (flips, burns, sprouts, waves), rule changes and regrow, scores and
speeds inside the CA (they stay with the referee), Spectre tiles (M4), the wire format for a CA
state, a browser GPU port, and anything about the full 242k-tile level-6 field. Each is a milestone of
its own and none is needed to show that the patterns can be recreated by a learned local rule.
