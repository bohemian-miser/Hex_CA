# Hex_CA

Experimenting with cellular automata as the foundation for the Spectacle
game hexagon.rodeo.

**Live demo:** https://bohemian-miser.github.io/Hex_CA/ — drag to draw a
line, right-drag to erase, paint walls, close a loop or bridge and watch the
inside fill.

## Inside, as a cellular automaton

A **field** is a 6-connected set of hexagon cells (a hexagon of radius R by
default, or any ragged subset). Every other cell is **dead**: never
updated, always read as a fixed value. On the field you paint a **picture**:
`OFF` (empty), `ON` (a line) or `WALL`. The rule turns that into a **state**
for every cell: `ON`/`WALL` as painted, or `FILLED`/`OFF` for the rest.

    fill(region) ⇔ !edge(region) || size(region) < M

- **A region** is a 6-connected group of `OFF` cells.
- **Exterior** is every dead cell, plus any `WALL` cell 6-connected to a dead
  cell through other `WALL` cells — the **rim**. A wall drawn in from the
  edge of the field carries the outside in with it; a wall island sitting
  alone in the middle does not.
- **edge(region)** holds when some cell of the region touches the exterior.
  `M` is the largest size among regions that do.
- **A loop:** walled off from the exterior (`edge` false) — always fills,
  whatever its size, however nested or however much of the field it
  encloses.
- **A bridge:** a line from edge to edge of the field splits the exterior
  into two regions; the smaller one fills.
- **The tie rule:** every edge region of exactly size `M` stays `OFF` — a
  bridge that halves the field fills neither side. A region smaller than `M`
  fills even when another region ties for the largest.

The fill depends only on the picture there is now, never on the order it was
painted in, and painting is piecewise: add or erase any cells, in any order,
a few at a time, whenever you like.

### The rule

`src/engine.ts` is the generic engine: named integer channels in a few
kinds, a radius-1 kernel (self plus six neighbours, any order), one `update`
called for every field cell. It knows nothing about fills or walls.
`src/fill.ts` is the rule: 22 channels across five kinds.

| kind | channels | what |
|---|---|---|
| const | `field id gpar area` | field membership; a random id 1…N; the static-tree parent's id; size weight (1) |
| input | `paint stamp` | the picture; the edit's serial |
| hidden | `epoch rim redge leader dist rpar cdone sub rsize` | the watched system below |
| gate | `gmax M qt run commit` | the certificate |
| out | `want state` | the unlatched answer; what is drawn |

The update, in prose, per field cell per step:

1. **Epoch reset.** `epoch` is a max-flood of itself and every neighbour's
   `stamp`/`epoch`. A cell whose epoch just rose is **fresh**; taps still on
   an older epoch are invisible to it (not refuted — just not read). This is
   the whole "reset": there is no explicit clearing step, so an edit mid-way
   through a settle is simply absorbed into the wave.
2. **Rim and region flood.** `rim` floods through `WALL` cells from the dead
   ring. On `OFF` cells, `redge` floods from any tap that is `rim` (or, with
   `wallsBound`, any `WALL` tap at all) or already `redge` — "this region
   touches the exterior."
3. **Leader flood and size convergecast.** Each `OFF` cell keeps the
   largest `id` reachable through same-epoch `OFF` taps as its region's
   `leader`, with a BFS `dist` to it and a parent `rpar` one step closer
   (ties broken by id, so the flood is tap-order-free). Once a cell's
   `leader`/`dist`/`rpar` stop changing (`settled`) and all its tree
   children report `cdone`, it sums its own weight and theirs into `sub`;
   `rsize` is that sum read back down from the parent, so the whole region
   agrees on one size.
4. **Static-tree gate.** `gpar` is a *second*, fixed spanning tree over the
   field (not the per-region one) used only to certify quiet. A cell is
   `qt` once nothing watched has changed and every static child is `qt`
   too; the root's `run` counts consecutive quiet steps and **fires** at
   `run == K + 1` (`K` = the field's radius in the static tree) — by then
   every watched channel has been at its true fixed point, everywhere, for
   at least `K` steps straight. On fire the root reads a lagged, unwatched
   maximum of edge-region sizes up its static tree as `M`, and commits
   `commit = epoch`. Every other cell inherits `commit`/`M` from its static
   parent, so both reach a cell exactly `depth(cell)` steps after the root.
5. **Latch.** A cell shows `FILLED` only while `commit == epoch`: its own
   fixed-point `rsize`/`redge` compared against the certified `M`. A cell
   whose epoch a newer edit has already moved past keeps showing its last
   latch instead of a half-settled one.

### Settling, the bound, and the no-overshoot invariant

Because each channel is the unique fixed point of its formula given the
channels above it (§5 of `DESIGN.md` spells out the induction), the whole
stack converges from **any** starting state — "Scramble" in the demo fills
every hidden/gate/out channel with garbage to show this; "Bump" re-stamps
the root with a fresh epoch and the same picture, which is what actually
makes recovery certain (garbage that shares the current epoch can outlive
it: `tests/garbage.test.ts` pins that a bump always recovers, and that a
board left unbumped recovers only most of the time).

A cell's commit is bounded by the epoch wave, the region's own radius, the
wall flood (if any), and the static-tree depth and gate margin:

    T_commit(c) ≤ t₀ + Dw + max(3·Er, Wd + 2·Er) + 2K + depth(c) + 3

which on a hexagon works out to about `11R + 3` steps after the last edit —
roughly 5.5× the field's diameter, down from an earlier design's ~127·R
(see "What changed" below). `settleBound()` in `src/oracle.ts` computes this
exactly from a field and the edited cells, and every test that waits for a
settle checks the real commit step against it rather than against a fixed
step count.

Measured on a Raspberry Pi 5 (`npm run bench` at R = 20, 40, 80; the
R = 6…40 series from the review's scaling runs, with straight rows and
regular loops):

- **Boot** (empty field, nothing painted) to full quiet: 49, 94, 115, 172,
  250, 349 steps at R = 6, 10, 14, 20, 28, 40, and 676 at R = 80 — the boot
  is the *tightest* case of the bound (`Dw = 0`), so it lands within a step
  or two of it.
- **A closed loop or bridge**, painted all but its last cell then closed, to
  every cell's commit: 48–61, 90–103, 109–133, 172–194, 267–281, 352–413
  steps at R = 6…40 — about 10·R. Random, wigglier shapes make longer,
  thinner regions and take longer: in `npm run bench` the root commits the
  closing cell at 168–186 steps (R = 20), 307–368 (R = 40) and 609–675
  (R = 80), each 10–25 % inside its own picture's `settleBound`; the bound is
  per picture, not a fixed multiple of R.
- **Per-step cost**: 0.08–1.13 ms raw per step while settling (R = 6…40, on
  the active set, not the whole board) — about 1.4 µs per cell computed, 3–4
  µs per cell that actually changed; a settled board costs well under 1 µs a
  step (`active == 0`, nothing to do).

**No-overshoot invariant**, checked every step of every fuzz/overshoot test,
not just at quiet: a cell shown `FILLED` is always the oracle's answer for
*some* picture that genuinely existed, and a given cell flips `FILLED` at
most once per edit. The cost of that guarantee: after an erase, the old
fill stays exactly as it was until the next commit reaches that cell — it
doesn't flicker through a wrong intermediate state, but it also doesn't
disappear instantly.

### Host contract

An edit's `stamp` must be `generation + 1` — the number of the step that
will absorb it — so that every change between two steps shares one serial
and serials strictly increase across steps. `paint()` enforces this; nothing
else is sound (reusing a serial across a whole stroke lets a cell already on
that epoch read new paint without going fresh, and settling degrades to
roughly N steps — measured about 7× slower at R = 20). The same rule is
meant to carry forward: anything that edits the board from inside the CA
later (§10 of `DESIGN.md`, line growth) stamps with `u.step`, not a value of
its own choosing.

### The active-set engine

`CA.step()` only recomputes the slots on its active list — every slot some
channel of which changed last step, plus each slot's own field taps — and
every read in a step sees the *pre-step* state (writes are buffered and
applied after the pass), so it is bit-identical to `stepFull()`, which
recomputes the whole field every time and exists as the reference sweep.
`write()` adds a slot and its taps to the active list whether or not the
value actually changes. A settled board has an empty active list and `step`
costs nothing; `tests/engine.test.ts` runs both sweeps, and a third copy
with a randomised per-slot tap order, in lockstep against the same edit
stream and checks every channel agrees at every step.

### Run it

```bash
npm install
npm test          # vitest: oracle pictures, the CA against the oracle, fuzz streams with a
                  # per-step latch check, overshoot, engine lockstep twins, scramble+bump
                  # recovery, rotation (FUZZ_SEEDS=200 for CI's longer run)
npm run typecheck
npm run build     # → dist/index.html, the demo (open it directly)
npm run bench     # settle steps / ms / µs-per-update at R = 20, 40, 80 (not a test)
```

The reference (`src/oracle.ts`) computes §4's predicate directly over the
whole field — the regions, which touch the exterior, the tie at `M` — with
no CA involved, and `settleBound()` computes the bound above by BFS. Both
are the thing the CA's fuzz tests check against on every step, not just at
the end.

Every push to `main` runs the typecheck, tests and build, and deploys the
demo to GitHub Pages. Pull requests run the same checks without deploying.

### Layout

```
src/hex.ts      hexagon board, axial coords, direction-indexed neighbours
src/engine.ts   the generic channel CA: ChannelSpec/Rule/Topology, active set, stepFull
src/field.ts    Field: masks, the static tree, ids, wall presets (hexagon/blob/lobes/ring)
src/fill.ts     the rule: channels, update, paint/bump/scramble, stateOf
src/oracle.ts   the reference predicate and settleBound
src/lines.ts    random pictures for tests and the demo (loops, bridges, scribbles, walls)
tests/          vitest: oracle, fill, fuzz, overshoot, engine, field, garbage, rotate
src/strand.ts   Spectacle's strand rules on its hex fields: rule table, split, rendering, walker
src/strand-nca.ts  the trained strand NCAs (FrameNCA, StrandNCA, v1) in TypeScript
src/draw.ts     hex drawing and pointer maths the pages share (fit, cell under a point, chord ranking, tiles)
src/game/       the hybrid game (docs/spectacle-ca-hybrid.md): host.ts, the referee; line-ca.ts, the line CA; area.ts,
                the area layer (one flood per player, cpuArea); area-gl.ts, the same on WebGL2 (glArea)
web/            the demo page (page.html + main.ts), the trained NCA's page (nca.html + nca.ts), the strand
                page (strand.html + strand.ts) and the game (game.html + game.ts), bundled by scripts/build-web.ts
scripts/bench.ts  settle steps / ms / µs-per-update at three field sizes
scripts/area-probe.ts  the flood on strand walls (loops, rim-to-rim claims) against the oracle, levels 2-4 (--gl: on WebGL2)
scripts/area-gl-parity.ts  glArea against cpuArea in headless Chromium, and both layers' ms per step
.github/        CI on pull requests; build and deploy to Pages from main
```

### The demo

Draw with the **Line** tool (left-drag), **Erase** (right-drag, clears both
`ON` and `WALL`) or **Wall**; keys `1`/`2`/`3` switch tools. **Random loop**
drops a closed line for a quick look. **Radius** (4–60) and **Preset**
(hexagon / blob / lobes / ring, as wall pictures) rebuild the field;
**Clear**, **Scramble** and **Bump** act on it as described above.
**Play/Pause** (space), **Step** and **Step to commit** control playback at
the **Steps/s** rate. **View** switches the board between the drawn state,
a want-vs-state diff, the epoch wave, or any single channel as a heat map.
**Grid** adds a small map of every channel under the board, grouped by kind
and each on its own live range; click one to put it on the board.
**Spectrum** gives each channel of the ticked groups (const is off at first)
its own hue and blends them per cell, weighted by each one's value on its
range, with lines and walls drawn as in the State view.
The **readout** shows generation, active/changed cells, FPS and steps/s,
the root's quiet run against `K`, commit latency since the last edit, the
filled count, and an oracle ✓/✗ once the board is quiet.

### Known limits

- **The gate is global.** Any edit anywhere restarts the quiet count for the
  whole field, so nothing fills while anything is still being drawn — the
  bound counts from the *last* edit, not the first. At R = 20 a loop alone
  fills in 162 steps; the same loop with a distant cell painted every 120
  steps takes 3789. Fine for painting by hand; a game with lines growing
  continuously needs the gate to work per-region instead (see "Open
  questions").
- **Erasing doesn't un-fill instantly.** A cell that was correctly `FILLED`
  stays `FILLED` until the next commit reaches it, even once the picture
  that justified it is gone — the no-overshoot invariant only promises no
  cell is ever shown filled for a picture that *never* existed, not that
  the display is live.
- **Spiral regions are the worst case.** A region whose leader is reached
  by a long, narrow path has `Er ≈ size / width`, which the bound (and real
  settle time) scales with directly — any flood-based scheme pays this.

### What changed from the earlier, min-plus-border design

The previous rule (kept in `DESIGN.md` §11 as the record of it, no longer
in `src/`) numbered the hexagon's border starting from one fixed corner and
had every region find its "openness" by flooding a label outward and then
counting that label's open border cells up a spanning tree; a region cut off
from its label source only discovered this by counting all the way up to N,
so one closure cost on the order of N steps at N work apiece, while most of
the board sat transiently `FILLED` in the meantime, and the border numbering
only made sense on an unbroken hexagon edge. This design replaces that count
with an **epoch**: a stale value is simply invisible rather than something
that has to be refuted, which turns an O(N) death into an O(diameter) one;
it names a region by the **largest id** in it instead of a border position,
so masked, ragged or irregular fields need no border walk at all; it
compares each region's size against the field's actual maximum through a
convergecast and a certificate on a fixed static tree, rather than counting
border cells specifically, so a wall and a line are "one thing" (anything
that blocks); and what's shown is **latched** behind that certificate, so a
half-settled board is never drawn as if it were finished.

### Open questions and next steps

- **Per-region gates.** The gate is global because quiet is checked once,
  at the root. A loop's fill doesn't actually depend on the rest of the
  field (its `redge` is 0 regardless of `M`), so only bridges truly need a
  field-wide maximum — splitting the gate so loops certify locally, while
  bridges still wait on the shared one, would let fills keep happening
  while an unrelated part of the field is still being edited. This needs a
  watched, *stable* per-region height channel to drive a local quiet count,
  which is a design of its own (the earlier design's boot problem came from
  almost exactly this).
- **The game on top.** Spectacle's rules are per line, not per wall: rival
  lines are captured rather than simply blocking, and each line claims only
  its own shorter side rather than one shared fill. `DESIGN.md` §10 sketches
  owners and patterns as extra input/region channels; building that out is
  the next real step once this base — the fill rule alone, under fuzzing —
  is trusted.

## Trained NCA

`nca/` and `web/nca.html` hold a second, learned answer to the fill rule
above, bridges included: a neural cellular automaton, distill.pub's [Growing
Neural Cellular Automata](https://distill.pub/2020/growing-ca/) style, and as
shipped now, nothing hand-written in it. Channel 0 is the wall input
(re-imposed every step), channel 1 the fill output (>0.5), channels 2-23
hidden, with one constant input, the board mask. Each step every cell runs
one learned 7-tap hex convolution over those 24 channels plus the mask, ReLU
into 256 hidden units, a zero-init linear layer added back into the state:
64,024 parameters, every one trained. Target: every enclosed region fills,
and a rim-to-rim wall fills every rim region but the largest, by area or by
the angular span of its rim contact, either acceptable when they disagree.

The page's **Board** selector swaps the hexagon (the Radius slider) for one
of Spectacle's own hex fields — the game's actual ragged tile patches at
levels 2, 3 and 4 (`nca/fields/*.json`, also `?map=l2`/`l3`/`l4` in the
URL) — with no change to the model or its inputs: the field's own missing
neighbours already read as "off board" to the trained rim perception.
Level 4 (3905 tiles) runs well under real time in plain JS (a few steps a
second, noted on the page); level 2 and 3 stay smooth.

The plain model got loops right quickly, but short attempts (hundreds of
iterations, `runs/experiments.md`) never learned the bridge half, so a
hybrid shipped first (PR #7): five hidden channels hand-written as exact
rim-angle floods, frozen, with only the readout trained on top. It worked,
but it was hand-crafted — exactly the shortcut the bitter lesson warns
against — so the owner's call was to go back and make the plain model learn
it properly. The fix wasn't a smarter architecture, just more honest
training: a persistent sample pool that keeps training on *edited*, not
just fresh, boards, run for tens of thousands of iterations on a GPU (~19
iterations/s) instead of a few hundred on a Pi. On held-out bridge boards at
R 8 it now gets 0.89 exact against the hybrid's 0.75.

`nca/train.py`'s default is that plain recipe: a persistent pool of boards
per radius, each kept in the state it was left in; most of a batch takes one
piece of accumulating damage (a wall edit, a toggle burst, an erased disc, a
stamped loop/bridge, or distill's own state-zeroing damage), a few restart
fresh, a few are brand new. Loss is the fill MSE against the *best* of a
board's acceptable targets, min not mean. A held-out quick check every 200
iterations includes the exact three-edits-in-a-row sequence the page does
with no reset between; `best.pt` keeps only the run's highest score.

Two failures on the way, fixed in the trainer, not the model:

- **Collapse at lr 2e-3**: two GPU runs peaked by iteration 2000-4000 then
  collapsed (score 0.86 → 0.24) as the weights grew until most hidden
  channels sat at the state clamp, zero gradient. Fixed with a lower
  default rate (5e-4), a decay schedule actually reached within a run, and
  a rollback guard: a collapsed or sharply worse check reloads `best.pt`,
  halves the lr scale, and starts a fresh optimiser with its own warm-up.
- **Larger boards at 3e-4 made it worse**: continuing at R 6, 8, 10 dropped
  bridge-exact from 0.88 to 0.33 in 2,500 iterations. Still running now at
  5e-5 instead (rollback tightened to a 15% drop) — not done as of writing.

### Honest numbers (shipped checkpoint, `runs/fb-r6816nt/best.pt`)

Launch 4 (see `docs/nca-journey.md`): launch 3's best, fine-tuned for 85 minutes on one L4 at R 6-16
with half the pool on ragged board outlines (random blobs and crops of Spectacle's hex fields) and
extra near-even bridges. `python -m nca.evaluate CKPT --radii 8 16 24 --n 100 --n-big 50 --mults 8 16`,
held-out boards read out at 16R steps, exact = every cell right against any acceptable target
(±2 standard errors), trivial = always-empty, both/none = share of bridge boards left
every-side-filled / two or more sides empty:

| R (n) | mix (trivial) | page loops | bridge | ragged bridge | bridge both / none |
|---|---|---|---|---|---|
| 8 (100) | 0.99 ±0.02 (0.34) | 1.00 | 0.99 ±0.02 | 0.98 ±0.03 | 0.01 / 0.00 |
| 16 (100) | 0.95 ±0.04 (0.30) | 0.99 | 0.98 ±0.03 | 0.91 ±0.06 | 0.00 / 0.00 |
| 24 (50) | 0.94 ±0.07 (0.22) | 1.00 | 0.76 ±0.12 | 0.88 ±0.09 | 0.02 / 0.10 |

R 24 is the open problem: every miss there is a near-even bridge (area ratio above 0.5), and a
further 100-minute stage at R 8-24 did not move it. The previous shipped checkpoint (trained at
R 4-6 only) scored 0.89 / 0.40 on bridges at R 8 / 16 and 0.21 on Spectacle's level-3 field outline.
The edit test (settle, one live wall edit, no reset, exact against the edited targets) at R 8:
mix 0.96, bridge 0.92, page loops 1.00. For the previous checkpoint, confirmed by hand too: driving `dist/nca.html`
headless (Playwright) past ~10R steps, seven live edits at R=6 (a loop drawn
cell by cell while running, opened, closed, a second loop, a rim-to-rim
wall, broken, restored) all matched with no reset; 30 checks of alternating
bridge/loop plus 3-cell erase/add matched 30/30 at R 8 and R 10, 28/30 at
R 6 (two small, short-lived over-fills, gone by the next edit; zero "both"
or "none"); a settled bridge held exactly 1500 steps later. ms/step: 8.5 /
14.3 / 28.6 at R 6 / 8 / 12, page responsive throughout, no console errors.
`src/nca.ts`, a float32 re-implementation, is pinned to the exported
weights by a parity fixture (`tests/nca.test.ts`, 1e-3).

### Running it

`python -m nca.train` runs the recipe above (every flag defaults; needs
`torch` + `numpy`); `python -m nca.progress RUN_DIR` gives one verdict from
a run's log in ~30s; `python -m nca.dashboard [--demo]` serves a local,
read-only page over `runs/`: progress plus the live pool as small
multiples, with a Play link into `dist/nca.html` against that run's own
weights, live; `python -m nca.evaluate CKPT` makes the table above;
`python -m nca.export CKPT` writes `web/nca-weights.json` and the parity
fixture. `nca/cloud/` runs the same training on one time-boxed, ledgered
Spot GPU VM (`nca/cloud/README.md`); a public dashboard tracks it live:
https://storage.googleapis.com/recipe-lanes-staging-hexca-runs/dash/index.html

The hybrid's code paths (`--floods`, `--aux`, `--teach`, pooled-max/min
perception) still exist behind flags — `nca/train.py`'s docstring has the
exact v4/v5/v6 recipes — but nothing shipped uses them now.

## Spectacle as a CA: the game

**Play:** https://bohemian-miser.github.io/Hex_CA/game.html. The design is `docs/spectacle-ca-hybrid.md`: the
lines are a hand-written local CA (every edge, chord, collision and wipe decided by a tile from itself and its six
neighbours), and each player's area is the trained flood fill with that player's lines as its walls. Players sit
at one screen: select one (keys 1-8) and tap for them; each has a rule of their own, sticky across visits. Score =
the tiles your lines are on + the tiles your flood fills for you alone. Knobs: growth until a line stops or a few
chords per tap (`fuel`), one-way taps, speed, flood steps per line step; boards are Spectacle's hex levels 2-4
(the flood is ~30-50 ms a step per player at level 3 on a Pi's CPU, so areas fill in over seconds there).
`?map=l2|l3|l4`, `?rules=<rule>,<rule>` (one per seat).

**Parity** (`docs/spectacle-ca-hybrid.md` §5.1). The line CA is checked in two ways. First, against Spectacle's own
engine on 400 scripted collision games: they agree at rest on 327. Second, against the strand simulator's CA mode
(`nca/strand/sim.py`). Every difference is classified, and none is unexplained:
- Most come from the simulator's own wave and head-timing races, which the CA shares.
- The rest come from the CA's own timing. With that timing added, the simulator matches the CA chord for chord,
  every step, on thousands of games.

```bash
npx tsx scripts/game-parity.ts       # LineCA on data/strand-v2/collide.json -> ca-collide.json, against the engine
python -m nca.strand.ca_parity       # (b) against sim.py's CA mode, (c) against the engine, every difference classed
```

## Strand NCA: Spectacle's lines, grown by a network

**Play:** https://bohemian-miser.github.io/Hex_CA/strand.html (e.g.
[`?rule=128·010100000`](https://bohemian-miser.github.io/Hex_CA/strand.html?rule=128%C2%B7010100000&map=l3),
the infinite-line rule). Pick one of Spectacle's hex fields (levels 2, 3, 4)
and a rule — a random held-out one (never seen in training), a random
training one, one typed in `describeRule` form, or a preset — and tap tiles.
A strand NCA from `nca/strand/train2.py` grows the strand of every tap from
the tapped chord; the true strand (Spectacle's rule and walker, in
`src/strand.ts`) is drawn thinly underneath, with an "exact ✓" badge per tap.
Taps share one board by default, so strands of different rules can run into
each other — never trained, just shown. Each tap's input is training's: the
rule's 53-bit code and the tapped chord's two edges on the tapped tile, held
every step (train2's models; the next ones take the tap once, below).

**When two patterns meet** (`npx tsx scripts/strand-meet.ts WEIGHTS`: the page's runner on a shared board, `tap-e2-l3` and `tap-e2-w-l3`:
pairs of taps of training rules whose strands share tiles, each strand drawn exactly when alone; 25 pairs on
levels 2 and 3). The game wants the line that is hit to disappear; these models do roughly the reverse:
- *A line already drawn, then a second tap whose strand runs into it:* the first line stays, losing 10 of its
  379 chords, most of them on the shared tiles. The newcomer stops within a chord of the first tile it shares
  with the old line on 32 of the 38 ways that reach it. It runs on only where the two strands merely cross in
  one tile.
- *Both taps from step 0:* both strands usually lose a few chords where they meet. Both are exact on 2 of 25
  pairs.
- The `-bc` and v1 models carry one rule on every tile, so their taps can't share a board.

**The tap as an event** (`nca/strand/train3.py`, `docs/spectacle-nca-taps.md`, launch 10). The next models take
a tap once, as the game does: its inputs on for one step (impulse), or a write of the tapped tile's state (fixed).
After that the line has to keep itself alive and grow a chord every two steps. When two patterns meet, both lines
have to go (Spectacle's rule: the pattern owns the tile). The targets come from `nca/strand/sim.py`. Such weights
say so (`tap` in the file, version 2). On the page a tap is fired into the running board and refused where
Spectacle would refuse it. Taking a tap away starts the board over, with the others fired again at step 0. Older
weights hold their taps as above. `strand-meet.ts` also counts the pairs where both lines end up gone, the game's
answer: `tap-e2-l3` (level 2) gets 0 of 8.

`python -m nca.strand.export CKPT` writes a strand checkpoint (FrameNCA at any
depth, the plain-conv StrandNCA, v1) as `web/strand-weights.json`'s format
(float32, exact; the run's held-out numbers from its `log.jsonl`); `--board-data`
writes `web/strand-data.json` (the rule table and the three Delta patches,
from `data/strand-v2`; each board cell carries its `geo` and Spectacle's `tile`
index), `--fixtures` the parity fixtures. `src/strand-nca.ts`
matches PyTorch to ~2e-6 over 8 steps (`tests/strand.test.ts`: option E at
depth 1 and 2, E-bc, C, A, v1, two and three taps on one board). The page's
default weights are `tap-e2-l3`'s best (option E, depth 2; held-out exact
0.31). The dashboard's ▶ Play opens a strand run on this page with its own
weights. A step runs the network on every tile on the main thread:
`npx tsx scripts/strand-bench.ts` — on a Raspberry Pi 5, ~8 ms at level 2,
~60 ms at level 3, ~460 ms at level 4 (3,905 tiles), so level 4 grows slowly.

### Licence

AGPL-3.0-only — see `LICENSE`.
