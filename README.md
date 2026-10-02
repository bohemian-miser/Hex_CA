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
web/            the demo page (page.html + main.ts), bundled by scripts/build-web.ts
scripts/bench.ts  settle steps / ms / µs-per-update at three field sizes
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

### Licence

AGPL-3.0-only — see `LICENSE`.
