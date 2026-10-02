# Hex_CA fill rule — the specification

One uniform, synchronous, deterministic radius-1 rule with 22 named integer channels: the inside of every painted loop fills, as does the smaller side of every edge-to-edge bridge. It is Design C (prototyped, judged correct) with two grafts from Design A: the global maximum is a *lagged, unwatched* convergecast read at certification time (~2K steps sooner), and the edit serial is nailed to the step counter so a host cannot lose the O(D) bound.

## 1. Field and terms

- **Board**: the padded axial square of `src/hex.ts`, unchanged. A **field** slot has `field = 1`; every other slot (padding or masked out) is **dead**: never updated, always reads its channel's `dead` value. Field slots must be 6-connected (asserted). The hexagon of radius R is the default; a masked field is any 6-connected subset.
- **Picture** = `paint`: `OFF 0`, `ON 1`, `WALL 3`. `FILLED 2` is a state, never a paint value. ON and WALL both block; the one difference: WALL cells 6-connected to a dead slot through WALL cells are **exterior** (the *rim*). So a ragged field is a hexagon with a painted wall ring, and a wall island is an obstacle a loop may enclose.
- **N** field cells, **CAP** = N + 1. **root** = the field cell nearest the centre (lowest slot on ties). **gpar** = static BFS tree over all field cells towards the root, parent = the highest-id neighbour one step closer, 0 at the root. **K** = the root's eccentricity (R on a hexagon). **id** = a seeded permutation of 1…N (random ids cut leader churn 1.5–2×). **area** = 1, the size weight.
- Per edit: **Dw** = the edited cell's eccentricity over field cells (≤ D, the diameter); **Er** = the largest eccentricity of a region's leader within its region; **Wd** = depth of the wall flood from the dead ring; **depth(c)** = static-tree depth.

Values are integers (`Int32Array`; `RGBA32I` on the GPU); only `stamp`, `epoch`, `commit` grow, bounded by the step count.

## 2. Channels

`w` = watched: the channels whose joint fixed point the gate certifies. Dead values equal init unless shown.

| channel | kind | values (init / dead) | meaning |
|---|---|---|---|
| field, id, gpar, area | const | 0/1; 1…N; 0…N; 1 | field slot; id; static parent's id (0 = root); size weight |
| paint | input | 0,1,3 (0 / 3) | the picture |
| stamp | input | serial (0) | edit serial at the edited cell |
| epoch | hidden w | serial (0) | reset wave |
| rim | hidden w | 0/1 (0 / 1) | wall connected to the dead ring |
| redge | hidden w | 0/1 | region touches the exterior |
| leader, dist | hidden w | 0…N; 0…CAP (CAP) | region's maximum id; BFS distance to it |
| rpar | hidden w | 0…N | region-tree parent's id |
| cdone, sub | hidden w | 0/1; 0…CAP | subtree counted; subtree size |
| rsize | hidden w | 0…CAP | region size, 0 = unknown |
| gmax | gate | 0…CAP | lagged max of edge-region sizes in the static subtree |
| M | gate | 0…CAP | certified global maximum, broadcast |
| qt, run | gate | 0/1 (0 / 1); 0…K+2 | static subtree quiet; root's quiet run |
| commit | gate | −1…serial (−1) | last certified epoch |
| want | out | 0/1 | the unlatched answer (diagnostic) |
| state | out | 0 off 1 on 2 filled 3 wall (0 / 3) | what is drawn |

GPU banks: B0 `state want epoch commit`, B1 `leader dist rpar cdone`, B2 `sub rsize redge rim`, B3 `gmax M qt run` (the four guaranteed MRT targets), B4 `paint stamp` + two spare (owner, pattern), B5 the constants.

## 3. The update

For one field cell. Unprimed names are old values, `x_k` is tap k's old value (k = 1…6, **any order**: every loop is a max, min, sum, any, all, or "the tap whose `id` is x"), primes are new. `F(k) = field_k == 1`.

```
e       = max(epoch, stamp, max_k epoch_k, max_k stamp_k)
fresh   = e > epoch
cur(k)  = !F(k) || epoch_k == e                 // taps on an older epoch are invisible
fr(k)   = F(k) && cur(k) && paint_k == OFF
edge    = any_k cur(k) && (rim_k == 1 || (wallsBound && F(k) && paint_k == WALL))
epoch'  = e
rim'    = paint == WALL && edge
if paint == OFF:
  redge'   = edge || any_{fr(k)} redge_k
  (leader', dist') = max of {(id, 0)} ∪ {(leader_k, dist_k + 1) : fr(k), leader_k > 0, dist_k + 1 < CAP}
                     // larger leader wins, then smaller dist
  rpar'    = dist' == 0 ? 0 : max{ id_k : fr(k), leader_k == leader', dist_k == dist' − 1 }
  kid(k)   = fr(k) && leader_k == leader' && rpar_k == id
  settled  = !fresh && leader' == leader && dist' == dist && rpar' == rpar && all_{fr(k)} leader_k == leader'
  cdone'   = settled && all_{kid(k)} cdone_k
  sub'     = cdone' ? min(CAP, area + Σ_{kid(k)} sub_k) : 0
  rsize'   = dist' == 0 ? sub' : rsize_k of the k with fr(k) && id_k == rpar' && leader_k == leader'  (else 0)
else:
  redge' = 0; leader' = 0; dist' = CAP; rpar' = 0; cdone' = 0; sub' = 0; rsize' = 0

skid(k) = F(k) && gpar_k == id                   // static children
gmax'   = max( paint == OFF && redge' ? rsize' : 0,  max_{skid(k)} gmax_k )
root    = gpar == 0
changed = any watched channel x: x' != x
qt'     = !changed && all_{skid(k)} qt_k
if root:  run' = qt' ? min(run + 1, K + 2) : 0
          fire = run' == K + 1
          commit' = fire ? e : commit ;  M' = fire ? gmax' : M
else:     run' = 0 ;  (commit', M') = (commit_k, M_k) of the k with F(k) && id_k == gpar
want'   = paint == OFF && rsize' > 0 && (!redge' || rsize' < M')
state'  = paint == ON ? ON : paint == WALL ? WALL
        : commit' == e ? (want' ? FILLED : OFF) : (state == FILLED ? FILLED : OFF)
```

No explicit reset exists: a fresh cell ignores every tap on an older epoch and `settled` is false; stamps are read from taps, so an edited cell's neighbours change epoch in the step they first see the new paint. `wallsBound` (uniform, default 0) makes every wall exterior.

## 4. Reference predicate (write the oracle from this alone)

```
X   = dead slots ∪ { field cells with paint WALL, 6-connected to a dead slot through WALL cells }
      (wallsBound: every WALL cell)
Φ   = field cells with paint OFF ;  regions = 6-connected components of Φ
edge(R) ⇔ some cell of R has a tap in X ;  size(R) = |R|
M   = max{ size(R) : edge(R) }, 0 if none
fill(R) ⇔ !edge(R) || size(R) < M
state(c) = ON / WALL by paint; FILLED iff c lies in a region with fill; else OFF
```

**Tie rule:** every edge region of size exactly M stays OFF; smaller edge regions fill. A bridge halving the field fills nothing; edge regions of 10, 10 and 4 fill only the 4. A region not touching X fills whatever its size (loop, nested loop, loop round more than half the field or round a wall island). Painting order is irrelevant.

## 5. Correctness, bound, latch

**Fixed-point ladder.** Within an epoch the picture is static, so each watched channel is the unique fixed point of its formula given those above it: `epoch` the max-flood (uniform E); `rim`, `redge` least floods from the dead ring; `leader`/`dist` a monotone max-flood carrying only the region's own ids (splits and merges need no refutation); `rpar` a BFS tree per region; `sub`, `rsize` its sizes. Together they are §4 up to the comparison with M.

**Certificate.** `qt_root(t) = AND_c unchanged(c, t − depth(c))` over watched channels. `run' == K + 1` at step t means `qt_root` held at t−K…t, so the common step t−K had no watched change anywhere: the watched system — closed and deterministic, it never reads `gmax`, `M`, `qt`, `run`, `commit`, `state` — was at its fixed point from t−K−1 on, with one epoch E everywhere. `gmax_root(t) = max_c contrib_c(t − depth(c))` with every `t − depth(c) ≥ t − K`, so the lagged maximum read at the fire step is exactly M for picture E. `commit = E` and `M` travel down the tree together; a cell latches `want` — its fixed-point `rsize`, `redge` and the certified M — only while its epoch is E; one already reached by a newer edit keeps its old latch. **Invariant: every FILLED shown is the oracle of a picture that existed, and a cell's FILLED flips at most once per edit.** The price: after an erase the old, valid fill stays until the next commit.

**Bound.** After the last edit at t₀: epoch wave Dw; leaders with exact distances Dw + Er; tree and `settled` +2; `cdone`/`sub` up and `rsize` down ≤ 2Er; `rim` Dw + Wd, then `redge` + 2Er — the redge flood starts at the region's exterior contact, not at its leader, and the far cell can be a whole diameter (≤ 2Er) from it. **T_fix ≤ t₀ + Dw + max(3Er, Wd + 2Er) + 2.** The root's `qt` lags the deepest cell by K and needs K + 1 quiet steps, so it fires at ≤ T_fix + 2K + 1 and depth j latches j later: **T_commit(c) ≤ t₀ + Dw + max(3Er, Wd + 2Er) + 2K + depth(c) + 3**, on a hexagon with blob-like regions (Dw, Er ≤ 2R, K = R, Wd small) **≤ 11R + 3 ≈ 5.5·D** (C measured 5.9–11.1·R before the graft; HEAD ≈ 127·R). The bound is tight: the boot (every cell fresh at step 1, Dw = 0) commits at exactly 3Er + 3K + 3, and a wall snake of length Wd > 2Er from the rim into a corridor region whose leader sits mid-corridor commits at Dw + Wd + 2Er + 3K + 1 (with `Wd + Er` the bound was short by Er − 1 there). Work is O(N · leader churn) per edit, half of it `qt`. A settled board costs nothing: `run` saturates at K + 2 and the active set empties. Spiral regions have Er ≈ N/width, inherent to any flood.

**What the user sees** (loop closed at R = 20, ~0.3 s at 600 steps/s): nothing while the epoch ring sweeps out, the leaders re-flood and the root counts K + 1 quiet steps; then the commit front leaves the root at a cell per step and the interior fills within its width.

**Known limits.** The epoch and the gate are global: any edit restarts the count everywhere, so nothing fills while anything is being drawn — the bound counts from the *last* edit (R = 20: a loop fills in 162 steps alone, 3789 with a distant cell painted every 120 steps). Fine for painting, so the global gate is kept for this milestone; lines growing continuously need §10's per-region gates. If the lagged `gmax` ever fails the latch-invariant test, fall back to C's original: mark `gmax` and `M` watched and compute `want` from the previous step's `M` — same answers, ~2K slower. If the channel gate itself proves fragile, latch on a host timer `age ≥ settleBound` instead (Design A's fallback).

## 6. Host contract, engine, API

**Serial.** The `stamp` an edit writes is **the number of the step that absorbs it**, `generation + 1`: all changes between two steps share one serial, serials strictly increase across steps. Nothing else is sound — a serial reused across steps lets a cell already on that epoch read new paint without going fresh, and settling degrades to ~N steps (7× slower measured at R = 20 with one serial per stroke). `paint()` enforces it.

```ts
// src/engine.ts — rule-agnostic
export type ChannelKind = 'const' | 'input' | 'hidden' | 'gate' | 'out';
export interface ChannelSpec { name: string; kind: ChannelKind; bank: number; init: number; dead: number; watch: boolean }
export interface Uniforms { N: number; CAP: number; K: number; step: number /* generation + 1 */; wallsBound: 0 | 1 }
export interface Rule {
  readonly channels: readonly ChannelSpec[];
  /** P[c*7 + t]: channel c at tap t, 0 = self, 1..6 = taps in any order. out[c] holds self on entry. */
  update(P: Int32Array, out: Int32Array, u: Uniforms): void;
}
export interface Topology { size: number; cells: Int32Array; nbr: Int32Array /* slot*6+k */; isField: Uint8Array }
export class CA {
  constructor(topo: Topology, rule: Rule, u: Omit<Uniforms, 'step'>, constants?: Record<string, ArrayLike<number>>);
  readonly ch: Int32Array[];                 // SoA, ch[c][slot]; dead slots hold `dead`
  generation: number; changed: number; active: number;
  tapOrder: Int32Array | null;               // tests only: per-slot tap permutation
  get(name: string, slot: number): number;
  write(name: string, slot: number, v: number): void;   // 'input' channels, field slots; activates slot + 6 taps
  step(): void;  stepFull(): void;           // active set / every field cell (the reference sweep)
  run(max?: number): number;                 // until changed == 0; steps taken
}
// src/fill.ts
export const OFF = 0, ON = 1, FILLED = 2, WALL = 3;
export const fillRule: Rule;
export function fillCA(field: Field, opts?: { wallsBound?: boolean }): CA;
export function paint(ca: CA, slots: Iterable<number>, v: 0 | 1 | 3): void;   // paint + stamp = generation + 1; skips dead and unchanged slots
export function bump(ca: CA): void;                                           // re-stamps the root: new epoch, same picture
export function stateOf(ca: CA): Uint8Array;                                  // per slot
export function scramble(ca: CA, rand: () => number): void;                   // in-range garbage into every hidden/gate/out channel
// src/field.ts
export interface Field { board: Board; topo: Topology; N: number; CAP: number; K: number; root: number; ids: Int32Array; gpar: Int32Array }
export function hexField(R: number, opts?: { idSeed?: number }): Field;
export function maskedField(board: Board, mask: Uint8Array, opts?): Field;   // throws unless 6-connected
export function presetWalls(field: Field, preset: 'hexagon' | 'blob' | 'lobes' | 'ring', seed?: number): number[];
// src/oracle.ts
export function oracle(field: Field, paint: (slot: number) => number, wallsBound?: boolean): Uint8Array;  // state per slot, §4
export function settleBound(field: Field, paint: (slot: number) => number, edited: number[]): number;    // §5 by BFS, depth = K
```

**Active set.** A step computes only the listed slots, every read seeing the pre-step state (writes are buffered and applied after the pass), so it is bit-identical to `stepFull`. The next list is every slot whose value changed plus its six field taps; `write` adds the slot and its taps; all field slots are listed before the first step. The engine never interprets a channel.

## 7. Files

```
src/hex.ts      kept as is
src/field.ts    new: Field, masks, presets, static tree, ids
src/engine.ts   replaced: ChannelSpec/Rule/Topology/CA, active set, stepFull
src/fill.ts     new: the rule and helpers            — replaces src/inside.ts (deleted)
src/oracle.ts   new: the predicate and settleBound   — replaces expectedFill in lines.ts
src/lines.ts    kept: rng and the random generators; expectedFill removed
scripts/bench.ts   settle steps, ms, µs/update at R = 20, 40, 80 (not a test)
web/page.html, web/main.ts   rewritten; scripts/build-web.ts kept
tests/*.test.ts, vitest.config.ts (testTimeout 60 000, FUZZ_SEEDS default 40)
```

## 8. Tests (vitest; R ≤ 15; under a minute on a Raspberry Pi)

1. **oracle.test.ts** — hand-counted pictures: loop, bridge, centre-line tie, near-tie, nested loops, loop > half, loop round a wall island, two parallel bridges, loop inside a ragged ring, bridge onto a ragged rim, empty board, `wallsBound` variants.
2. **fill.test.ts** — the same through the CA on hexagon and masked fields: run to quiet, `stateOf == oracle`, every commit ≤ `settleBound`.
3. **fuzz.test.ts** — `it.each` over seeds: 40 % ragged (`blob` walls + an island), 2–5 shapes drawn in two halves with random gaps, 30 % with an erased cell, edits every 0–3 steps, walls in the stream; at quiet `stateOf == oracle`; at **every step** the latch invariant: where `commit == epoch` and paint is OFF, FILLED ⇔ the oracle of that epoch's picture.
4. **overshoot.test.ts** — one edit on a settled board: at every step FILLED ⊆ before ∪ after, each cell flipping at most once; a stroke painted a cell every two steps settles within the bound after its last cell (pins the serial contract).
5. **engine.test.ts** — lockstep twins on the fuzz streams: `step` vs `stepFull`, and a random `tapOrder` vs none, identical in every channel every step; a settled board has `active == 0`; `write` activates slot + taps; a disconnected mask throws.
6. **garbage.test.ts** — `scramble` then `bump`: correct within the bump's bound (the latch may hold garbage until the bump's commit: assert the end state only). Without a bump nothing is guaranteed and the test must not expect it: the design refutes nothing (§11), so garbage that shares the top epoch survives. `leader`/`dist` do count to CAP in ≤ N steps, but `rim` and `redge` are OR-floods over the taps alone: in a region that touches nothing exterior a garbage 1 next to a triangle sticks forever (the gate then certifies the wrong picture — a loop round more than half the field shows its outside filled), and a checkerboard on a bipartite region (two OFF cells in a row of ON) flips every step and the board never goes quiet. Any edit resets everything, since the epoch wave crosses the whole field; a host that scrambles must bump.
7. **rotate.test.ts** — the six rotations of a picture give the rotated final `state` (hidden channels differ with the ids).

## 9. Demo (plain TS, esbuild, Canvas2D; the existing page's layout and tokens)

- **Tools:** paint ON (left drag; skips walls), erase (right drag or tool; clears ON and WALL), wall; keys 1/2/3; strokes interpolated along hex lines; one `paint` batch per pointer event (strokes get increasing serials for free).
- **Field:** radius 4–60 (CPU; beyond needs the GPU port); presets as wall pictures on a full hexagon — `blob` a wobbly rim, `lobes` two discs joined by a neck, `ring` a rim ring plus a central wall island; clear, scramble, bump.
- **Playback:** run/pause, step, step-to-commit, steps/s (capped per frame).
- **Views:** state; any channel as a heat map (ids hashed to hue, counters scaled); want-vs-state diff; epoch wave (newest epoch bright).
- **Readout:** generation, active and changed cells, FPS, steps/s, root `run`/K, commit latency (steps from the last edit to its commit), filled count, oracle ✓/✗ at quiet.

## 10. Extensibility

- **More states.** `paint` is an enum, `state` derived; the fill asks only `blocks(paint) = paint != OFF`. A new state declares whether it blocks and whether it joins the rim.
- **Owners and patterns.** Inputs in B4's spare slots. Cheap (3 channels): `rownLo`/`rownHi`, min/max floods over the region of the adjacent blockers' owners; latched `fowner = rownLo` when equal, else contested. Spectacle's exact rule — your circuit ignores rival lines and captures them — needs one 9-channel region bank per owner slot with `blocks = own lines`, each with its gate: ~100 channels for 10 owners, or owners time-multiplexed over steps.
- **Lines by pattern.** Constants `kind`, `rot` (tile-local edge numbers as in Spectacle), a uniform table `partner[kind][pattern][edge]`; channels `lines` (6-bit edge mask), `lown`, `lpat`, `head` (1 + exit edge), `hphase`. A tap k with `head_k = 1 + opp(k)` enters through edge k and leaves through the partner edge; a chord conflict or two heads entering at once cuts both, and a `cut` wave runs back along the mask — the one unavoidable use of absolute directions. A cell whose `blocks()` changes inside the CA sets `epoch' = u.step + 1`, the serial of the batch in which the change becomes visible, so host and CA edits never collide.
- **Per-region gates (the starvation fix).** A loop's decision is local (`redge = 0` fills regardless of M); only bridges need the global maximum. Next: an epoch flooding through free cells only (an edit then resets exactly the regions it touches, since split and merged regions all touch the edited cell) and a leader-rooted quiet run certifying non-edge regions locally, the global gate kept for edge regions. The local run needs a watched, stable region-height channel — what broke Design A's boot — so it is a design of its own, not a knob.
- **GPU.** B0–B3 ping-pong as four `RGBA32I` MRT targets, B4 via `texSubImage2D`, B5 static, taps by `texelFetch` at the six axial offsets, the same integer code; the gate is in channels, so no readback.

## 11. What the previous attempt did and why this differs

HEAD (`src/inside.ts`) labels each region by the smallest *border index* it reaches through a min-plus flood, numbers the border ring from the east corner, counts open border cells up a spanning tree and fills when `2b < T`. Right at rest, but a label whose source is cut off dies only by counting up to N: one closure costs ~N steps at ~N work each (~21 s at R = 40) while ~90 % of the board transiently reads filled; the numbering is hexagon-only; walls and lines are one thing. PR #1/#2's pulse race was local but needed one host cell per step and broke on a circuit drawn in two halves. Here an **epoch** per edit makes stale values invisible instead of refuting them (no count-to-infinity: O(D)); regions are named by their **maximum id**, so masked, ragged and later irregular fields need no border walk; size is a **convergecast** compared with a **global maximum** over a host-constant static tree, so the smaller side is by area; the visible state is **latched by a certificate**, so no wrong fill is ever shown; and the engine gains an active set, named channel specs and a tap-symmetric update that ports to GLSL line for line.
