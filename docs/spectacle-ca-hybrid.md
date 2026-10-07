# Spectacle as a hybrid CA: hand-written local rules for the lines, the trained flood for the area (2026-10-08)

A design for coding agents to build from; nothing here is implemented. It departs on purpose from the learned-strand
work (`spectacle-nca-taps.md`, launches 6-10): every edge, chord and strand mechanic is a **deterministic, synchronous,
radius-1 integer CA** written by hand, each cell reading itself and its six neighbours plus static tile facts and a
chord table; **area control** is the already-trained flood-fill NCA (`web/nca-weights.json`, `runs/fb-r6816nt`) with
line tiles as its walls. The owner's words: "all of the edges and the patterns are done deterministically, still locally
at the cell level, but we use the trained flood fill for the area control portion." **Measured** = read from code or
logs today; the rest is judgement. Spectacle facts are from `shared/game/engine.ts`, `strand.ts`, `field.ts`,
`knobs.ts` (read today) and `CLAUDE.md`'s settled decisions.

Owner decisions already in force and taken as given: the pattern owns the tile (any rival line on an entered tile is a
hit; Spectacle's `crossingMode` moves to `'tile'`), collisions kill both lines (`mutualCut`), your own lines overlap or
join, rules are unique per player in Normal mode (so *rule = player* for the CA), the infinite-line rule need not be hidden.

## 0. Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| 1 | **Which growth semantics first.** Spectacle's (a tapped line grows until it stops) or the bounded variant ("a tap only flips a few tiles; to close large loops you keep tapping") | **Spectacle's first, with the bounded knob built in from day one** (`fuel`, §2.8: an 8-bit counter a tap sets and each chord decrements; 0 = unbounded). Reasons: the parity harness (sim.py's CA mode, the `--collide` fixture, the engine) exists only for unbounded growth, so it is the only semantics we can *check*; the knob costs one state field and three lines; and whether bounded growth plays better is a balance question only the playable page can answer. Play-test the knob in the second week, not the first |
| 2 | **What score counts.** Spectacle today: the distinct tiles your lines are on (`scoreTiles`; interiors are drawn, never scored — measured, `hold`/`heldTiles`). The flood would be decorative under that rule | **Score = line tiles + the tiles your flood fills for you alone** (`scoreFill`, default on; off = Spectacle's count). Contested tiles (inside two players' fills) score nobody until captures land (decision 8) |
| 3 | **Whose "smaller side".** Spectacle: per claim, the smaller *polygon area* of line + outline arc, other lines ignored (`boundaryRegion`). The flood: per owner, every rim-touching region but the largest fills, by cell count, learned | **Accept the flood's answer.** It is what the model does; the two agree for one claim except at near ties, and differ for two rim-to-rim claims by one owner (§3.3). Nothing in the game needs the polygon |
| 4 | **A closed flag.** No mechanic needs it: the flood decides "inside", and a loop has no end to turn (§2.9). It is wanted for the UI (colour a circuit) and for the parity check on `closed` | **Build the loop probe (§2.9) behind a knob**, ~120 bits of optional state and ~40 lines. If the owner would rather not: the host's union-find readout gives the same answer in O(N) |
| 5 | **Taps grow both ways** (as sim.py, the training data and the fixture) rather than Spectacle's one way | Yes, `oneWay` as a knob. A one-way tap is one tip instead of two |
| 6 | **Heads.** Spectacle counts heads per path (a `back` head counts two); a path needs line identity, which is not local | **`maxTips` per player** (default 2 = one two-way line); captures add tips later. A tip is a per-edge bit, counted by a scan |
| 7 | **Speed.** Spectacle: tiles/s ∝ score, per player | **Uniform period 2 in v1** (one chord per two steps, every player: the setting the parity harness pins). Per-player `go` pulses are the knob (§2.3), the host decides their rate |
| 8 | **Captures and conversion.** v1 has none: a rival line inside your fill stays theirs and its tiles are contested | **v1.1: Normal-mode conversion host-side** (wipe + sprout your rule on its tiles, §3.4); Conquest's relabel later |
| 9 | **Flood fine-tune.** The model was trained on blobs, polygons, rings, strokes, bridges and spirals on ragged masks, not on strand-shaped walls | **Measure first** (`scripts/area-probe.ts`, §3.6); spend ≤ $2 of the ~$5 only if level-3 exact on strand walls is under 0.95 |
| 10 | **One frame.** The strand boards (`web/strand-data.json`, row/col with `geo`) and the flood's maps (`nca/fields`, S×S) use different frames (measured) | **The strand `Board` is the frame**; the area layer translates into the NCA's S×S array (§3.1). The export gains Spectacle's tile index per cell for the engine fixture |

## 1. Scope

| Mechanic | v1 | How | Simplified from Spectacle |
|---|---|---|---|
| Tap | yes | host validates (nearest chord, refusals), writes two tips | one-way is a knob; `tapOwnLine`'s recolour/promote have no counterpart (one pattern per player) |
| Growth, two-way | yes | §2.3-2.4, one chord per `period` steps | speed uniform (decision 7) |
| Tails, board edge | yes | a tip with no chord ahead or no neighbour stops; the chord stays | — |
| Collisions, mutual | yes | §2.5: the tile rule; both lines wiped by waves | wipe is a wave at one tile a step, not instant (§2.10) |
| Own-line join / overlap | yes | §2.6: set semantics — a tip onto its own drawn chord stops; beside it, draws | no layering of two paths on one chord (Spectacle's `join` is the common case anyway) |
| Loop closure | yes (knob) | §2.9 probe round the loop, `closed` announced along it | informational only |
| Edge-to-edge claim | yes | nothing in the line CA: the flood fills the smaller side of a rim-to-rim wall (§3.3) | smaller by cells, not polygon area; a tail on a rim tile also claims (§3.3) |
| Area, score | yes | §3: one flood instance per player, walls = that player's line tiles; score = line tiles + sole fill | Spectacle scores line tiles only (decision 2) |
| Respawn delay | yes | host: a `hit` pulse per cell → `respawnAt[owner]` | — |
| Head limit | yes | `maxTips` (decision 6) | not per path |
| Bounded growth | knob | §2.8 `fuel` | new |
| Captures, conversion | v1.1 | host: a component wholly inside a rival's fill → convert (wipe + sprout) | Conquest relabel later |
| Flips between own patterns, regrow on rule change, speed ∝ score, points scoring, Spectre view | later / never | — | one pattern per player; a rule change = leave and rejoin |

## 2. The line CA

### 2.1 Field, frame, static facts

Cells are the strand `Board`'s (`src/strand.ts`): axial `(row, col)` in an `h × w` box, `nbr[i*6+d]` in `DROW/DCOL`
order (E, NE, NW, W, SW, SE; `opp(d) = d+3 mod 6`), -1 off the board. Per cell the constant `geo = type·12 + rot·2 +
mirror` (108 values). The chord table is uniform: `partner[r][geo][d]` = the direction a line of rule `r` entering a tile
of this `geo` across direction `d` leaves by, or -1 (no chord through `d`): 648 bytes per rule, built from
`RuleTable.render` / `exits` (`src/strand.ts:184`, verified against Spectacle on 131,840 taps, `strand-data.md`). Rules are
host-assigned small integers (1..1023; 0 = none); in Normal mode a rule is a player, and `owner` is kept beside it only so
Conquest's captured patterns can come later.

The engine is `src/engine.ts`'s `CA` (Int32 channels, synchronous, active set, `step == stepFull` by construction). Taps
there are in direction order (`Topology.nbr`), which this rule needs; `tapOrder` symmetry does not apply to it.

### 2.2 Cell state

| field | bits | meaning |
|---|---|---|
| `rule` | 10 | the rule of every chord on this tile (0 none). Stays while any `W` pulse is live, so a wiped tile is *hot* for one step |
| `owner` | 4 | the player (0 none); Normal mode: `owner = ownerOf[rule]` |
| `D[6]` | 6 | edge `d` carries a drawn chord end. Under one rule a tile's chords never share an edge, so `D` fixes the chords: chord `(d, pt(d))` is drawn iff `D[d]` (and then `D[pt(d)]`) — the owner's observation. Two chords of one strand on a tile (14 % of strand cells, measured) are two pairs of bits |
| `T[6]` | 6 | a tip: the line ends here and will leave across `d` |
| `W[6]` | 6 | a wipe wave: my chord through `d` was erased this step; the wave leaves across `d`. A one-step pulse |
| `H` | 1 | hit pulse (victim, hitter or crash this step), for the host's respawn delay and the UI's sparks |
| `fuel` | 8 | bounded growth (§2.8); unused when the knob is 0 |
| `closed[6]` | 6 | the chord through `d` is on a confirmed loop (§2.9, knob) |
| `probe[6]` | 6 × 20 | a loop probe in flight out through `d`: the origin's cell index + 1, 0 none (§2.9, knob) |

Core state is 41 bits (fits one Int32 with `fuel` in a second); the engine keeps one Int32 channel per field, SoA. Invariant
(asserted in tests): `rule ≠ 0 ⇔ (∃d D[d] ∨ ∃d W[d])`; `T[d] ⇒ D[d]`; `D[d] ⇒ D[pt(d)]`; one rule per tile by construction.

### 2.3 The update

One synchronous step for cell `me`; unprimed = old values, `N_d` = the neighbour across `d` (⊥ off board), subscript `d`
= that neighbour's field. Uniforms: `period`, `step`, `go[owner]` (v1: `go[p] = (step mod period = 0)` for every `p`),
`fuelMax`. Everything a cell reads is its own state, its six neighbours' state and `geo`, and the table.

```
pt(d)      = rule ≠ 0 ? partner[rule][geo][d] : -1           // the other end of my chord through d
aimed(d)   = N_d ≠ ⊥ ∧ T_d[opp(d)] ∧ partner[rule_d][geo][d] ≥ 0   // a tip in N_d pointed at me, with a chord here under its rule
waveIn(d)  = N_d ≠ ⊥ ∧ W_d[opp(d)] ∧ (rule_d = rule ∨ rule_d = 0)   // a wipe wave arriving across d along my strand
hitAt(d)   = T[d] ∧ N_d ≠ ⊥ ∧ rule_d ∉ {0, rule}                    // my tip is aimed at a rival's tile (drawn, or hot)

// 1. collisions — the tile rule, mutual, symmetric
victim  = rule ≠ 0 ∧ ∃d: aimed(d) ∧ rule_d ≠ rule
crash   = rule = 0 ∧ |{ rule_d : aimed(d) }| ≥ 2                      // tips of two rules into one empty tile on one step
hitter  = ∃d hitAt(d)
H'      = victim ∨ crash ∨ hitter

// 2. erasing: a hit takes every chord on the tile; a wave or my own hit takes the chord it reaches
gone(d) = D[d] ∧ (victim ∨ waveIn(d) ∨ waveIn(pt(d)) ∨ hitAt(d) ∨ hitAt(pt(d)))
D'[d]   = D[d] ∧ ¬gone(d)
W'[d]   = (gone(d) ∧ ¬waveIn(d)) ∨ (crash ∧ aimed(d))                 // out of every erased end but the one the wave came in by

// 3. tips: stop, retire, or wait
tail(d)  = N_d = ⊥ ∨ partner[rule][geo_d][opp(d)] < 0 ∨ (fuelMax > 0 ∧ fuel = 0)
moved(d) = N_d ≠ ⊥ ∧ rule_d = rule ∧ D_d[opp(d)]                     // the neighbour holds my chord through our edge: I moved, or met my own line
T'[d]    = T[d] ∧ ¬gone(d) ∧ ¬tail(d) ∧ ¬moved(d)                      // else: wait (the neighbour draws on its go step; I see it next step)

// 4. receiving a tip across d
enter(d) = aimed(d) ∧ go[owner_d] ∧ (rule = 0 ∨ rule = rule_d) ∧ ¬victim ∧ ¬crash ∧ ¬D[d] ∧ ¬W[d]
f(d)     = partner[rule_d][geo][d]                                     // my exit for that line
D'[d]   |= enter(d) ∨ ∃e: enter(e) ∧ f(e) = d
T'[d]   |= ∃e: enter(e) ∧ f(e) = d ∧ ¬enter(d)                         // the new tip, unless a tip also arrived there (a loop closing on a fresh tile)
rule'    = ∃d enter(d) ? (rule ≠ 0 ? rule : rule_d of any entering d) : (∃d D'[d] ∨ ∃d W'[d]) ? rule : 0
owner'   = likewise; fuel' = (rule = 0 ∧ ∃d enter(d)) ? min_{enter(d)} fuel_d − 1 : fuel
```

Reading it: a tip is one bit on one edge and *waits* until the tile ahead has drawn the continuation; the tile ahead
draws only on its owner's `go` step, so a chord costs `period` steps and a tip is never in two places. `W` is a one-step
pulse: the chord goes in the step the wave reaches it and the pulse carries the wave one tile further, so a wipe runs at
one tile a step and a tip (at 1/`period`) is always caught (§2.10). `rule` outlives the chords by the pulse's one step,
which is the sim's "lingering" (a tap there is refused, a rival tip entering it hits it and dies). `¬D[d]` in `enter` is
absorption (§2.6); `¬W[d]` stops a tip from redrawing the hole a wave just made (the far tip of a cut loop would otherwise
run round and regrow the line, sim.py's case).

### 2.4 Taps (host, between steps)

The host picks the chord nearest the pointer (`web/strand.ts`'s `rankChords`, Spectacle's `nearestChord`) and refuses,
in Spectacle's order: respawn delay running; `rule_c ∉ {0, r}` (a rival's tile, hot ones included); `D_c[d0]` or
`W_c[d0]` (that chord drawn or just wiped — the next nearest free chord is tried first, as `freeChord` does); `tips(owner)
≥ maxTips`; the tile filled for a rival by the area layer (Spectacle's `insideRivalCircuit`, §3.4). Accepted: write
`rule, owner, D[d0] = D[d1] = 1, T[d1] = 1` and `T[d0] = 1` unless `oneWay`, `fuel = fuelMax`. `retap` on a drawn chord of
your own sets `T` on each of its loose ends that has somewhere to go (Spectacle's `tapOwnLine` → `turnRound`/`setBack`):
with two-way growth and no fuel it is a no-op, which matches — a stuck end here is a tail, and tails are static.

### 2.5 Entering a tile, case by case

| The tile ahead | The CA | Spectacle |
|---|---|---|
| off the board | `tail`: `T` off, chord stays | `stepForward` dead → `stuck` (an edge claim if the line started at the rim: §3.3 covers it) |
| no chord through this edge under my rule | `tail` | dead → `stuck` |
| empty | draws `(d, f(d))`, new tip at `f(d)` | `addStep` |
| my own rule, chord through this edge not drawn | draws beside (a second chord of one strand on the tile) | grows on over the top (`overlapOwnLines`); chords of one rule never cross |
| my own rule, chord through this edge drawn | `moved`: my tip retires, nothing drawn. A loose end met (join) or a loop closed (§2.9) | `join` (the other line's loose end on the same chord) or `closeCircuit` |
| a rival's rule, drawn or hot | I am `hitter`: my chord goes, a wave runs back along me; the tile is `victim`: every chord on it goes, waves run out along each | `cutRivals` (tile mode) drops every rival path on the tile; `mutualCut` drops the hitter |
| two tips of different rules arrive together | `crash`: nothing drawn, both hitters' chords go | the second mover hits the first's step — both die |
| two tips of one rule arrive together | both chords drawn (one, if the same chord: a loop closed on a fresh tile); no tip where a tip came in | the second joins the first |
| a wave of my rule arrives across the edge I'm about to cross | `gone`: my chord goes with the wave (head-on with my own wipe) | n/a (instant) |

**Where Spectacle has an order, the CA has symmetry.** Spectacle moves players in join order and paths in array order; the
order decides which chord is drawn for one tick in a race, never who survives (both die under `mutualCut`). Every rule above
is symmetric in the neighbours, so no player or rule order is used anywhere. The one asymmetry in Spectacle that is visible
at rest — a join folds the *met* path into the joiner, which keeps the id and the points — is about ids and points,
which the CA does not have.

### 2.6 Joins and overlap

Lines of one rule are a set of chords. A tip reaching a chord already drawn under its rule retires (`moved`); the two
stretches are one line. This is Spectacle's `join` whenever the meeting is at a loose end — and under a clean rule it
always is: a chord's continuation across an edge is unique, so two stretches of one strand can only meet end to end. A
tip reaching a tile that holds *another* chord of its rule draws beside it. Two paths layered on one chord (Spectacle's
"over the top", reachable only through flips and regrow pieces, which v1 has not) have no representation here.

### 2.7 Wipe waves

A hit erases a tile and starts a wave out of each erased chord end; a wave arriving across `d` erases the chord through
`d` and continues out of its other end. It follows the strand through twice-visited tiles chord by chord (case 9 of the
taps doc) and never jumps to a neighbouring chord of the same tile. It stops at a loose end (nothing beyond holds the rule)
and kills any tip whose chord it takes. It crosses between touching stretches of one rule, because they are one line
(sim.py's rule; Spectacle wipes a *path*, and after a `join` that is the same set).

### 2.8 The bounded-growth knob

`fuelMax > 0`: a tap writes `fuel = fuelMax` on its tile; a tile drawn by an entering tip takes `fuel_d − 1`; a tip on a
tile with `fuel = 0` is a `tail`. So a tap lays at most `fuelMax` chords each way, and `retap` on the stuck end (the host
rewrites `fuel` and `T`) extends it by another `fuelMax`: "to close large loops you need to keep tapping, and the NCA
will help flip tiles as you go" — the fill appears as soon as the chain closes. Nothing else changes; waves, hits and
joins are the same. `fuelMax = 0` is Spectacle's semantics. (Multi-head cascades are bounded in v1 by `maxTips` anyway:
there are no spawned pieces, so a player never has more tips than taps.)

### 2.9 Loop closure: the probe (knob `probe`)

A loop needs no flag to *work* — the flood fills it. To *know* it is a loop locally: when a tip of my rule lands on my
drawn chord through `d` (`aimed(d) ∧ rule_d = rule ∧ D[d]`, or a loop closing on a fresh tile: `enter(d) ∧ enter(f(d))`),
I send a probe carrying my own cell index out through the chord's far end. Cells relay it along their chord, one tile a
step; it dies at a loose end (the next tile holds no chord to pass it to) or with an erased chord. If it comes back to me
across the other end of that chord, the line is a loop:

```
met(d)     = (aimed(d) ∧ rule_d = rule ∧ D[d] ∧ ¬gone(d)) ∨ (enter(d) ∧ enter(f(d)))   // a tip of my line landed on my chord through d
in(d)      = N_d ≠ ⊥ ∧ rule_d = rule ∧ D[d] ∧ D_d[opp(d)] ? probe_d[opp(d)] : 0
probe'[d]  = ¬D'[d] ? 0 : met(pt(d)) ? me+1 : in(pt(d)) ≠ me+1 ? in(pt(d)) : 0     // start, relay, or consume my own
loop(d)    = in(d) = me+1 ∨ in(pt(d)) = me+1
closed'[d] = D'[d] ∧ (loop(d) ∨ closed[d] ∨ closed[pt(d)] ∨ (N_d ≠ ⊥ ∧ rule_d = rule ∧ D_d[opp(d)] ∧ closed_d[opp(d)]))
```

`closed` is an OR-flood along the loop from the origin (both ways: L/2 steps after the probe's L). Two probes in flight pass
each other (one per edge per direction). A join of two open stretches sends a probe that reaches a loose end and dies:
nothing announced, correctly. Cut loops lose `closed` with their chords. Cost: 6 × 20 bits + 6 bits per cell, only in the
line cells. The host's `components()` (union-find over drawn chords) gives the same answer in O(N) and is the test oracle.

### 2.10 Timing: where the CA necessarily differs from Spectacle, and how parity is still tested

| | Spectacle | The CA | At rest |
|---|---|---|---|
| wipe | the whole path in one tick | a wave, one tile a step, from the hit; a fleeing tip is caught after `2d` steps (period 2) and its extra chords go too | same: both lines gone |
| the hitter's step into the tile | drawn for one tick | never drawn | same |
| growth | `stepIntervalMs`, fractional progress | one chord per `period` steps, phase-gated: a tap on an odd step draws its first chord one step late | same chords |
| a tile just wiped | free at once | hot for one step: a tap is refused, a rival tip entering it dies | a race only |
| a line heading for a loose end that is hit meanwhile | the wipe is instant, so the newcomer passes through and survives | the wave takes `d` steps; if the newcomer joins first, the wave takes it too | **can differ**: the one at-rest divergence, a race between a wave and a join. sim.py's CA mode has the same semantics |
| region / inside | instant polygon | the flood settles over ~8-16 R steps (§3.5) | same side, near ties and pockets aside (§3.3) |

Parity is checked at settled states, three ways (§5): (a) a single tap against `walk` at `period` × its length (sim's
own test); (b) against **sim.py's CA mode** (`simulate(..., wave=1, sequential=False)`) chord for chord, on and off times
within ±1 step (the phase), on `draw_taps` episodes of every kind; (c) against **Spectacle's engine** with `crossingMode:
'tile'` on the existing `--collide` fixture (`data/strand-v2/collide.json`: 400 episodes, 18 boards, `COLLIDE_KNOBS`),
comparing accepted taps, survivors and chord coverage at rest, with every mismatch classified as the wave-vs-join race or
reported as a failure. sim.py's game mode already reproduces the engine tick for tick on all 400 (measured), so (b) and (c)
together pin the CA to the game.

## 3. Area control with the trained flood

### 3.1 Lines to walls, and the frame

Per player `p`, `walls_p[c] = 1` iff `owner_c = p ∧ ∃d D_c[d]` — the player's line tiles, whole. One `HexNCA` instance per
player over one shared `NCAWeights` (`src/nca.ts:254`; instances are independent, measured). The strand `Board`'s
`(row, col)` translates into the NCA's S×S array (`cell (q, r) at [r+R][q+R]`, the same six axial neighbour vectors as
`DROW/DCOL`, measured) with `S = max(h, w) + 2`, rounded up to odd (a one-cell margin), `slot = (row+1)·S + col+1`; the mask is the
board. `partner[0][·][·] = -1`, so nothing is ever aimed from or into "no rule". The flood sees the rim as cells with off-mask neighbours — exactly how it was trained on ragged masks.

**Watertight.** On a hex grid the flood's 6-neighbourhood *is* edge adjacency (no corner-only neighbours), and a line
crosses between tiles only through shared edges, so a loop's tiles are a 6-connected ring. A free cell is wholly on one side
of the loop's polygon (the polygon passes only through line tiles), and two 6-adjacent free cells are on the same side (the
shared edge is not a chord end of a free cell), so the enclosed regions are unions of 6-components of free cells: the flood
cannot leak. Twice-visited tiles and two touching strands of one player are single wall cells; a pinch that encloses no free
cell fills nothing, as in Spectacle (no tile to score). **Pockets** are the one divergence: a free cell boxed in by line
tiles yet outside the polygon (a sharp U-turn) is "inside" for the flood and outside for Spectacle. It matters only for tap
refusal and captures (Spectacle never scores interiors), and is cheap to count (§7).

### 3.2 Several players

K instances, K wall sets, rival lines transparent to each (Spectacle's "other lines ignored"). Nesting: a rival's loop can
only be inside yours if it was there when yours closed — a rival cannot start inside your fill (tap refused) or grow in
(a hit). So a rival line wholly inside your fill is exactly Spectacle's capture set (§3.4). Until captures are built, tiles
inside both fills are *contested* and score nobody; with v1.1 the inner line converts and its fill goes with it.

### 3.3 Edge-to-edge claims = the flood's bridge

A line from rim to rim is, as walls, a rim-to-rim chain; the free cells split into two or more rim-touching regions and the
flood fills all but the largest: for one claim, the smaller side. Spectacle's `region` (the smaller shoelace area of line +
outline arc) agrees except: (i) **near ties** — cell count versus polygon area, and the model's own weakness there (bridge
exact 0.98 at R 16, 0.76 at R 24, every miss a near-even bridge; measured); (ii) **two claims by one owner**: Spectacle fills
each claim's own smaller side (both ends of a board cut in three), the flood fills every region but the largest (one end
and the middle); (iii) **a tail on a rim tile** whose exit does not reach the outline is no claim in Spectacle
(`startsAtEdge` + dead on the boundary) but its wall chain still separates regions, so the flood fills a side. (iii) is a
small gift to the player; (i) and (ii) are decision 3.

### 3.4 Score, taps, captures — the readouts

Every `scoreEvery` steps (default: each line step) the host reads each instance's `channel(1) > 0.5`:
`territory[c] = owner_c` if a line is on `c`, else the one player whose fill covers `c`, else 0, else -1 (contested). Score =
tiles per owner, shown with a two-read hysteresis so a settling flood does not flicker the HUD. Tap refusal "inside a rival's
circuit" = `territory[c] ∉ {0, me}` (Spectacle tests the tile centre against the polygon; the flood's answer may lag a
closure by its settle time, §3.5). **v1.1 conversion** (Normal mode): a component (`components()`) of player `q` whose cells
are all filled for `p ≠ q` is converted — the host writes a wave (`W` on its end chords) or clears it outright, then
sprouts `p`'s rule on those tiles as taps with no tips (Spectacle's `convertPath` → `sprout`, pieces growing is left out).

### 3.5 Live edits and settle time

The flood was trained on live edits (damage pool; edit test exact 0.92-0.96 at R 8, measured) and tracks walls as they
change; here a wall changes at every tip every `period` steps. It is right once it has settled — read-outs are at 16 R
steps in evaluation — so after a closure the fill spreads in over ~8-16 R flood steps (level 3, R 31: 250-500). The host
runs `floodPerStep` flood steps per line step (default 1, raised while the frame budget allows, as `web/nca.ts` does), so
at level 3 a closure fills in 10-25 s of line time at 20 line steps/s, visibly spreading — the owner's "the NCA will help
flip tiles as you go". Growth itself never waits on the flood.

### 3.6 Quality, and a small cloud plan (optional)

What the model has seen: closed blobs/polygons/rings, random-walk strokes, 1-3 rim-to-rim bridges (thickened), spirals,
noise, on discs and on ragged masks (blobs, l3/l4 crops; measured, `nca/data.py`, `masks.py`). What strands look like: one-cell
chains turning in 60° steps, 14 % of cells visited twice, loops of 3-4 tiles enclosing 0-1 cells (strand length p50 3-4
chords), many loops of one owner side by side, corridors one cell wide between parallel strands. Measured on its own kinds:
~0.97 exact on whole level-3 maps; unknown on strand walls. **First task (B1): `scripts/area-probe.ts`** — 200 random
(rule, tap) loops and claims from `allStrands`/`walk` on l2/l3/l4, 1-3 per board, one owner; run the shipped weights to 16 R;
exact against `targets()`; per level and per enclosed-size bucket. ≥ 0.95 at level 3 → skip the fine-tune.

Otherwise (≤ $2 of the ~$5): `nca/data.py` kind `strand` — walls rendered from `nca/strand/walker.py` + `rules.py` on the
l2/l3 fields and l4 crops (1-3 loops or claims of one rule), 30 % of new pool boards, `--mask-mix` with the whole l3 field;
one plan line, 90 min from `bucket:runs/fb-r6816nt/best.pt` at `--lr 5e-5 --R 8 12 16 24`; `evaluate` with a `strand` set;
`python -m nca.export` → `web/nca-weights.json` + the parity fixture (`tests/nca.test.ts` re-pins). ~2.5 Spot-L4 hours ≈ $1,
plus one reclaim. Level 4 whole (R 82) is beyond anything trained and settles in ~1,300 steps; it is a GPU-and-measure
item (§4.4), not a v1 target.

## 4. Architecture and layout

TypeScript, in hex_ca, browser-first; no Python twin of the CA (the Python side compares timelines, §5).

```
src/game/line-ca.ts     A  the line CA: channels, the §2.3 update as an engine Rule, taps, readouts, components()
src/game/area.ts        B  the area layer over src/nca.ts: K instances, frame translation, territory, score
src/game/area-gl.ts     B2 (optional) the same AreaLayer on WebGL2
src/game/host.ts        C  the referee: players, rules, nearest-chord taps, go pulses, respawn, scores, events, knobs
web/game.html, game.ts  C  the page: board, rule picker, 1-4 players, a bot tapper, channel views
src/draw.ts             C  hex drawing + pointer→cell lifted from web/nca.ts (fitGeom, centreOf, cellAtPoint, hexes, Fills)
scripts/area-probe.ts   B1 §3.6
scripts/game-parity.ts  D  replays collide.json taps through LineCA → data/strand-v2/ca-collide.json
nca/strand/ca_parity.py D  compares it with sim.py's CA mode and the engine (reusing sim_parity's views)
tests/line-ca.test.ts A · tests/area.test.ts B · tests/game-parity.test.ts D · tests/game.test.ts C
```

### 4.1 Interfaces (frozen by this doc; agents build against them)

```ts
// src/game/line-ca.ts
export interface LineKnobs { period: number; oneWay: boolean; fuel: number; probe: boolean; maxTips: number }
export const DEFAULT_LINE_KNOBS: LineKnobs = { period: 2, oneWay: false, fuel: 0, probe: true, maxTips: 2 };
export type Refusal = 'no chord' | 'rival tile' | 'drawn' | 'hot' | 'heads' | 'respawn' | 'territory';
export interface TapRequest { cell: number; d0: number; d1: number; rule: number }
export class LineCA {
  constructor(board: Board, table: RuleTable, knobs?: Partial<LineKnobs>);
  readonly ca: CA;                                   // src/engine.ts; channels: geo(const) rule owner D T W H fuel closed probe0..5
  addRule(rule: Rule, owner: number): number;        // → rule index ≥ 1; builds partner[ix][geo][d]
  tap(req: TapRequest, blocked?: (cell: number) => boolean): Refusal | null;   // the local refusals + the host's `blocked`
  retap(cell: number, d0: number, d1: number, rule: number): boolean;
  step(go?: Uint8Array): void;                       // go per owner; default: all on steps ≡ 0 mod period
  readonly ch: { rule: Int32Array; owner: Int32Array; D: Int32Array; T: Int32Array; W: Int32Array; H: Int32Array; closed: Int32Array };  // views on ca.ch
  changed(): Int32Array;                             // cells whose D or owner changed in the last step
  countTips(owner: number): number;  lineTiles(owner: number): number;
  components(): { owner: number; rule: number; cells: Int32Array; ends: number; closed: boolean }[];
}
// src/game/area.ts
export interface AreaLayer {
  readonly owners: number;
  update(lines: LineCA, changed: Int32Array): void;  // re-derives walls_p for the changed cells; setWall per instance
  step(n?: number): void;
  fill(owner: number): Uint8Array;                   // per board cell, channel(1) > 0.5
  territory(): Int8Array;                            // owner | 0 | -1 contested; line tiles → their owner
}
export function cpuArea(board: Board, weights: NCAWeights, owners: number): AreaLayer;
export function glArea(gl: WebGL2RenderingContext, board: Board, weights: NCAWeights, owners: number): AreaLayer; // B2
// src/game/host.ts
export interface GameKnobs extends LineKnobs { scoreFill: boolean; respawnMs: number; stepMs: number; floodPerStep: number; scoreEvery: number }
export type GameEvent = { t: number; kind: 'tap' | 'refused' | 'hit' | 'tail' | 'closed' | 'convert'; owner: number; cell: number; why?: Refusal };
export class Game {
  constructor(board: Board, table: RuleTable, weights: NCAWeights, area?: AreaLayer, knobs?: Partial<GameKnobs>);
  addPlayer(name: string, rule: Rule): number;       // owner 1..15; refuses a rule another player holds
  tap(owner: number, x: number, y: number): Refusal | null;   // board units → cell + rankChords; applies between steps
  tick(nowMs: number): void;                         // due line steps; flood steps; readouts; respawn; (v1.1) conversion
  readonly lines: LineCA; readonly area: AreaLayer; scores(): Int32Array; drain(): GameEvent[];
}
```

A, B and D depend only on `src/strand.ts` (`Board`, `RuleTable`, `walk`, `allStrands`), `src/engine.ts` and `src/nca.ts`;
C depends on A and B through the interfaces above and can start on stubs the same hour.

### 4.2 Performance per step

| level | cells | line CA, JS (active set; full sweep) | flood, scalar JS, per player | flood, WebGL2, per player (est.) | flood settle after a closure (8-16 R steps) |
|---|---|---|---|---|---|
| 2 | 63 | ≪ 1 ms | ~4 ms | — | 80-160 |
| 3 | 496 | ≪ 1 ms (0.1 ms) | ~30 ms (measured range 30-45) | < 1 ms | 250-500 |
| 4 | 3,905 | < 1 ms (1 ms) | ~250 ms (measured) | ~1 ms | 650-1,300 |
| 5 | 30,744 | ~1 ms (6 ms) | ~2 s | ~5 ms | 1,900-3,800 |
| 6 | 242,047 | ~2 ms (50 ms) | ~16 s | ~30-60 ms | 7,000-14,000 |

Line CA: ~0.2 µs per active cell-step of integer work; only line cells and their neighbours are active. Flood: 65 µs per
cell-step scalar (measured 8.5/14.3/28.6 ms at R 6/8/12); on a GPU the 49k MACs per cell are 0.2 GMAC a step at level 4 and
12 GMAC at level 6. **The limit past level 4 is the settle time, not the step**: at level 6 a closure takes minutes to fill
on any hardware, and the model is untested above R 24. v1 plays levels 2-3 on the CPU; level 4 wants B2; levels 5-6 want a
faster-settling fill (the hand CA's O(D) gate, or a multi-scale model) — out of scope, named in §7.

### 4.3 The WebGL2 flood (B2, when level 4 matters)

24 channels as six RGBA32F textures, ping-pong, written with MRT (two passes of 3 if `MAX_DRAW_BUFFERS` is 4); `w1` as a
256 × 175 float texture, `w2` 24 × 256; one fragment shader does the 7-tap gather, ReLU into 256 and the residual add,
`highp`; walls and mask uploaded with `texSubImage2D` for the changed cells; K players as K layers of a 2D array texture or
K passes; read-back of channel 1 once per `scoreEvery` steps (`readPixels` of one R8 texture). Software renderers
(SwiftShader on CI and this sandbox) are detected on a throwaway canvas and fall back to `cpuArea`, as Spectacle's
`tiles-gl.ts` does. Parity with `cpuArea` to 1e-4 on the thresholded fill.

## 5. Test plan

**Unit tests per rule** (`tests/line-ca.test.ts`, hand-built 3-7 cell boards from `strand-data.json`'s l2 geo, each
checked step by step): growth timing (chord k at tap + `period`·k, both ways); tail at a no-chord tile and at the rim; draw
beside; absorb at a loose end (join) and a loop closing on a drawn chord and on a fresh tile (probe returns, `closed`
floods round in ≤ 1.5 L); hit: victim and hitter waves at one tile a step, a tip `d` ahead caught at `2d`, the hot step;
head-on rival tips; `crash`; a wave through a twice-visited tile takes one chord and leaves the other; a wave stops at a
loose end; tap refusals (each `Refusal`); `fuel`: a tap lays `fuelMax` chords and `retap` extends; `oneWay`; per-owner `go`
pulses; `H` pulses exactly once per hit.

**Invariants, every step, in fuzz** (random legal taps of 2-4 rules on l2/l3 for 2,000 steps, 40 seeds): the §2.2
invariants (`rule ⇔ D∨W`, `T ⇒ D`, `D[d] ⇒ D[pt(d)]`, no tile holds two rules, `owner = ownerOf[rule]`); every drawn chord's
continuation across an edge is drawn under the same rule or that end is a tip or a tail (lines are stretches of strands:
checked with `walk`); no `W` lasts two steps; no tip waits more than `period` steps with an empty tile ahead; `step` ==
`stepFull` (engine twin); after a hit both lines' chords are gone within `L + 2` steps and no other line lost a chord
(collateral 0); `closed` ⇔ `components().closed`; `countTips` never exceeds `maxTips` per owner.

**Parity** (§2.10): (a) `walk` at double time, 300 taps per level; (b) sim.py's CA mode on `draw_taps` episodes of every
kind (single, collide, control, own), 200 each, levels 2-3: accepted taps equal, every chord's on/off within ±1 step, rest
state equal; (c) the engine on `collide.json` (400): accepted, survivors and coverage at rest, mismatches classified.
`tests/game-parity.test.ts` runs (a) and (c) from committed fixtures; (b) is `python -m nca.strand.ca_parity` on the Pi.

**Area** (`tests/area.test.ts`): walls derived from a `LineCA` equal the per-owner line tiles; a loop's fill never covers a
wall cell or an off-board cell; fill ⊆ the oracle's enclosed set after settle on 50 strand loops at level 2 (`targets()`
from `src/nca.ts`); two players' fills resolve to `territory` with contested cells marked; a pocket case documented.
**Game** (`tests/game.test.ts`): a scripted two-player game at level 2, 500 ticks, deterministic replay byte-identical;
respawn blocks taps for `respawnMs`; score = line tiles + sole fill at every read.

### 5.1 Parity, measured (D2, D3; `LineCA` of PR #33)

`npx tsx scripts/game-parity.ts` replays a fixture of timed taps (collide.json's format) through `LineCA` and writes
every chord's on and off step; `python -m nca.strand.ca_parity` runs (b) and (c) and gives each difference a class;
`tests/game-parity.test.ts` runs (a) and pins (c) to `tests/fixtures/ca-collide-classes.json` (collide.json itself is
now under `tests/fixtures`). All measured, at `maxTips` 99 (the engine's `maxHeads` 12 never binds on the fixture).

- **(a) the walker at double time:** 900 single taps (300 each on l2, l3, l4; random rules; taps at t = 0-3). Every
  chord k ≥ 1 from the tap (a loop: the shorter way round) is drawn at t + 2k for a tap on an even step and
  t + 2k - 1 on an odd one (the go steps are even, so the first chord comes one step *early*, not late as §2.10
  says). At rest the line is the whole strand: 900/900.
- **(b) sim.py's CA mode:** 1,000 episodes, 200 each of `draw_taps`' four kinds plus `busy` (3-6 players, up to 4
  taps each, as collide.json's), L2/L3, seed 1. Equal at rest (accepted taps and every chord): single 200, control
  200, own 199, collide 175, busy 144. On and off steps within 1 of the sim's: on 99.9 %, off 80 %. The off steps
  come 1-3 steps early because the CA's collisions are early (below). Every difference is timing.
  `nca.strand.ca_parity._CATiming` is the sim's CA mode with the CA's own timing added, as seven switches (phase
  and the six below). Against it, `LineCA` matches every chord's on and off step in 1,000/1,000 episodes, and in
  8,500 more (seeds 2-4 on L2/L3, 7,500; seed 5 on 1,000 L4 crops). 0 unexplained.
- **(c) Spectacle's engine** (collide.json, 400 episodes):

  | measure | result |
  |---|---|
  | accepted taps equal | 359/400 |
  | end state equal | 339/400 |
  | both equal | 327/400 (sim.py's CA mode: 318) |
  | survival agrees | 1,702 of the engine's 1,800 lines (1,070 survive) |
  | chords at rest | engine 15,020, `LineCA` 13,261, both 12,791 |

  The 73 differing episodes:

  | class | episodes | what it is |
  |---|---|---|
  | as sim.py's CA mode: waves | 35 | `LineCA` = the sim at rest; instant wipes (`wave=0`) would make it the engine |
  | as sim.py's CA mode: waves + head timing | 12 | the same |
  | as sim.py's CA mode: head timing | 4 | the engine's second-head bookkeeping (sim_parity) |
  | the CA's own timing | 22 | phase 16, phase + early 2, early 1, hitter 1, early + cold 1, phase + early + tap window 1 |

  0 unexplained. Against the sim with the CA's timing, every chord's on and off step is equal in 400/400.

**The wave-vs-join race is not the only divergence at rest.** These are the CA-only events in the 47 wave-caused
episodes; one episode can have several:

| event | episodes | what happens |
|---|---|---|
| worm hits | 27 | a dying line's fleeing tip hits a third line before its wave catches it, and the third line dies |
| taps refused on a dying line | 18 | |
| hits on a dying line | 16 | |
| dying-tile hits | 6 | |
| contact | 5 | §2.10's join race: a line that joins a dying line goes with it |
| waves met head on | 1 | |

All of these are the wipe taking a step a chord: the "a race only" rows of §2.10 do change rest states.

**Timing beyond §2.10's table.** Each item is a `_CATiming` switch, and each one is needed somewhere:

- **early:** collisions do not wait for `go`. `aimed`, `hitAt`, `victim` and `crash` read tips on any step, so a
  tip placed on a go step collides on the next step, one before it would move.
- **hitter:** a hitter's last chord goes on the hit step, both ends and a fresh tap's other tip with it. W goes
  out of both ends.
- **tap window:** W refuses a tap for exactly one step: on a wiped chord, on a crash tile, and where a wave is about
  to cross in (`waveAt`, PR #33).
- **wait:** a tip facing a hot or hit tile of its own rule waits; it does not die.
- **cold:** W leaves only by ends no wave came in by, so a chord that two waves meet in leaves its tile free the
  next step.
- **cross:** a wave reaching a tile on the step a tip of its rule draws there finds nothing and is spent. The
  newcomer lives on, as it would in the engine, where the old line is already gone.

## 6. Work breakdown

Agents A-D in parallel from hour 0; interfaces are §4.1. Times are coding-agent estimates.

| # | Task | Owner | Est. | Depends on |
|---|---|---|---|---|
| A1 | `line-ca.ts`: channels, `partner` tables, the §2.3 update as a `Rule`, `tap`/`retap`, readouts | A | 4 h | — |
| A2 | `components()`, the probe (§2.9), `fuel`, `oneWay`, `go` | A | 2 h | A1 |
| A3 | `tests/line-ca.test.ts`: the unit cases and the fuzz invariants of §5 | A | 3 h | A1 |
| B1 | `scripts/area-probe.ts` (§3.6) → numbers into this doc; the fine-tune go/no-go | B | 1.5 h | — |
| B2 | `area.ts`: frame translation, K instances, `update` from `changed`, `territory`, `tests/area.test.ts` | B | 3 h | — (uses a stub `LineCA` until A1) |
| B3 | `nca/data.py` kind `strand`, the plan line, launch, evaluate, export (only if B1 says so) | B + lead | 3 h + ~$1 | B1 |
| B4 | `area-gl.ts` (§4.3), parity with `cpuArea` | B | 6 h | B2 |
| C1 | `src/draw.ts` lifted from `web/nca.ts`; `host.ts` with stubs; `web/game.html/ts`: board, players, rule picker, taps, HUD, knobs | C | 5 h | — |
| C2 | integrate A and B; bot tapper; channel views; `scripts/build-web.ts`; `tests/game.test.ts` | C | 3 h | A1, B2 |
| C3 | v1.1 conversion (§3.4), `GameEvent`s for sparks, hysteresis | C | 2 h | A2, C2 |
| D1 | `strand-export.ts`: add Spectacle's tile index per cell to `strand-data.json`; regenerate `data/strand-v2` (`--rule-table --collide`) | D | 1 h | — |
| D2 | `scripts/game-parity.ts` → `ca-collide.json`; `tests/game-parity.test.ts` for (a) and (c) | D | 3 h | A1 |
| D3 | `nca/strand/ca_parity.py`: the sim CA-mode comparison (b), mismatch classes | D | 3 h | D2 |
| — | Integration day: all tests green, the page on Pages, the owner plays level 3 with two bots | lead | 0.5 d | all |

Critical path: A1 → A2/A3 and D2 → D3 (about 10 h); C and B run beside it. The first two hours give B1's number and A1's
skeleton, which decide B3 and confirm the state layout before anyone else depends on it.

## 7. Risks, each with its cheapest check

| Risk | Cheapest check | If it bites |
|---|---|---|
| The flood is poor on strand-shaped walls (tiny loops, corridors, many loops of one owner) | B1, hour 1 | B3 fine-tune (≤ $2); meanwhile the UI shows fills with hysteresis and the score tolerates it |
| The wave-vs-join race makes the engine parity noisy | D2's mismatch classes on the 400 episodes: expect a few percent, all classified | accept and document; or `period` 3 so joins lose more races the same way the engine would |
| A rule I have not seen breaks the "two stretches meet only end to end" argument (§2.6), e.g. class-0 chords | the fuzz invariant "lines are stretches of strands" over all 7 subsets (hex class 0 is a midpoint contract — measured — so it should hold) | treat the case as `crash` |
| The probe's 120 bits make the engine's active set slow on long loops | A3's timing on the level-3 infinite-line rule (loops of 400+) | probe off by default; `components()` for the UI |
| Pockets (§3.1) are common enough to be felt in tap refusals | count free cells with 6 line neighbours outside the polygon in 100 bot-game snapshots (10 min with `allStrands`) | treat 6-walled single cells as not-inside in `territory` |
| Settle time makes the area feel laggy at level 3 | the page at 20 line steps/s and `floodPerStep` 1-4 | more flood steps per frame; B4 |
| Float flood differs across devices (not bit-reproducible) | roll `cpuArea` on two machines, compare thresholded fills | the host that runs the flood is authoritative, as Spectacle's server is |
| `data/strand-v2` is a symlink into a scratchpad and vanishes | D1 regenerates it (~1 min) | commit `collide.json` (766 KB) under `tests/fixtures` |
| Level 4+ is wanted soon | §4.2 | B4 for the step; the settle time needs a different fill — a design of its own |
| The engine's `write` accepts only `input` channels | A1's first hour | declare the tap-written channels writable, or add `writeState` |
