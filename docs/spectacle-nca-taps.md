# The tap as a state set, lines that last, lines that die (2026-10-07)

The next stage of the strand NCA after `spectacle-nca-options.md` (the input rule, option E) and launch 9
(`strand-data.md` "Launch 8's plateau"; `nca/cloud/plan-9.txt`). Today the trainer HOLDS the tap on its cell
every step and teaches the line to regrow from it after damage; the owner wants the opposite: the tap is an
event that sets state once, the line keeps itself alive, and when another pattern hits it, it disappears
(Spectacle's `mutualCut`: both lines go). This is a design for the owner to review and a coding agent to
build from; nothing here is implemented. **Measured** = read from code or logs today; the rest is judgement.

Where things stand (plan-9, option E, depth 2, held-out rules, levels 3-4; measured from the l4 logs): exact
0.29 / 0.27 / 0.31 for E at hidden 128 / E-bc / E at hidden 256, short strands ~0.55, medium and long ~0;
right-exit rate 0.81-0.86 (1-chord cells 0.95, 2-chord 0.73-0.80, 3-chord 0.41-0.45). The code-fidelity probe
reads 85 % of the 53 code bits from a line cell's hidden state but has every bit right on only 0.4-2 % of
cells. So two things are unfinished before anything below: routing (which exit pairs with the entry edge) and
carrying the code faithfully. The design here does not fix them; it changes what the line is asked to do with
them, and the compute plan keeps them as the first thing to read on the dashboard.

## 0. Decisions for the owner

**Owner's answers (2026-10-07): all recommendations accepted.** On the hit rule: "pattern owns the
tile" -- a rival's line owns its whole tile for growth as well as for taps, so the tile rule is the
game's rule, not only the CA's; Spectacle's default `crossingMode` moves from `geometric` to `tile`
(a Spectacle PR, the owner merges). Both tap arms (impulse and fixed write) run in launch 10.

| # | Decision | Recommendation |
|---|---|---|
| 1 | **How the tap enters.** A one-step input impulse (the 59 tap planes on the tapped cell for one step, then zero), a fixed write of the code and the tapped chord into named state channels, or a learned linear encoder added to the state once | **Impulse as the product, fixed write as a diagnostic arm.** The encoder is dropped: the update's first layer already *is* a learned linear map of the tap planes, and the impulse lets the write depend on what is already on the cell (§1) |
| 2 | **Who is who.** No owner channel this stage: in Normal mode rules are unique per player, so *same code = same player* (owner's word). Own-line rules apply between lines of one code, collision rules between different codes | Yes. Conquest's captured patterns (your line, their code) break it; that is the owner slot of the options doc §4, later |
| 3 | **When a tip "hits".** Spectacle's default is geometric: the entered tile holds a rival chord that crosses the tip's chord or shares an edge midpoint with it (`chordsConflict`, `touchCounts: true`); a non-conflicting rival chord on the tile is passed over. The alternative is the engine's `crossingMode: 'tile'`: any rival line on the tile | **The tile rule for the CA**, and the engine run with `crossingMode: 'tile'` for the parity fixture. Reason: it is the one rule a single-code cell can decide (§2, case 5); the geometric rule needs cells that hold two lines of two codes. Taps already use the tile rule (`chordBlocked`: a rival's line owns its whole tile) |
| 4 | **Growth speed.** A wipe in Spectacle is instant; in a CA it is a wave at one tile a step. A tip that grows one chord a step runs away from the wave for ever (§2, case 6). Either cap growth at **one chord per two steps** so the wave catches a fleeing tip, or keep one a step and accept a cut line's tip running on as a "worm" until its strand ends | **Cap growth at 1/2.** It is the game's own notion (lines grow at a finite pace, `stepIntervalMs`), it is the only setting where "a hit line disappears" is locally true, and it gives the per-cell update two steps per hop for the routing it gets wrong today. The worm is the fallback if the cap proves hard to learn (§5) |
| 5 | **Two-way growth stays.** Spectacle's tap grows one way (a random end); the CA grows both ways from the tap, as the data, eval sets and legacy numbers assume | Keep two-way this stage. A one-way tap is one more bit on the tap event (which end), a small later change |
| 6 | **What damage means now.** Drop all three of train2's kinds: "state" (a disc wiped, the line regrows from the tap) and "move" (the tap moves, the old line vanishes) both teach the tap-as-source habit; "edit" re-types a tile, which the game never does and has no defined answer once lines persist. Damage becomes: noise on line cells (the line must survive it), collisions (the line must die), and new taps landing mid-episode | Yes |
| 7 | **Size.** Channels 96, hidden 256, depth 2 (`tap-e2-w`'s shape) for both arms, warm-started from `tap-e2-w-l3`'s best checkpoint; the consts layout is unchanged, so the weights load | Yes; §3 has the bits and the probe that would say it is too small |
| 8 | **Compute.** ~$6.80 ≈ 16 Spot L4 hours. Launch 10: the two arms, ~6 h ≈ $2.5; launch 11: the winner, longer, ~6 h ≈ $2.5; the rest is Spot reclaims and margin | Yes, serially, with the owner's read between them (§4.6) |

## 1. The tap as a state set

A tap in Spectacle starts a path on the chord of the tile nearest the finger, drawn at once (`startStep`), then
`advance` does the rest. The tap event is: this cell, this rule (the 53-bit code), this chord (its two edges).
Those 59 planes exist today as consts on the tapped cell, held every step. Three ways to make them a one-time
event:

| | I. one-step impulse | F. fixed write | L. learned encoder |
|---|---|---|---|
| What happens | the 59 planes are consts for `--tap-steps` steps (1, or 2) after the tap, then zero everywhere | at the tap step the host assigns state: edge planes ch1-6 ← the tapped chord's two edges, named hidden channels ← the code (±1); the update runs from there. A FrameNCA gotcha: its directional groups are read rotated by the cell's frame, so a fixed write must go to *scalar* channels, and at 96 channels with the default 6 groups there are 47 of them — fewer than the 53 one-hot planes. So F writes the code in its 35-plane form (8 class bits + 9 digits in 3-bit binary, the options doc's T2) and `--dir-groups` stays as trained | a 59 × C matrix, trained with the model, maps (code, chord) to a state increment added once to the tapped cell |
| What the net must learn | to *latch*: write what it reads of the tap into its own hidden state within a step or two, then hold it (the first layer sees the planes directly, so any function of them can land in the ~83 hidden channels in one step; the second step adds a nonlinearity over state + tap) | to hold what was written and to copy it on; the format at the tapped cell is dictated, the format elsewhere is still its own | as I, minus the latching |
| Standing under the owner's rule | **the purest**: the tap stays an input event, nothing about the state is prescribed | the owner's own description ("force the state into the pattern and set some hidden thing"). My view: it dictates a *storage format* at one cell at one moment and does none of the computing (routing, copying, persistence, dying); it is the I/O contract, not a hand-crafted channel. Its real costs: it commits 53 of 83 hidden channels at that cell, and it overwrites whatever is there, so a tap on a tile with a line of yours (allowed per chord, §2 case 7) destroys that line's state unless the write is additive | learned, but it is I with the write forbidden to look at the cell's state — a strict subset of what I's first layer can do, for an extra module |
| Carry to the play page | `setTaps` becomes "taps for the next step": the consts' part of w1 is folded into a per-cell bias once per step (`src/strand-nca.ts` already folds it per set of taps), then cleared | `setTaps` assigns state; no bias | an extra matrix in the weights file |
| Pi test | the plan-9 benchmark: one rule, 4 taps, level 2, 32 channels, damage: train2 reached exact 1.0 at ~300 iterations with the held tap; run the same with the impulse and report iterations to 1.0 and **exact at ages 100 / 300 / 600 after the tap is gone** (persistence). ~15 min a variant | same | same |

Recommendation: I is the product and F the diagnostic arm of launch 10. If F learns and I does not, latching is
the bottleneck (fix: `--tap-steps 2`, or a short lr warm-up on the tap step); if both learn alike the question
is closed and launch 11 drops F. The impulse also answers the owner's capacity question in the only way that
counts: a net that can latch 59 planes in one step and carry them for 600 steps has the scope; one that cannot
would not be rescued by a different write.

## 2. The edge-case table

Conventions. A *cell* is a hex tile; its *chords* under a code are a function of (type, code) that the net has
to compute anyway (§2 of the options doc). The *line* through a cell is the set of its drawn chords. A *tip* is a
cell whose line ends there and is still growing. *Light cone*: news travels one cell a step, so a cell `k`
chords along a line from an event cannot act before `k` steps after it; targets are **don't-care** for
`slack` = 2 steps after that and fixed on either side. Growth: a chord `k` from the tap must not be drawn before
step `2k − 1` and must be drawn by `2k + slack` (decision 4; the same window both ways round a loop). A wipe
from a hit at step `h` must leave the chord `k` along the line drawn until `h + k`, don't-care to `h + k + slack`,
0 after. The episode generator (§4.3) computes these per edge as two planes, `on_at` and `off_at`, and the loss
is the don't-care-weighted MSE as now.

| # | Case | Spectacle | What the CA must do, locally | What the cell needs, and where it comes from | Target | Measure |
|---|---|---|---|---|---|---|
| 1 | A tip enters a tile holding a rival's line | with the tile rule every rival path on that tile is cut (`cutRivals`), and with `mutualCut` the hitter draws its step then goes too (`addStep`): both whole lines vanish in that tick | the entered cell sees a tip of code B heading at it across an edge while it holds code A: it becomes a *collision cell* — its chords go, and it starts a dying wave along its own line; the tip's cell sees its tip has nowhere to go and dies, starting the wave back along B | the neighbour's state: code (≠ mine), tip flag, heading edge; my own code and chords. All in the 7-tap neighbourhood | both lines: `off_at = h + k` along each from the hit; the hit cell itself `h` | **wipe completeness**: share of both lines' chords gone by `h + L + slack`; **false wipes**: chords lost on lines nobody hit |
| 2 | Two tips meet head-on | the first to move enters the other's tile: case 1. Same tick or not, both die | the two tip cells see each other: a tip of another code heading across the shared edge. Both die, both waves start | as 1; synchronous updates make it symmetric | as 1, `h` = the meeting step | as 1 |
| 3 | Two tips enter the same empty tile on one step | the engine moves players in turn: the first lays its chord, the second hits it (case 1). Both die | the empty cell sees two tips of different codes heading into it: a collision cell for both; it draws nothing lasting | two neighbours' tip states | both lines wipe from `h`; the entered cell 0 after `h + slack` (Spectacle draws the hitter's chord for one tick: don't-care) | as 1; plus the share of such cells drawn after `h + slack` |
| 4 | A tip meets its own player's line (same code) | never a collision. The loose end on the same chord: `join` (one line, the far end grows on). Otherwise `overlapOwnLines`: it grows over the top; chords of one rule on a tile never cross | same code → never die. A tip stepping onto a chord already drawn under its code is **absorbed**: tip flag off, nothing else changes (the far end, if growing, is already a tip). A tip entering a tile holding another chord of its code just draws its own chord beside it | the neighbour's code (= mine) and the drawn planes | the union of the strands; both tips absorbed when they reach drawn chords of their code; `off_at` never | exact on the union; **phantom tips**: cells still flagged tip past absorption time |
| 5 | A tip crosses a rival line on a tile where the chords don't conflict | allowed under the default geometric rule (chords that neither cross nor share an edge midpoint: e.g. (0,1) and (3,4)); the tap rule is already tile-level | **not allowed here (decision 3)**: a rival on the tile is a hit. Why: a cell holding two lines of two codes needs two codes in state, and every message about "my line" becomes "which of my lines". Possible later with a second code slot | — | as case 1 | the parity fixture runs the engine with `crossingMode: 'tile'` so this is never a disagreement |
| 6 | A line is wiped while still growing | the whole path, head included, goes in one tick | the dying wave runs along the line at 1 a step; the tip flees at 1/2; the wave reaches a tip `d` away after `2d` steps and kills it (decision 4). At speed 1 the tip would never be caught (a worm until its strand ends) | the wave: a neighbour with my code, dying, across an edge my chord uses | the fleeing part's chords drawn at `2k` are still due `off_at = h + k'` with `k'` their distance along the line from the hit, so the generator simulates both the tip and the wave per step | **worm length**: chords drawn by a line after its hit; should be ≤ `d` and then gone |
| 7 | A new tap lands on a tile with a line | refused on any rival's tile; on your own line's tile refused only on a chord your line runs on or crosses (`freeChord`), so a free chord of the same rule takes the tap | the host validates taps as the engine does; the CA never sees a refused one. A legal tap on a tile of your own code adds a chord and a tip beside the existing chord: the write must be additive (I does this by learning; F must OR the edge planes) | nothing new | no illegal taps in the data; legal own-tile taps are case 4 | the play page: refuse, and show why, as Spectacle does |
| 8 | A wipe wave meets a growing tip of the same line | n/a (instant) | the tip cell receives dying along its own chord: it dies, drawing nothing further | as 6 | as 6 | as 6 |
| 9 | A wipe wave meets a tile shared with another line | a rival cut wipes one *path*; another path of yours on that tile survives (separate paths), unless joined (then it is one path and all of it goes). Under the tile rule a hit on the shared tile cuts every line on it | the wave arrives across edge `e`: erase the chord using `e`, pass the wave out of that chord's other edge, leave other chords of my code alone. A cell the strand passes twice (14 % of strand cells) is reached twice, once per chord. A collision cell erases everything and sends waves along every chord | which edge the dying came in on; my chords | per chord `off_at`, from the generator's per-path accounting | **collateral**: chords of untouched lines lost within `L + slack` of a wipe |
| 10 | A line's state is perturbed (noise) versus genuinely cut | n/a | a cut arrives as a *neighbour's* dying flag with my code across my chord's edge; noise is on my own channels and no neighbour agrees with it. The line is a chain: two neighbours hold copies of the code and the flags, so a cell can repair itself by consensus (MNIST / EngramNCA) | the two line neighbours' states | noise: target unchanged; disc wipes are **not** trained — nothing in the game zeroes a cell but the CA's own wipe | exact at `age + 50` after noise at σ ∈ {0.1, 0.3} on the hidden channels; code-probe exact before and after |
| 11 | Later: capture relabel, conversion | capture (Conquest): owner changes, pattern stays — a relabel along the line. Conversion (Normal): the enclosed rival line is wiped and the captor's pattern sprouts on its tiles | a *code change travelling along my own chords from a cell that was my line* — the second kind of message in case 12. Needs the owner slot (decision 2) for capture; conversion is a wipe plus a sprout with the enclosing circuit's code, which the territory fill would have to carry inward | the enclosing line's code at the boundary; later | out of scope | — |
| 12 | The owner's question: "meeting a cell that has a particular line versus that line just changing" | the engine never confuses them: it has every path's identity | **distinguishable locally, given the state carries code + tip/heading + alive.** A rival arrives *across* an edge, from a cell with a *different* code, as a *tip heading at me*. My own line changes *along* one of my chords' edges, from a cell that *held my code* last step, with *no tip pointed at me*. Three independent tells in one 7-tap read. The one configuration that breaks it is a neighbour holding two lines of two codes — excluded by decision 3 | — | — | the cases above are its measurements: false wipes (a change mistaken for a hit) and phantom survivors (a hit mistaken for a change) |

**The key argument.** Every line cell can recompute its own chords from (type, code), so "which edges are
mine" is never in doubt, and every message a cell receives is tagged by the sender's code. Hence a cell can sort
what it sees into: *mine, along my chord* (dying, or later a new code: my line changing), *not mine, across an
edge, a tip aimed at me* (a hit), *not mine, not aimed at me* (a rival passing nearby: ignore), *mine, across an
edge my chord does not use* (another line of my code touching: ignore, draw beside it). Two things are genuinely
ambiguous: (i) a cell holding lines of two codes — the tile rule removes it by making it a collision; (ii) a cell
whose *own* code is wrong (noise) versus legitimately changed — resolved by its line neighbours, who hold copies,
which is why noise damage is kept. Timing is the remaining softness: a CA cannot do "instant", so every target
above is a light cone with slack, and the one place that changes the game visibly — a cut line's fleeing tip —
is what decision 4 is for.

## 3. State and capacity

What a line cell has to hold, per cell, between steps:

| | Bits | Note |
|---|---|---|
| The code | 53 planes, 21 bits of information (log₂ 1.95 M) | the whole code, not just this tile's digit: the next tile may be any type. The net may compress it; the probe below checks what it keeps |
| Which chords are drawn | 6 edge planes (output) | up to 3 chords; re-derivable from the code, but kept as the drawing |
| Tip and heading | 1 + 3 (which edge it will leave by; two tips at a two-way start) | |
| Alive / dying, and which chord the dying is on | 1-2 | |
| Growth phase (the 1/2 speed) | 1 | a bit that flips each step |
| Later: closed flag (6 planes), owner slot (2-4) | | M1b's planes exist in the layout already (ch7-12) |

About 30-65 bits in 83 hidden channels clamped to [-2, 2] — room enough in principle, at hidden 128 or 256. The
measured problem is not room but fidelity: the linear probe decodes 85 % of code bits, 0.4-2 % of cells whole;
E-bc (the code on every cell) routes slightly better than E, so what is lost is the code in flight. Hidden 256
helps a little on both counts (exit 0.86 vs 0.81, code-exact 2 % vs 0.4 %). Recommend channels 96, hidden 256,
depth 2 (`tap-e2-w`'s shape, checkpoint loadable; memory ~7 GB an arm at batch 16, window 48, by
`memprobe`); not more channels this launch — a wider state needs a cold start, and the budget has no room for
two cold arms. The capacity test: (a) the code probe made **non-linear** (a 2-layer MLP from hidden state to the
53 bits, minutes on the pool) and reported exact by distance from the tap and by age — if the MLP probe reads the
code whole and the linear one cannot, the code is there and the net's own readout is the problem; if neither can,
widen; (b) **persistence**: exact at ages 2× and 4× ideal after the tap is gone, from the pool — decay here
with the code still readable is a holding problem, not a capacity one.

## 4. Training plan

### 4.1 Remove

train2's damage: "state" (regrow from the tap), "move" (vanish without the tap), "edit" (re-typed tiles). The
held tap. The `a` / `b` / `oc` planes and `erase_after`, replaced by `on_at` / `off_at` (§2).

### 4.2 Keep

Option E's `FrameNCA` and inputs (the tap planes as the only non-static consts); depth 2; `--last-k 48`; the
pool with ages, worst-slot restart and fresh slots (the pool is what asks a line to still be there 600 steps
later); the quick check, `q.exit`, `q.code` (+ the MLP probe), `bySubset`, `byLen`; warm start from
`tap-e2-w-l3` (the consts layout is unchanged, so `--init` loads; the routing it half-learned is the hard part
and carries over; the regrow habit is what the first stage untrains). From scratch only if the first check shows
the warm start clinging to the tap — see §5.

### 4.3 Episode generation: `nca/strand/sim.py`

A small event simulator over the walker: taps with times `(cell, chord, code, t)`; per step, each tip advances
one chord every second step (the cap), a step into a cell holding another code is a hit, a hit starts dying
waves along every line on that cell and back along the hitter at one cell a step, a tip reached by its wave dies,
a tip stepping onto a drawn chord of its own code is absorbed. It outputs, per sample, the 6-plane `on_at` and
`off_at` (int16, INF = never) for the loss, the tip timeline for the metrics, and the list of lines with their
fate. Single-strand episodes must reduce to today's walker targets with the speed doubled (a unit test against
`walker.walk`). Taps are drawn so that the pair's strands share a tile and the hit lands inside the gradient
window (the walker knows both strands, so the intersection is known before the episode runs); same-code pairs
are drawn so their strands meet at a loose end (join) or beside each other (two chords of one rule on one tile).
Throughput: today a fresh sample costs ~1 ms; two strands plus the timeline should stay under 5 ms (bench it;
`dataSec` is already 25-30 % of an iteration).

**Parity with Spectacle.** `scripts/strand-export.ts --collide`: the engine (`shared/game/engine.ts`, imported
read-only by path as the exporter does today) on the parity boards with knobs `{mode: 'normal', crossingMode:
'tile', mutualCut: true, overlapOwnLines: true, maxHeads: 12, speedOffset/baseStepMs for a constant pace,
scoreTiles: true}`, a scripted sequence of taps, ticked to rest; written per episode: the chords each surviving
path covers and which paths were wiped by whom. `python -m nca.strand.sim --parity` replays the same taps and
compares survivors and coverage (not timing: the engine is instant where the CA is a wave). Spectacle's tap grows
one way (decision 5), so the fixture taps the engine twice per CA tap — once per end (`tapOwnLine` turns a line
round; simpler: two paths started on the two ends, both owned by the player). ~2 h; the only piece that can
tell us the sim's rule is Spectacle's rule.

### 4.4 Curriculum

| Stage | What | Episodes | Pass |
|---|---|---|---|
| **T1 persistence** | one strand, one-shot tap, 1/2 speed, noise damage (σ 0.1-0.3 on hidden channels of line cells, p 0.3 per kept slot); the pool ages to 4× ideal | as today, `on_at` doubled, `off_at` INF | exact ≥ plan-9's at the same iteration count on the legacy set (no regression from losing the tap); persistence at 4× ideal ≥ 0.95 of exact at 1×; `q.exit` not below plan-9 |
| **T2 collisions** | 2-4 taps of different codes, staggered tap times (0 to half the window), chosen to collide inside the window; a quarter of slots with non-colliding pairs (control) | sim episodes | wipe completeness ≥ 0.9 by `h + L + slack`; false wipes ≤ 0.02 of chords; no regrowth (chords on a wiped line drawn after `h + L + slack`) ≤ 0.01; worm ≤ `d` |
| **T3 own-line meetings** | 2-3 taps of one code: joins at loose ends, chords beside each other, a legal second tap on a line's tile; mixed with T2's episodes | sim episodes | exact on the union ≥ T1's exact for the same lengths; phantom tips ≤ 0.02; collateral ≤ 0.02 |

Levels: T1 on 2 then 2 + 3 (level 4 crops only in launch 11); T2 and T3 on 2 + 3. Each stage `--init`s from the
previous best and re-mixes a third of its slots from the earlier stages, so T1's persistence does not decay while
T2 trains. The m1b closed-flag task waits until T3 holds.

### 4.5 Metrics (the quick check)

Unchanged: `exact`, `balanced`, `q.exit`, `q.code` (+ MLP probe, `byAge`). New, from the sim's timeline on
held-out rules: `persist` (exact at 2× and 4× ideal, tap gone), `speed` (chords a step per tip, expected ≤ 0.5),
`wipe` (completeness at `h + L + slack`), `falseWipe`, `regrow`, `worm`, `phantom`, `collateral`, each as in
§2's table. Dashboard: `q.exit.rate` and `persist` first (they move early), `wipe` once T2 starts.

### 4.6 Compute

~$6.80 ≈ 16 h of Spot L4 at ~$0.41/h (g2-standard-4). Launches are serial (one GPU quota); arms share the GPU.

| Launch | Arms | Stages | h | ~$ | Reads |
|---|---|---|---|---|---|
| **10** | I (impulse) and F (fixed write), both 96 / 256 / depth 2, warm from `tap-e2-w-l3` | T1 l2 45 → T1 l3 120 → T2 90 → T3 60 min = 315 + 25 setup | ~5.7 | ~2.3 | I vs F at the first T1 check (latching); `persist`; `wipe` / `falseWipe` after T2 |
| **11** | the winner alone (so ~2× the iterations an hour) | T2 + T3 mixed 180 → l4 crops 90 → m1b 45 | ~5.7 | ~2.3 | the §4.4 bars at level 4; the play page's `strand-meet` rerun |
| reserve | Spot reclaims (~$0.25 each: 25 min setup + ≤ 500 iterations), `evaluate` on the VM | | ~4 | ~1.6 | |
| margin | | | | ~0.6 | |

Fallbacks. A reclaim mid-stage: `startup.sh` resumes the chain from the last stage's `best.pt` (as plan-9's
chaining already does); two reclaims in one launch: drop T3 from launch 10 and shorten T1 l3 to 90 — T1 and T2
are the stage's claim, T3 is polish. Spot unusable for a day: on-demand g2-standard-4 is ~$0.70-0.75/h, so
launch 11 alone still fits (~$4) if launch 10 ran on Spot. Everything that is not a GPU run — the sim, the parity
fixture, both Pi overfits of §1, the selftest, the MLP probe — runs on the Pi for free and should be done before
launch 10 starts, so the GPU hours buy only the question that needs them.

## 5. Risks

| Risk | Cheapest test | Fix if it bites |
|---|---|---|
| The impulse is not latched: a one-step input is ignored, the line never starts | the §1 Pi overfit (15 min): iterations to exact 1.0 vs the held-tap 300 | `--tap-steps 2`; the F arm; as a last resort F as the product (an I/O convention, not an algorithm) |
| The line decays once the tap is gone (persistence) | `persist` at the first T1 check; the pool's oldest slots in `pool.npz` | longer pool ages; noise on the code-carrying channels (EngramNCA) so the copy is redundant; the MLP probe says whether the code or the drawing decays first |
| The 1/2 speed cap slows learning or confuses routing | the Pi overfit with doubled `on_at`: iterations to 1.0 and the settle ratio | the worm semantics (speed 1, a cut tip runs until its strand ends), with `worm` reported so the owner sees the cost |
| Dying leaks across codes (false wipes) or along touching own lines (collateral) | T2's control pairs (non-colliding) and T3's beside-each-other pairs; `falseWipe`, `collateral` | more control episodes; a longer T1 remix share |
| The hit line's wave never reaches a tip (phantom survivors) | `wipe` completeness split by "victim was growing / stuck" | the speed cap is the fix; if the cap is learned loosely (speed ~0.6), tighten `on_at`'s not-before to `2k` |
| The warm start clings to the tap (reads the code from consts, not state) | the first T1 check's `exact` vs plan-9 at the same iteration | a from-scratch arm in launch 11 instead of F |
| The sim's rule differs from the engine's | the parity fixture, before launch 10 | fix the sim; the knobs chosen (decision 3) make the engine match a one-code cell |
| Routing stays at 0.8 and nothing above is reachable at medium length | `q.exit.c2 / c3` through T1 | this doc does not fix routing; the routing probe says depth 2 learns it at 3 M samples — give T1 l3 the hours before spending any on T2 |
| `dataSec` doubles with two-strand episodes and the timeline | `rules --bench`-style bench of `sim.py` | pre-simulate per slot in a worker (the review's item 5) |
| Spot reclaims eat the budget | the ledger | §4.6's fallbacks |

### Build list (a coding agent; two in parallel ≈ 6-7 h)

| # | Item | Where | Est. |
|---|---|---|---|
| 1 | `sim.py`: taps with times, 1/2-speed tips, tile-rule hits, dying waves, absorption, `on_at` / `off_at`, the timeline; unit test: a single strand = the walker's targets at double time | `nca/strand/sim.py`, `tests` | 2.5 h |
| 2 | Spectacle parity fixture: `--collide` in the exporter (engine with the §4.3 knobs, scripted taps, survivors and coverage), `sim --parity` | `scripts/strand-export.ts`, `sim.py` | 2 h |
| 3 | `train3.py` (train2 with: `--tap impulse|fixed --tap-steps`, `on_at` / `off_at` loss, damage = noise + sim events, multi-tap slots, stage flags T1 / T2 / T3, the remix share); `selftest3` on the Pi | `nca/strand/train3.py`, `selftest3.py` | 3 h |
| 4 | Metrics: `persist`, `speed`, `wipe`, `falseWipe`, `regrow`, `worm`, `phantom`, `collateral`; the MLP code probe with `byAge`; dashboard series | `train3.py`, `nca/dashboard` | 1.5 h |
| 5 | The §1 Pi overfits (I, F, held) and the speed-cap overfit; numbers into this doc | `selftest3.py` | 1 h |
| 6 | Play page: tap events for one step, bias per step, `setTaps` for F; `strand-meet.ts` rerun on the new weights | `src/strand-nca.ts`, `web/strand.ts`, `scripts/strand-meet.ts` | 1 h |
| 7 | `plan-10.txt`, launch, watch | `nca/cloud/` | 0.5 h |

Not in this document: implementation, Conquest (captures, the owner slot), conversion, one-way taps, rule
changes and regrow, the closed flag beyond its place in the curriculum, and routing itself.

## 6. Built: the trainer side (2026-10-07)

Build items 3-7. Items 1-2 (`nca/strand/sim.py` and its parity with Spectacle's engine) are PR #24. This section
records what was built, where it departs from §1-§5, and the Pi overfits of §1 and §5. **Measured** as above.

| Piece | Where |
|---|---|
| The trainer | `nca/strand/train3.py`; `train2.py` is unchanged (launches 6-9) |
| Episodes: the adapter over `sim.py` | `nca/strand/episodes.py` |
| Self-check; the §1 overfits (`--overfit`) | `nca/strand/selftest3.py` |
| Training memory | `nca/strand/memprobe.py --train3` |
| Play page | `src/strand-nca.ts`, `web/strand.ts`, `nca/strand/export.py` (weights version 2 carry `tap`) |
| Two patterns meeting | `scripts/strand-meet.ts` (event taps; counts "both lines gone") |
| Dashboard | the default strand series |
| Launch 10 | `nca/cloud/plan-10.txt` |

**What it does.** `--tap impulse` puts the 59 tap planes on the tapped cell for `--tap-steps` steps (t + 1 ..
t + tapSteps), then zero. `--tap fixed` never sets the tap planes; at age t the host writes the cell's state, then
the update runs. The write sets edge channels 1 + d0 and 1 + d1 to max(itself, 1), so it adds to a line already
there (§2 case 7). It also sets the code's 35-plane form (8 class bits, then each digit as 3 bits MSB first, ±1)
into the last 35 channels, which are scalar: at 96 channels and 6 directional groups, channels 61-95. `--tap held`
is train2's tap, kept as the overfits' baseline.

The targets are `sim.py`'s. Every edge has up to K = 4 intervals [on, off) in nominal steps (chord k of a tap at
t + 2k). The windows are `sim.target_weight`'s: required from on + 2, don't-care from on - 1 and for 2 steps after
off. A slot holds up to four taps with their times. Episodes come from `sim.draw_taps`:
- t1: one tap.
- t2: 2-4 codes that collide. The draw is repeated until the first hit is by `--hit-max` (48).
- control: a quarter of t2 slots, two codes whose strands share no cell.
- t3: own meetings, redrawn until two taps stood.

Taps the host refuses are dropped from the slot, since the CA never sees one. Stages are T1 / T2 / T3 with
`--remix` (1/3). Damage:
- noise on the hidden channels of line cells (p 0.3, σ U[0.1, 0.3]);
- the episodes' collisions and staggered taps (by age `--stagger`, 24);
- from T2 on, a mid-episode tap on 10 % of kept slots. In T2 it is a rival on a strand through the slot's lines;
  in T3 it is half own, half rival. The slot is re-simulated and its past is unchanged.

**The metrics** (`train3.py`'s docstring has the exact definitions; `selftest3` checks every one against an oracle
that draws the targets, and against variants that decay, never wipe, regrow or drop part of a meeting):
- `speed`: from the run of chords drawn out of each tap.
- `persist` `x1` / `x2` / `x4`: exact at 1, 2 and 4 × each tap's ideal, on 48 level-3 taps.
- `collide`: wipe, falseWipe, regrow, worm against wormMax.
- `own`: collateral, phantom, stray.
- The code probes: the linear one and the MLP (`codeMlp`), each with `byAge` (x1 / x2 / x4) on the persist set's
  line cells.

The score that best.pt and the collapse guard use is the mean of balanced, persist.x4,
wipe × (1 − falseWipe) and own.exact, over those a stage measures.

### Where it departs from §1-§5

1. **Timing is `sim.py`'s**, not the "2k − 1 / 2k + slack" written in §2. Chord k is nominally at t + 2k. It is
   due from 2k + 2 and may be drawn from 2k − 1. A wipe at the step h + j the wave reaches the chord leaves it
   drawn until h + j − 1, don't-care to h + j + 1. Erased chords linger 2 steps. `--early -1` drops the
   not-before entirely. With `--speed 1 --slack 1 --early -1`, one tap's target is train2's exactly (checked at
   every age).
2. **phantom** cannot be "cells still flagged tip": the CA has no tip output. It is the share of own-line
   episodes whose drawing still changes after ideal + slack (a tip still running, or flicker).
3. **wipe** counts only the doomed edges the model drew while they were due. Otherwise a model that draws nothing
   scores 1.0. **worm** is reported next to **wormMax**, the most the targets allow (the chords due after the
   hit, given the slack), rather than as "≤ d".
4. **The overfits** use 48 channels, not 32. The fixed write needs 35 scalar channels after 13 + 6 × groups, so
   all arms use `--dir-groups 0`. They also use hidden 64 at depth 2, and lr 1e-3 (warm-up 50) to fit the Pi's
   time box. Their exact is read as train2 reads it, after max(8 S, ideal + 8) steps. "onTime" is exact at the
   ideal step itself.
5. **The persist set** is level 3 only in plan-10: at level 4 most taps' 4 × ideal passes the 1,200-step cap.
6. **The warm start's best.pt** is iteration 0 of tap-e2-w-l3, that is tap-e2-w-l2's last best. The l3 stage
   never beat it on the balanced score (0.217), although its ckpt.pt, 7.7k iterations on, has the higher
   right-exit rate (0.90 vs 0.85). plan-10 uses best.pt as §0 says; ckpt.pt is in the bucket if the lead prefers
   the routing.
7. **Draws** are `sim.draw_taps`'. Its collisions come early: the first hit at median age 2-3 on levels 2 and 3.
   Its own draws often place a second tap on a drawn chord, and the host refuses it. That is why "two taps stood"
   is a redraw condition.
8. **The broadcast inputs** (`e-bc`, `c-bc`) are refused by train3: the rule on every tile is not a one-time tap.
9. **The quick check** costs about 2.1× plan-9's (rollout work counted per set): legacy and wide at the doubled
   ideal are 1.86×, and persist plus the episode sets add the rest. Expect ~5 min a check on the L4 where plan-9
   took 85-155 s; plan-10 spaces checks ~20 min of training apart.
10. **`--hit-min 8` in plan-10's T2 and T3.** sim's collision draws hit at age 2 (the median; 87 % by age 7 at
    level 2, 69 % at level 3), before the lines have grown. Requiring the first hit at 8 or later moves the median
    hit to 13-18, with lines of 18-38 chords, so the waves and the fleeing tips are what T2 trains. Mid-episode
    taps cover lines that are hit late in life. The collide eval set uses the same draws.
11. **Pool ages.** `--new-share` and `--no-worst-restart` exist (§5's "longer pool ages"), but plan-10 keeps
    train2's defaults: on the Pi, keeping old slots stopped learning (below).

### The Pi overfits (§1, §5)

`python -m nca.strand.selftest3 --overfit --stage T1|T2 --arms ...` runs `train3 --overfit 4`:
- **Episodes.** One train rule, `023468·120000000`, with four level-2 episodes. In T1 these are single taps on
  strands of 24, 4, 7 and 4 chords (ideal 6-34 steps). In T2 the same rule collides with `128·011110001`; with
  `--hit-min 16 --overfit-len 4 40` the first hit comes at age 16-36 after lines of 10-38 chords have grown.
- **Net.** Option E at 48 channels with no hidden directional groups (the fixed write's minimum), hidden 64,
  depth 2. Batch 8, pool 32, bptt and last-k 48, lr 1e-3 (warm-up 50), noise damage 0.3. From scratch unless
  marked.
- **Budget.** 15 minutes an arm on the Pi, which was shared with other work at load average 5-6: 1.8-2.4 s an
  iteration, so 350-450 iterations. A check every 50 iterations.
- **Columns.** `exact` is read as train2 reads it, after max(8 S, ideal + 8) = 96 steps. `on time` is exact at
  the ideal step itself. a100, a300 and a600 are exact at those ages, with the tap long gone. steps = the settle
  step over ideal. One seed each, so 50-100 iterations is within the noise.

**T1 persistence.**

| Arm | Speed cap | Iterations to exact 1.0 | ... to on time 1.0 | At the end: exact / on time / a100 / a300 / a600 | steps |
|---|---|---|---|---|---|
| held (plan-9's benchmark) | 1/2 | 300 | - | 1.0 / 0.5 / 1.0 / 0 / 0 (350 it.) | 1.09 |
| impulse, 1 step | 1/2 | - (0.75 at 300-350) | - | 0.5 / 0.75 / 0 / 0 / 0 (400 it.) | 1.0 |
| impulse, `--tap-steps 2` | 1/2 | 400 | 450 | 1.0 / 1.0 / 1.0 / 0 / 0 (450) | 0.86 |
| fixed write | 1/2 | 450 | - | 1.0 / 0.25 / 1.0 / 0 / 0 (450) | 1.21 |
| held | none | 200 | 450 | 1.0 / 1.0 / 1.0 / 0 / 0 (450) | 0.90 |
| impulse, 1 step | none | 300 | - | 1.0 / 0.25 / 0 / 0 / 0 (350; at 300: a100 1.0, a300 0.75) | 1.43 |
| fixed write | none | - (0.5 at 250-400) | - | 0 / 0.75 / 0 / 0 / 0 (450) | - |
| impulse, old pool (`--no-worst-restart --new-share 1/16`) | 1/2 | - | - | 0 everywhere; loss 0.3-0.6 throughout (450) | - |
| held, old pool | 1/2 | - | - | 0 everywhere (450) | - |

**T2 collisions** (one pair of rules). Here `grown` is the share of the doomed edges the model drew at all while
they were due. `wipe` is measured on those edges. The targets allow at most 7.5 chords of worm.

| Arm | wipe | grown | regrow | worm | exact (best check) |
|---|---|---|---|---|---|
| impulse, from scratch | - (draws nothing in time) | ~0 | 0-0.82 | 0-22 | "1.0", empty: the hit lines are gone by the read-out, so drawing nothing scores |
| fixed write, from scratch | 0.33 -> 0.75 | - | 0.16 at 450 | 3.75 | 0.5 at 200 |
| impulse, `--init` its T1 run | 0 -> 0.84 (400), 0.77 (450) | 0.20 -> 0.42 | 0.57 -> 0.29 | 10.4 -> 4.6 | 0 |
| fixed write, `--init` its T1 run | 0.03 -> 0.86 (400), 0.77 (450) | 0.15 -> 0.67 | 0.53 -> 0.33 | 9.8 -> 2.9-5.1 | 0.5 at 400 |

What this says:

1. **The impulse latches.** One step of input starts lines and draws them on time: 3 of 4 episodes at 300-350
   iterations under the cap, all 4 at 300 without it. With `--tap-steps 2` it reaches 1.0 at 400 and is on time
   at 450, steadier than one step. The fixed write gets there at 450. The held tap, plan-9's benchmark, gets
   there at 300 under the cap; plan-9 reported ~300 for its held tap, and this one reached 200 without the cap.
   So a one-time tap costs little at this size, and "F learns and I does not" did not happen: no sign that
   latching is the bottleneck. If launch 10's first t1l2 check shows the impulse arm well behind tap-f,
   `--tap-steps 2` is the cheapest fix (§5). plan-10 keeps one step, as the doc chose.
2. **The 1/2 cap costs iterations but is learnable.** Held needs 300 instead of 200. The one-step impulse is
   noisier under it (its loss jumped at 400).
3. **Persistence holds only inside the ages the pool trains.**
   - Every arm, the held tap included, is exact at age 100 (3 × the longest episode's ideal) and wrong at ages
     300 and 600. The pool's slots live to about age 300, median ~150, under train2's renewal (1/8 of each batch
     new, and the batch's worst slot restarted every iteration).
   - Past that the held model does not fade, it overgrows. Over the four episodes there are 219 stray edges by
     age 200 and 468 by age 600, with 51 of 78 right edges still there.
   - The obvious knob makes it worse. Keeping old slots (no worst restart, 1/16 renewal) stops learning
     altogether at this budget: the broken old slots swamp the gradient.
   - So plan-10's persist set measures 1-4 × ideal, which at level 3 (100-200 steps a visit, ~5 visits a slot)
     lies inside the pool's ages. Ages far past the pool's, such as the 600 here, are §5's persistence risk,
     measured and open. The next candidates are a share of pool slots aged without gradient before they are
     trained, or a longer no-gradient prefix. Neither is built.
4. **Collisions need T1 first and more than the Pi's 450 iterations.**
   - From scratch the impulse arm finds the degenerate answer: draw nothing in time. A collision episode's lines
     are gone by the read-out, so nothing drawn scores exact, which is why `wipe` counts only drawn edges and
     `grown` is reported.
   - From scratch the fixed write is better off, since its write draws the tapped chord at once and gives the
     wipe something to act on.
   - Initialised from their T1 runs, as the curriculum does, both arms learn every part: they draw more of the
     lines before the hit (grown up to 0.42 and 0.67), wipe them (0.77-0.86), regrow less (about 0.3) and lay
     fewer chords after the hit (2.9-5.1 against the 7.5 allowed). Neither converged in 450 iterations.
5. **The warm start under one-time taps holds nothing.** `scripts/strand-meet.ts` (level 2, 150 training-rule
   taps, 220 steps) on tap-e2-w-l3's best.pt exported with `tap: impulse` or `tap: fixed` gets 0 taps exact when
   alone, both ways, so there are no pairs to meet. The default page weights (tap-e2-l3, held) are unchanged on
   the same script and arguments: 8 pairs, A 1.00 -> 0.97 when B arrives, and both lines gone in 0 of 8 (the
   new count; the game's answer is 8). The script's rerun on trained event weights waits for launch 10.

### Memory (launch 10)

`python -m nca.strand.memprobe --train3 --channels 96 --hidden 256 --depth 2 --batch 8 --window 4 20 --last-k 48`
measures the peak RSS per backprop step at batch 8 (saved tensors in brackets):
- level-3 boards (S 37): 52.5 MB impulse, 50.3 MB fixed (38.7);
- level-4 crops (S 42): 76.9 / 71.4 MB (50), against plan-9's 68-73 MB (49) for the same net in train2.

At batch 16 and window 48 that is ~5.0 GB an arm, ~11 GB for both arms with pools and CUDA contexts, of the L4's
~22.5 GB.
