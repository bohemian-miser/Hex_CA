# Review: the hex flood-fill NCA (2026-10-05, ~12:30 UTC)

A frank review of `nca/` as it stands today, with prioritised improvements and a GPU queue for the
next hours. Written while launch 3 (`ha-big` / `hb-big`) was running. "Measured" = numbers I read off
logs or produced with read-only CPU probes today (scripts noted); "inference" = my judgement.

## Summary

- **The approach is right and it works.** A plain NCA (walls + board mask in, fill out, learned 7-tap
  hex perception, pool with accumulating damage, min-over-acceptable-targets loss) learns loops,
  bridges and live edits. The hybrid detour was a budget error, not a modelling one; the journey doc
  says so and the evidence agrees.
- **The large-radius problem is smaller than the README says, and it is not a horizon problem.**
  Launch 3's `hb-big` checkpoint (trained only at R 5-10, lr 5e-5) already scores 0.85 bridge-exact
  at R 16 zero-shot (README's shipped model: 0.40) and 0.55 at R 24. Measured: every remaining R 16
  failure is a near-tie (area-ratio bins 1.00 / 1.00 / 0.83 / 0.33-0.44), the model reaches a fixed
  point by ~5.5R steps and running 32R steps changes nothing. So: the global *comparison* loses
  precision as boards grow; curriculum and near-tie data come first, capacity second, steps not at all.
- **Both live runs plateaued 45 minutes into a 290-minute box**, and neither will reach its lr decay
  (it is tied to `--iters` the box cannot reach). That is the one wasteful thing running right now.
  Owner's call whether to cut it; the review's queue assumes the GPU is free from ~14:00 UTC.
- **Evaluation hygiene is decent but the headline numbers hide the shape of the error**: the bridge
  set is 40 % lopsided boards (trivially right) and the near-tie bin carries all the failures; R 16 is
  n = 30 (±0.18); the only out-of-distribution set is "page loops".
- **The model does not work on Spectacle's board shape.** Measured today: with the level-3 hex patch
  (496 tiles, a ragged blob) as the mask, bridge-exact is 0.21 and "none" 0.79 (disc at the same R:
  0.79 / 0.13). It has only ever seen a convex rim. Ragged masks in the pool are the first item of
  any game-facing stage; the model needs no change.
- **The browser cannot run this model at game size.** Scalar JS at ~67 µs per cell-step means 0.7 s per
  step on Spectacle's level-4 hex field. A GPU path (WebGL2/WebGPU) or a smaller model is needed
  before any game milestone; a hidden-size ablation should run alongside the next stage.
- **Top five actions**: (1) finish large radii with a curriculum fine-tune from `hb-big` (R 6-16, then
  to 24), lr fixed low, decay actually reached, **with ragged masks in the pool from the start**;
  (2) add near-tie-heavy bridge data and A/B it; (3) evaluate with n ≥ 200 per radius, per-bin, at
  8R/16R/32R, plus a ragged-mask set; (4) split `secPerIter` into data vs GPU and move board
  generation off the training thread (GPU is at 65 %); (5) size ablation (hidden 64/128/256) for the
  browser's sake.

## 1. What is there (short)

| Piece | State today |
|---|---|
| Model (`nca/model.py`) | 24 state channels (ch0 walls re-imposed, ch1 fill, 22 hidden), 1 const (mask), masked 3×3 conv = 7 hex taps → ReLU(256) → 1×1 zero-init → residual; hard clamp [-2, 2]; synchronous (`fire_rate` 1.0, no CLI flag). 64,024 parameters counting the two masked corner taps; 51,224 effective. |
| Data (`nca/data.py`) | 8 board kinds (blobs, polys, rings, leaky, messy, open, noise, bridge), 25-30 % of boards with 2+ rim regions; targets = every enclosed region fills, all rim regions but one "largest" (max area, or rim-arc span within EPS 0.02) fill, every candidate acceptable; damage = edit / burst / erase / stamp / state. |
| Recipe (`nca/train.py`) | pool per radius (512), batch 32, worst sample restarts, batch/8 new boards, p = 0.5 one damage each; T ~ U[aR, bR] with BPTT through all T; loss = min over targets of fill MSE over the last 8 steps; Adam + per-parameter gradient L2 normalisation; lr step decay at 60 % / 85 % of `--iters`, warm-up, rollback guard; quick check every 200-500 its (mix / bridge / page / 3-edit sequence, n = 128 per set per radius on the GPU). |
| Evaluation (`nca/evaluate.py`) | fresh rollouts at 8R and 16R, exact / acc / IoU / trivial, bridge per area-ratio bin, both / none; one-edit test at R ≤ 12. |
| Tooling | `progress.py` (noise-aware verdict), dashboard, cloud scripts (plan, ledger, watch, budget cap), selftest, parity fixture for `src/nca.ts`. |
| Shipped | `runs/gb-r456b/best.pt` (R 4-6, 29k iterations): README table — bridges 0.91 / 0.89 / 0.77 / 0.40 at R 6 / 8 / 10 / 16. |
| Live | launch 3: `ha-big` (6R-10R rollouts) and `hb-big` (3R-6R), both from `gb-r456b`, R 5 6 8 10, lr 5e-5, 290-min boxes. At 12:13 UTC: PLATEAU, best quick score 0.975 / 0.982, bridges 0.95 / 0.96 all radii, R 10 bridges 0.72 → 0.93-0.97. |

## 2. What the numbers show and do not show

### 2.1 Measured today (read-only CPU probes on the Pi; the `evaluate` rows are the commands shown, the trace, area-vs-span and ragged rows are ~50-line throwaway scripts, not committed)

`hb-big/best.pt` (iteration 20,000, best quick score 0.982; the file was refreshed to iteration 39,000
by the time of the last probe — noted per row). `python -m nca.evaluate ... --sets bridge`:

| Probe | Result |
|---|---|
| R 16 bridges, n = 40, read at 8R / 16R / 32R steps | exact **0.825 / 0.850 / 0.850**; per-bin [0,.25) [.25,.5) [.5,.75) [.75,1] = **1.00 / 1.00 / 0.83 / 0.33→0.44** (n 17/8/6/9); both 0.000; none **0.125** at every readout |
| R 16 step trace, 24 bridge boards (mean ch1 on the side that should fill vs the side that should not) | fill side 0.38 @8 → 0.70 @32 → 0.86 @88 then **flat to 320**; empty side ≤ 0.04 throughout; exact-vs-primary 0.79 from step 88 on. The undecided boards are flat at zero from the start, not slowly rising |
| R 24 bridges, n = 20, 8R / 16R | exact 0.45 / 0.55; bins 1.00 / 0.2→0.6 / 0.0 / 0.0 (n 8/5/3/4); none 0.45 / 0.35; **page loops 1.00 / 1.00** |
| Area vs rim-arc on boards where the two acceptable targets differ (iteration 39k) | R 8: 24 boards in 85 draws (28 % of bridge boards!) — picks area 20, span 2, neither 2. R 10: area 15, span 5, neither 4 |
| CPU data path per board (Pi 5) | `targets` 1-6 ms, `damage_walls` 2-8 ms, `random_walls` 2-5 ms at R 6-24; per iteration ≈ 20 boards' worth ≈ 50-100 ms of Python |
| VM at 12:13 UTC | GPU util 65 %, 4.2 / 23 GB, load 2.3 on 4 vCPU with two runs |
| Ragged board (Spectacle's level-3 hex patch, 496 tiles, as the mask at R 19; area-only targets; iteration 39k) | bridges **0.21** exact, none **0.79**; disc at the same R: 0.79 / 0.13 — see §2.4 |

(Note on the eval sets: the `evaluate.py` bridge generator draws 1 board in 6 point-symmetric, so exact
ties are ~17 % of the top bin by construction; for a tie either side is acceptable, "none" is the
failure.)

### 2.2 What this says

- **Curriculum to R 10 at a low lr transfers to R 16 and partly to R 24** (0.40 → 0.85 at R 16 with
  nothing trained above R 10). The earlier "larger boards made it worse" was the learning rate
  (3e-4), not the radii — the journey doc notes lr and radii were changed together; the zero-shot
  numbers above are the missing half of that ablation (inference, but strong).
- **Failures are near-ties only.** Lopsided boards are 100 % at R 16 and R 24. The model has the
  right mechanism; what degrades with size is the *resolution* of its size comparison. (Measured.)
- **Not a horizon problem.** 8R → 32R steps changes nothing at R 16; the trace settles by 5.5R and
  stays. Longer rollouts at test time are pointless; longer *training* rollouts were the losing arm
  of launch 3 (`ha` 6R-10R vs `hb` 3R-6R: `hb` is ahead and 25 % cheaper per iteration). (Measured.)
- **Not "fails by filling no side" as a mode of its own**: "none" at R 16 is 0.125 and sits entirely
  inside the near-tie bins — the model abstains when its comparison is a coin flip. That is the
  better failure (a wrong fill in the game is worse than no fill). (Measured + inference.)
- **The learned rule is area, mostly.** On boards where area and rim-arc disagree the model follows
  area 4:1. Spectacle's `boundaryRegion` compares polygon *areas* too (`field.ts:618`), so the target
  the game needs is the one the model already prefers. Dropping the span alternative from the target
  would make the metric stricter but is cheap for the model. (Measured; the Spectacle fact is read
  from the code.)
- **Noise.** Quick checks at n = 128 per radius give 2se ≈ 0.025-0.03 on bridge exact; `progress.py`
  called PLATEAU on both runs while R 10 bridges were still moving 0.93 → 0.97 inside the averaged
  number. The README's R 16 row is n = 30: 0.40 ± 0.18. Any claim finer than ±0.05 needs n ≥ 300.
- **The mix metric is soft.** Its trivial baseline is 0.30-0.40 (a third of mix boards have nothing
  to fill), so 0.98 mix exact is ~0.97 on the boards that matter. Page loops are easy (1.00 at
  R 24 zero-shot) and are the only OOD set.

### 2.3 Telling the four hypotheses apart (cheaply)

| Hypothesis for the large-R failure | Discriminating test | Cost | What we already know |
|---|---|---|---|
| Horizon (not enough steps) | read the same rollout at 8R / 16R / 32R; per-step side trace | 4 min CPU | **Ruled out** at R 16 (flat from 5.5R) |
| Receptive field / communication (signal cannot cross the board) | lopsided-bin accuracy vs R; page loops at R 24 | done | **Ruled out**: lopsided 1.00 at R 24, loops 1.00 at R 24 |
| Precision of the global comparison (near ties) | per-bin table vs R; "none" share vs ratio | done | **This is it**: [.75,1] bin 0.77 (R 6) → 0.41 (R 10, shipped) → 0.33-0.44 (R 16) → 0.0 (R 24) |
| Curriculum / data (never saw large boards or enough near ties) vs capacity | fine-tune at R 6-16 with near-tie-heavy bridges (Q1 + Q3 below); if the top bin at R 16 does not pass ~0.8 in 60-90 min while R 8 holds, run the width ablation | 2 × 90 GPU-min | Open; the R 10 jump (0.72 → 0.97) once R 10 was trained on says curriculum first |

### 2.4 Ragged boards (Spectacle has no hexagon-shaped field)

Spectacle's hex field is an exact axial lattice (0 off-lattice centres at levels 3-6, measured with
`buildField`) but a ragged blob: level 3 = 496 tiles in a 30×37 box, level 4 = 3,905 in 101×109,
level 5 = 30,744 in 243×298. The current model has only ever seen a hexagonal disc; concave rims are a
local pattern it has never met. Probe (`scratchpad/ragged.py`): the level-3 patch embedded as the mask
at R 19, area-only targets, `hb-big` iteration 39k, 240 steps, n = 24 per row:

| mask | kind | exact | both | none | boards with 2+ rim regions |
|---|---|---|---|---|---|
| disc R 19 | mix | 0.917 | – | – | 0.46 |
| disc R 19 | bridge | 0.792 | 0.042 | 0.125 | 1.00 |
| disc R 19 | straight cut | 0.917 | 0.000 | 0.042 | 1.00 |
| **ragged (level-3 patch)** | mix | **0.333** | 0.000 | – | 0.96 |
| **ragged** | bridge | **0.208** | 0.000 | **0.792** | 1.00 |
| **ragged** | straight cut | **0.167** | 0.000 | **0.750** | 1.00 |

**The model does not transfer to a ragged board.** On Spectacle's own field shape it fills no side
of three bridges in four and gets a third of mixed boards right (the ragged "mix" is 96 % multi-rim,
so it is mostly a bridge test too). It never fills both sides, so the failure is again abstention.
A concave rim is a local pattern it has never seen; a disc's rim is always convex. This is the first
thing the next stage has to fix, and the fix is data, not architecture: random supertile patches and
crops as masks in the pool (the mask input already supports it; the trainer builds consts per radius
today, so masks become per board like walls — a small refactor). Measured; caveat: n = 24 per row,
area-only targets, one mask shape.

## 3. Review by area

### 3.1 Model — good, one open question

- Good: residual zero-init update, learned perception (MNIST paper: learned 3×3 beat fixed Sobel),
  hex-masked conv (the Textures paper's hex result says the square recipe carries over), the mask as
  the only const. Nothing hand-written remains in the default path.
- Risk: the hard clamp [-2, 2]. The 2e-3 collapse was weights growing until 75-85 % of hidden cells sat
  on the clamp with zero gradient. The literature's answers are an *overflow loss* (Textures, ViTCA:
  penalise |state| beyond the box, keep the gradient) or clipping at [-3, 3] with a loss on the vote
  channels (MNIST: L2 to a bounded target makes updates decay as agreement forms). The rollback
  guard hides the symptom; the fix is cheap to A/B (§5, item 6).
- Open: whether 24 channels is the right *communication* width. Pathfinding NCA got its best size
  generalisation from 48 channels with weight sharing and smaller hidden (22k params), i.e. more
  state, less MLP. The next stage's size ablation should vary channels and hidden independently.
- `fire_rate` exists in the model but the trainer hard-codes 1.0 and there is no `--fire-rate`.
  Every reference NCA trains at 0.5-0.8 and reports robustness from it; here synchronous was the
  right first choice (determinism for a game), but the comparison has never been run (§5, item 7).
- `--init` refuses any change of `channels`/`hidden`; `load_expanded` widens only consts/pool. A
  zero-padded widening loader (new channels and hidden units start at zero, old behaviour exact)
  would make capacity A/Bs a 10-minute fine-tune instead of a 2-hour restart.

### 3.2 Data — good generator, two gaps

- Good: the mix was tuned for 25-30 % multi-rim boards; the pool's accumulating damage is the single
  change that made bridges learnable (journey doc); targets accept every defensible answer.
- Gap 1: **near-tie coverage**. The bridge generator's arc length is `sqrt(u)`-distributed, so the
  cut-off area is roughly uniform in ratio; the failing bin [.75, 1] is ~22 % of bridge boards and the
  only hard part. A generator knob for the ratio distribution (e.g. half the bridges drawn with ratio
  ≥ 0.6) is an hour's work and is the cheapest lever on the remaining error.
- Gap 2: **board shape**. Only discs of radius R. The game needs ragged masks; the trainer's `consts`
  are built per R, so "a pool of masks" is a small refactor (mask per board, like walls).
- Minor: the span alternative in `targets()` is not what Spectacle computes (area). Keep it for the
  demo if the owner likes it; for the game stage use area only (see the plan).

### 3.3 Training recipe — good, three things to fix

- Good: pool + worst-restarts + damage, random rollout length (Graph-CA: fixed t gives cycles, random
  t gives fixed points), gradient normalisation, min-over-targets, the collapse guard, best.pt by a
  held-out score, the 3-edit sequence in the quick check.
- Fix 1: **the lr schedule never fires inside a time box**. Decay is at 60 % / 85 % of `--iters`; the
  plans ask for 200k-300k iterations in boxes that reach ~150k-190k. Either size `--iters` to the box
  (the README says to; the plan didn't) or decay on wall time / on the PLATEAU verdict.
- Fix 2: **BPTT through the whole rollout** (up to 100 steps at R 10, 160 at R 16) is the memory and
  time cost. `hb` showed 3R-6R with persistent pool states is at least as good — the Reasoning-NCA
  recipe (rollout 100 in chunks of 100 with a replay pool; 9×9 → 500× generalisation) is the same
  idea. Keep short rollouts and let settling span pool visits.
- Fix 3: **the data path is on the training thread.** ~20 Python board operations per iteration at
  2-8 ms each is the same order as the GPU step at 10 it/s; GPU util is 65 % with two runs. Either a
  worker process producing (walls, targets, damage) for the next batch, or vectorised labelling.
  This is worth more than a bigger VM.
- Wasteful today: two runs on a 290-min box after a 45-min plateau, no decay coming. ~4 GPU-hours.
- Attribution still missing: which of {budget, curriculum from R 4-6, lr, damage pool} made the plain
  model work. Not worth a dedicated run; the queue below gives some of it for free.

### 3.4 Evaluation — honest but under-powered and in-distribution

- n = 100 (R ≤ 12) and 30 (R 16) per set; bins at R 16 have n = 6-9. The README should carry ± 2se.
- All sets but "page loops" come from the training generator. Add: ragged masks; near-tie-only
  bridges; thin corridors / nested regions (Pathfinding NCA's adversarially evolved mazes are the
  precedent — hard boards are where the model is wrong).
- `evaluate.py`'s edit test is one edit (journey's correction); the trainer's 3-edit sequence should
  be the one reported, with "hold" (how often the old answer is still acceptable) alongside.
- Report settle time (steps to the fixed point, measured by the first step after which the readout
  does not change) — it is the number the browser needs, and it shows horizon vs precision at a glance.
- The quick-check score averages four numbers at 8R only; fine for best.pt, but the per-radius bridge
  rows are where the signal is. `progress.py` could watch the worst radius, not the mean.

### 3.5 Tooling — strong

Progress verdicts with 2se, the ledger and budget cap, the whitelisted public bucket, `selftest.sh`,
the parity fixture for `src/nca.ts`, the dashboard with pool small-multiples: this is better than most
research code. Two wants: a data/GPU split in `secPerIter`, and `evaluate.py` on the VM at the end of
every stage (seconds on the GPU; an hour on the Pi).

### 3.6 Process

- The journey doc is right about the hybrid: ~6,300 iterations across 21 Pi runs proved nothing when
  the loops-only model had needed 9,200. The lesson is already in memory; keep it.
- Launch plans changed two things at once twice (lr + radii; now rollout length is the only variable —
  good). Keep one variable per A/B.
- `train.py` defaults (96 / 16) are not the shipped model (256 / 24); every continuation must pass the
  sizes. Make the shipped sizes the default or put them in the plan template.

## 4. Literature: what to copy, what not to

Read for this review (links at the end): Growing NCA, Self-classifying MNIST, Self-Organising
Textures, Adversarial Reprogramming, the DSOS thread, Isotropic NCA, Learning Graph CA, Pathfinding
NCA (2023), Reasoning with NCA (2026), DiffLogic CA (2025), E(n)-GNCA, MeshNCA, Med-NCA / M3D-NCA,
ViTCA, NoiseNCA, plus three 2026 papers on consensus and attractors.

| Trick | Source | Here? |
|---|---|---|
| Pool (1024) + reseed worst + damage best 3/8, batch 8, 64-96 steps, lr 2e-3 → 2e-4, gradient norm | Growing NCA | Already in, adapted (pool 512, batch 32, damage p 0.5). Keep. |
| Random rollout length; fixed t → periodic orbits, U[10,20] → fixed points | Graph CA | In. The reason `--steps-mult` must stay a range. |
| Short chunks with a persistent replay pool; AdamW 4e-4 constant, grad clip 1.0, EMA 0.999; damage p 0.1, target swap 0.1; 9×9 mazes → 201×201 at 100 % | Reasoning NCA 2026 | `hb` is this recipe in spirit. Add EMA weights for evaluation (free), grad clip instead of per-param normalisation as an A/B. |
| L2 to a bounded one-hot for "vote" channels, small update noise σ 0.02; agreement peaks then erodes | MNIST | The fill channel is already L2 to {0,1}. Noise: try only if flicker on near-ties appears. Read out in a window, not "at convergence". |
| Overflow loss instead of a hard clamp | Textures, ViTCA | Yes: A/B against the clamp at a higher lr (item 6). |
| Fire rate 0.5 (0.8; 60 %) for robustness; a seeded mask keeps it deterministic | Growing, Reasoning, DiffLogic | A/B (item 7). For the game, a host-seeded mask makes it reproducible. |
| Hex kernels work with the square recipe | Textures | Confirms the design. |
| Learned perception beats fixed Sobel | MNIST | Confirms the design. |
| Train on small patches, run on the whole image; steps ∝ diameter | Med-NCA, M3D-NCA, DiffLogic | The size-generalisation recipe: train on crops / small masks, evaluate on big ones. |
| Reflect padding so the border is not a position cue; feed the rim as a channel | Med-NCA | Here the rim *is* the task (bridges), so the mask input is right; but train on ragged masks so "rim" is not "disc edge". |
| Spatial max-pool for a global quantity (diameter) | Pathfinding NCA | **Do not**: a non-local op is the engineered hint the owner rejected. The paper's warning stands: global aggregation by purely local rules generalises worst; budget for it. |
| Diffusion-of-mass as the size signal (equilibrium concentration 1/area) | folk (IsoNCA Laplacian) | Not as an input or fixed channel. Worth knowing as the kind of thing the model may be discovering; a hidden-channel PCA (Guichard 2026) on near-tie boards would show whether it has a conserved-mass-like channel. |
| Adversarially evolved hard boards | Pathfinding NCA | Yes, cheaply: oversample the boards the current model gets wrong when refilling the pool. |
| Discrete logic-gate CA for exactness and bit-reproducibility | DiffLogic CA | Not now. The right tool if server/client determinism ever becomes a hard requirement. |
| Float non-associativity alone breaks symmetry | IsoNCA | A trap for any server/client replay of a float NCA (see the plan). |
| Edge-attributed graph NCA is universal for finite-state rules on a graph | Graph CA | The Spectre-tile architecture (plan). |

## 5. Prioritised improvements

Costs are GPU-minutes on the L4 (two runs share it; ~11 it/s per run at R 5-10, ~3-4 it/s at R 16,
estimate) and engineering hours. "Falsified if" is the result that would make me drop the item.

| # | What | Why / evidence | Expected effect | Cost | Test | Falsified if |
|---|---|---|---|---|---|---|
| 1 | **Large-radius curriculum fine-tune**: `--init hb-big/best.pt --R 6 8 12 16 --steps-mult 3 6 --lr 5e-5 --batch 16 --iters` sized to the box so decay happens; then a stage at R 8 12 16 24 | zero-shot R 16 is 0.85 from R ≤ 10 training; R 10 went 0.72 → 0.97 once trained on | R 16 bridges ≥ 0.9, R 24 ≥ 0.8 | 150 + 120 GPU-min; 0 eng | quick check per radius; `evaluate` n = 200 at R 16 and 24 after | R 16 top bin does not pass 0.7 within 90 min while R 8 holds → capacity (item 4) |
| 1b | **Ragged masks in the pool** (per-board mask: random crops of the disc, random supertile patches from Spectacle's field, blobs with concave rims), a ragged held-out set | §2.4: bridges 0.21 / none 0.79 on the level-3 patch | the game's board shape works; the rim is learned as "cells with an off-board neighbour", not "the disc edge" | +0 GPU if folded into item 1 (same stage); ~2 h eng (mask per board, rim/targets on any mask) | the ragged set in `evaluate` | ragged bridges stay under 0.7 after 90 min while disc numbers hold → a model question |
| 2 | **Near-tie-heavy bridges** (generator knob: ratio ≥ 0.6 for half the bridges; plus oversample pool boards the model last got wrong) | all residual error is in [.5, 1]; Pathfinding NCA's evolved mazes | top-bin accuracy up at every R; "none" down | 90 GPU-min as the A/B arm of 1; ~1 h eng | same metrics, same iterations as 1 | top bin at R 16 not above arm 1 by > 2se |
| 3 | **Evaluation hygiene**: n ≥ 200 per radius, bins with ± 2se, 8R / 16R / 32R, settle time, ragged-mask set, near-tie-only set, 3-edit sequence in `evaluate.py` | §2; R 16 is ±0.18 today | trustworthy tables; horizon vs precision visible | 0 GPU (seconds on the VM at stage end); ~2 h eng | — | — |
| 4 | **Capacity ablation** (channels 24/32/48 × hidden 64/128/256), via a zero-padded widening loader so each is a fine-tune | Pathfinding NCA: more channels, fewer hidden generalised best; the browser needs the smallest model that works | know the cheapest adequate size; maybe near-tie precision | 3 × 60 GPU-min; 2 h eng for the loader | same as 1 on the three sizes | sizes tie within 2se → keep 256/24 and stop |
| 5 | **Data path off the training thread** + `secPerIter` split | GPU at 65 %; ~50-100 ms Python per iteration | +30-50 % it/s (estimate) | 0 GPU; 2-3 h eng | it/s before/after | GPU util does not rise |
| 6 | **Overflow loss (or AdamW weight decay) vs hard clamp**, A/B at lr 5e-4 from `hb-big` | the 2e-3 collapse mechanism; Textures / ViTCA | stability at higher lr, fewer rollbacks | 2 × 60 GPU-min; 1 h eng | rollbacks and score over 60 min | both arms collapse alike |
| 7 | **`--fire-rate` flag; A/B 0.5 vs 1.0 fine-tune**, evaluate at p = 1 and at p = 0.5 with a seeded mask | every reference NCA; asynchrony is where robustness comes from; the game may want per-player rates | same exactness at p = 1, robustness to async stepping | 60 GPU-min; 15 min eng | exact at p = 1 before/after; exact at p = 0.5 | exact at p = 1 drops > 2se |
| 8 | **lr schedule tied to the box** (iters sized, or decay on PLATEAU) | both live runs will never decay | the last 20 % of accuracy at the end of a stage | 0 GPU; 30 min eng | — | — |
| 9 | **EMA weights for evaluation / best.pt** | Reasoning-NCA; free | smoother best.pt, less quick-check noise | 0 GPU; 30 min eng | compare EMA vs raw on the quick check | no difference |
| 10 | **Hidden-channel look** on near-tie boards (PCA of channels along a bridge) | Guichard 2026; tells whether a size-like signal exists | diagnosis only | 10 min CPU | — | — |

Deliberately *not* proposed: any non-local pooling, any engineered input (angles, rim sources), any
frozen channel. Those are the owner's settled no.

## 6. Experiment queue for this task's next GPU hours

Assumes the lead frees the GPU around 14:00 UTC (both runs plateaued at 12:13 and will not decay;
`down.sh` pulls `hb-big/best.pt`, iteration ≥ 39k, as the new base). Two runs share the GPU; the plan
file format is `nca/cloud/plan.txt`'s. Total ≈ 6 h of GPU-slot time ≈ 3 h wall ≈ $1.3, leaving the
rest of the window to the next stage (see `spectacle-nca-plan.md` §8 for the combined timeline).

| Order | Slot A | Slot B | Minutes | Go / no-go |
|---|---|---|---|---|
| 1 | `fa\|r6816`: item 1 stage 1 (R 6 8 12 16, 3R-6R, lr 5e-5, batch 16, eval-n 64, iters sized to ~90 min), with ragged masks (item 1b) if the refactor lands in time, else disc only | `fb\|r6816nt`: the same with item 2's near-tie knob (if landed) else `--bridge-frac 0.5` | 90 | go if R 16 bridge quick ≥ 0.9 in either arm and R 8 ≥ 0.93; else item 4 next |
| 2 | `fa\|r81224`: stage 2 (R 8 12 16 24, batch 8) from the better arm | `fc\|clamp`: item 6 A/B at lr 5e-4 (60) then item 7 (60) | 120 | R 24 bridge ≥ 0.8 → done with radii for now; else stop and ablate size |
| 3 | `evaluate` on the VM: n = 200, R 6 8 10 12 16 24, mults 8 16 32, ragged set | — | 10 | tables into the README with ± 2se |

What the owner should see at the end: a checkpoint ≥ 0.9 bridge-exact at R 16 and ≥ 0.8 at R 24 with
per-bin tables, or the clean negative result that the near-tie bin does not move with data and
curriculum — which hands the question to capacity, not to more steps or more features.

## 7. The browser cost

Measured (README / journey): `src/nca.ts`, scalar float32 JS, ~67 µs per cell-step for the 256/24
model (8.5 ms per step at R 6). Linear scaling (inference):

| Board | Cells (array) | ms per step (JS) | ~6R settle | Note |
|---|---|---|---|---|
| R 16 | 1,089 | ~70 | ~7 s | the page's current ceiling in practice |
| Spectacle hex level 3 | 1,110 box / 496 tiles | ~75 (box) | — | just usable for a demo |
| level 4 | 11k box / 3,905 tiles | ~740 | — | unusable |
| level 6 | 694k box / 242k tiles | ~46,000 | — | — |

The model is ~64k MACs per cell-step; level 4 is ~0.7 GFLOP per step, level 6 ~45 GFLOP. A GPU path is
unavoidable for game boards: WebGL2 fragment shaders with the state in RGBA textures (the hand-written
CA's `DESIGN.md` already lays out `RGBA32I` MRT banks; a 256-wide hidden layer is 64 RGBA targets, i.e.
8 passes of 8 MRT targets, or WebGPU compute with a plain matmul — the better target in 2026). On an
integrated GPU (~0.5 TFLOP/s effective) level 4 is ~3 ms per step and level 6 ~100 ms: level 4 live,
level 6 not at this width. Hence item 4: a model 4× cheaper (hidden 64) would make level 6 borderline.
Growing NCA's demo quantised weights and activations to 8 bits and most models survived; worth a check
once a GPU path exists. Also: `tf32` is off and `cudnn.benchmark` on in training — fine; keep float32
throughout so the browser and PyTorch agree to the 1e-3 the parity fixture pins.

## Sources

- Growing NCA — https://distill.pub/2020/growing-ca/
- Self-classifying MNIST — https://distill.pub/2020/selforg/mnist/
- Self-Organising Textures — https://distill.pub/selforg/2021/textures/
- Adversarial Reprogramming of NCA — https://distill.pub/selforg/2021/adversarial/
- DSOS thread — https://distill.pub/2020/selforg/
- Growing Isotropic NCA — https://arxiv.org/abs/2205.01681 ; Steerable NCA https://arxiv.org/abs/2302.10197
- Learning Graph CA — https://arxiv.org/abs/2110.14237
- Pathfinding NCA — https://arxiv.org/abs/2301.06820
- Reasoning with NCA (2026) — https://arxiv.org/abs/2609.36126
- Differentiable Logic CA — https://arxiv.org/abs/2506.04912
- E(n)-equivariant Graph NCA — https://arxiv.org/abs/2301.10497
- MeshNCA — https://arxiv.org/abs/2311.02820
- Med-NCA — https://arxiv.org/abs/2302.03473 ; M3D-NCA — https://arxiv.org/abs/2309.02954
- ViTCA — https://arxiv.org/abs/2211.01233
- NoiseNCA — https://arxiv.org/abs/2404.06279
- Hidden channels in NCA (2026) — https://arxiv.org/abs/2609.21870
- Consensus in NCA (2026) — https://arxiv.org/abs/2606.21202 ; global majority limits — https://arxiv.org/abs/2603.19472
- Project history: `docs/nca-journey.md`; failed attempts: `runs/experiments.md`.
