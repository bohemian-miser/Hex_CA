# Hex_CA: from a hand-written flood fill to a trained NCA

The whole project so far in one place, for the owner and for the next agent. It covers 2026-10-01 to
2026-10-05 12:17 UTC. Times are UTC. The owner's clock was UTC+10 until 2026-10-04 and UTC+11 after, so
the commit dates on GitHub show local time.

Each number below is marked as **measured** (from a log, an evaluation output, the ledger or a test run)
or as **inference** (a reading of the evidence, not tested). Unmarked figures were taken from a source
named nearby. Where the sources disagree, the "Corrections" section at the end says which one is right.

Repo: https://github.com/bohemian-miser/Hex_CA (local `/home/dog/projects/hex_ca`). The game it serves is
Spectacle (`/home/dog/projects/Spectacle`). Players there pick a rule for how lines cross each tile, tap
a tile, and a line grows tile to tile. Closed circuits and edge-to-edge claims score. The long-term aim
is to run that game as a cellular automaton. Filling the inside of a circuit is the first piece.

## Timeline

| When (UTC) | What | Where |
|---|---|---|
| 10-01 04:50 to 23:10 | An earlier bot's three PRs. Its last version filled correctly but took Θ(N) steps per edit and showed ~90% of the board filled while settling | PRs #1-#3 |
| 10-01 23:24 | Owner: "I gave another bot this task and it did a terrible job". Wants loops and bridges to fill on a hex CA, hand-written rules preferred, states off/on/filled/wall | session start |
| 10-02 10:07-10:10 | Owner: the effort has gone beyond the scope ("all I wanted was ... the flood fill but with a cellular automata") | feedback |
| 10-02 10:23 | Hand-written CA merged: 22 integer channels, settles in ~10R steps, 212 tests | PR #4 |
| 10-02 11:32 | Channel grid and spectrum overlay in the demo | PR #5 |
| 10-02 11:57 | Owner asks for a trained NCA, distill.pub style, "fill enclosed regions" only | request |
| 10-02 13:29 | A review finds the training data held only perfect hexagon rings. Generator widened | in PR #6 |
| 10-02 15:30 | Trained NCA (loops only) merged: ~99.8% exact held-out, live edits work | PR #6 |
| 10-02 22:49-22:51 | Owner asks for variable board size and bridges ("either side is acceptable so long as only 1 side is filled") | request |
| 10-03 | On the Pi, 21 short pure-training attempts fail on bridges. A hybrid with hand-written frozen floods works | `runs/experiments.md` |
| 10-05 07:22-07:31 | Hybrid merged: bridges 75% exact at R 8, needs a state reset on every edit | PRs #7, #8 |
| 10-05 07:41 | Owner: "it works when we reset the state every time, but fails if i continue to edit it". Proposes distill's pool + damage | request |
| 10-05 07:58 | Owner: "i don't even want to have the current hand crafted input layers, remember the bitter lesson?" | feedback |
| 10-05 ~08:30-11:21 | Plain model trained on the Pi (`pure-a`). Bridges rise from 0 to 0.5-0.7 on small boards | Pi |
| 10-05 08:53 | GPU cost investigation reported (the July Vertex bill was ~$200, driven by volume) | `gcp-gpu-cost.md` |
| 10-05 08:56 | Owner approves $3 of Spot GPU | |
| 10-05 09:58-10:12 | Launch 1: ~19 it/s per run, both runs collapse at ~4-5k iterations | ledger |
| 10-05 10:38-11:20 | Launch 2: stage 1 at R 4-6 reaches 98-99% mix and 95-96% bridges. Stage 2 at R 6-10 makes both worse and is stopped | ledger |
| 10-05 11:22 | Launch 3 starts (A/B on rollout length, R 5-10, lr 5e-5). Hard limit 5.5 h | running |
| 10-05 11:35-11:51 | Stage-1 larger model evaluated on the Pi and driven live in the page: 7/7, 28/30, 30/30, 30/30 | eval |
| 10-05 12:06 | Owner: summarise, review, then plan the next stage (Spectre patterns, a replacement for Spectacle). $20 of Spot over 24 h | request |
| 10-05 12:10 | Plain model shipped to the page. Cloud budget raised to 45 VM-hours | PR #9 |

## What is being computed

There are three different "smaller side" rules in play. Anyone joining the work on Spectacle has to
reconcile them.

- **Hand-written CA (PR #4, `DESIGN.md` section 4).** A region of open cells fills if it doesn't touch
  the exterior. A region that does touch the exterior fills only if its size is below `M`, the largest
  exterior-touching region's size. So **ties fill nothing**, and with edge regions of 10, 10 and 4 only
  the 4 fills. The measure is area only.
- **Trained NCA (spec v2, later v4, `nca/data.py targets`).** Every enclosed region fills. With two or
  more rim regions, all of them fill **except exactly one**, a "largest" one. Largest means maximal area
  *or* maximal angular span of rim contact, within EPS 0.02 (the owner's "where the ends are closer
  along the rim"). Any such choice counts as exact. So with 10, 10 and 4, a 10 also fills, and a tie
  fills one side.
- **Spectacle's engine** (Spectacle `CLAUDE.md`): a line that runs edge to edge closes as a circuit
  "whose polygon is the line plus the smaller arc of `fieldOutline`". The 10-02 brief read `field.ts`
  as a polygon-area comparison. Confirm which before the next stage.

## Stage 0: the earlier bot (PRs #1-#3, 10-01)

- **PR #1 and #2.** An event-driven pulse race round the border. The host had to feed one line cell per
  step, so a circuit drawn in two halves never closed.
- **PR #3.** A static, self-stabilising, layered CA. It was correct when settled, but a cut-off label
  only died by counting up to N. **Measured** in the 10-02 brief: steps/N was 1.03-1.33 for N = 127 to
  4,921, about 21 s per edit at R 40, and ~90% of the board read `filled` mid-settle. Its tests had no
  vitest timeout, and the 150-seed fuzz test timed out on slow machines.

## Stage 1: the hand-written CA (PRs #4, #5, 10-02)

Design C (an NCA-shaped integer CA) was chosen after Designs A and C were both prototyped, with two
grafts from A. It is in `src/engine.ts`, `src/field.ts`, `src/fill.ts`, `src/oracle.ts` and `DESIGN.md`.
It is uniform, synchronous and radius-1, with 22 Int32 channels in five kinds.

How it works, in order:

1. **Epoch wave.** A max-flood of edit serials makes stale values invisible instead of having to refute
   them.
2. **Floods.** A rim flood runs through walls, and a region-touches-exterior flood runs over open cells.
3. **Leader and size.** Each region elects its largest random id as leader and builds a BFS tree to
   it. A convergecast sums the region's size up that tree, and the result is read back down.
4. **Gate.** A second, static BFS tree certifies quiet. When it fires, the root commits a lagged
   maximum `M` and broadcasts it down the tree.
5. **Latch.** A cell shows `FILLED` only for a certified epoch, so nothing is ever shown filled for a
   picture that never existed.

**Measured** (README, `npm run bench` on the Pi 5):

- Settling takes about 10R steps from R 6 to R 40. The proven bound is `11R + 3` on a hexagon
  (`settleBound()`).
- A step costs 0.08-1.13 ms while settling, about 1.4 µs per computed cell.
- 212 tests run in about 8 s.
- An adversarial review found two errors, both fixed with a regression picture: `settleBound`
  undercounted the rim branch, and the docs falsely claimed garbage state recovers without a bump.

**Limits.** The quiet gate is global, so any edit anywhere restarts it. At R 20 a loop alone fills in
162 steps, but with a distant cell painted every 120 steps it takes 3,789. Erasing does not un-fill
until the next commit. A game where lines grow continuously would need per-region gates, which are
not designed.

PR #5 added a grid of all 22 channels and a "spectrum" overlay (one hue per channel). The owner asked
for them at 11:11, along with "could this really be replaced with matrix maths?", which led to the
trained track.

**Owner feedback on this stage (10-02 10:10):** the work went past the ask. This is recorded in memory
as `feedback-scope-tight`: build the stated milestone, keep the process to one designer plus one
reviewer, and put future phases in a short note, not in the code.

## Stage 2: trained NCA, loops only (PR #6, 10-02)

The spec is v1 (`nca-spec.md`).

**Board and model.** The board is a hexagon of radius R stored in an S×S array, S = 2R+1. Neighbours
are a 3×3 kernel with two corners masked off. The state has 16 channels:

- ch0 is the walls, re-imposed every step;
- ch1 is the fill, read as filled above 0.5;
- the other 14 are hidden;
- the board mask is one constant input.

Each step runs a masked 3×3 conv into 64 hidden units, a ReLU, then a zero-initialised 1×1 conv whose
output is added back to the state, clamped. That is 10,896 parameters with synchronous updates. Python
(`nca/`) trains it. `src/nca.ts` runs it in the browser and is pinned to PyTorch by a parity fixture
(`tests/nca.test.ts`, 1e-3). The page is `web/nca.html`.

**Training.** About 2.5 h on the Pi at ~0.8 s/iteration, 9,200 iterations in all:

- `p1`: 1,400 fresh-start iterations at R 6;
- `p2`: 1,800 pool iterations at R 8 on the first generator;
- `p3`: 6,000 more pool iterations at R 8 on the widened generator, lr 1e-3.

**The data bug.** The page review (13:29) found that the wall generator drew only perfect hexagon rings.
The run on that data was stopped. The generator was widened to grown blobs, thin polygon loops, messy
boards and leaky shapes, and edits now change the answer about two times in three.

**Measured** on 500 held-out boards: mixed boards 0.998 exact at R 8 and 0.998/1.000 at R 11; the page's
own random loops 1.000; after a live wall edit with no reset 1.000.

## Stage 3: bridges on the Pi, then the hybrid (PRs #7, #8, 10-02 to 10-05)

The owner's request (10-02 22:51): "fill the smaller side when a bridge is made, in the case where the
side formed where the ends are closer is the larger area, either side is acceptable so long as only 1
side is filled", plus variable board size.

### The pure attempts that "failed"

All of these ran on a Pi 5 that was shared with Chromium and other agents (load 5-15). Most used batch
8 and a flat lr (screens set `--iters 2000`, so the decay never came). Each run was chunked into
8.5-minute `--resume` pieces. Most trained at R 6-10 from the start. The record is
`runs/experiments.md` (git-ignored, on the Pi only). The specs are `nca-spec-v2.md` to `nca-spec-v6.md`.

| Spec | Idea | Runs (iterations) | Best bridge exact | What it showed |
|---|---|---|---|---|
| v2 | fill loss only, from `p3` | x1-x6 (155-415) | 0.22 once (x5 at 150), then 0.00-0.05 | Failure mode always "no side fills". lr 1e-3 swung or diverged, 3e-4 was stable but slow |
| v3 | angle inputs theta1/theta2 | v3a, v3b (258-342), v3-small-a/b fresh at R 4/5 (500 each) | 0.06 at R 6-10; **0.2-0.3 at R 4/5, with or without angles** | Angle weights barely moved. Small boards learned partial bridges |
| v3 aux | supervise floods in ch2-5 | aux-small (473), aux-a/b/c (297-298) | 0.03 at R 6-10 | Floods learned only slowly. The threshold rule "arc < 0.5" with *exact* floods is right on only ~0.5 of bridge boards |
| v4 | pooled max/min perception, span measure, comparative rule | v4-smoke-a (250), v4-smoke (451) | 0.03 | Exact comparative floods would give ~0.82 of bridge boards (1.0 off near-ties). Channels held board-wide defaults, not floods |
| v5 | dense per-step flood targets, fresh only | 3 runs (182-427) | 0.03 | Learned a position-blind time ramp (aux MSE 0.053 vs the ramp's 0.052), not a flood |
| v6 teach | teacher forcing on the floods | v6-smoke-a (100), v6-smoke (300) | 0.02 | The one-step rule was learned (0.0009 vs copy 0.0056), but free-running cells drift +0.003-0.009 per step |

**In total:** 21 runs and about 6,300 iterations, none longer than 500 (**measured**, summed from the
table in `experiments.md`). The loops-only model had needed 9,200.

### The hybrid (spec v6)

`model.install_floods` hand-writes 26 of the 64 hidden units and the output rows of channels 2-7, and
freezes them:

- four gated max-floods of rim-angle sources (channels 2-5);
- the region's rim span (channel 7);
- a board-wide max-flood of that span (channel 6).

The inputs gain six constants (theta1, theta2 and four rim sources). Perception gains per-channel
max/min pooling. Only the readout trains: 9,770 of 16,400 parameters. The run was v6-floods, 4,000
iterations at ~2.5 s each, best at iteration 3,600.

**Measured** at R 8 on 300 boards:

- mixed boards 0.930;
- page loops 1.000;
- bridges 0.753 overall, 0.947 / 0.852 / 0.636 / 0.500 by area-ratio bin from most lopsided to
  near-even;
- every side filled 0.120, no side filled 0.037.

At R 24 (50 boards): mixed 0.94, bridges 0.84.

**Why it failed in use:** a max-flood can only go up. Adding a wall that splits a region needs the
values to fall, and frozen weights cannot learn that. So the page reset the state on every edit
(`pool: false` in the weights meta), and continued editing without a reset failed, as the owner found
on 10-05 07:41.

## Stage 4: back to a plain model (10-05 07:58 onward)

The owner rejected hand-crafting on principle (memory `feedback-bitter-lesson`): no hand-written
channels, engineered inputs or frozen weights. The `nca.train` default is now the pure recipe:

- **Inputs:** walls (ch0) and the board mask only.
- **Model:** a learned 7-tap conv, ReLU, a zero-initialised linear update and clamp [-2, 2].
  `--hidden 96 --channels 16` by default. The shipped model uses 256 and 24 (64,024 parameters).
- **Pool:** a persistent pool per radius with accumulating damage. Each iteration:
  - the worst board restarts fresh;
  - batch/8 boards are brand new;
  - each of the rest takes one damage with probability 0.5: a small wall edit, a burst of 3-20
    toggles, an erased disc, a stamped loop or bridge, or distill's state zeroing in a disc.
  - Targets are recomputed and boards go back to the pool. Nothing reverts to an original.
- **Loss:** min over acceptable targets of the fill MSE, over the last 8 steps of a T ~ U[6R, 10R]
  rollout. Adam with per-parameter gradient normalisation.
- **Quick check** every `--eval-every` iterations, on fixed held-out boards: mix, bridge (with
  none/both), page loops, and the **edit sequence**: settle, then three successive damages with 6R steps
  after each and no reset. `best.pt` keeps the best mean score.
- **Tooling:** `nca/progress.py` gives a noise-aware verdict (2 standard errors).
  `nca/dashboard.py` serves the pool and charts on `localhost:8765` and builds a static copy for the
  bucket.

**Pi run `pure-a`** (fresh, hidden 96, batch 16, lr 2e-3, R 4-6, ~08:30-11:21). **Measured** quick checks
(32 boards per set per radius, noisy):

| Iteration | Bridges exact |
|---|---|
| 0 | 0.00 |
| 800 | 0.27 |
| 1,400 | 0.52 |
| 3,400 | 0.70 (best score 0.835) |
| 4,000-7,200 | 0.33-0.68, drifting down to 0.41 |

The drift came at the same constant lr that later collapsed on the GPU. Its iteration-3,200 checkpoint
seeded launch 1's `ga` run. It ran at about 0.7 it/s.

## Stage 5: cloud setup and cost

### The Vertex history (July, the ncawords project)

From `gcp-gpu-cost.md` (scratchpad), drawn from Vertex's own job records:

- 581 one-GPU jobs ran from 07-17 to 07-24, about 450 GPU-hours, with a peak of 16 at once.
- About 40% of the hours were Spot L4 jobs that ran for up to 13 h and then ended FAILED on stockout.
- The models were tiny and launch-bound.
- The estimated bill is **about $200, plausibly $150-300** (inference; only the billing console has
  the real figure). The cause was volume under an autonomous agent with a 1,000-job grant, not
  Vertex's rate. Vertex added roughly 25-30% over plain Compute Engine.
- Leftover Vertex Spot quotas in `recipe-lanes-staging` (~32 GPUs) were left in place by the owner's
  call.

### The setup (10-05 09:00-09:58)

- **Project.** `recipe-lanes-staging`. A dedicated project was impossible: billing account
  is at its 5-project limit. Everything carries the label `purpose=hexca`.
- **GPU quota.** `GPUS_ALL_REGIONS` = 1, granted in a minute. This caps the project at one GPU VM.
- **VM.** One Spot `g2-standard-4` (1×L4) named `hexca-train` in us-central1-a, falling back to b and
  then c. It boots the Deep Learning VM image `common-cu129-ubuntu-2404-nvidia-580` with the driver
  preinstalled. `--max-run-duration` deletes it whatever happens. Its service account
  `hexca-vm@...` can only read and write the bucket. At DONE the VM powers off, and `down.sh` deletes it.
- **Bucket.** `gs://recipe-lanes-staging-hexca-runs`, public by the owner's choice, no expiry. Uploads
  are whitelisted file names only. The dashboard polls fixed files and never lists the bucket (the old
  bucket's ~$14 came from listing).
- **Scripts** (`nca/cloud/`):
  - `launch.sh` and `down.sh`: lead only;
  - `watch.sh`, which exits on a verdict;
  - `pull.sh` and `publish.sh`;
  - `startup.sh`, which runs a plan of staged runs in parallel on the one GPU, syncs every minute, and
    resumes after a preemption;
  - `selftest.sh`: everything against a local directory, no cloud.
- **Ledger.** `ledger.tsv` (git-ignored) records create and delete times. `launch.sh` refuses when
  ledger hours plus `MAX_HOURS` would pass `BUDGET_HOURS`. These were 6 and 6.5 under the $3 approval,
  and are 12 and 45 since commit 3e67fb1 for the $20 / 24 h budget.

## The three launches

### Launch 1 (09:58:06-10:12:00, 0.23 VM-h)

Two runs at R 4-6, batch 32, pool 512, constant lr 2e-3 (the decay was set at 60% of 40,000
iterations, never reached):

- `ga`: hidden 96, continuing `pure-a` from iteration 3,200;
- `gb`: hidden 256, 24 channels, from scratch.

**Measured:**

- Throughput was 18.7 and 18.0 it/s, the two runs sharing the one L4. The first training log line came
  2m47s after the ledger's create.
- Both runs peaked at iteration 2,000: `ga` score 0.857 (bridges 0.742), `gb` 0.809 (bridges 0.672,
  from scratch, under two minutes of training).
- By iteration 5,000 both had collapsed: `ga` score 0.244, bridges 0.008; `gb` 0.163.
- `watch.sh` exited DIVERGED and the VM was deleted.

**Diagnosis.** **Measured** from the checkpoints: the first-layer weight norm grew from 17 to 25-33, and
75-85% of hidden-channel cells sat at the clamp, where the gradient is zero. **Inference, not tested:**
per-parameter gradient normalisation makes every step move the weights by about lr, so a constant 2e-3
over thousands of steps keeps inflating them.

**Fixes** (commit c99b4f6):

- default lr 5e-4, with a decay that a run actually reaches, a warm-up after every fresh optimiser, and
  a floor of 1e-5;
- a **rollback guard**: a quick check below 0.6 of the best score, or a loss over 4× the recent median,
  reloads `best.pt`, halves the lr scale, restarts the pool states and gives the optimiser a fresh
  warm-up. It stops cleanly after 6 rollbacks. Replayed on launch 1's logs it fires on both runs; it
  doesn't fire on `pure-a`'s;
- later stages start from the previous stage's `best.pt`.

A bug of the lead's own was found at the same time: `down.sh`'s "nothing left" check reported a false
alarm, because it read gcloud warning text as resource names. It was checked by hand (no instances,
disks, snapshots, addresses or images) and fixed.

### Launch 2 (10:38:35-11:20:53, 0.70 VM-h)

Both models continued from launch 1's iteration-2,000 best.

**Stage 1** (R 4-6, lr 5e-4 decaying to 5e-5, 30,000 iterations in about 28 minutes, 18.3 and 17.6
it/s, no rollbacks). **Measured** quick checks at the end, 128 boards per set per radius:

| | ga (hidden 96) | gb (hidden 256) |
|---|---|---|
| Mixed exact | 0.984 | 0.989 |
| Bridges exact | 0.950 | 0.958 |
| Page loops | 1.000 | 1.000 |
| Three edits in a row | 0.97 | 0.97 |
| Every side filled | 0.003 | 0.002 |
| Best checkpoint (iteration within the stage) | 0.9796 (27,500) | 0.9837 (27,000) |

**Stage 2** (R 6-10, lr 3e-4) made both worse within 10 minutes. `gb`'s bridges fell from 0.875 to 0.331
by iteration 2,500 and swung from 0.40 to 0.69 after that. `ga` fell from a score of 0.91 to 0.59-0.90.
No rollback fired, because the drops stayed above 0.6 of best. The lead stopped the launch with
`down.sh`.

Before any training at those sizes, `gb` already carried over to R 8-10 better than `ga` (bridges at R 8
and R 10: 0.93 and 0.72 against 0.82 and 0.59). That is why launch 3 continues `gb`.

### Launch 3 (11:22:21 onward; hard limit 5.5 h, so deleted by 16:52 at the latest)

Both runs start from `gb-r456b/best.pt` (hidden 256, 24 channels) on R 5, 6, 8 and 10, with:

- lr 5e-5 (where stage 1 ended);
- the rollback threshold tightened to a 15% drop (`--collapse-frac 0.85`);
- the small radii kept in the mix;
- a 290-minute box each.

The only difference is rollout length per iteration: `ha` unrolls 6R-10R steps, `hb` 3R-6R.

**Measured** as of 12:13 (~47 minutes of training, 8.6 vs 10.9 it/s, the GPU at 100%):

| | Start | ha-big (6R-10R) | hb-big (3R-6R) |
|---|---|---|---|
| Iterations | 0 | 24,250 | 30,950 |
| Best score (iteration) | 0.952 | 0.975 (17,000) | 0.982 (20,000) |
| Bridges exact, all radii (latest check) | 0.898 | 0.951 | 0.957 |
| Bridges exact at R 10 (latest) | 0.719 | 0.93 | 0.945 (0.969 at 25,500-27,000) |
| Every side / no side filled (latest) | 0.002 / 0.078 | 0.002 / 0.022 | 0.004 / 0.010 |

Both now read PLATEAU on the noise rule, and no rollback has fired. The gain is concentrated at R 10.
`hb` is cheaper per iteration and slightly ahead, but outside R 10 the gap is within noise. These are
quick checks at 8R steps, not `nca.evaluate`, so they are not directly comparable to the tables below.

## The shipped model and its honest numbers

PR #9 (merged 12:10, Pages deployed 12:10:49) ships `runs/gb-r456b/best.pt`: 29,000 iterations in all
(2,000 in launch 1 plus 27,000 in launch 2), trained only at R 4-6. **Measured** with
`python -m nca.evaluate`, 100 held-out boards per radius (30 at R 16), read out at 8R steps:

| R | Mixed exact (trivial) | Page loops | Bridges exact | By ratio bin, lopsided to even | Both | None |
|---|---|---|---|---|---|---|
| 6 | 0.980 (0.34) | 1.000 | 0.910 | 1.00 / 1.00 / 0.94 / 0.77 | 0.000 | 0.010 |
| 8 | 0.960 (0.34) | 1.000 | 0.890 | 1.00 / 0.95 / 0.88 / 0.75 | 0.000 | 0.070 |
| 10 | 0.940 (0.30) | 0.990 | 0.770 | 1.00 / 1.00 / 0.73 / 0.41 | 0.000 | 0.220 |
| 16 | 0.767 (0.40) | 1.000 | 0.400 | 0.86 / 0.00 / 0.00 / 0.00 | 0.000 | 0.600 |

**Edit test** (one edit; see the corrections): mixed / bridges / loops were 0.99 / 0.97 / 1.00 at R 6,
0.95 / 0.95 / 0.98 at R 8, and 0.97 / 0.77 / 0.99 at R 10.

The small model (`ga-r456b`) ties at R 6 but falls off faster: bridges 0.84 at R 8 and 0.55 at R 10.

**Page, driven headless on the Pi with no reset** (measured):

- **Hand sequence at R 6:** draw a loop, open it, close it, draw a second loop, a rim-to-rim wall,
  break it, restore it. All 7 steps matched the oracle.
- **Ten rounds of (bridge or loop, erase 3, add 3):** 28/30 at R 6. The two misses were small
  short-lived pockets, never both sides or no side. 30/30 at R 8 and at R 10.
- **Settled bridge:** it held exactly for 1,500 further steps.
- **Speed:** 8.5 ms/step at R 6, 14.3 at R 8, 28.6 at R 12, with no console errors. The page's buttons
  make easier boards than the held-out bridge set (inference from the gap between 30/30 and 0.77).

## What each approach taught

- **Hand-written CA.** Exact, O(R) and live-editable fill is possible by hand in 22 integer channels.
  But it took a careful design, an adversarial review and 212 tests, and its gate is still global. It
  remains the correctness and speed reference (~1.4 µs per cell-step against the NCA's ~67 µs per
  cell-step at R 6; inference, since these are different harnesses).
- **Loops-only NCA.** A plain NCA learns "enclosed" readily, and a pool with wall edits buys live
  editing. Data diversity mattered more than the model: ring-only data looked fine until a review read
  the generator.
- **The failed attempts.** Each produced true local facts:
  - the threshold arc rule caps at ~0.5;
  - the comparative rule at ~0.82;
  - min-floods starting at 0 can't be reached by a max rule;
  - dense targets invite a position-blind ramp;
  - teacher-forced floods drift when run free.

  The general conclusion drawn from them was wrong (next section). The plain model now beats the
  hand-designed comparative rule's own ceiling: 0.89-0.91 at R 6-8 against ~0.80-0.83 for exact floods.
  The board sets differ slightly, but the gap is larger than the difference (inference). The target
  accepts area *or* span, and the hand rule only had span.
- **Hybrid.** Hand-crafting one mechanism blocked another: max-floods can't forget, so live editing
  broke. It also ran against the owner's stated principle.
- **Plain model with a damage pool.** It learns bridges at small R, holds an answer through repeated
  live edits, and never filled both sides in any evaluation. Scaling to larger R is the open part.
- **Cloud.** One L4 ran two runs at ~18-19 it/s each at R 4-6 (~25× the Pi per run, at twice the
  batch), and ~8.5-11 it/s at R 5-10. The real risk was control, not price. The guardrails held:
  quota 1, a hard run limit, the ledger, labels, a whitelisted bucket and verdict-driven watching.
- **Learning rate.** A constant 2e-3 eventually collapses or degrades: launch 1 collapsed, and `pure-a`
  and the v2 screens swung. 5e-4 decaying to 5e-5 was stable. Fine-tuning at larger R at 3e-4 hurt.
  Launch 3 changed both lr (5e-5) and the radius mix at once, so which of the two mattered is not known.

### Why "pure training fails" was wrong

The lead declared pure learning a failure on 10-03 and shipped the hybrid. The evidence did not support
that:

1. **Budget.** The longest attempt was 500 iterations, mostly at batch 8 on a loaded Pi, about 6,300
   iterations across 21 runs. The loops-only model had needed 9,200. A single GPU stage later ran
   30,000 iterations at batch 32 in 28 minutes.
2. **The signal was there and was dismissed.** The fresh small-radius arms (v3-small-a/b, aux-small at
   R 4/5) reached 0.2-0.3 bridge exact in 250-500 iterations, with or without angle inputs. `pure-a` was
   at 0.27 at iteration 800 and 0.52 at 1,400 on the same kind of board. They were on the same
   trajectory and were stopped.
3. **Wrong radii first.** Most attempts trained at R 6-10 from the start, or from `p3`, whose hidden
   channels were all in use. What worked was a curriculum from R 4-6.
4. **Learning rate.** The screens held lr flat at 1e-3 to 2e-3, and the same instability showed up
   there: x2 diverged, x5 rose to 0.22 and fell back.
5. **No accumulating damage.** v5 and v6 were fresh-start only. The earlier pools only edited walls.
   The plain recipe trains on states mid-transition, the setting where "un-flooding" has to be learned.

No ablation separates these factors. `pure-a` learned at lr 2e-3, so budget, small radii and a fresh
model probably mattered most for getting started, and lr for staying stable (inference).

## Process lessons

- **Scope.** On 10-02 the owner felt the CA work went beyond the ask: a judged panel of designs and
  extensibility sections. The rule now is to restate the milestone in one line and size the process to
  it (memory `feedback-scope-tight`).
- **Give a learned method a real budget before calling it failed, and say what budget was spent**
  (memory `feedback-bitter-lesson`).
- **Verification that caught real bugs:**
  - the PR #6 review read the data generator and found ring-only data;
  - the PR #4 adversarial review found `settleBound`'s undercount and a false recovery claim;
  - `watch.sh` verdicts caught the launch 1 collapse within minutes;
  - a hand check of `down.sh`'s alarm found the check itself was wrong;
  - the headless page check found the page opening at a radius the model wasn't trained on (fixed).
- **What verification missed:** the hybrid's failure under live editing. Its evaluation only ever ran
  from a fresh state. The fix was to test what the user does: the three-edit sequence is now in every
  quick check, and the page is driven with no reset.
- **Read the right signal.** The pool loss has a floor (half of each batch was just damaged), so a flat
  loss is not a stall. The held-out checks are the signal, read against their noise (2 standard errors
  in `progress.py`). PLATEAU fires often on runs near their ceiling: `WATCH_IGNORE=PLATEAU`.
- **Like-for-like comparisons.** Quick checks at R 4-6 were briefly compared with the hybrid's R 8
  evaluation; see the corrections.

## Costs

The ledger times are **measured**. Dollars are inference at ~$0.41 per VM-hour (Spot $0.403 + disk
$0.007 + IP ~$0.003, us-central1, checked 10-05; Spot prices move daily).

| Launch | Create to delete (UTC) | VM-hours | ~$ |
|---|---|---|---|
| 1 | 09:58:06 to 10:12:00 | 0.23 | 0.10 |
| 2 | 10:38:35 to 11:20:53 | 0.70 | 0.29 |
| 3 | 11:22:21 to (running) | 0.85 at 12:13, at most 5.5 | 0.35 so far, at most 2.26 |
| **Total** | | **1.79 so far** | **~$0.73 so far; ~$2.4-2.6 if launch 3 runs its plan** |

- Bucket storage and operations are cents (inference).
- A stopped VM still costs ~$0.007/h for its disk until `down.sh`.
- The owner's new budget is $20 until ~10-06 12:00 UTC, with everything since launch 1 counted
  (`env.sh` comment). With GPU quota 1, at most ~24 VM-hours fit in that window, about $10 on
  `g2-standard-4`. The window, not the dollars, is the real limit.
- The July Vertex history (~$200) is separate and belongs to ncawords.

## Where everything lives

| What | Where |
|---|---|
| Repo | https://github.com/bohemian-miser/Hex_CA. Local `/home/dog/projects/hex_ca`. `main` has PR #9; branch `claude/nca-pure` (head 3e67fb1) is what was merged |
| Hand-written CA | `src/` (engine, field, fill, oracle, hex, lines), `DESIGN.md`, `tests/`, demo `web/page.html` + `web/main.ts` |
| Trained NCA, Python | `nca/`: `train.py` (docstring = the recipes, v4-v6 included), `data.py`, `model.py`, `hexgrid.py`, `evaluate.py`, `export.py`, `progress.py`, `dashboard.py`, `selftest.py` |
| Trained NCA, browser | `src/nca.ts`, `web/nca.html` + `web/nca.ts`, shipped weights `web/nca-weights.json`, parity fixture `tests/fixtures/nca-parity.json` |
| Cloud | `nca/cloud/`: `README.md` (the cold-start guide), `env.sh`, `plan.txt` (launch 3, with its reasoning in comments), `plan.example.txt`, `ledger.tsv` (git-ignored) |
| Runs | `runs/` (git-ignored, Pi only): `experiments.md` (the failed-attempts record), `p3`, `v6-floods`, `pure-a`, `ga-*`, `gb-*`, `ha-big`, `hb-big`, and `_cloud/` (the VM's `status.json` and `startup.log` mirror) |
| Live demos | https://bohemian-miser.github.io/Hex_CA/ (hand CA) and https://bohemian-miser.github.io/Hex_CA/nca.html (plain NCA) |
| Public dashboard | https://storage.googleapis.com/recipe-lanes-staging-hexca-runs/dash/index.html. Heartbeat at `/status.json`, raw files under `/runs/`. Play opens the board with a run's current best weights |
| Local dashboard | http://localhost:8765 (`python -m nca.dashboard --port 8765`, already running on the Pi) |
| Python with torch | `/home/dog/projects/ncawords/.venv/bin/python` (torch 2.5.1, CPU). The VM uses torch 2.14.1+cu130 |
| GCP | Project `recipe-lanes-staging`, us-central1-a, VM `hexca-train`, label `purpose=hexca`, bucket `gs://recipe-lanes-staging-hexca-runs`. |
| Design notes (ephemeral) | `/tmp/claude-1000/-home-dog-projects-Spectacle/fbf60643-.../scratchpad/`: `brief.md`, `nca-spec.md`, `nca-spec-v2.md` to `-v6.md`, `nca-snapshot-spec.md`, `gcp-gpu-cost.md`, `ncawords-notes.md`. **/tmp may not survive.** This document carries what matters from them |
| Memory | `/home/dog/.claude/projects/-home-dog-projects-Spectacle/memory/` (scope, bitter lesson, project, cloud) |
| Related | Spectacle `/home/dog/projects/Spectacle`. Spectre `/home/dog/projects/Spectre` (https://github.com/bohemian-miser/Spectre). Reference NCA project `/home/dog/projects/ncawords` |

## How to resume from cold

1. **Read** this file, the README's "Trained NCA" section and `nca/cloud/README.md`. Run
   `git log --oneline -5` and `gh pr list --state all --limit 3`.
2. **Is a VM up?** Run `tail -3 nca/cloud/ledger.tsv`, then `nca/cloud/watch.sh 40; echo "exit $?"`.
   The README's table maps exit codes to actions. `curl -s` on the public `status.json` also works.
   - An agent never runs `launch.sh` or `down.sh`, never deletes anything in the bucket, and never runs
     a gcloud command that creates, changes or deletes. Only the lead or the owner does.
   - Launch 3 ends with DONE (exit 2) around 16:15 UTC on 10-05. The VM then powers itself off and
     `down.sh` must delete it (exit 9 means stopped, not deleted).
3. **Read a run:** `nca/cloud/pull.sh`, then
   `/home/dog/projects/ncawords/.venv/bin/python -m nca.progress runs/ha-big` (and `hb-big`).
4. **After launch 3:**
   - Run `down.sh`, which pulls the checkpoints.
   - Evaluate both bests:
     `python -m nca.evaluate runs/hb-big/best.pt --radii 6 8 10 16 --n 100 --n-big 30`.
   - Compare against the table above, and with the R 16 column in particular.
   - If one is better, `python -m nca.export` it, run `npm test` (parity), drive `dist/nca.html`
     headless with no reset, then open a PR. Merging to main deploys Pages.
5. **Launching more training:**
   - Write `nca/cloud/plan.txt` with new stage names; a name already in the bucket counts as done.
     Don't pass `--name`, `--resume`, `--minutes` or `--device`.
   - Commit and push. `launch.sh` refuses an unpushed HEAD or a dirty `nca/`.
   - Upload any `--init` checkpoint to `bucket:init/` or reference `bucket:runs/...`.
   - `nca/cloud/launch.sh --dry-run plan.txt`, then the real launch.
   - The budget is 45 VM-hours (`BUDGET_HOURS`) with a 12 h maximum per launch, counted from launch 1
     in the ledger.
6. **The Pi:** never run two trainings at once. Keep a process under ~1.5 GB. Use `--threads 2` when
   the Pi is shared. Chunk long Pi runs with `--minutes` and `--resume`.

## Open problems

- **Large radii.** Nothing has been trained above R 10. The shipped model's R 16 bridges are 0.40, and
  it fails by filling no side (0.60). Launch 3 is lifting R 10 (0.72 to ~0.95 on quick checks). A
  learned rule is not yet scale-free, and its settle time grows with R: it reads out at 8R steps.
  Spectacle's boards are far larger than R 32, the page's slider maximum.
- **Near-even bridges.** The [.75, 1] area-ratio bin is the weakest: 0.75 at R 8, 0.41 at R 10. That
  bin includes exact ties, where any one side is acceptable.
- **Browser cost.** The plain model is ~6× the parameters of the loops-only one. At ~67 µs per
  cell-step, R 32 (3,169 cells) would be ~200 ms per step, and a settle of ~8R = 256 steps would take
  ~50 s (inference, by linear scaling). A WebGL or matrix version of `src/nca.ts` would be the fix; it
  doesn't exist.
- **Semantics.** The three "smaller side" rules disagree (see "What is being computed"). The NCA has
  only seen hexagon-shaped boards: the mask is an input but ragged fields are untested. The hand CA's
  gate is global.
- **Attribution.** No ablation says which of budget, radius curriculum, lr or damage pool made the
  plain model work. The launch 3 A/B on rollout length is still within noise outside R 10.
- **Evaluation gaps.** `evaluate.py`'s edit test is a single edit (the three-edit sequence exists only
  in the trainer's quick check and the page drive). The page checks use easier boards than the held-out
  sets.
- **The next stage, still undefined.** "Recreate the patterns from the Spectre project and act as a
  replacement for Spectacle" means strands growing tile to tile by a rule (an edge subset plus a
  matching per tile type), circuits, tails, owners and captures. Spectacle runs on hexagon *and*
  Spectre tilings. The current NCA lives on a regular hex grid; Spectre tiles have 14 edges in ~6
  seams and no regular neighbourhood. The target, the board representation (grid or neighbour graph)
  and the first milestone are not yet chosen. Per the scope rule, pick one concrete milestone first.
- **Risk items left as the owner chose:** the bucket and its checkpoints are public, and the leftover
  Vertex Spot quotas in `recipe-lanes-staging` were not reduced.

## Corrections to other records

Found while writing this; the sources above are the evidence.

- **README, Trained NCA:** "The edit test (settle, 3 live edits, no reset ...)". `nca/evaluate.py`'s
  `edit_run` applies **one** edit, then runs as many steps again. The three-edit sequence is the
  trainer's quick check.
- **README:** "the plain model passed the hybrid's own numbers within a couple of minutes". That compares
  R 4-6 quick checks (`gb` bridges 0.67 at iteration 2,000) with the hybrid's R 8 evaluation (0.753).
  It is not like for like; the lead said so at the time. The fair comparison is the R 8 evaluation:
  0.89 against 0.753.
- **Memory `hexca-cloud-training`:** "training starts ~8 min after create". The logs show 2m47s, 3m20s
  and 3m16s from the ledger's create to the first training line. "~19 it/s per run" holds at R 4-6
  only; at R 5-10 it is 8.5-11 it/s. Its budget line (6.5 VM-hours) is now 45 (commit 3e67fb1).
- **`gcp-gpu-cost.md`:** "Deep Learning VM images can't be the boot disk on G2". Launches 1-3 booted
  `g2-standard-4` from `common-cu129-ubuntu-2404-nvidia-580` and saw the L4 (measured).
- **PR #7 body and memory `hex-ca-project`:** the hybrid had "five hidden channels" of floods. It
  actually hand-wrote 26 hidden units and the output rows of six channels (2-7): four floods, the span
  and the board-wide max span. The same memory dates the trained-NCA request 10-03. It was 10-02 11:57
  UTC, which was 10-02 21:57 on the owner's clock.
- **`nca/train.py` defaults** (16 channels, hidden 96) are not the shipped model (24 and 256). Any
  continuation of the shipped model needs `--hidden 256 --channels 24`, and `--init` refuses other
  sizes.
