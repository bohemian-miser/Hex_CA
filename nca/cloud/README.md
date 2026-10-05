# nca/cloud: train on one Spot GPU VM, watch it from here

One Compute Engine Spot VM (`hexca-train`, g2-standard-4: 1x L4) runs a **plan** of `nca.train` stages,
several runs at once on the one GPU, and copies everything back to a bucket every minute. This machine
pulls from the bucket for the local dashboard; a static copy of the dashboard is public.

**Public dashboard:** https://storage.googleapis.com/recipe-lanes-staging-hexca-runs/dash/index.html
(polls every 15 s; each run shows the time of its newest data, so a stale page is visible as such).

**Only the lead / owner runs `launch.sh` and `down.sh`.** Everything else here only reads the cloud
(`publish.sh` writes the dashboard to the bucket and nothing else).

## The bucket is public: nothing secret, ever

`gs://recipe-lanes-staging-hexca-runs` is readable by anyone (allUsers objectViewer, no expiry), so the
checkpoints, logs and the VM's log tail under it are public. What goes there:

| Path | Written by | What |
|---|---|---|
| `runs/<run>-<stage>/{log.jsonl,pool.npz,ckpt.pt,best.pt,stdout.log}` | the VM, every 60 s | only these whitelisted names |
| `status.json` | the VM, every 60 s | the heartbeat: phase, GPU use, load, each run's stage and verdict |
| `startup.log` | the VM, every 60 s | the last 300 lines of its boot script's log |
| `DONE` | the VM, at the end | `{launch, time, ok, reason, runs}` |
| `dash/` | `publish.sh` (the VM as `vm`, this Pi as `pi`) | the static dashboard |
| `init/` | the lead, by hand | checkpoints a plan starts from (`--init bucket:init/x.pt`) |

Never put an env file, a token, a key, the ledger or anything from a home directory in the bucket, and
never write a secret into a plan or into instance metadata (both end up in public logs).

## Files

| File | Who | What |
|---|---|---|
| `env.sh` | | the settings: project, region, zones, bucket, service account, machine, `MAX_HOURS`, `BUDGET_HOURS` |
| `launch.sh [--dry-run] PLAN` | **lead only** | checks, then creates the VM (see below) |
| `down.sh` | **lead only** | final pull with checkpoints, deletes the VM, checks nothing of ours is left |
| `watch.sh [minutes]` | anyone | pulls every 2 min, prints a block, exits when something needs a look |
| `pull.sh [--ckpt]` | anyone | bucket -> `runs/` (checkpoints only with `--ckpt`) |
| `publish.sh [--every SEC] RUN...` | anyone with bucket write | the static dashboard of local runs to `dash/` |
| `startup.sh`, `shutdown.sh` | the VM | its boot and shutdown scripts (passed as metadata by `launch.sh`) |
| `plan.example.txt` | | the plan format, a benchmark plan and a curriculum |
| `selftest.sh` | anyone | all of the above against a local directory: no cloud, no money |
| `ledger.tsv` | `launch.sh`, `watch.sh`, `down.sh` | local record of each VM's create / stop / gone / delete |
| `../progress.py` | anyone | `python3 -m nca.progress runs/X`: one run's verdict from its log |

The project `recipe-lanes-staging` is shared with another product: every command passes `--project`,
everything created carries the label `purpose=hexca`, and the checks filter on that label (or on the name
`hexca-*`). The VM's service account can only read and write the bucket: it cannot delete its VM, so at the
end it powers off and `down.sh` deletes it.

## Budget

Prices for us-central1, checked 2026-10-05 (Spot prices move daily):

| Item | $/hour |
|---|---|
| g2-standard-4 Spot (1x L4, 4 vCPU, 16 GB) | 0.403 (on-demand would be 0.707) |
| 50 GB pd-balanced boot disk ($0.10/GiB-month) | 0.007 |
| ephemeral external IPv4 on a Spot VM | ~0.003 (unverified) |
| **running VM** | **~0.41** |
| stopped VM (after DONE, before `down.sh`): the disk only | 0.007 (~$5/month: run `down.sh`) |

- One launch is at most `MAX_HOURS` = 6 h (`--max-run-duration`, then Compute Engine deletes the VM):
  **at most ~$2.50.**
- `launch.sh` refuses a launch when the ledger's VM hours so far + `MAX_HOURS` would pass `BUDGET_HOURS`
  = 6.5 h: **all launches together, at most ~$2.70**. A launch's hours run from its create to its first
  stop / gone / delete in the ledger; with none recorded yet, to now, capped at its `MAX_HOURS`.
  With 6 + 6.5 that means one full launch; a relaunch (after a preemption) only fits if the first used
  under half an hour. Raise `BUDGET_HOURS` in `env.sh` deliberately if more is wanted.
- The bucket: a few tens of MB stored (cents a month); ~20 writes a minute while the VM runs (~$0.04 per
  6 h); downloads at $0.12/GB (more to Australia): `pull.sh` without checkpoints and one open dashboard
  tab are each well under a cent an hour. Checkpoints (~7 MB a run) come down only with `--ckpt`
  (`down.sh` does that once).
- GPU quota in the project is 1, so a second GPU VM can't start anyway; `launch.sh` also refuses if any
  `purpose=hexca` instance exists.

## A plan

One stage per line, `run-name | stage-name | minutes | nca.train args` (see `plan.example.txt`):

- the same run-name: stages one after another, in `runs/<run>-<stage>`; a later stage starts with `--init`
  from the previous stage's `ckpt.pt` unless it names its own `--init`; `bucket:PATH` is fetched first;
- different run-names: in parallel, a process each on the one GPU (`--threads` = 4 / runs unless given);
- minutes: the stage's time box (`--minutes`). Each run's total + `SETUP_MIN` (25) must fit in `MAX_HOURS`.
  The lr decays at 60% and 85% of `--iters`, so choose `--iters` that the minutes reach;
- don't pass `--name`, `--resume`, `--minutes` or `--device`;
- on a relaunch (after a preemption) the VM pulls `runs/` from the bucket: a stage that ended
  (`{"stopped": "done"}`, or its time box used) is skipped, one that was cut off resumes from its
  `ckpt.pt` with the minutes it has left. So stage names already in the bucket count as done: use new names
  for new work.

## Launching and ending (lead)

```bash
cd ~/projects/hex_ca
git push                                         # the VM clones the pushed HEAD from GitHub
gcloud storage cp runs/pure-a/best.pt gs://recipe-lanes-staging-hexca-runs/init/pure-a.pt   # if the plan uses it
nca/cloud/launch.sh --dry-run my-plan.txt        # checks + the exact command
nca/cloud/launch.sh my-plan.txt                  # creates hexca-train (zone a, else b, else c)
nca/cloud/publish.sh --every 60 pure-a &         # optional: the Pi's own run on the public page too
nca/cloud/watch.sh 40; echo "exit $?"            # or hand this to an agent (next section)
nca/cloud/down.sh                                # when it says so: pull, delete, verify -> one line
```

`launch.sh` refuses unless: the plan parses and fits; the budget allows it; HEAD is the pushed tip of its
branch and `nca/` has no uncommitted changes; no `purpose=hexca` instance exists; every `bucket:` file is
there. It records `create` in `ledger.tsv`.

## Monitoring from cold (a new Claude session, or a person)

You have 30 seconds and no context. In `~/projects/hex_ca`:

1. `tail -3 nca/cloud/ledger.tsv`: the last launch, and whether it was stopped / deleted already.
2. `nca/cloud/watch.sh 40; echo "exit $?"`: pulls every 2 minutes and prints a block like

   ```
   [14:02:11] hexca-train RUNNING  phase train  heartbeat 41s ago  gpu 71% 3.2/22.5 GB  load 3.9
     pure-b         r456 (1/2)   it 20350/120000 22.4 it/s     PROGRESS  score +0.071 (2se 0.031); loss -6.2%
     b64            r456 (1/1)   it 8150/100000 9.1 it/s       WARMUP    5 quick check(s) so far; ...
   ```

   and exits when something needs a look (or after the minutes, with 0). Then act on the exit code:

| Exit | Means | Do |
|---|---|---|
| 0 | all runs PROGRESS / WARMUP / between stages | run `watch.sh` again |
| 3 | a run is PLATEAU: no quick-check metric beat the noise over its last 6 checks and its loss fell < 3% | note it. If every run in the block is PLATEAU, run `WATCH_IGNORE=PLATEAU nca/cloud/watch.sh 60`; if they are all still PLATEAU after that hour, tell the lead to run `down.sh` |
| 4 | a run is STALLED: no new log line for > 10 min (or 5x its usual gap) while the VM runs | read `runs/<dir>/stdout.log` and `runs/_cloud/startup.log`; tell the lead (likely `down.sh`) |
| 5 | a run DIVERGED: non-finite loss, the skipped-step guard firing, or the loss doubled | that run wastes its GPU share; tell the lead. Others go on (`WATCH_IGNORE=DIVERGED` to keep watching them) |
| 6 | a run ENDED: all its stages done, or one failed (block says which) | if failed, read its `stdout.log`; keep watching the rest with `WATCH_IGNORE=ENDED` |
| 2 | DONE: the plan is over (`ok` or the reason it gave up); the VM powers itself off | tell the lead to run `down.sh` (deletes the stopped VM, pulls the checkpoints) |
| 9 | the VM is **stopped, not deleted** (it still costs its disk) | tell the lead to run `down.sh` |
| 8 | the VM is gone without DONE: Spot preemption or the 6 h limit | the work up to the last minute is in the bucket; the lead may `launch.sh` the same plan again (it resumes) if the budget allows |
| 7 | the heartbeat is > 10 min old (or none 15 min after the launch) | read `runs/_cloud/startup.log`; if nothing moves, tell the lead (`down.sh`) |
| 1 | usage / settings error | read the message |

   Once you have seen a PLATEAU / DIVERGED / ENDED and decided to carry on, `WATCH_IGNORE=PLATEAU,ENDED`
   (any of PLATEAU STALLED DIVERGED ENDED) stops those waking you again; DONE, a stopped or gone VM and a
   stale heartbeat always wake you.

3. More detail on one run: `python3 -m nca.progress runs/<run>-<stage>` (exit 0 PROGRESS / WARMUP / FINISHED,
   3 PLATEAU, 4 STALLED, 5 DIVERGED); the curves: http://localhost:8765 (pulled cloud runs are listed next to
   the Pi's) or the public link above.

An agent never runs `launch.sh` or `down.sh`, never deletes anything in the bucket, and never runs a gcloud
command that creates, changes or deletes. Reading (`watch.sh`, `pull.sh`, `progress`) is always fine.

## The first real boot: what to expect, what to check

| After create | Expect |
|---|---|
| ~1 min | VM RUNNING; first `status.json` (phase `driver`) in the bucket, `watch.sh` shows a heartbeat |
| ~5-12 min | the NVIDIA driver (Google's `cuda_installer.pyz`, repo mode, prod branch) installs and the VM reboots once; the heartbeat pauses for 1-2 min across the reboot |
| ~3-6 min more | phase `venv`: `pip install torch numpy` (the default Linux wheel brings its own CUDA runtime, ~3 GB of downloads) |
| then | `runs/_cloud/startup.log` has `torch <version> NVIDIA L4`. If it says "torch sees no GPU" instead, the VM gives up (DONE, ok false) rather than train on its CPU: the driver and the wheel's CUDA disagree; set `TORCH` in `env.sh` to an older build, e.g. `torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`, and launch again |
| ~15-20 min | phase `train`; each stage's first log line has `"device": "cuda"` and `"gpu": "NVIDIA L4"`; the first quick check follows within a minute; `dash/` appears on the next sidecar tick |

`SETUP_MIN` (25) budgets for all of that inside `MAX_HOURS`. Check on the first boot: the driver install
time and whether it rebooted (`startup.log`), the torch version and that it found the GPU, `gpu` utilisation
in `status.json` (several runs should keep it well above 30%), and each run's it/s in the watch block, to
size `--iters` for the next plan.

## Local test

```bash
PY=/home/dog/projects/ncawords/.venv/bin/python WORK=runs/_smoke-cloud nca/cloud/selftest.sh
```

runs `startup.sh` in its test mode (`HEXCA_TEST=1`: a local directory as the bucket, no driver / venv /
clone, never powers anything off) through a 2-run plan, a preemption (TERM) and a resume on a fresh "VM",
then `pull.sh`, every `watch.sh` exit, the ledger, `launch.sh --dry-run` refusals and `publish.sh`
(~3 minutes; the training is tiny: R 3/4, hidden 16). `SKIP_VMS=1` reuses the last bucket.
