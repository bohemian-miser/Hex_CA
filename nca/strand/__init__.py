"""Strand data and trainers for the M1 neural CA: Spectacle's line patterns on the hex lattice.

v1 (pre-rendered chords as inputs; launch 5, kept as it was):
walker   -- the lattice conventions and a walker that follows a strand from the 15 chord-bit planes alone
loader   -- reads the .npz files of scripts/strand-export.ts and makes M1a / M1b batches
parity   -- `python -m nca.strand.parity`: the walker against Spectacle's walkStrand (the exported walks)
stats    -- `python -m nca.strand.stats`: the numbers in docs/strand-data.md
train    -- `python -m nca.strand.train`: the M1 trainer (M1a a strand from a tap, M1b the closed flag)
selftest -- `python -m nca.strand.selftest`: targets vs walkStrand, oracle = exact 1.0, rollback, resume

v2 (the rule at the tap: static board facts per cell, the rule's code on the tapped cell only; launch 6):
rules     -- the whole hex kernel (1,953,569 rules) from `strand-export.ts --rule-table`: split, code, rendering,
             static input planes; `python -m nca.strand.rules --split --parity --bench`
nets      -- StrandNCA (HexNCA's step with --depth) and FrameNCA (option E, local-frame perception)
train2    -- `python -m nca.strand.train2`: the trainer (--inputs a | c | c-bc | d | d-cs | e | e-bc; m1a / m1b)
probe     -- `python -m nca.strand.probe`: the per-cell lookup probe (static planes + code -> chords, an MLP)
memprobe  -- `python -m nca.strand.memprobe`: training memory per arm (peak RSS of a gradient window), to size batches
selftest2 -- `python -m nca.strand.selftest2`: data, inputs, targets, oracles, E's equivariance, runs, resume
export    -- `python -m nca.strand.export CKPT`: a strand checkpoint (v1 or v2) as the strand play page's weights
             (web/strand.html); --board-data, --fixtures: the page's rule table and boards, the TS parity fixtures

v3 (the tap as a one-time event: lines that last, lines that die; docs/spectacle-nca-taps.md, launch 10):
sim       -- the multi-strand event simulator: taps with times, tips at one chord per two steps, the tile rule's
             hits and dying waves, absorption, refusals; the targets' interval planes and windows; episode draws
episodes  -- train3's adapter over sim: a pool slot's taps, its kinds' draws (single / collide / control / own),
             mid-episode taps
train3    -- `python -m nca.strand.train3`: --tap impulse | fixed | held, stages T1 / T2 / T3, noise and
             mid-episode taps as damage, the quick check's persist / speed / collide / own and the MLP code probe
selftest3 -- `python -m nca.strand.selftest3`: episodes vs sim, the target windows, tap events, the metrics
             against oracles, runs; --overfit: the doc's §1 Pi overfits
"""
