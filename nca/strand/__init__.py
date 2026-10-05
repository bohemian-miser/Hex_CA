"""Strand data for the M1 neural CA: Spectacle's line patterns on the hex lattice.

walker  -- the lattice conventions and a walker that follows a strand from the 15 chord-bit planes alone
loader  -- reads the .npz files of scripts/strand-export.ts and makes M1a / M1b batches
parity  -- `python -m nca.strand.parity`: the walker against Spectacle's walkStrand (the exported walks)
stats   -- `python -m nca.strand.stats`: the numbers in docs/strand-data.md
train   -- `python -m nca.strand.train`: the M1 trainer (M1a a strand from a tap, M1b the closed flag)
selftest -- `python -m nca.strand.selftest`: targets vs walkStrand, oracle = exact 1.0, rollback, resume
"""
