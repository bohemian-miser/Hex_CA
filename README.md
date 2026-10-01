# Hex_CA
Experimenting with cellular automata as the foundation for the spectacle  game hexagon.rodeo 

## Exercise 1 — flood the shorter side of a line

A hexagon-shaped field of hexagons. A *line* of active cells crosses it, border
to border. Its two ends (the *feet*) split the border ring into two arcs. The
automaton must flood the region between the line and the **shorter arc** (the
same "smaller arc of the outline" rule Spectacle uses for an edge-to-edge
claim). A tie floods nothing.

Every cell runs one rule, synchronously, reading only its own state and its
six neighbours' (direction-indexed: E, NE, NW, W, SW, SE). No cell can see the
board, count, or know where the line is. The difficulty is that "which arc is
shorter" is a global question. The trick is to answer it with a race.

### The rule

| # | Phase | Who | What |
|---|-------|-----|------|
| 1 | border | any cell | A cell with an `outside` neighbour sets `border`. On a hexagon-shaped field each border cell has exactly two border neighbours, so the border is a ring every cell can follow locally. |
| 2 | arm | fresh-line cell on the border (a foot) | `None → Armed`. Next step: `Armed → Listening`, and both ring neighbours of an armed foot pick up a pulse. |
| 3 | race | border cell, not fresh line | A pulse is stored as the direction it came from (`pulse`, 6 bits). Each step it moves to the cell's *other* ring neighbour. Pulses pass through one another and through old lines; nothing else on the ring stops them. |
| 4 | judge | listening foot | Hearing a pulse heading into it from exactly one ring neighbour d: `Decided(d)`. From both at once: `Tie`. Later pulses are swallowed and ignored. |
| 5 | flood | not outside, not fresh line | Flooded if a neighbour is flooded, or the neighbour in direction d is a foot `Decided(opp(d))` (the seed: the first cell of the winning arc). |

Why it works: both feet fire at the same step. The pulse foot A sends round
arc *k* reaches foot B after |arc *k*| steps, and B's pulse reaches A along
the same arc at the same moment. So the first pulse either foot hears came
along the shorter arc, and both feet hear it at step `2 + min(|arc|)`. They
always agree, with no messages between them. The flood starts at both ends of
the shorter arc and spreads one cell per step. It cannot cross the line, so it
fills exactly that side.

State per cell: terrain (outside / empty / fresh line / old line), `border`
(1 bit), foot (none, armed, listening, tie, decided×6), `pulse` (6 bits; at
most two are ever set), `flood` (1 bit). All finite.

### What the line must be (and what the generator guarantees)

- It touches the border only at its two feet, and the feet are not ring
  neighbours. Both arcs have at least one cell.
- It is an *induced* path: no two cells of it are adjacent unless they are
  consecutive. On a hex grid such a line splits the field into exactly two
  pieces, one per arc, with no pockets.

`randomLine` builds a cheapest path between two border cells under a random,
smooth cost field. With positive costs, a cheapest path is always induced,
because any shortcut would be cheaper.

### Several lines

Lines are resolved one at a time, as in the game. `Automaton.commit(line)` is
the one input from outside the rule: the fresh line becomes an old line
(`Wall`), the round's flood is handed back, and a new fresh line is laid. Old
lines are ordinary ground: pulses run through their feet and the flood covers
them, the way a Spectacle circuit's polygon takes in everything inside it.
Only the fresh line's feet judge, so old feet never confuse the race.

### Run it

```bash
npm install
npm test         # CA vs. a global BFS reference: 400 random single lines (ties included),
                 # decision timing, quiescence, and 6-line sequences over old lines
npm run typecheck
npm run build    # → dist/hex-ca.html, a self-contained animated demo
```

### Layout

```
src/hex.ts     hexagon board, axial coords, direction-indexed neighbours, border ring (for checks only)
src/ca.ts      the cell state, the rule, and the synchronous stepper
src/lines.ts   random line generator and the reference fill (global; never used by the rule)
tests/         vitest
web/           the demo page (page.html + main.ts), bundled by scripts/build-web.ts
```

### Open questions and next steps

- **Ties.** A symmetric rule cannot pick a side on a tie: rotating the board
  half a turn swaps the arcs. It could break ties using absolute direction
  (cells do know which way is east), at the cost of a less symmetric rule.
- **Length measure.** The arc is measured in ring cells. To measure exposed
  edge length instead (a corner cell has 3 outside edges, a side cell 2), a
  corner could hold a pulse one extra step.
- **Irregular outlines.** Spectacle's hex field is a substitution patch, not
  a hexagon. On a ragged outline a border cell can have more than two border
  neighbours, so the ring needs a wall-following rule (keep the outside on
  your left) in place of "the other border neighbour".
- **Concurrent lines.** Two lines finishing in the same step would hear each
  other's pulses. Pulses carrying a small line tag, or a refractory period,
  would separate them.
- **Area instead of perimeter.** "Smaller *area*" is a different and harder
  question for a CA, because flood time measures depth, not cell count.
