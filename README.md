# Hex_CA
Experimenting with cellular automata as the foundation for the spectacle  game hexagon.rodeo 

**Live demo:** https://bohemian-miser.github.io/Hex_CA/ — drag on the field to draw.

## Circuits and bridges as a cellular automaton

A player draws a line on a hexagon-shaped field of hexagons. Two things can
close it, as in Spectacle:

- **Circuit.** The line touches itself. Flood the inside of the loop.
- **Bridge.** The line runs from the border back to the border. It splits the
  border ring into two arcs. Flood the side of the **shorter arc** (the same
  "smaller arc of the outline" rule Spectacle uses for an edge-to-edge claim).
  A tie floods nothing.

Every cell runs one rule, synchronously, reading only its own state and its six
neighbours' (direction-indexed: E, NE, NW, W, SW, SE). No cell can see the
board, count, or know where the line is. Both questions are global ("which
side is inside?", "which arc is shorter?"). The rule answers them with signals
that travel along the line and along the border.

### The input

The only thing that comes from outside the rule is `place(cell)`: the line's
next cell is here, next to the head. At most one per step, the way a line grows
in the game. The placed cell stores the direction of the cell before it
(`pred`). The step after, the cell before learns its successor (`next`). So the
line is a wire: every cell of it knows both of its neighbours on it.

### What the new head sees

The step after it lands, the head looks round. Everything below starts there.

| Check | Condition (all local) | Result |
|---|---|---|
| circuit | A line cell is next to the head that is neither `pred` nor the cell before that. The cell before that is found by adding two direction vectors, so a sharp 120° turn doesn't count. | `Circuit`. The touched cell sees the same thing and marks itself the **anchor**. |
| bridge | The head is on the border, and `pred` is "left the border". Every line cell carries `anch` ∈ {free, on, left}, copied forward from `pred`: on if it is a border cell, else left if `pred` was on or left, else free. | `Bridge`. The head arms. |

A circuit takes precedence. Once either happens the line is closed and takes no
more cells.

### Circuit: which side is inside?

**Pass 1, count the turns.** A signal walks back along the line, from the head
towards the anchor. It carries a sum. Each cell adds the turn the loop makes
there, in 60° steps: `turn(pred, opposite(next))` ∈ {−2…2}. The anchor adds its
own turn and the head's (it can read the head's `pred`, next to it). A simple
closed loop of lattice cells turns exactly ±360°, which is ±6 steps. So the sum
**mod 8** comes out `6` (the walk ran anticlockwise) or `2` (clockwise). Three
bits are enough, however long the loop. The tests check that no other value
ever appears.

**Pass 2, fill.** The head reads the answer from the anchor beside it and sends
it back along the loop. Each cell it passes floods its neighbours on the inside
of its own turn: the left of the walk if the walk ran anticlockwise, the right
if clockwise.

Time: about twice the loop's length, then the flood spreads one cell per step.

### Bridge: which arc is shorter?

**Arm.** The head fires a pulse both ways round the border ring. On a
hexagon-shaped field, each border cell has exactly two border neighbours, so
the ring is a cycle every cell can follow locally.

**Echo.** A pulse moves one ring cell per step, away from where it came from.
It is stored as that direction (`pulse`, 6 bits). It turns back where the ring
meets the line. Old lines don't stop it.

**Judge.** The first echo back came along the shorter arc. It arrives
`2·|arc| + 2` steps after the head arms. The head floods the ring cell on that
side, and the flood fills the region. Echoes from both sides at once are a tie.

### Flood

A cell floods if a neighbour is flooded or it is a seed. The flood spreads to
every neighbour that isn't outside and isn't on the line. Old lines don't stop
it, the way a Spectacle circuit takes in everything inside it.

### State per cell

All finite and independent of the board's size:

- terrain: outside, empty, line or old line
- `border`
- `pulse` (6 bits)
- `flood`
- line cells only:
  - `pred` and `next` (7 values each)
  - `anch` (4 values)
  - `event`
  - bridge judge state (10 values)
  - `anchorDir`
  - pass-1 sum (9 values)
  - the anchor's answer
  - pass-2 orientation
  - the head's direction to the anchor

### Drawing by hand (the demo)

Dragging across hexagons wiggles. A wiggle can brush the line just behind the
head and close a "circuit" of four or five cells that holds nothing. The rule
treats any touch as a circuit. The demo's drawing tool keeps the last two
cells back while you drag, and cuts such corners before they reach the board.
A touch at least 6 cells back is a real circuit. This is the player's input
device, not the rule.

### Several lines

Lines are resolved one at a time. Once the board is quiet, `commit()` turns the
line into an old line (`Wall`) and hands back the round's flood, and a new line
can start. That is the second input from outside the rule. Old lines are
ordinary ground: pulses run through them and floods cover them. Only the
current line's cells reflect, judge or carry signals.

### Run it

```bash
npm install
npm test         # CA vs. a global reference on random bridges, loops (both ways round),
                 # freehand scribbles (sharp turns, border runs, lassos, open lines),
                 # bridge timing, and rounds over old lines
npm run typecheck
npm run build    # → dist/index.html, the demo (open it directly)
```

The reference (`expectedOutcome`) replays a drawing to find where it closes. A
circuit's inside is every cell that can't reach the border without crossing
the line. For a bridge it walks the ring from the head both ways to the line
and fills from the shorter arc.

Every push to `main` runs the typecheck and tests, builds the demo and deploys
it to GitHub Pages (`.github/workflows/pages.yml`, Pages source "GitHub
Actions"). Pull requests run the same checks without deploying (`ci.yml`).

### Layout

```
src/hex.ts     hexagon board, axial coords, direction-indexed neighbours, border ring (for checks only)
src/ca.ts      the cell state, the rule, the synchronous stepper, place() and commit()
src/lines.ts   random bridges, loops and scribbles, and the reference outcome (global; never used by the rule)
tests/         vitest
web/           the demo page (page.html + main.ts), bundled by scripts/build-web.ts
.github/       CI on pull requests; build and deploy to Pages from main
```

### Open questions and next steps

- **Many players at once.** One line resolves at a time. With several players
  drawing, signals need to know whose line they are on. A circuit's signals
  stay on their own line already. A bridge's pulses share the border ring, so
  they would need a small owner tag, or the ring would have to be lent to one
  bridge at a time.
- **Committing.** The demo commits once the board is quiet, a global
  observation. In the game a new line can start while an old flood is still
  spreading. That needs the flood to know which line bounds it (for example a
  line tag), so the old line can turn into a wall without letting it leak.
- **Ties.** A symmetric rule can't pick a side on a tie, because rotating the
  board half a turn swaps the arcs. It could break ties by absolute direction
  (cells do know which way is east).
- **Length measure.** The arc is measured in ring cells. To measure exposed
  edge length instead (a corner cell has 3 outside edges, a side cell 2), a
  corner could hold a pulse one extra step.
- **Irregular outlines.** Spectacle's hex field is a substitution patch, not a
  hexagon. On a ragged outline a border cell can have more than two border
  neighbours, so the ring needs a wall-following rule (keep the outside on one
  side) in place of "the other border neighbour".
- **Area instead of perimeter.** "Smaller *area*" is a harder question for a
  CA, because flood time measures depth, not cell count.
