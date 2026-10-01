# Hex_CA
Experimenting with cellular automata as the foundation for the spectacle  game hexagon.rodeo 

**Live demo:** https://bohemian-miser.github.io/Hex_CA/ — drag to paint walls.

## Inside, as a cellular automaton

Walls are painted on a hexagon-shaped field: any cells, in any order, a piece
at a time, added or erased whenever you like. The walls cut the field into
regions (6-connected groups of non-wall cells). A region is **inside**, and
fills, when it holds less than half of the open border:

    fill(region) ⇔ 2·b < T     b = the region's open border cells (border cells that aren't walls)
                                T = open border cells on the whole field

- **A circuit:** a region walled off from the border has b = 0, so it always fills.
- **A bridge:** a wall from edge to edge splits the open border into two arcs,
  and the side of the shorter arc fills.
- **A tie:** two equal halves both stay empty.

At most one region can hold half or more of the open border, and that one is
the outside. The fill depends only on the walls there are now, not on the
order they were painted in or when.

### The rule

The state is a stack of integer **layers**, one value per cell per layer.
Each step, for every cell at once:

1. **Perceive.** Every layer is read through the same kernel bank: the cell
   itself and its six neighbours (taps E, NE, NW, W, SW, SE). These seven taps
   per layer are all a cell ever sees.
2. **Update.** One rule maps that perception to the cell's new value in every
   layer.

`src/engine.ts` is the generic engine: it holds the layers, applies the kernel
bank and calls the rule, and knows nothing about walls or fills. Cells past
the edge hold fixed values. The host may write only the rule's input layers.
`src/inside.ts` is the rule: the layer list and the update.

The only input is the `wall` layer and the output is `fill`; the rest are
hidden layers. N is the number of cells, P the number of border cells.

| Layer | Update, from the kernel taps | Values |
|---|---|---|
| `out` | Fixed: 1 past the edge, 0 on the board. A border cell reads its `out` taps to find `prev`, the tap just before its run of outside neighbours going anticlockwise. | bit |
| `wall` | Input: written by the host, never by the rule. | bit |
| `idx` | 0 at the east corner (outside exactly at taps 5, 0, 1), else `idx(prev) + 1`. The cell's position round the border. | 0…P |
| `cnt` | `open + cnt(prev)`, restarting at the east corner. Open border cells so far. | 0…P |
| `tot` | At the east corner `cnt(prev)`, which is the total T; elsewhere `tot(prev)`. | 0…P |
| `lab`, `d` | Lexicographic min-plus: `min( (idx, 0) if an open border cell, (lab_k, d_k + 1) over non-wall taps k )`, with `d` capped at N. | 0…P or none; 0…N |
| `par` | The first tap with the same label and `d − 1`. | tap or none |
| `s` | `open + Σ s(k)` over taps k whose `par` points back here. Open border cells in my subtree. | 0…P |
| `v` | At the root (`d = 0`), `2·s < tot`; elsewhere `v(par)`. | bit |
| `fill` | `not wall and (lab = none or v)` | bit |

**What each layer is for:**
- `lab` names a region by its smallest border position, and `d` is the
  distance to that cell. Together they also detect a walled-in region: a label
  whose source has been cut off has no cell holding `d = 0`, so its `d` counts
  up to the cap and it dies. A region with no border keeps no label at all.
- `par` makes each region a spanning tree, rooted at its label's source.
- `s` counts that region's open border cells up the tree, so the root learns b.
- `v` is the root's verdict, copied down the tree to every cell in the region.

**Is it a cellular automaton?** Yes: it is uniform, synchronous,
deterministic, and reads only radius-1 kernel taps. The input is one layer,
and the host never writes any other.

- **Absolute directions.** They appear in exactly two places: the east corner
  anchors the border numbering, and `par` breaks ties by lowest direction.
  Neither changes the answer; the tests check that a rotated picture gives the
  rotated fill.
- **State size.** It is not finite in the strict, board-independent sense:
  the counters go up to N and P, so a cell holds O(log N) bits. That is the
  price of a rule that is static and recovers from any state.
  - An event-driven version with constant state can't tell a wall painted
    long ago from one painted just now. That is exactly how drawing a circuit
    in two halves broke the earlier version.

### Settling

Each hidden layer is a fixed point of its update, given the layers above it
in the table. So the stack settles to the same answer from **any** starting
state, and a wall painted or erased mid-way is simply absorbed. "Scramble
state" in the demo fills every hidden layer with garbage to show it, and the
layer picker shows any single layer as a heat map.

After the last change, settling takes:

1. about P steps for the border numbering and count;
2. up to N steps for a cut-off label to count up and die;
3. twice the region's depth for `s` and `v`.

Step 2 is the slow one: a circuit fills about N steps after it closes (about
600 steps at radius 14). Each step is cheap (about 0.2 ms at radius 14 in
JavaScript), so the demo runs hundreds of steps per frame.

### Run it

```bash
npm install
npm test         # vs. a whole-board reference: random wall pictures, a circuit drawn in two
                 # halves, bridges painted in random order, edits mid-settle (adding and
                 # erasing, several per step), recovery from garbage state, rotation
npm run typecheck
npm run build    # → dist/index.html, the demo (open it directly)
```

The reference (`expectedFill`) does the same job over the whole board: it
finds the regions, counts b and T, and fills when `2·b < T`. Breaking the
rule's tie test, its label cap or its verdict copy each makes the tests fail.

Every push to `main` runs the typecheck and tests, builds the demo and deploys
it to GitHub Pages. Pull requests run the same checks without deploying.

### Layout

```
src/hex.ts     hexagon board, axial coords, direction-indexed neighbours, border ring (for checks only)
src/engine.ts  the generic layered CA: layers, the hex kernel bank, one update per cell
src/inside.ts  the rule: the layer list and the update (input `wall`, output `fill`)
src/lines.ts   random walls (bridges, loops, scribbles, specks) and the reference fill
tests/         vitest
web/           the demo page (page.html + main.ts), bundled by scripts/build-web.ts
.github/       CI on pull requests; build and deploy to Pages from main
```

### Open questions and next steps

- **Faster death for cut-off labels.** A label counts up to N before it dies.
  A retraction wave (a cell whose parent is gone tells its children at once)
  would make it proportional to the region's depth. The risk is oscillation,
  so it wants the same garbage-state tests.
- **The game on top.** Spectacle's semantics are per line: rival lines are
  captured rather than walled off, and each line claims its own shorter side.
  They can be layered on as patterns and owners once this base is trusted.
- **Irregular outlines.** Spectacle's hex field is a substitution patch, not a
  hexagon. On a ragged outline a border cell can have more than two border
  neighbours, so `next` and `prev` would need a wall-following rule.
- **Length measure.** The border is measured in cells. Exposed edge length
  (a corner cell has 3 outside edges, a side cell 2) would weight `open` by
  the number of outside neighbours.
