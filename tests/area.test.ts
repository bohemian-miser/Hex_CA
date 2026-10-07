// The area layer (src/game/area.ts, docs/spectacle-ca-hybrid.md §3 and §5 "Area"): the frame, walls from the line
// CA's channels, the fill against the oracle on strand loops, territory with contested cells, and the pocket.
// The line CA here is a stub of §4.1's `LineCA.ch` (owner and D only), drawn straight from `allStrands`.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { areaFrame, CpuArea, cpuArea, territoryOf, type LineChannels } from '../src/game/area.js';
import { rng } from '../src/lines.js';
import { hexDist, loadWeights, NEIGHBOURS, targets } from '../src/nca.js';
import { allStrands, Board, DCOL, DROW, RuleTable, type BoardData, type Rule, type Strand, type StrandData } from '../src/strand.js';

const readJson = (path: string): unknown => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const data = readJson('../web/strand-data.json') as StrandData;
const table = new RuleTable(data);
const weights = loadWeights(readJson('../web/nca-weights.json'));
const boards = Object.fromEntries(Object.entries(data.boards).map(([k, b]) => [k, new Board(b)]));

/** §4.1's LineCA channels, owner and D, drawn by hand. */
class StubLines implements LineChannels {
  readonly ch: { owner: Int32Array; D: Int32Array };
  constructor(readonly board: Board) {
    this.ch = { owner: new Int32Array(board.n), D: new Int32Array(board.n) };
  }

  /** Draws a strand's chords for `owner`; the cells it touched. */
  strand(owner: number, s: Strand): Int32Array {
    const out = new Set<number>();
    for (let k = 0; k < s.rows.length; k++) {
      const c = this.board.cellOf[s.rows[k] * this.board.w + s.cols[k]];
      this.ch.D[c] |= (1 << s.ins[k]) | (1 << s.outs[k]);
      this.ch.owner[c] = owner;
      out.add(c);
    }
    return Int32Array.from(out);
  }

  /** Marks whole cells as `owner`'s line tiles (a chord through edges 0 and 3: walls only care that one is there). */
  cells(owner: number, cells: number[]): Int32Array {
    for (const c of cells) { this.ch.D[c] = owner ? 0b1001 : 0; this.ch.owner[c] = owner; }
    return Int32Array.from(cells);
  }

  wipe(cells: ArrayLike<number>): Int32Array {
    for (let k = 0; k < cells.length; k++) { this.ch.D[cells[k]] = 0; this.ch.owner[cells[k]] = 0; }
    return Int32Array.from(cells);
  }
}

const randomRule = (rand: () => number): Rule => {
  const s = Math.floor(rand() * table.nSub);
  return { s, digits: table.nOpt[s].map((n) => Math.floor(rand() * n)) };
};

/** A hexagon of radius r as a strand Board (every tile geo 0): small and symmetric, for hand-drawn walls. */
function hexBoard(r: number): Board {
  const side = 2 * r + 1;
  const geo: number[] = [];
  for (let row = 0; row < side; row++) for (let col = 0; col < side; col++) geo.push(hexDist(col - r, row - r) <= r ? 0 : -1);
  const bd: BoardData = { level: 0, root: 'hex', h: side, w: side, mirror: 1, tiles: geo.filter((g) => g >= 0).length, geo };
  return new Board(bd);
}
/** The cells of `board` (a hexBoard of radius r) at hex distance exactly k from its centre. */
function ring(board: Board, r: number, k: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < board.n; i++) {
    const row = Math.floor(board.pos[i] / board.w);
    const col = board.pos[i] % board.w;
    if (hexDist(col - r, row - r) === k) out.push(i);
  }
  return out;
}

describe('the frame (§3.1)', () => {
  it('places every board cell in an odd S×S array with a one-cell margin, and keeps every neighbour', () => {
    // The NCA's six neighbour vectors are the strand board's own (as sets).
    const ours = new Set(DROW.map((dr, d) => `${dr},${DCOL[d]}`));
    expect(new Set(NEIGHBOURS.map(([dr, dc]) => `${dr},${dc}`))).toEqual(ours);
    for (const [name, b] of Object.entries(boards)) {
      const f = areaFrame(b);
      expect(f.S % 2, name).toBe(1);
      expect(f.S).toBeGreaterThanOrEqual(Math.max(b.h, b.w) + 2);
      expect(f.S).toBeLessThanOrEqual(Math.max(b.h, b.w) + 3);
      expect(f.R).toBe((f.S - 1) / 2);
      expect(f.mask.reduce((a, v) => a + v, 0)).toBe(b.n);
      for (let i = 0; i < b.n; i++) {
        const row = Math.floor(f.slot[i] / f.S);
        const col = f.slot[i] % f.S;
        expect(f.mask[f.slot[i]]).toBe(1);
        for (let d = 0; d < 6; d++) {
          const r2 = row + DROW[d];
          const c2 = col + DCOL[d];
          expect(r2 >= 0 && r2 < f.S && c2 >= 0 && c2 < f.S).toBe(true); // the margin: a rim cell's off-board side is in the array
          const j = b.nbr[i * 6 + d];
          if (j >= 0) expect(f.slot[j]).toBe(r2 * f.S + c2);
          else expect(f.mask[r2 * f.S + c2]).toBe(0);
        }
      }
    }
  });
});

describe('walls from the line channels', () => {
  it('equal each owner’s line tiles, through draws, a change of owner and wipes', () => {
    const b = boards.l3;
    const area = new CpuArea(b, weights, 3);
    const lines = new StubLines(b);
    const rand = rng(11);
    const drawn: Int32Array[] = [];
    // Three owners, a few strands each; a strand that would share a tile with another owner's is skipped
    // (the tile rule: one rule per tile).
    for (let p = 1; p <= 3; p++) {
      const strands = allStrands(table.exits(randomRule(rand), b), b);
      for (let k = 0; k < 4 && strands.length; k++) {
        const s = strands.splice(Math.floor(rand() * strands.length), 1)[0];
        const cells = s.rows.map((r, t) => b.cellOf[r * b.w + s.cols[t]]);
        if (cells.some((c) => lines.ch.owner[c] && lines.ch.owner[c] !== p)) continue;
        const ch = lines.strand(p, s);
        drawn.push(ch);
        area.update(lines, ch);
      }
    }
    const check = () => {
      for (let p = 1; p <= 3; p++) {
        let n = 0;
        for (let i = 0; i < b.n; i++) {
          const want = lines.ch.D[i] !== 0 && lines.ch.owner[i] === p ? 1 : 0;
          expect(area.ncas[p - 1].walls[area.frame.slot[i]]).toBe(want);
          expect(area.ncas[p - 1].channel(0)[area.frame.slot[i]]).toBe(want);
          n += want;
        }
        expect(area.wallCount[p - 1]).toBe(n);
        // Nothing off the board is ever a wall.
        for (let s = 0; s < area.frame.S ** 2; s++) if (!area.frame.mask[s]) expect(area.ncas[p - 1].walls[s]).toBe(0);
      }
      for (let i = 0; i < b.n; i++) expect(area.lineOwner[i]).toBe(lines.ch.D[i] ? lines.ch.owner[i] : 0);
    };
    expect(drawn.length).toBeGreaterThan(4);
    check();
    // A tile changing hands (as a conversion would): it leaves one owner's walls and joins the other's.
    const moved = drawn[0].slice(0, 2);
    const to = (lines.ch.owner[moved[0]] % 3) + 1;
    for (const c of moved) lines.ch.owner[c] = to;
    area.update(lines, moved);
    check();
    // Wipes, half of them; an update listing unchanged cells is harmless.
    for (let k = 0; k < drawn.length; k += 2) area.update(lines, lines.wipe(drawn[k]));
    area.update(lines, Int32Array.from({ length: b.n }, (_, i) => i));
    check();
  });

  it('refuses an owner the layer has no flood for', () => {
    const b = boards.l2;
    const area = cpuArea(b, weights, 2);
    const lines = new StubLines(b);
    expect(() => area.update(lines, lines.cells(3, [0]))).toThrow(RangeError);
    expect(() => area.fill(3)).toThrow(RangeError);
    expect(() => cpuArea(b, weights, 0)).toThrow(RangeError);
  });
});

describe('the fill on strand loops at level 2', () => {
  it('after 16 R steps, never covers a wall or an off-board slot, fills only what some acceptable answer fills, and is exact on ≥ 48 of 50 loops', () => {
    // "Inside an acceptable target" is per cell (the union of targets()'s answers), not per board: a loop that cuts
    // the rim into near-even sides may fill more than one of them (the model's near-tie weakness, §3.3 (i)); the
    // exact count bounds how often a board is off at all.
    const b = boards.l2;
    const area = new CpuArea(b, weights, 1);
    const lines = new StubLines(b);
    const { S, R, slot, mask } = area.frame;
    const rand = rng(5);
    let tested = 0;
    let exact = 0;
    let enclosing = 0;
    while (tested < 50) {
      const loops = allStrands(table.exits(randomRule(rand), b), b).filter((s) => s.closed);
      if (!loops.length) continue;
      const drawn = lines.strand(1, loops[Math.floor(rand() * loops.length)]);
      area.update(lines, drawn);
      area.step(16 * R);
      const fill = area.fill(1);
      const walls = new Uint8Array(S * S);
      for (let i = 0; i < b.n; i++) if (lines.ch.D[i]) walls[slot[i]] = 1;
      const tg = targets(walls, R, mask);
      let best = Infinity;
      for (const t of tg) {
        let wrong = 0;
        for (let i = 0; i < b.n; i++) if (fill[i] !== t[slot[i]]) wrong++;
        best = Math.min(best, wrong);
      }
      for (let i = 0; i < b.n; i++) {
        if (lines.ch.D[i]) expect(fill[i]).toBe(0);
        if (fill[i]) expect(tg.some((t) => t[slot[i]])).toBe(true);
      }
      const nca = area.ncas[0];
      for (let s = 0; s < S * S; s++) if (!mask[s]) for (let c = 0; c < nca.channels; c++) expect(nca.state[c * nca.N + s]).toBe(0);
      if (best === 0) exact++;
      if (tg[0].some((v) => v)) enclosing++;
      tested++;
      area.update(lines, lines.wipe(drawn)); // no walls left: the flood goes dormant and resets
      expect(area.wallCount[0]).toBe(0);
      expect(nca.steps).toBe(0);
    }
    expect(enclosing).toBeGreaterThan(5); // not only loops round nothing
    expect(exact).toBeGreaterThanOrEqual(48);
  });
});

describe('territory', () => {
  it('resolves two players’ fills: line tiles to their owner, a cell inside both contested, the rest 0', () => {
    // Player 1 rings the centre at distance 1, player 2 at distance 2. Rival lines are transparent, so player 2's
    // fill is the centre and player 1's ring; the centre is inside both.
    const r = 4;
    const b = hexBoard(r);
    const area = cpuArea(b, weights, 2);
    const lines = new StubLines(b);
    const centre = ring(b, r, 0);
    const r1 = ring(b, r, 1);
    const r2 = ring(b, r, 2);
    area.update(lines, lines.cells(1, r1));
    area.update(lines, lines.cells(2, r2));
    area.step(16 * areaFrame(b).R);
    const f1 = area.fill(1);
    const f2 = area.fill(2);
    for (let i = 0; i < b.n; i++) {
      expect(f1[i]).toBe(centre.includes(i) ? 1 : 0);
      expect(f2[i]).toBe(centre.includes(i) || r1.includes(i) ? 1 : 0);
    }
    const t = area.territory();
    for (let i = 0; i < b.n; i++) {
      const want = centre.includes(i) ? -1 : r1.includes(i) ? 1 : r2.includes(i) ? 2 : 0;
      expect(t[i]).toBe(want);
    }
  });

  it('territoryOf: one fill → its owner, two → -1, a line tile → its owner whatever covers it', () => {
    const lineOwner = Int32Array.from([0, 0, 0, 0, 2, 0]);
    const fills = [Uint8Array.from([1, 1, 0, 0, 1, 0]), Uint8Array.from([0, 1, 1, 0, 1, 0]), Uint8Array.from([0, 0, 0, 0, 0, 0])];
    expect(Array.from(territoryOf(lineOwner, (p) => fills[p - 1], 3))).toEqual([1, -1, 2, 0, 2, 0]);
  });

  it('a pocket: a free cell whose six neighbours are one player’s line tiles fills, closed loop or not', () => {
    // Spectacle decides "inside" by the line's polygon, so a cell boxed in by an open line's tiles (a sharp U-turn,
    // or a line that comes round to its own start without joining it) is outside there; the flood sees only the
    // tiles, so it is inside here (§3.1 "Pockets"). Real strands do this: on l2, rule 01234568·143100431's
    // rim-to-rim strand through (2, 8) boxes in cell (3, 7). Here: the ring round the centre, drawn as tiles.
    const r = 3;
    const b = hexBoard(r);
    const area = cpuArea(b, weights, 1);
    const lines = new StubLines(b);
    area.update(lines, lines.cells(1, ring(b, r, 1)));
    area.step(16 * areaFrame(b).R);
    const t = area.territory();
    const [c] = ring(b, r, 0);
    expect(t[c]).toBe(1);
    expect(area.fill(1).reduce((a, v) => a + v, 0)).toBe(1);
  });

  it('the l2 pocket is real: that strand is open and its tiles box in a free cell', () => {
    const b = boards.l2;
    const rule = table.parse('01234568·143100431') as Rule;
    const ex = table.exits(rule, b);
    const s = allStrands(ex, b).find((x) => x.rows.some((rw, k) => rw === 2 && x.cols[k] === 8 && x.ins[k] + x.outs[k] === 7 && x.ins[k] * x.outs[k] === 10));
    expect(s).toBeDefined();
    expect(s!.closed).toBe(false);
    const wall = new Set(s!.rows.map((rw, k) => b.cellOf[rw * b.w + s!.cols[k]]));
    const pocket = b.cellOf[3 * b.w + 7];
    expect(wall.has(pocket)).toBe(false);
    for (let d = 0; d < 6; d++) expect(wall.has(b.nbr[pocket * 6 + d])).toBe(true);
  });
});

describe('dormant floods', () => {
  it('are not stepped while their owner has no walls, and fill nothing', () => {
    const b = boards.l2;
    const area = new CpuArea(b, weights, 3);
    const lines = new StubLines(b);
    area.update(lines, lines.cells(2, [0, 1, 2]));
    area.step(5);
    expect(area.ncas.map((n) => n.steps)).toEqual([0, 5, 0]);
    expect(area.fill(1).every((v) => v === 0)).toBe(true);
    expect(area.ncas[0].state.every((v) => v === 0)).toBe(true);
  });
});
