// Hex drawing and pointer maths shared by the pages (web/nca.ts, web/strand.ts, web/game.ts) and by the game
// host's taps (src/game/host.ts): pointy-top hexagons in axial (q, r), circumradius 1 in board units, cell
// (q, r) centred at (√3·(q + r/2), 1.5·r). Direction d (0..5: E, NE, NW, W, SW, SE, src/strand.ts's order)
// points at −60·d degrees, so edge d's midpoint is √3/2 out that way. Nothing here touches the DOM at load:
// the canvas functions take their context, so the host and the tests can import the pure part under Node.

export const SQ3 = Math.sqrt(3);

/** Screen placement: board units × size, then offset by (ox, oy). */
export interface Geom { size: number; ox: number; oy: number }

/** The centre of axial (q, r) in board units (size 1, origin 0). */
export const unitCentre = (q: number, r: number): [number, number] => [SQ3 * (q + r / 2), 1.5 * r];

/** The centre of axial (q, r) on screen. */
export const centreOf = (g: Geom, q: number, r: number): [number, number] =>
  [g.ox + g.size * SQ3 * (q + r / 2), g.oy + g.size * 1.5 * r];

/** The six corners of a unit hexagon, k = 0..5, at 60·k − 30 degrees. */
export const CORNERS: ReadonlyArray<readonly [number, number]> = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 180) * (60 * k - 30);
  return [Math.cos(a), Math.sin(a)] as const;
});

/** The midpoint of edge d of a hexagon of circumradius `size` centred at (x, y), `f` of the way out. */
export function edgeMid(x: number, y: number, d: number, size: number, f = 1): [number, number] {
  const a = (-Math.PI / 3) * d;
  const r = size * (SQ3 / 2) * f;
  return [x + r * Math.cos(a), y + r * Math.sin(a)];
}

/** A bounding box of unit centres. */
export interface Bounds { minX: number; maxX: number; minY: number; maxY: number }

/** The bounding box of the unit centres of axial cells, fed one at a time (`add`), so no array is built. */
export function bounds(each: (add: (q: number, r: number) => void) => void): Bounds {
  const b: Bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
  each((q, r) => {
    const x = SQ3 * (q + r / 2);
    const y = 1.5 * r;
    if (x < b.minX) b.minX = x;
    if (x > b.maxX) b.maxX = x;
    if (y < b.minY) b.minY = y;
    if (y > b.maxY) b.maxY = y;
  });
  return b;
}

/**
 * The largest grid that fits a w×h box with `pad` to spare across (and down), centred on the cells' own box
 * rather than on any enclosing hexagon — tight around a ragged field. `padY` is the extra room down, in cell
 * units: the pages have always used 4 (fill page) or 2 (strand page).
 */
export function fitGeom(b: Bounds, w: number, h: number, pad: number, padY = 2): Geom {
  const size = Math.max(0.5, Math.min((w - pad) / (b.maxX - b.minX + SQ3), (h - pad) / (b.maxY - b.minY + padY)));
  return { size, ox: w / 2 - (size * (b.minX + b.maxX)) / 2, oy: h / 2 - (size * (b.minY + b.maxY)) / 2 };
}

/** Fractional axial → the axial cell holding it. */
export function cubeRound(fq: number, fr: number): [number, number] {
  const fs = -fq - fr;
  let q = Math.round(fq);
  let r = Math.round(fr);
  const s = Math.round(fs);
  const dq = Math.abs(q - fq);
  const dr = Math.abs(r - fr);
  const ds = Math.abs(s - fs);
  if (dq > dr && dq > ds) q = -r - s;
  else if (dr > ds) r = -q - s;
  return [q, r];
}

/** The axial cell whose hexagon holds screen point (x, y) under `g` (it may be off any board). */
export function axialAt(g: Geom, x: number, y: number): [number, number] {
  const fx = (x - g.ox) / g.size;
  const fy = (y - g.oy) / g.size;
  return cubeRound((SQ3 / 3) * fx - fy / 3, (2 / 3) * fy);
}

/**
 * A tile's chords ranked by how near each runs to the point (px, py) (point-to-segment distance between the
 * two edge midpoints): the chord a tap there takes first, and the next nearest to fall back on. A stable sort,
 * so taps near the same spot always pick the same chord (they never cycle through a tile's chords).
 * (x, y, size): the tile's centre and circumradius in the same units as the point.
 */
export function rankChords(chords: readonly (readonly [number, number])[], x: number, y: number, size: number,
  px: number, py: number): [number, number][] {
  const distOf = (ch: readonly [number, number]): number => {
    const [ax, ay] = edgeMid(x, y, ch[0], size);
    const [bx, by] = edgeMid(x, y, ch[1], size);
    const vx = bx - ax;
    const vy = by - ay;
    const t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
    return (ax + vx * t - px) ** 2 + (ay + vy * t - py) ** 2;
  };
  return chords.map((c) => [c[0], c[1]] as [number, number]).sort((a, b) => distOf(a) - distOf(b));
}

/** The canvas-path methods a hexagon needs: a CanvasRenderingContext2D or a Path2D. */
interface PathSink { moveTo(x: number, y: number): void; lineTo(x: number, y: number): void }

/** One hexagon of circumradius s at (x, y), closed by a line back to its first corner: closePath() costs more
 * the longer the path is in Chromium, and a big board's path is long. */
export function hexPath(path: PathSink, x: number, y: number, s: number): void {
  path.moveTo(x + s * CORNERS[0][0], y + s * CORNERS[0][1]);
  for (let k = 1; k <= 6; k++) path.lineTo(x + s * CORNERS[k % 6][0], y + s * CORNERS[k % 6][1]);
}

/** A fresh path of hexagons on `ctx`, one per cell, centred at (xs[i], ys[i]). */
export function hexes(ctx: CanvasRenderingContext2D, xs: ArrayLike<number>, ys: ArrayLike<number>,
  cells: Iterable<number>, s: number): void {
  ctx.beginPath();
  for (const i of cells) hexPath(ctx, xs[i], ys[i], s);
}

/** Cells grouped by colour, so a frame sets each colour once. */
export class Fills {
  private groups = new Map<string, number[]>();
  constructor(private readonly ctx: CanvasRenderingContext2D, private readonly xs: ArrayLike<number>,
    private readonly ys: ArrayLike<number>, private readonly s: number) {}
  add(colour: string, cell: number): void {
    let g = this.groups.get(colour);
    if (!g) this.groups.set(colour, (g = []));
    g.push(cell);
  }
  draw(): void {
    for (const [colour, cells] of this.groups) {
      this.ctx.fillStyle = colour;
      hexes(this.ctx, this.xs, this.ys, cells, this.s);
      this.ctx.fill();
    }
  }
}

/** Strand-shaped input for `strandPaths` (src/strand.ts's Strand): tile k is (rows[k], cols[k]), entered across
 * ins[k] and left across outs[k]. */
interface StrandLike { rows: readonly number[]; cols: readonly number[]; ins: readonly number[]; outs: readonly number[]; closed: boolean }

/** Every strand's chords on screen under `g`, circuits into `solid` and tails into `dash` (the strand Board's
 * frame: tile (row, col) is axial (col, row)). Build once per (strands, camera); stroking a cached path is cheap. */
export function strandPaths(strands: readonly StrandLike[], g: Geom): { solid: Path2D; dash: Path2D } {
  const solid = new Path2D();
  const dash = new Path2D();
  for (const st of strands) {
    const path = st.closed ? solid : dash;
    for (let k = 0; k < st.rows.length; k++) {
      const [x, y] = centreOf(g, st.cols[k], st.rows[k]);
      path.moveTo(...edgeMid(x, y, st.ins[k], g.size));
      path.lineTo(...edgeMid(x, y, st.outs[k], g.size));
    }
  }
  return { solid, dash };
}

/** A rule's whole pattern, faint, under everything else: circuits solid, tails dashed, with a halo underneath so
 * it still reads over a saturated tile (the casing the lines themselves get). `s`: the cell size on screen. */
export function strokePattern(ctx: CanvasRenderingContext2D, paths: { solid: Path2D; dash: Path2D }, s: number,
  halo: string, muted: string): void {
  ctx.strokeStyle = halo;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = Math.max(2, s * 0.14);
  ctx.stroke(paths.solid);
  ctx.stroke(paths.dash);
  ctx.strokeStyle = muted;
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = Math.max(1, s * 0.07);
  ctx.stroke(paths.solid);
  ctx.setLineDash([Math.max(2, s * 0.22), Math.max(2, s * 0.18)]);
  ctx.stroke(paths.dash);
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

/** Spectacle's tile colours (shared/tiles/colors.ts PALETTE_BRIGHT), by leaf type. */
export const TYPE_RGB: Record<string, [number, number, number]> = {
  Delta: [220, 220, 220], Theta: [255, 191, 191], Lambda: [255, 160, 122], Xi: [255, 242, 0], Pi: [135, 206, 250],
  Sigma: [245, 245, 220], Phi: [0, 255, 0], Psi: [0, 255, 255], Gamma: [255, 255, 255],
};

/** What `drawTiles` needs of a strand Board (src/strand.ts). */
interface TileBoard { n: number; w: number; pos: ArrayLike<number>; type(p: number): number; edgeDir(p: number, k: number): number }

/** CSS colour → [r, g, b]: #rrggbb, or the numbers of an rgb()/rgba() string. */
export function cssRgb(css: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(css.trim());
  if (m) { const n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  const nums = css.match(/[\d.]+/g)?.map(Number) ?? [255, 255, 255];
  return [nums[0], nums[1], nums[2]];
}

/**
 * The tiles of a strand board on screen under `g` (a w×h CSS-pixel view; tiles out of view skipped): Spectacle's
 * type colours mixed `mix` of the way from `cell` (or plain `cell` without `types`), outlines once there is room,
 * and each tile's rotation (a dart at local edge 0, as Spectacle's board draws it) once there is more.
 */
export function drawTiles(ctx: CanvasRenderingContext2D, board: TileBoard, leafOrder: readonly string[], g: Geom,
  w: number, h: number, o: { types: boolean; cell: string; edge: string; muted: string; mix: number }): void {
  const cell = cssRgb(o.cell);
  const groups = new Map<string, Path2D>();
  const s = g.size * 0.985;
  const outline = new Path2D();
  const darts = new Path2D();
  for (let i = 0; i < board.n; i++) {
    const p = board.pos[i];
    const [x, y] = centreOf(g, p % board.w, Math.floor(p / board.w));
    if (x < -g.size || y < -g.size || x > w + g.size || y > h + g.size) continue;
    let fill = o.cell;
    if (o.types) {
      const c = TYPE_RGB[leafOrder[board.type(p)]] ?? [255, 255, 255];
      fill = `rgb(${c.map((v, k) => Math.round(cell[k] + (v - cell[k]) * o.mix)).join(' ')})`;
    }
    let path = groups.get(fill);
    if (!path) groups.set(fill, (path = new Path2D()));
    hexPath(path, x, y, s);
    if (g.size > 5) hexPath(outline, x, y, s);
    if (o.types && g.size >= 9) {
      const d = board.edgeDir(p, 0);
      const [tx, ty] = edgeMid(x, y, d, g.size, 0.62);
      const a = (-Math.PI / 3) * d;
      const bx = x + g.size * 0.12 * Math.cos(a);
      const by = y + g.size * 0.12 * Math.sin(a);
      const wd = g.size * 0.17;
      darts.moveTo(tx, ty);
      darts.lineTo(bx - wd * Math.sin(a), by + wd * Math.cos(a));
      darts.lineTo(bx + wd * Math.sin(a), by - wd * Math.cos(a));
      darts.closePath();
    }
  }
  for (const [fill, path] of groups) {
    ctx.fillStyle = fill;
    ctx.fill(path);
  }
  ctx.strokeStyle = o.edge;
  ctx.lineWidth = 0.6;
  ctx.stroke(outline);
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = o.muted;
  ctx.fill(darts);
  ctx.globalAlpha = 1;
}
