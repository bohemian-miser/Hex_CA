// The demo page (DESIGN.md §9): draw lines, erase and paint walls, watch the
// fill rule settle, and check the settled state against the oracle.

import { CA, type ChannelKind } from '../src/engine.js';
import { Field, hexField, presetWalls } from '../src/field.js';
import { FILLED, OFF, ON, WALL, bump, fillCA, fillRule, paint, scramble, stateOf } from '../src/fill.js';
import { coordsOf, hexDistance, indexOf } from '../src/hex.js';
import { randomLoop, rng } from '../src/lines.js';
import { oracle } from '../src/oracle.js';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

/** Channel indices, by name (every fill CA has fillRule's channel order). */
const NAMES = fillRule.channels.map((s) => s.name);
const chan = (name: string): number => {
  const c = NAMES.indexOf(name);
  if (c < 0) throw new Error(`no channel ${name}`);
  return c;
};
const PAINT = chan('paint'), EPOCH = chan('epoch'), RUN = chan('run'), COMMIT = chan('commit');
const WANT = chan('want'), STATE = chan('state');

/** Channels holding ids: drawn as hashed hues, not a scale. */
const HASHED = new Set(['id', 'gpar', 'leader', 'rpar']);
/** Region channels: meaningless on a blocked cell, which keeps its paint colour. */
const REGION = new Set(['redge', 'leader', 'dist', 'rpar', 'cdone', 'sub', 'rsize', 'want']);
/** Bits and enums: the grid and the spectrum read these on their fixed range (0..1, 0..3), not the board's. */
const FIXED: Record<string, number> = { field: 1, rim: 1, redge: 1, cdone: 1, qt: 1, want: 1, paint: 3, state: 3 };
/** Channel kinds, in the order the rule declares them. */
const KINDS = [...new Set(fillRule.channels.map((s) => s.kind))];

/** "Step to commit" gives up after this many steps. */
const SEEK_MAX = 20000;
/** Per frame: at most this many steps, and no more than this much time stepping. */
const FRAME_STEPS = 5000;
const FRAME_MS = 14;
/** Heat maps are quantised so a frame fills one path per colour, not one per cell. */
const LEVELS = 32;

type Tool = 'on' | 'erase' | 'wall';
type View = 'state' | 'diff' | 'epoch' | 'channel' | 'grid' | 'spectrum';
type Preset = 'hexagon' | 'blob' | 'lobes' | 'ring';

const settings = { radius: 20, speed: 600, preset: 'hexagon' as Preset };
let field: Field;
let ca: CA;
let root = 0;
let tool: Tool = 'on';
let view: View = 'state';
let shownChannel = chan('leader');
let playing = true;
/** The board needs a redraw; the readout needs one (cheap, and the step count moves on a quiet board). */
let dirty = true;
let readoutDirty = true;

/** The last edit: the commit it waits for and the generation it was made at. */
let pending: { serial: number; from: number } | null = null;
/** Steps from the last edit to the root's commit of it. */
let latency: number | null = null;
/** The oracle check, made once the board is quiet; null while it is not. */
let check: { ok: boolean; filled: number } | null = null;
/** "Step to commit": the root's commit when it started and the steps left. */
let seek: { from: number; left: number } | null = null;

// ── The automaton ───────────────────────────────────────────────────────────

/** A fresh automaton for the current radius and preset; the walls go in with the boot. */
function newField(): void {
  field = hexField(settings.radius);
  ca = fillCA(field);
  root = field.root;
  const walls = presetWalls(field, settings.preset);
  paint(ca, walls, WALL);
  // The boot commits epoch 0, or 1 when the walls were stamped with step 1.
  pending = { serial: walls.length ? 1 : 0, from: 0 };
  latency = null;
  check = null;
  seek = null;
  layout();
  dirty = true;
}

/** One stroke's cells as one `paint` call: one serial per pointer event. */
function edit(cells: Iterable<number>, v: 0 | 1 | 3): void {
  const p = ca.ch[PAINT];
  const isField = field.topo.isField;
  const todo: number[] = [];
  for (const c of cells) {
    // A line goes round walls, never over them.
    if (c >= 0 && isField[c] && p[c] !== v && !(v === ON && p[c] === WALL)) todo.push(c);
  }
  if (!todo.length) return;
  paint(ca, todo, v);
  touched({ serial: ca.generation + 1, from: ca.generation });
}

/** The picture or state changed: the old check and the latency no longer hold. */
function touched(next: { serial: number; from: number } | null): void {
  pending = next;
  latency = null;
  check = null;
  dirty = true;
}

function advance(): void {
  ca.step();
  stepsDone++;
  if (pending && ca.ch[COMMIT][root] >= pending.serial) {
    latency = ca.generation - pending.from;
    pending = null;
  }
  readoutDirty = true;
  if (ca.changed > 0) {
    check = null;
    dirty = true;
  } else if (!check) {
    const got = stateOf(ca);
    const p = ca.ch[PAINT];
    const want = oracle(field, (slot) => p[slot], ca.u.wallsBound === 1);
    let ok = true;
    let filled = 0;
    for (const c of field.topo.cells) {
      if (got[c] !== want[c]) ok = false;
      if (got[c] === FILLED) filled++;
    }
    check = { ok, filled };
  }
}

// ── Pointer: draw, erase, wall ──────────────────────────────────────────────

let stroke: 0 | 1 | 3 | null = null;
let lastCell = -1;
const SQ3 = Math.sqrt(3);

function cellAt(ev: PointerEvent): number {
  const rect = canvas.getBoundingClientRect();
  return cellAtPoint(geom, ev.clientX - rect.left, ev.clientY - rect.top);
}

/** Where the field sits in a w×h box: hexagon size and centre, in CSS pixels. */
interface Geom { size: number; ox: number; oy: number }

/** The largest hexagon of the field's radius that fits a w×h box with `pad` to spare, centred. */
function fitGeom(w: number, h: number, pad: number): Geom {
  const R = field.board.radius;
  return { size: Math.min((w - pad) / (SQ3 * (2 * R + 1)), (h - pad) / (3 * R + 2)), ox: w / 2, oy: h / 2 };
}

/** The centre of axial cell (q, r) under `g`. */
function centreOf(g: Geom, q: number, r: number): [number, number] {
  return [g.ox + g.size * SQ3 * (q + r / 2), g.oy + g.size * 1.5 * r];
}

/** The field slot whose hexagon holds point (x, y) under `g`, or −1. */
function cellAtPoint(g: Geom, x: number, y: number): number {
  const fx = (x - g.ox) / g.size;
  const fy = (y - g.oy) / g.size;
  const [q, r] = cubeRound((SQ3 / 3) * fx - fy / 3, (2 / 3) * fy);
  if (hexDistance(q, r) > field.board.radius) return -1;
  const slot = indexOf(field.board, q, r);
  return field.topo.isField[slot] ? slot : -1;
}

function cubeRound(fq: number, fr: number): [number, number] {
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

/** Cells on the straight hex line from a to b, both ends included. */
function hexLine(a: number, b: number): number[] {
  const board = field.board;
  const [aq, ar] = coordsOf(board, a);
  const [bq, br] = coordsOf(board, b);
  const n = hexDistance(bq - aq, br - ar);
  const out = [a];
  for (let k = 1; k <= n; k++) {
    const t = k / n;
    const [q, r] = cubeRound(aq + (bq - aq) * t + 1e-6, ar + (br - ar) * t + 1e-6);
    out.push(indexOf(board, q, r));
  }
  return out;
}

function strokeTo(cell: number): void {
  if (stroke === null || cell < 0) return;
  edit(lastCell >= 0 ? hexLine(lastCell, cell) : [cell], stroke);
  lastCell = cell;
}

const TOOL_PAINT: Record<Tool, 0 | 1 | 3> = { on: ON, erase: OFF, wall: WALL };

canvas.addEventListener('pointerdown', (ev) => {
  const cell = cellAt(ev);
  if (cell < 0) return;
  canvas.setPointerCapture(ev.pointerId);
  // The right button always erases.
  stroke = ev.button === 2 ? OFF : TOOL_PAINT[tool];
  lastCell = -1;
  strokeTo(cell);
});
canvas.addEventListener('pointermove', (ev) => {
  if (stroke !== null) strokeTo(cellAt(ev));
});
const endStroke = () => {
  stroke = null;
  lastCell = -1;
};
canvas.addEventListener('pointerup', endStroke);
canvas.addEventListener('pointercancel', endStroke);
canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

// ── Drawing ─────────────────────────────────────────────────────────────────

/** The board's geometry. */
let geom: Geom = { size: 10, ox: 0, oy: 0 };
/** Cell centres in CSS pixels, per slot (field slots only). */
let cx = new Float64Array(0);
let cy = new Float64Array(0);

function layout(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  geom = fitGeom(rect.width, rect.height, 16);
  cx = new Float64Array(field.board.size);
  cy = new Float64Array(field.board.size);
  for (const i of field.topo.cells) {
    const [q, r] = coordsOf(field.board, i);
    [cx[i], cy[i]] = centreOf(geom, q, r);
  }
}

/** Unit hexagon corners (pointy-top). */
const CORNERS = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 180) * (60 * k - 30);
  return [Math.cos(a), Math.sin(a)] as const;
});

/**
 * Hexagons at `slots`, filled or stroked as paths of at most CHUNK cells:
 * building one path of thousands of subpaths costs far more than its share
 * (half a second for 11k cells in Chromium, against ~30 ms chunked).
 */
const CHUNK = 256;
function hexes(slots: readonly number[], s: number, paint: () => void): void {
  for (let i = 0; i < slots.length; i += CHUNK) {
    ctx.beginPath();
    const end = Math.min(slots.length, i + CHUNK);
    for (let j = i; j < end; j++) {
      const x = cx[slots[j]];
      const y = cy[slots[j]];
      ctx.moveTo(x + s * CORNERS[0][0], y + s * CORNERS[0][1]);
      for (let k = 1; k < 6; k++) ctx.lineTo(x + s * CORNERS[k][0], y + s * CORNERS[k][1]);
      ctx.closePath();
    }
    paint();
  }
}

/** Cells grouped by colour, so a frame sets each colour once. */
class Fills {
  private groups = new Map<string, number[]>();
  constructor(private readonly s: number) {}
  add(colour: string, slot: number): void {
    let g = this.groups.get(colour);
    if (!g) this.groups.set(colour, (g = []));
    g.push(slot);
  }
  draw(): void {
    for (const [colour, slots] of this.groups) {
      ctx.fillStyle = colour;
      hexes(slots, this.s, () => ctx.fill());
    }
  }
}

let css: CSSStyleDeclaration;
const tok = (name: string) => css.getPropertyValue(name).trim();

function isDark(): boolean {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/** A #rrggbb colour as [r, g, b], or null when it is not one. */
function rgbOf(hex: string): [number, number, number] | null {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const css3 = ([r, g, b]: readonly number[]) => `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)})`;

/** a + (b − a)·t for two #rrggbb colours; a when either is not one. */
function mix(a: string, b: string, t: number): string {
  const pa = rgbOf(a);
  const pb = rgbOf(b);
  if (!pa || !pb) return t < 0.5 ? a : b;
  return css3(pa.map((x, k) => x * (1 - t) + pb[k] * t));
}

/** hsl (degrees, percent, percent) as [r, g, b] in 0…255. */
function hslRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100;
  l /= 100;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)));
  };
  return [f(0), f(8), f(4)];
}

const heatHsl = (t: number): [number, number, number] =>
  isDark() ? [200 - 160 * t, 55, 18 + 40 * t] : [200 - 160 * t, 65, 92 - 45 * t];
const hashedHsl = (v: number): [number, number, number] =>
  isDark() ? [(v * 137.5) % 360, 45, 42] : [(v * 137.5) % 360, 60, 72];

function heat(t: number): string {
  const [h, s, l] = heatHsl(t);
  return `hsl(${h} ${s}% ${l}%)`;
}

function hashed(v: number): string {
  const [h, s, l] = hashedHsl(v);
  return `hsl(${h} ${s}% ${l}%)`;
}

/** The board's colours, read from the tokens once per draw. */
function colours() {
  css = getComputedStyle(document.documentElement);
  return {
    cell: tok('--cell'), edge: tok('--cell-edge'), line: tok('--line'), wall: tok('--wall'),
    flood: tok('--flood'), accent: tok('--accent'), pulse: tok('--pulse'),
  };
}
type Colours = ReturnType<typeof colours>;

const paintColour = (col: Colours, v: number) =>
  v === ON ? col.line : v === WALL ? col.wall : v === FILLED ? col.flood : col.cell;

/**
 * A channel read over the field: which cells it means anything on (region
 * channels skip blocked cells), its live min…max there, and the lo…hi a
 * value is normalised over. `fixed` reads bits and enums on their whole
 * range; otherwise lo…hi is the live range, as the single-channel view has it.
 */
interface Scale { shown: (slot: number) => boolean; region: boolean; min: number; max: number; lo: number; hi: number; hashed: boolean }

function scaleOf(c: number, fixed: boolean): Scale {
  const name = NAMES[c];
  const v = ca.ch[c];
  const p = ca.ch[PAINT];
  const region = REGION.has(name);
  const cells = field.topo.cells;
  let min = Infinity;
  let max = -Infinity;
  for (let k = 0; k < cells.length; k++) {
    const slot = cells[k];
    if (region && p[slot] !== OFF) continue;
    if (v[slot] < min) min = v[slot];
    if (v[slot] > max) max = v[slot];
  }
  const top = fixed ? FIXED[name] : undefined;
  return {
    shown: region ? (slot: number) => p[slot] === OFF : () => true,
    region, min, max, lo: top === undefined ? min : 0, hi: top ?? max, hashed: HASHED.has(name),
  };
}

/** A value on its scale, 0…1, quantised to LEVELS. */
const norm = (sc: Scale, v: number) =>
  sc.hi > sc.lo ? Math.round(Math.max(0, Math.min(1, (v - sc.lo) / (sc.hi - sc.lo))) * LEVELS) / LEVELS : 0;

function draw(): void {
  dirty = false;
  if (view === 'grid') gridDirty = true;
  const col = colours();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const stateColour = (v: number) => paintColour(col, v);
  const cells = field.topo.cells;
  const cellList = Array.from(cells);
  const p = ca.ch[PAINT];
  const st = ca.ch[STATE];
  const s = geom.size * 0.97;
  const fills = new Fills(s);
  /** The picture (lines, walls) drawn small over a view that colours every cell. */
  const inset = new Fills(s * 0.45);

  if (view === 'state') {
    for (const c of cells) fills.add(stateColour(st[c]), c);
  } else if (view === 'diff') {
    const want = ca.ch[WANT];
    for (const c of cells) {
      const differs = (want[c] === 1) !== (st[c] === FILLED);
      fills.add(differs ? col.pulse : mix(col.cell, stateColour(st[c]), 0.35), c);
    }
  } else if (view === 'epoch') {
    const ep = ca.ch[EPOCH];
    let top = -Infinity;
    for (const c of cells) if (ep[c] > top) top = ep[c];
    const span = 2 * Math.max(1, field.K);
    for (const c of cells) {
      const age = top - ep[c];
      // Newest bright; older fades by age, quantised to a few shades.
      const t = age === 0 ? 1 : Math.round((0.12 + 0.3 * Math.exp(-age / span)) * LEVELS) / LEVELS;
      fills.add(mix(col.cell, col.accent, t), c);
      if (p[c] !== OFF) inset.add(stateColour(p[c]), c);
    }
  } else if (view === 'spectrum') {
    spectrum(col, fills);
  } else {
    // 'channel', and 'grid', whose board is the channel picked in the grid.
    const v = ca.ch[shownChannel];
    const sc = scaleOf(shownChannel, false);
    for (const c of cells) {
      if (!sc.shown(c)) {
        fills.add(stateColour(p[c]), c);
        continue;
      }
      fills.add(sc.hashed ? (v[c] === 0 ? col.cell : hashed(v[c])) : heat(norm(sc, v[c])), c);
      if (p[c] !== OFF) inset.add(stateColour(p[c]), c);
    }
  }
  fills.draw();
  inset.draw();
  if (geom.size > 6) {
    ctx.strokeStyle = col.edge;
    ctx.lineWidth = 0.5;
    hexes(cellList, s, () => ctx.stroke());
  }
  readoutDirty = true;
}

// ── Spectrum: every selected channel blended into one colour per cell ─────────

/** Channel kinds the spectrum blends; const is static, so off to start with. */
const spectrumKinds = new Set<ChannelKind>(KINDS.filter((k) => k !== 'const'));

const spectrumChannels = (): number[] =>
  fillRule.channels.flatMap((sp, c) => (spectrumKinds.has(sp.kind) ? [c] : []));

/** Channel i of n's hue: i/n of the way round the spectrum. */
const spectrumRgb = (i: number, n: number) => hslRgb((360 * i) / n, isDark() ? 80 : 85, isDark() ? 60 : 48);

/** Spectrum colours are snapped to this step per component, so cells share fills. */
const RGB_STEP = 8;
/** How far a blended hue is pushed back out from grey. */
const CHROMA_LIFT = 1.8;

/**
 * Each channel's normalised value weights its hue; the cell takes the
 * weighted average of the hues (chroma lifted back up, since an average of
 * many hues greys out), laid over the cell colour by the strongest weight —
 * so a cell where every channel is at its minimum stays the board's own.
 */
function spectrum(col: Colours, fills: Fills): void {
  const sel = spectrumChannels();
  const n = sel.length;
  const hue = sel.map((_, i) => spectrumRgb(i, n));
  const scales = sel.map((c) => scaleOf(c, true));
  const vals = sel.map((c) => ca.ch[c]);
  const bg = rgbOf(col.cell) ?? [255, 255, 255];
  const p = ca.ch[PAINT];
  const snap = (x: number) => Math.max(0, Math.min(255, Math.round(x / RGB_STEP) * RGB_STEP));
  for (const c of field.topo.cells) {
    if (p[c] !== OFF) {
      fills.add(paintColour(col, p[c]), c);
      continue;
    }
    let sum = 0, top = 0, r = 0, g = 0, b = 0;
    for (let i = 0; i < n; i++) {
      const w = norm(scales[i], vals[i][c]);
      if (w <= 0) continue;
      sum += w;
      if (w > top) top = w;
      r += w * hue[i][0];
      g += w * hue[i][1];
      b += w * hue[i][2];
    }
    if (sum === 0) {
      fills.add(col.cell, c);
      continue;
    }
    r /= sum;
    g /= sum;
    b /= sum;
    const m = (r + g + b) / 3;
    const lift = (x: number) => m + (x - m) * CHROMA_LIFT;
    fills.add(
      css3([snap(bg[0] + (lift(r) - bg[0]) * top), snap(bg[1] + (lift(g) - bg[1]) * top), snap(bg[2] + (lift(b) - bg[2]) * top)]),
      c,
    );
  }
}

// ── Grid: every channel as a small multiple ─────────────────────────────────

const gridPanel = $('gridPanel');
const specGroups = $('specGroups');
/** CSS size of a tile's canvas. */
const MINI_W = 120;
const MINI_H = 108;
/** Above this radius the grid redraws every GRID_EVERY frames, not every frame. */
const GRID_FAST_R = 30;
const GRID_EVERY = 3;

interface Tile { c: number; el: HTMLButtonElement; ctx: CanvasRenderingContext2D; range: HTMLElement; text: string }
const tiles: Tile[] = [];
let gridDirty = true;
/** Device pixel → field slot (−1 for none), shared by every tile; rebuilt with the field or the pixel ratio. */
let mini: { map: Int32Array; img: ImageData; px: Uint32Array; field: Field; dpr: number } | null = null;

function buildGrid(): void {
  for (const kind of KINDS) {
    const group = document.createElement('div');
    group.className = 'tgroup';
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = kind;
    const row = document.createElement('div');
    row.className = 'tiles';
    group.append(label, row);
    fillRule.channels.forEach((sp, c) => {
      if (sp.kind !== kind) return;
      const el = document.createElement('button');
      el.className = 'tile';
      el.type = 'button';
      el.title = `Show ${sp.name} on the board`;
      const cv = document.createElement('canvas');
      cv.style.width = `${MINI_W}px`;
      cv.style.height = `${MINI_H}px`;
      const head = document.createElement('span');
      head.className = 'thead';
      head.innerHTML = `<b>${sp.name}</b><i>${sp.kind}</i>`;
      const range = document.createElement('span');
      range.className = 'trange';
      el.append(cv, head, range);
      el.addEventListener('click', () => pickChannel(c));
      row.append(el);
      tiles.push({ c, el, ctx: cv.getContext('2d')!, range, text: '' });
    });
    gridPanel.append(group);
  }
}

function miniLayout(): NonNullable<typeof mini> {
  const dpr = window.devicePixelRatio || 1;
  if (mini && mini.field === field && mini.dpr === dpr) return mini;
  const w = Math.round(MINI_W * dpr);
  const h = Math.round(MINI_H * dpr);
  const g = fitGeom(MINI_W, MINI_H, 4);
  const map = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) map[y * w + x] = cellAtPoint(g, (x + 0.5) / dpr, (y + 0.5) / dpr);
  }
  for (const t of tiles) {
    t.ctx.canvas.width = w;
    t.ctx.canvas.height = h;
  }
  const img = new ImageData(w, h);
  return (mini = { map, img, px: new Uint32Array(img.data.buffer), field, dpr });
}

/** [r, g, b] as one ImageData pixel (little-endian RGBA). */
const pixel = ([r, g, b]: readonly number[]) => ((255 << 24) | (Math.round(b) << 16) | (Math.round(g) << 8) | Math.round(r)) >>> 0;

function drawGrid(): void {
  gridDirty = false;
  const m = miniLayout();
  const col = colours();
  const heatLut = Array.from({ length: LEVELS + 1 }, (_, k) => pixel(hslRgb(...heatHsl(k / LEVELS))));
  const cellPx = pixel(rgbOf(col.cell) ?? [255, 255, 255]);
  const paintPx = [cellPx, pixel(rgbOf(col.line) ?? [0, 0, 0]), cellPx, pixel(rgbOf(col.wall) ?? [128, 128, 128])];
  const hashLut = hashPixels(cellPx);
  const tone = new Uint32Array(field.board.size);
  const p = ca.ch[PAINT];
  const cells = field.topo.cells;
  for (const t of tiles) {
    const v = ca.ch[t.c];
    const sc = scaleOf(t.c, true);
    const span = sc.hi - sc.lo;
    // The single-channel view's colouring, inlined: this runs for every cell of every channel.
    for (let k = 0; k < cells.length; k++) {
      const c = cells[k];
      const x = v[c];
      if (sc.region && p[c] !== OFF) tone[c] = paintPx[p[c]];
      else if (sc.hashed) tone[c] = x > 0 && x < hashLut.length ? hashLut[x] : x === 0 ? cellPx : pixel(hslRgb(...hashedHsl(x)));
      else tone[c] = heatLut[span > 0 ? Math.round(Math.max(0, Math.min(1, (x - sc.lo) / span)) * LEVELS) : 0];
    }
    const { map, px } = m;
    for (let i = 0; i < map.length; i++) px[i] = map[i] < 0 ? 0 : tone[map[i]];
    t.ctx.putImageData(m.img, 0, 0);
    const text = sc.min > sc.max ? '—' : sc.min === sc.max ? String(sc.min) : `${sc.min} … ${sc.max}`;
    if (text !== t.text) t.range.textContent = t.text = text;
    t.el.setAttribute('aria-pressed', String(t.c === shownChannel));
  }
}

/** Hashed id colours as pixels, ids 1…N (0 is the cell colour); kept while the field and scheme last. */
let hashKept: { field: Field; dark: boolean; lut: Uint32Array } | null = null;
function hashPixels(cellPx: number): Uint32Array {
  const dark = isDark();
  if (hashKept && hashKept.field === field && hashKept.dark === dark) return hashKept.lut;
  const lut = new Uint32Array(field.N + 1);
  lut[0] = cellPx;
  for (let v = 1; v <= field.N; v++) lut[v] = pixel(hslRgb(...hashedHsl(v)));
  hashKept = { field, dark, lut };
  return lut;
}

/** A tile clicked: its channel is the board's single channel. */
function pickChannel(c: number): void {
  shownChannel = c;
  channelSel.value = String(c);
  setView(view === 'grid' ? 'grid' : 'channel');
}

// ── Readout ─────────────────────────────────────────────────────────────────

let frames = 0;
let stepsDone = 0;
let fps = 0;
let stepsPerSec = 0;
let rateFrom = performance.now();

function readout(): void {
  readoutDirty = false;
  const run = ca.ch[RUN][root];
  let filled = 0;
  for (const c of field.topo.cells) if (ca.ch[STATE][c] === FILLED) filled++;
  const ref = !check
    ? '<span class="pill wait">wait</span>'
    : check.ok ? '<span class="pill good">matches</span>' : '<span class="pill bad">differs</span>';
  const lat = latency !== null ? `${latency} steps`
    : pending ? `waiting (${ca.generation - pending.from})` : '—';
  const rows: Array<[string, string]> = [
    ['Step', String(ca.generation)],
    ['Active', `${ca.active} / ${field.N}`],
    ['Changed', String(ca.changed)],
    ['FPS', String(fps)],
    ['Steps/s', String(stepsPerSec)],
    ['Root run / K', `${run} / ${field.K}`],
    ['Commit latency', lat],
    ['Filled', String(filled)],
    ['Oracle', ref],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  let status: string;
  if (seek) status = `Stepping to the next commit… (${SEEK_MAX - seek.left} steps)`;
  else if (pending) status = `Settling: the root has seen <b>${run}</b> of ${field.K + 1} quiet steps it needs to commit.`;
  else if (check && !check.ok) status = 'Quiet, but not the right answer: garbage that agrees with itself survives. Press Bump.';
  else if (check) status = `Settled: <b>${check.filled}</b> cells filled. Drag to draw; right-drag erases.`;
  else if (latency !== null) status = 'Certified: the commit spreads out from the root, a cell per step.';
  else status = 'Drag to draw a line; close a loop and its inside fills.';
  $('status').innerHTML = status;
}

function legend(): void {
  const sw = (colour: string, label: string) =>
    `<span><i class="sw" style="background:${colour}"></i>${label}</span>`;
  const channel = HASHED.has(NAMES[shownChannel])
    ? ['ids hashed to hues', sw('var(--cell)', '0')]
    : [sw(heat(0), 'board minimum'), sw(heat(1), 'maximum')];
  const sel = spectrumChannels();
  const items: Record<View, string[]> = {
    state: [sw('var(--line)', 'line'), sw('var(--wall)', 'wall'), sw('var(--flood)', 'filled'), sw('var(--cell)', 'off')],
    diff: [sw('var(--pulse)', 'want ≠ state'), sw('var(--flood)', 'filled (dimmed)')],
    epoch: [sw('var(--accent)', 'newest epoch'), sw('var(--cell)', 'older, fading with age')],
    channel,
    grid: [`<span class="note">Board: <b>${NAMES[shownChannel]}</b>. Click a tile to show another.</span>`, ...channel],
    spectrum: sel.length
      ? ['<span class="note">Hue: the channel. Strength: its value on its own range.</span>',
        ...sel.map((c, i) => sw(css3(spectrumRgb(i, sel.length)), NAMES[c]))]
      : ['<span class="note">No channel group selected.</span>'],
  };
  const el = $('legend');
  el.innerHTML = items[view].join('');
  el.classList.toggle('strip', view === 'spectrum');
}

// ── Loop and controls ───────────────────────────────────────────────────────

let last = performance.now();
let acc = 0;
let frameNo = 0;
function frame(now: number): void {
  const dt = Math.min(100, now - last);
  last = now;
  const t0 = performance.now();
  if (seek) {
    let n = 0;
    while (seek && n < FRAME_STEPS && performance.now() - t0 < FRAME_MS) {
      advance();
      n++;
      seek.left--;
      // Done when the root commits, when nothing is left to change, or at the cap.
      if (ca.ch[COMMIT][root] !== seek.from || ca.active === 0 || seek.left <= 0) seek = null;
    }
  } else if (playing) {
    acc += (dt * settings.speed) / 1000;
    let n = 0;
    while (acc >= 1 && n < FRAME_STEPS && performance.now() - t0 < FRAME_MS) {
      advance();
      acc -= 1;
      n++;
    }
    // Capped this frame: drop the backlog rather than spiral.
    if (acc > 1) acc = 1;
  }
  frames++;
  if (now - rateFrom >= 1000) {
    fps = Math.round((frames * 1000) / (now - rateFrom));
    stepsPerSec = Math.round((stepsDone * 1000) / (now - rateFrom));
    frames = 0;
    stepsDone = 0;
    rateFrom = now;
    readoutDirty = true;
  }
  if (dirty) draw();
  if (view === 'grid' && gridDirty && (field.board.radius <= GRID_FAST_R || frameNo % GRID_EVERY === 0)) drawGrid();
  frameNo++;
  if (readoutDirty) readout();
  requestAnimationFrame(frame);
}

function setPlaying(on: boolean): void {
  playing = on;
  $('play').textContent = playing ? 'Pause' : 'Run';
}

function setTool(t: Tool): void {
  tool = t;
  $<HTMLInputElement>(`tool-${t}`).checked = true;
}

function setView(v: View): void {
  view = v;
  $<HTMLInputElement>(`view-${v}`).checked = true;
  gridPanel.hidden = v !== 'grid';
  specGroups.hidden = v !== 'spectrum';
  legend();
  dirty = true;
}

function bindSlider(id: 'radius' | 'speed', onChange?: () => void): void {
  const input = $<HTMLInputElement>(id);
  const out = $(id + 'Out');
  input.value = String(settings[id]);
  out.textContent = input.value;
  input.addEventListener('input', () => {
    settings[id] = Number(input.value);
    out.textContent = input.value;
    onChange?.();
  });
}

bindSlider('radius', newField);
bindSlider('speed');
$<HTMLSelectElement>('preset').addEventListener('change', (ev) => {
  settings.preset = (ev.target as HTMLSelectElement).value as Preset;
  newField();
});
$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  seek = null;
  advance();
});
$('toCommit').addEventListener('click', () => {
  setPlaying(false);
  seek = { from: ca.ch[COMMIT][root], left: SEEK_MAX };
  dirty = true;
});
$('loop').addEventListener('click', () => {
  const p = ca.ch[PAINT];
  const blocked = new Set<number>();
  for (const c of field.topo.cells) if (p[c] !== OFF) blocked.add(c);
  const loop = randomLoop(field.board, rng((Math.random() * 2 ** 32) >>> 0), { blocked });
  if (loop) edit(loop, ON);
});
$('clear').addEventListener('click', () => edit(field.topo.cells, OFF));
$('scramble').addEventListener('click', () => {
  scramble(ca, Math.random);
  touched(null);
});
$('bump').addEventListener('click', () => {
  const next = { serial: ca.generation + 1, from: ca.generation };
  bump(ca);
  touched(next);
});
for (const t of ['on', 'erase', 'wall'] as const) {
  $<HTMLInputElement>(`tool-${t}`).addEventListener('change', () => setTool(t));
}
for (const v of ['state', 'diff', 'epoch', 'channel', 'grid', 'spectrum'] as const) {
  $<HTMLInputElement>(`view-${v}`).addEventListener('change', () => setView(v));
}
const channelSel = $<HTMLSelectElement>('channelSel');
channelSel.innerHTML = NAMES.map((name, c) => `<option value="${c}"${c === shownChannel ? ' selected' : ''}>${name}</option>`).join('');
channelSel.addEventListener('change', () => pickChannel(Number(channelSel.value)));
buildGrid();
for (const kind of KINDS) {
  const box = $<HTMLInputElement>(`spec-${kind}`);
  box.checked = spectrumKinds.has(kind);
  box.addEventListener('change', () => {
    if (box.checked) spectrumKinds.add(kind);
    else spectrumKinds.delete(kind);
    legend();
    dirty = true;
  });
}
window.addEventListener('keydown', (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const keyTool: Record<string, Tool> = { '1': 'on', '2': 'erase', '3': 'wall' };
  if (keyTool[ev.key]) {
    ev.preventDefault();
    setTool(keyTool[ev.key]);
  } else if (ev.key === ' ' && !(ev.target as HTMLElement).closest('button, select, input')) {
    // Space on a focused control is that control's own.
    ev.preventDefault();
    setPlaying(!playing);
  }
});
new ResizeObserver(() => {
  if (!field) return;
  layout();
  dirty = true;
}).observe(canvas);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  legend();
  dirty = true;
});
new MutationObserver(() => {
  legend();
  dirty = true;
}).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

// For the console and headless checks: the automaton, and where a cell is on screen.
Object.assign(window, {
  hexca: {
    get ca() { return ca; },
    get field() { return field; },
    /** Client coordinates of the cell at axial (q, r). */
    point(q: number, r: number): [number, number] {
      const rect = canvas.getBoundingClientRect();
      const [x, y] = centreOf(geom, q, r);
      return [rect.left + x, rect.top + y];
    },
  },
});

newField();
legend();
requestAnimationFrame(frame);
