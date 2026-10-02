// The trained NCA's page (web/nca.html): paint walls, run the network, and
// check its fill channel against the oracle. The weights are bundled in.

import weightsJson from './nca-weights.json';
import { HexNCA, cellCoords, cellIndex, enclosed, hexDist, loadWeights } from '../src/nca.js';
import { coordsOf, indexOf, makeBoard } from '../src/hex.js';
import { randomLoop, rng } from '../src/lines.js';

const weights = loadWeights(weightsJson);
const meta = weights.meta;
const C = weights.channels;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

/** Per frame, no more than this much time stepping; a backlog beyond it is dropped. */
const FRAME_MS = 12;
/** Colour ramps are quantised so a frame fills one path per colour, not one per cell. */
const LEVELS = 32;
/** The channel tiles and the readout redraw at most this often while running. */
const GRID_MS = 100;
const READOUT_MS = 150;

const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));
const settings = {
  radius: clampInt(typeof meta.trainedR === 'number' ? meta.trainedR : 8, 4, 14),
  speed: 60,
};

let m: HexNCA;
/** enclosed() of the current walls, and how many cells it fills. */
let oracle: Uint8Array = new Uint8Array(0);
let oracleCount = 0;
let playing = true;
let tool: 'wall' | 'erase' = 'wall';
/** The channel on the big board, or null for the fill view. */
let shown: number | null = null;
/** Edit-trained weights (meta.pool) carry on across edits; the rest start over. */
let resetOnEdit = meta.pool !== true;
let notice = '';
let dirty = true;
let gridDirty = true;
let readoutDirty = true;

// ── The automaton ───────────────────────────────────────────────────────────

/** A model of radius R, keeping the walls that still fit, from the fresh state. */
function newModel(R: number): void {
  const old = m as HexNCA | undefined;
  m = new HexNCA(weights, R);
  if (old) {
    for (const i of old.cells) {
      if (!old.walls[i]) continue;
      const [q, r] = cellCoords(old.R, i);
      if (hexDist(q, r) <= R) m.setWall(cellIndex(R, q, r), 1);
    }
  }
  m.reset();
  wallsChanged();
  layout();
}

function wallsChanged(): void {
  oracle = enclosed(m.walls, m.R);
  oracleCount = 0;
  for (const i of m.cells) oracleCount += oracle[i];
  notice = '';
  stateChanged();
}

function stateChanged(): void {
  dirty = gridDirty = readoutDirty = true;
}

/** One stroke's cells: walls in or out, then a fresh state if the box says so. */
function edit(cells: Iterable<number>, v: 0 | 1): void {
  let changed = false;
  for (const i of cells) if (m.setWall(i, v)) changed = true;
  if (!changed) return;
  if (resetOnEdit) m.reset();
  wallsChanged();
}

// ── Geometry and pointer ────────────────────────────────────────────────────

const SQ3 = Math.sqrt(3);
interface Geom { size: number; ox: number; oy: number }
let geom: Geom = { size: 10, ox: 0, oy: 0 };

/** The largest radius-R hexagon of pointy-top cells that fits a w×h box with `pad` to spare, centred. */
const fitGeom = (R: number, w: number, h: number, pad: number): Geom =>
  ({ size: Math.max(0.5, Math.min((w - pad) / (SQ3 * (2 * R + 1)), (h - pad) / (3 * R + 2))), ox: w / 2, oy: h / 2 });

const centreOf = (g: Geom, q: number, r: number): [number, number] =>
  [g.ox + g.size * SQ3 * (q + r / 2), g.oy + g.size * 1.5 * r];

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

/** The cell index whose hexagon holds (x, y) under `g`, or −1 off the board. */
function cellAtPoint(g: Geom, R: number, x: number, y: number): number {
  const fx = (x - g.ox) / g.size;
  const fy = (y - g.oy) / g.size;
  const [q, r] = cubeRound((SQ3 / 3) * fx - fy / 3, (2 / 3) * fy);
  return hexDist(q, r) <= R ? cellIndex(R, q, r) : -1;
}

/** Cells on the straight hex line from a to b, both ends included. */
function hexLine(a: number, b: number): number[] {
  const [aq, ar] = cellCoords(m.R, a);
  const [bq, br] = cellCoords(m.R, b);
  const n = hexDist(bq - aq, br - ar);
  const out = [a];
  for (let k = 1; k <= n; k++) {
    const [q, r] = cubeRound(aq + ((bq - aq) * k) / n + 1e-6, ar + ((br - ar) * k) / n + 1e-6);
    out.push(cellIndex(m.R, q, r));
  }
  return out;
}

let stroke: 0 | 1 | null = null;
let lastCell = -1;

function cellAt(ev: PointerEvent): number {
  const rect = canvas.getBoundingClientRect();
  return cellAtPoint(geom, m.R, ev.clientX - rect.left, ev.clientY - rect.top);
}

function strokeTo(cell: number): void {
  if (stroke === null || cell < 0) return;
  edit(lastCell >= 0 ? hexLine(lastCell, cell) : [cell], stroke);
  lastCell = cell;
}

canvas.addEventListener('pointerdown', (ev) => {
  const cell = cellAt(ev);
  if (cell < 0) return;
  canvas.setPointerCapture(ev.pointerId);
  // The right button always erases.
  stroke = ev.button === 2 || tool === 'erase' ? 0 : 1;
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

// ── Colours ─────────────────────────────────────────────────────────────────

function colours() {
  const css = getComputedStyle(document.documentElement);
  const tok = (name: string) => css.getPropertyValue(name).trim();
  return {
    cell: tok('--cell'), edge: tok('--cell-edge'), line: tok('--line'), flood: tok('--flood'),
    neg: tok('--neg'), pos: tok('--pos'), bad: tok('--bad'),
  };
}
type Colours = ReturnType<typeof colours>;

/** A #rrggbb colour as [r, g, b] (grey when it is not one). */
function rgbOf(hex: string): [number, number, number] {
  const mm = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!mm) return [128, 128, 128];
  const n = parseInt(mm[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const mixRgb = (a: string, b: string, t: number): [number, number, number] => {
  const pa = rgbOf(a);
  const pb = rgbOf(b);
  return [0, 1, 2].map((k) => Math.round(pa[k] + (pb[k] - pa[k]) * t)) as [number, number, number];
};
const css3 = ([r, g, b]: readonly number[]) => `rgb(${r} ${g} ${b})`;
/** [r, g, b] as one ImageData pixel (little-endian RGBA). */
const pixel = ([r, g, b]: readonly number[]) => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;

/** Fill intensity: a value clamped to 0…1, quantised. */
const fillLevel = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * LEVELS);

/** The diverging map's range: the export's clamp, else symmetric round zero. */
const [LO, HI] = weights.clamp ?? [-1, 1];
/** A value on the diverging map as a level in −LEVELS…LEVELS (negative: towards --neg). */
function divLevel(v: number): number {
  const t = v >= 0 ? (HI > 0 ? Math.min(1, v / HI) : 0) : LO < 0 ? -Math.min(1, v / LO) : 0;
  return Math.round(t * LEVELS);
}
/** The colours of fill levels 0…LEVELS and diverging levels −LEVELS…LEVELS (index + LEVELS). */
function ramps(col: Colours) {
  const fill = Array.from({ length: LEVELS + 1 }, (_, k) => mixRgb(col.cell, col.flood, k / LEVELS));
  const div = Array.from({ length: 2 * LEVELS + 1 }, (_, k) => {
    const t = (k - LEVELS) / LEVELS;
    return mixRgb(col.cell, t < 0 ? col.neg : col.pos, Math.abs(t));
  });
  return { fill, div };
}

// ── The board ───────────────────────────────────────────────────────────────

/** Cell centres in CSS pixels, per index (on-board cells only). */
let cx = new Float64Array(0);
let cy = new Float64Array(0);

function layout(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  geom = fitGeom(m.R, rect.width, rect.height, 16);
  cx = new Float64Array(m.N);
  cy = new Float64Array(m.N);
  for (const i of m.cells) {
    const [q, r] = cellCoords(m.R, i);
    [cx[i], cy[i]] = centreOf(geom, q, r);
  }
  dirty = true;
}

const CORNERS = Array.from({ length: 6 }, (_, k) => {
  const a = (Math.PI / 180) * (60 * k - 30);
  return [Math.cos(a), Math.sin(a)] as const;
});

function hexes(cells: readonly number[], s: number): void {
  ctx.beginPath();
  for (const i of cells) {
    ctx.moveTo(cx[i] + s * CORNERS[0][0], cy[i] + s * CORNERS[0][1]);
    for (let k = 1; k < 6; k++) ctx.lineTo(cx[i] + s * CORNERS[k][0], cy[i] + s * CORNERS[k][1]);
    ctx.closePath();
  }
}

/** Cells grouped by colour, so a frame sets each colour once. */
class Fills {
  private groups = new Map<string, number[]>();
  constructor(private readonly s: number) {}
  add(colour: string, cell: number): void {
    let g = this.groups.get(colour);
    if (!g) this.groups.set(colour, (g = []));
    g.push(cell);
  }
  draw(): void {
    for (const [colour, cells] of this.groups) {
      ctx.fillStyle = colour;
      hexes(cells, this.s);
      ctx.fill();
    }
  }
}

/** On-board cells where the thresholded fill and the oracle disagree. */
function differing(): number[] {
  const fill = m.channel(1);
  const out: number[] = [];
  for (const i of m.cells) if ((fill[i] > 0.5 ? 1 : 0) !== oracle[i]) out.push(i);
  return out;
}

function draw(): void {
  dirty = false;
  const col = colours();
  const rp = ramps(col);
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const s = geom.size * 0.97;
  const fills = new Fills(s);
  const inset = new Fills(s * 0.45);
  if (shown === null) {
    const fill = m.channel(1);
    for (const i of m.cells) fills.add(m.walls[i] ? col.line : css3(rp.fill[fillLevel(fill[i])]), i);
  } else {
    const v = m.channel(shown);
    for (const i of m.cells) {
      fills.add(css3(rp.div[divLevel(v[i]) + LEVELS]), i);
      if (m.walls[i] && shown !== 0) inset.add(col.line, i);
    }
  }
  fills.draw();
  inset.draw();
  if (geom.size > 6) {
    ctx.strokeStyle = col.edge;
    ctx.lineWidth = 0.5;
    hexes(Array.from(m.cells), s);
    ctx.stroke();
  }
  if (shown === null) {
    // A dot on every cell the network gets wrong.
    const bad = differing();
    if (bad.length) {
      ctx.fillStyle = col.bad;
      ctx.beginPath();
      const rad = Math.max(1.5, s * 0.22);
      for (const i of bad) {
        ctx.moveTo(cx[i] + rad, cy[i]);
        ctx.arc(cx[i], cy[i], rad, 0, 2 * Math.PI);
      }
      ctx.fill();
    }
  }
}

// ── Channel tiles ───────────────────────────────────────────────────────────

const NAMES = Array.from({ length: C }, (_, c) => (c === 0 ? 'wall' : c === 1 ? 'fill' : 'hidden'));
interface Tile { c: number; el: HTMLButtonElement; cv: HTMLCanvasElement; ctx: CanvasRenderingContext2D; range: HTMLElement; text: string }
const tiles: Tile[] = [];
/** Device pixel → cell index (−1 for none), shared by every tile. */
let mini: { map: Int32Array; img: ImageData; px: Uint32Array; R: number; w: number; h: number } | null = null;
let lastGrid = 0;

function buildTiles(): void {
  const box = $('tiles');
  for (let c = 0; c < C; c++) {
    const el = document.createElement('button');
    el.className = 'tile';
    el.type = 'button';
    el.title = `Show channel ${c} (${NAMES[c]}) on the board`;
    const cv = document.createElement('canvas');
    const head = document.createElement('span');
    head.innerHTML = `<b>${c}</b> <i>${NAMES[c]}</i>`;
    const range = document.createElement('i');
    el.append(cv, head, range);
    el.addEventListener('click', () => show(c));
    box.append(el);
    tiles.push({ c, el, cv, ctx: cv.getContext('2d')!, range, text: '' });
  }
}

function miniLayout(): NonNullable<typeof mini> | null {
  const rect = tiles[0].cv.getBoundingClientRect();
  if (rect.width < 4) return null;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (mini && mini.R === m.R && mini.w === w && mini.h === h) return mini;
  const g = fitGeom(m.R, rect.width, rect.height, 4);
  const map = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) map[y * w + x] = cellAtPoint(g, m.R, (x + 0.5) / dpr, (y + 0.5) / dpr);
  }
  for (const t of tiles) {
    t.cv.width = w;
    t.cv.height = h;
  }
  const img = new ImageData(w, h);
  return (mini = { map, img, px: new Uint32Array(img.data.buffer), R: m.R, w, h });
}

function drawGrid(): void {
  const g = miniLayout();
  if (!g) return;
  gridDirty = false;
  const lut = ramps(colours()).div.map(pixel);
  const tone = new Uint32Array(m.N);
  const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2));
  for (const t of tiles) {
    const v = m.channel(t.c);
    let lo = Infinity;
    let hi = -Infinity;
    for (const i of m.cells) {
      const x = v[i];
      if (x < lo) lo = x;
      if (x > hi) hi = x;
      tone[i] = lut[divLevel(x) + LEVELS];
    }
    const { map, px } = g;
    for (let p = 0; p < map.length; p++) px[p] = map[p] < 0 ? 0 : tone[map[p]];
    t.ctx.putImageData(g.img, 0, 0);
    const text = `${fmt(lo)} … ${fmt(hi)}`;
    if (text !== t.text) t.range.textContent = t.text = text;
  }
}

function show(c: number | null): void {
  shown = c;
  for (const t of tiles) t.el.setAttribute('aria-pressed', String(t.c === c));
  $('showFill').setAttribute('aria-pressed', String(c === null));
  legend();
  dirty = true;
}

// ── Readout ─────────────────────────────────────────────────────────────────

let stepsDone = 0;
let stepMs = 0;
let msPerStep = 0;
let stepsPerSec = 0;
let rateFrom = performance.now();
let lastReadout = 0;
let shownBadge = '';

function readout(): void {
  readoutDirty = false;
  const fill = m.channel(1);
  let filled = 0;
  for (const i of m.cells) if (fill[i] > 0.5) filled++;
  const off = differing().length;
  const badge = off === 0
    ? '<span class="pill good">Matches oracle</span>'
    : `<span class="pill bad">Differs from oracle (${off} cell${off === 1 ? '' : 's'})</span>`;
  const rows: Array<[string, string]> = [
    ['Step', String(m.steps)],
    ['Filled', `${filled} / ${m.cells.length}`],
    ['Oracle fills', String(oracleCount)],
    ['Steps/s', String(stepsPerSec)],
    ['ms/step', msPerStep ? msPerStep.toFixed(2) : '—'],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  if (badge !== shownBadge) $('verdict').innerHTML = shownBadge = badge;
  $('status').innerHTML = notice || (resetOnEdit
    ? 'Drag to paint walls. Each edit restarts the network from the fresh state.'
    : 'Drag to paint walls. Edits go in live: the network carries on from where it was.');
}

function metaLine(): string {
  const parts: string[] = [];
  if (typeof meta.trainedR === 'number') parts.push(`trained at R=${meta.trainedR}`);
  if (Array.isArray(meta.steps)) parts.push(`${meta.steps[0]}–${meta.steps[1]} steps`);
  if (typeof meta.iterations === 'number') parts.push(`${meta.iterations} iterations`);
  parts.push(meta.pool === true ? 'edit-trained (pool)' : meta.pool === false ? 'fresh starts only' : 'pool not recorded');
  parts.push(`${C} channels, ${weights.hidden} hidden, clamp ${weights.clamp ? `[${weights.clamp.join(', ')}]` : 'none'}`);
  const note = typeof meta.note === 'string' && meta.note ? ` · “${meta.note}”` : '';
  return `Weights: ${parts.join(' · ')}${note}`;
}

function legend(): void {
  const sw = (colour: string, label: string, cls = 'sw') => `<span><i class="${cls}" style="background:${colour}"></i>${label}</span>`;
  $('legend').innerHTML = shown === null
    ? [sw('var(--line)', 'wall'), sw('var(--flood)', 'fill 1'), sw('var(--cell)', 'fill 0'), sw('var(--bad)', 'wrong vs oracle', 'sw dot')].join('')
    : [`<span>Board: channel <b>${shown}</b> (${NAMES[shown]})</span>`,
      sw('var(--neg)', String(LO)), sw('var(--cell)', '0'), sw('var(--pos)', String(HI))].join('');
}

// ── Loop and controls ───────────────────────────────────────────────────────

let last = performance.now();
let acc = 0;
function frame(now: number): void {
  const dt = Math.min(100, now - last);
  last = now;
  if (playing) {
    acc += (dt * settings.speed) / 1000;
    const t0 = performance.now();
    let n = 0;
    while (acc >= 1 && performance.now() - t0 < FRAME_MS) {
      m.step();
      acc -= 1;
      n++;
    }
    if (acc > 1) acc = 1;
    if (n) {
      stepsDone += n;
      stepMs += performance.now() - t0;
      stateChanged();
    }
  }
  if (now - rateFrom >= 1000) {
    stepsPerSec = Math.round((stepsDone * 1000) / (now - rateFrom));
    if (stepsDone) msPerStep = stepMs / stepsDone;
    stepsDone = 0;
    stepMs = 0;
    rateFrom = now;
    readoutDirty = true;
  }
  if (dirty) draw();
  if (gridDirty && now - lastGrid >= GRID_MS) {
    drawGrid();
    lastGrid = now;
  }
  if (readoutDirty && now - lastReadout >= READOUT_MS) {
    readout();
    lastReadout = now;
  }
  requestAnimationFrame(frame);
}

function setPlaying(on: boolean): void {
  playing = on;
  $('play').textContent = playing ? 'Pause' : 'Run';
}

function setTool(t: 'wall' | 'erase'): void {
  tool = t;
  $<HTMLInputElement>(`tool-${t}`).checked = true;
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

function addLoop(): void {
  const board = makeBoard(m.R);
  const blocked = new Set<number>();
  for (const i of m.cells) {
    if (!m.walls[i]) continue;
    const [q, r] = cellCoords(m.R, i);
    blocked.add(indexOf(board, q, r));
  }
  const loop = randomLoop(board, rng((Math.random() * 2 ** 32) >>> 0), { blocked });
  if (!loop) {
    notice = 'No room for another loop here: clear some walls or raise the radius.';
    readoutDirty = true;
    return;
  }
  edit(loop.map((slot) => {
    const [q, r] = coordsOf(board, slot);
    return cellIndex(m.R, q, r);
  }), 1);
}

bindSlider('radius', () => newModel(settings.radius));
bindSlider('speed');
$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  m.step();
  stateChanged();
});
$('reset').addEventListener('click', () => {
  m.reset();
  stateChanged();
});
$('loop').addEventListener('click', addLoop);
$('clear').addEventListener('click', () => edit(Array.from(m.cells), 0));
$('showFill').addEventListener('click', () => show(null));
const resetBox = $<HTMLInputElement>('resetOnEdit');
resetBox.checked = resetOnEdit;
resetBox.addEventListener('change', () => {
  resetOnEdit = resetBox.checked;
  readoutDirty = true;
});
for (const t of ['wall', 'erase'] as const) {
  $<HTMLInputElement>(`tool-${t}`).addEventListener('change', () => setTool(t));
}
window.addEventListener('keydown', (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.key === '1' || ev.key === '2') {
    ev.preventDefault();
    setTool(ev.key === '1' ? 'wall' : 'erase');
  } else if (ev.key === ' ' && !(ev.target as HTMLElement).closest('button, select, input')) {
    ev.preventDefault();
    setPlaying(!playing);
  }
});
new ResizeObserver(() => {
  if (!m) return;
  layout();
  gridDirty = true;
}).observe(canvas);
const restyle = () => {
  legend();
  dirty = gridDirty = true;
};
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', restyle);
new MutationObserver(restyle).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

// For the console and headless checks.
Object.assign(window, {
  hexnca: {
    get model() { return m; },
    get oracle() { return oracle; },
    /** Client coordinates of the cell at axial (q, r). */
    point(q: number, r: number): [number, number] {
      const rect = canvas.getBoundingClientRect();
      const [x, y] = centreOf(geom, q, r);
      return [rect.left + x, rect.top + y];
    },
  },
});

$('meta').textContent = metaLine();
buildTiles();
newModel(settings.radius);
addLoop();
show(null);
setPlaying(true);
requestAnimationFrame(frame);
