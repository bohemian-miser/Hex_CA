// The trained NCA's page (web/nca.html): paint walls, run the network, and
// check its fill channel against the oracle (any of its acceptable targets).
// The weights are bundled in. With ?weights=<url>&name=<label> (the training
// dashboard's Play) the page loads a run's weights instead, falling back to the
// bundled ones, and can reload them while the run trains, keeping the walls.
// A strand run's weights (format "hexca-strand") send the page on to strand.html.

import weightsJson from './nca-weights.json';
import { HexNCA, cellCoords, cellIndex, fieldMask, hexDist, loadWeights, randomBridge, side, targets, type NCAWeights } from '../src/nca.js';
import { coordsOf, indexOf, makeBoard } from '../src/hex.js';
import { randomLoop, rng } from '../src/lines.js';
import { axialAt, bounds, centreOf, cubeRound, Fills, fitGeom, hexes, type Geom } from '../src/draw.js';
import {
  css3, divLevel as sharedDivLevel, fillLevel, fitPca3, hslRgb, hueBlend, LEVELS, pcaColour, pixel,
  ramps as sharedRamps, rgbOf, type Pca3, type Ramps,
} from '../src/ramp.js';
import hexL2 from '../nca/fields/hex-l2.json';
import hexL3 from '../nca/fields/hex-l3.json';
import hexL4 from '../nca/fields/hex-l4.json';

/** The bundled weights until ?weights= brings others (setWeights). */
let weights = loadWeights(weightsJson);
let meta = weights.meta;
let C = weights.channels;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

/** Per frame, no more than this much time stepping; a backlog beyond it is dropped. */
const FRAME_MS = 12;
/** The channel tiles and the readout redraw at most this often while running. */
const GRID_MS = 100;
const READOUT_MS = 150;

/** The radius slider's top (web/nca.html's max too): about 40 ms a step there in Chromium on a Raspberry Pi 5. */
const MAX_R = 32;
/** A board this many steps per unit of radius past its last edit has had time to settle. */
const SETTLE_PER_R = 8;
/** Past this many ms/step, the status line says why a big field (Spectacle level 4: 3905
 * cells) runs slower than the speed slider asks for — scalar JS costs ~80-90 µs per
 * cell-step, and every cell keeps changing every step (measured: no cell ever goes
 * inactive, even settled, so there is no cheap way to skip work here). */
const SLOW_STEP_MS = 50;

const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));
/** The largest radius the weights were trained at (meta.trainedR is a number or a list), else a fallback. */
const trainedRs = Array.isArray(meta.trainedR) ? meta.trainedR : typeof meta.trainedR === 'number' ? [meta.trainedR] : [];
const settings = {
  radius: clampInt(trainedRs.length ? Math.min(10, Math.max(...trainedRs)) : 8, 4, MAX_R), // capped: a big board is slow in the browser
  speed: 60,
};

// ── Maps: the hexagon (radius slider) or one of Spectacle's own hex fields ─────────────────

/** A board to run the automaton on: the hexagon (R from the slider, no mask) or a fixed,
 * ragged field (R and mask fixed, from `fieldMask`). */
interface MapSpec {
  id: string;
  label: string;
  /** Cell count, for the info line; 0 for the hexagon (it varies with the radius). */
  count: number;
  /** Fixed R and mask, or undefined for the hexagon (uses settings.radius, the default full board). */
  fixed?: { R: number; mask: Uint8Array };
}

interface FieldJson { count: number; qr: [number, number][] }
const asField = (json: unknown) => json as FieldJson;

const fieldMap = (id: string, label: string, json: unknown): MapSpec => {
  const { count, qr } = asField(json);
  return { id, label, count, fixed: fieldMask(qr) };
};

const MAPS: MapSpec[] = [
  { id: 'hex', label: 'Hexagon', count: 0 },
  fieldMap('l2', 'Spectacle hex, level 2', hexL2),
  fieldMap('l3', 'Spectacle hex, level 3', hexL3),
  fieldMap('l4', 'Spectacle hex, level 4', hexL4),
];
const mapById = new Map(MAPS.map((map) => [map.id, map]));
let currentMap: MapSpec = MAPS[0];
$<HTMLSelectElement>('map').append(...MAPS.map((map) => new Option(map.label, map.id)));

let m: HexNCA;
/** targets() of the current walls (the primary first), and how many cells each fills. */
let oracles: Uint8Array[] = [];
let oracleCounts: number[] = [];
/** The step count at the last wall change (0 after a reset): live edits don't restart the count. */
let editedAt = 0;
let playing = true;
let tool: 'wall' | 'erase' = 'wall';
/** The channel on the big board, 'super' for every hidden channel superimposed, or null for the fill view. */
type ShownView = number | 'super' | null;
let shown: ShownView = null;
/** Edit-trained weights (meta.pool) carry on across edits; the rest start over. */
let resetOnEdit = meta.pool !== true;
let notice = '';
let dirty = true;
let gridDirty = true;
let readoutDirty = true;

// ── Superimposed view: every (hidden) channel folded into one picture ───────
// Two false-colour modes (src/ramp.ts): PCA's top 3 directions of a chosen set of channels → red/green/blue
// (the default: it finds whatever is most different across the board, however many channels there are), or
// each channel its own fixed hue, weighted by |value| and added together (washes out with many channels, but
// needs no basis). The PCA basis is refit only now and then (SUPER_REFIT_STEPS, or the Recompute button) —
// projecting a fitted basis every frame is cheap, refitting it isn't worth doing every frame.
type SuperScope = 'hidden' | 'all' | 'noOut';
type SuperMode = 'pca' | 'hue';
let superScope: SuperScope = 'hidden';
let superMode: SuperMode = 'pca';
let superOverlay = true;
let superBasis: Pca3 | null = null;
let superHues: [number, number, number][] = [];
let superFitStep = -1;
const SUPER_REFIT_STEPS = 50;
/** Snap step for the superimposed view's colours (RGB units): a frame fills one path per colour, same idea as
 * the quantised ramps above, so a board of thousands of nearly-but-not-quite-equal cells doesn't mean
 * thousands of separate fill() calls. */
const SUPER_RGB_STEP = 8;

/** localStorage: whether the superimposed view was picked last (sticky across visits; ?view= still wins). */
const VIEW_STORAGE_KEY = 'hexca.nca.view';
function loadStoredView(): 'super' | null {
  try {
    return localStorage.getItem(VIEW_STORAGE_KEY) === 'super' ? 'super' : null;
  } catch {
    return null; // private window, or site data cleared/blocked: just not sticky
  }
}
function storeView(v: 'super' | null): void {
  try {
    if (v) localStorage.setItem(VIEW_STORAGE_KEY, v);
    else localStorage.removeItem(VIEW_STORAGE_KEY);
  } catch {
    // ditto: the choice just isn't sticky across visits
  }
}

/** The channels the superimposed view blends, under the current scope. */
function superChannels(): number[] {
  const out: number[] = [];
  for (let c = 0; c < C; c++) {
    if (superScope === 'hidden' && c < 2) continue; // 0 wall, 1 fill
    if (superScope === 'noOut' && c === 1) continue; // fill is the one trained output
    out.push(c);
  }
  return out;
}

/** Hue mode's fixed hue per selected channel (by position, not channel number): recomputed only when the
 * selection changes, not per frame. */
function refreshSuperHues(): void {
  const n = superChannels().length;
  superHues = Array.from({ length: n }, (_, i) => hslRgb((360 * i) / Math.max(1, n), 75, 55));
}

/** Refits the PCA basis if it is missing, or `force`, or it has been SUPER_REFIT_STEPS since the last fit. */
function fitSuperBasis(force = false): void {
  if (superMode !== 'pca') return;
  if (!force && superBasis && m.steps - superFitStep < SUPER_REFIT_STEPS) return;
  const sel = superChannels();
  if (!sel.length) {
    superBasis = null;
    return;
  }
  const chans = sel.map((c) => m.channel(c));
  superBasis = fitPca3((i, k) => chans[k][i], sel.length, m.cells, superBasis ?? undefined);
  superFitStep = m.steps;
}

/** Every on-board cell's superimposed colour, added to `fills` (web/nca.ts's draw()). */
function drawSuperFills(fills: Fills, col: Colours): void {
  const sel = superChannels();
  if (!sel.length) {
    for (const i of m.cells) fills.add(col.cell, i);
    return;
  }
  const chans = sel.map((c) => m.channel(c));
  const snap = (x: number) => Math.max(0, Math.min(255, Math.round(x / SUPER_RGB_STEP) * SUPER_RGB_STEP));
  if (superMode === 'pca') {
    fitSuperBasis();
    const basis = superBasis;
    const row = new Float64Array(sel.length);
    for (const i of m.cells) {
      if (!basis) {
        fills.add(col.cell, i);
        continue;
      }
      for (let k = 0; k < sel.length; k++) row[k] = chans[k][i];
      const [r, g, b] = pcaColour(basis, row);
      fills.add(css3([snap(r), snap(g), snap(b)]), i);
    }
  } else {
    const bg = rgbOf(col.cell);
    const hueWeights = new Array<number>(sel.length);
    const span = Math.max(1e-6, HI);
    for (const i of m.cells) {
      for (let k = 0; k < sel.length; k++) hueWeights[k] = Math.min(1, Math.abs(chans[k][i]) / span);
      const [r, g, b] = hueBlend(hueWeights, superHues, bg);
      fills.add(css3([snap(r), snap(g), snap(b)]), i);
    }
  }
}

// ── The automaton ───────────────────────────────────────────────────────────

/** A model of the current map, keeping the walls that still fit, from the fresh state. */
function newModel(): void {
  const old = m as HexNCA | undefined;
  const R = currentMap.fixed ? currentMap.fixed.R : settings.radius;
  m = new HexNCA(weights, R, currentMap.fixed?.mask);
  if (old) {
    for (const i of old.cells) {
      if (!old.walls[i]) continue;
      const [q, r] = cellCoords(old.R, i);
      if (hexDist(q, r) > R) continue;
      const j = cellIndex(R, q, r);
      if (m.mask[j]) m.setWall(j, 1);
    }
  }
  m.reset();
  wallsChanged();
  layout();
  superBasis = null; // the cell set just changed under it
  superFitStep = -1;
}

function wallsChanged(): void {
  oracles = targets(m.walls, m.R, currentMap.fixed?.mask);
  oracleCounts = oracles.map((t) => m.cells.reduce((n, i) => n + t[i], 0));
  editedAt = m.steps;
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

let geom: Geom = { size: 10, ox: 0, oy: 0 };

/**
 * The largest hexagonal-cell grid that fits a w×h box with `pad` to spare, centred on the
 * board's own cells rather than on the full R hexagon — tight around a ragged field (a
 * ratio of its own cells to the bounding hexagon's, e.g. Spectacle's level 4 is under a
 * fifth), and equivalent to the old fixed formula for the hexagon itself.
 */
function fitBoard(R: number, mask: Uint8Array, w: number, h: number, pad: number): Geom {
  const S = side(R);
  return fitGeom(bounds((add) => {
    for (let row = 0; row < S; row++) for (let col = 0; col < S; col++) if (mask[row * S + col]) add(col - R, row - R);
  }), w, h, pad, 4);
}

/** The cell index whose hexagon holds (x, y) under `g`, or −1 off the board (off `mask`, not just off the full hexagon of radius R). */
function cellAtPoint(g: Geom, R: number, mask: Uint8Array, x: number, y: number): number {
  const [q, r] = axialAt(g, x, y);
  if (hexDist(q, r) > R) return -1;
  const i = cellIndex(R, q, r);
  return mask[i] ? i : -1;
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
  return cellAtPoint(geom, m.R, m.mask, ev.clientX - rect.left, ev.clientY - rect.top);
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

/** The diverging map's range: the export's clamp, else symmetric round zero. */
let [LO, HI] = weights.clamp ?? [-1, 1];
/** A value on the diverging map as a level in −LEVELS…LEVELS (negative: towards --neg). */
function divLevel(v: number): number {
  return sharedDivLevel(v, LO, HI);
}
/** The colours of fill levels 0…LEVELS and diverging levels −LEVELS…LEVELS (index + LEVELS). */
function ramps(col: Colours): Ramps {
  return sharedRamps(col.cell, col.flood, col.neg, col.pos);
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
  geom = fitBoard(m.R, m.mask, rect.width, rect.height, 16);
  cx = new Float64Array(m.N);
  cy = new Float64Array(m.N);
  for (const i of m.cells) {
    const [q, r] = cellCoords(m.R, i);
    [cx[i], cy[i]] = centreOf(geom, q, r);
  }
  dirty = true;
}

/** On-board cells where the thresholded fill and the closest of the oracle's targets disagree. */
function differing(): number[] {
  const fill = m.channel(1);
  let best: number[] = [];
  oracles.forEach((target, k) => {
    const out: number[] = [];
    for (const i of m.cells) if ((fill[i] > 0.5 ? 1 : 0) !== target[i]) out.push(i);
    if (k === 0 || out.length < best.length) best = out;
  });
  return best;
}

function draw(): void {
  dirty = false;
  const col = colours();
  const rp = ramps(col);
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const s = geom.size * 0.97;
  const fills = new Fills(ctx, cx, cy, s);
  const inset = new Fills(ctx, cx, cy, s * 0.45);
  if (shown === null) {
    const fill = m.channel(1);
    for (const i of m.cells) fills.add(m.walls[i] ? col.line : css3(rp.fill[fillLevel(fill[i])]), i);
  } else if (shown === 'super') {
    drawSuperFills(fills, col);
    if (superOverlay) for (const i of m.cells) if (m.walls[i]) inset.add(col.line, i);
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
    hexes(ctx, cx, cy, m.cells, s);
    ctx.stroke();
  }
  if (shown === null || (shown === 'super' && superOverlay)) {
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

let NAMES = Array.from({ length: C }, (_, c) => (c === 0 ? 'wall' : c === 1 ? 'fill' : 'hidden'));
/**
 * A state-channel tile is clickable (shows that channel on the board, diverging scale);
 * a const tile (mask, and in version 2 theta1/theta2) just shows its value, plain 0…1 scale.
 */
interface Tile { kind: 'state' | 'const'; index: number; el: HTMLButtonElement; cv: HTMLCanvasElement; ctx: CanvasRenderingContext2D; range: HTMLElement; text: string }
const tiles: Tile[] = [];
/** Device pixel → cell index (−1 for none), shared by every tile. */
let mini: { map: Int32Array; img: ImageData; px: Uint32Array; mask: Uint8Array; w: number; h: number } | null = null;
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
    tiles.push({ kind: 'state', index: c, el, cv, ctx: cv.getContext('2d')!, range, text: '' });
  }
  weights.consts.forEach((name, k) => {
    // A disabled button, not a plain div, so it keeps the same card chrome (background,
    // border) as the clickable tiles above — just inert, with no click handler.
    const el = document.createElement('button');
    el.className = 'tile const';
    el.type = 'button';
    el.disabled = true;
    el.title = `Constant input: ${name} (0…1, not shown on the board)`;
    const cv = document.createElement('canvas');
    const head = document.createElement('span');
    head.innerHTML = `<i>${name}</i>`;
    const range = document.createElement('i');
    el.append(cv, head, range);
    box.append(el);
    tiles.push({ kind: 'const', index: k, el, cv, ctx: cv.getContext('2d')!, range, text: '' });
  });
}

function miniLayout(): NonNullable<typeof mini> | null {
  const rect = tiles[0].cv.getBoundingClientRect();
  if (rect.width < 4) return null;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (mini && mini.mask === m.mask && mini.w === w && mini.h === h) return mini;
  const g = fitBoard(m.R, m.mask, rect.width, rect.height, 4);
  const map = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) map[y * w + x] = cellAtPoint(g, m.R, m.mask, (x + 0.5) / dpr, (y + 0.5) / dpr);
  }
  for (const t of tiles) {
    t.cv.width = w;
    t.cv.height = h;
  }
  const img = new ImageData(w, h);
  return (mini = { map, img, px: new Uint32Array(img.data.buffer), mask: m.mask, w, h });
}

function drawGrid(): void {
  const g = miniLayout();
  if (!g) return;
  gridDirty = false;
  const rp = ramps(colours());
  const divLut = rp.div.map(pixel);
  const fillLut = rp.fill.map(pixel);
  const tone = new Uint32Array(m.N);
  const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(2));
  for (const t of tiles) {
    const v = t.kind === 'state' ? m.channel(t.index) : m.consts[t.index];
    const lut = t.kind === 'state' ? divLut : fillLut;
    let lo = Infinity;
    let hi = -Infinity;
    for (const i of m.cells) {
      const x = v[i];
      if (x < lo) lo = x;
      if (x > hi) hi = x;
      tone[i] = lut[t.kind === 'state' ? divLevel(x) + LEVELS : fillLevel(x)];
    }
    const { map, px } = g;
    for (let p = 0; p < map.length; p++) px[p] = map[p] < 0 ? 0 : tone[map[p]];
    t.ctx.putImageData(g.img, 0, 0);
    const text = `${fmt(lo)} … ${fmt(hi)}`;
    if (text !== t.text) t.range.textContent = t.text = text;
  }
}

function show(c: ShownView): void {
  shown = c;
  for (const t of tiles) if (t.kind === 'state') t.el.setAttribute('aria-pressed', String(t.index === c));
  $('showFill').setAttribute('aria-pressed', String(c === null));
  $('showSuper').setAttribute('aria-pressed', String(c === 'super'));
  $('superGroup').hidden = c !== 'super';
  if (c === 'super') {
    refreshSuperHues();
    fitSuperBasis(true);
  }
  legend();
  dirty = true;
  const u = new URL(location.href);
  if (c === 'super') u.searchParams.set('view', 'super');
  else u.searchParams.delete('view');
  history.replaceState(null, '', u);
  storeView(c === 'super' ? 'super' : null);
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
  const settling = m.steps - editedAt < SETTLE_PER_R * m.R ? ' (still settling)' : '';
  const rows: Array<[string, string]> = [
    ['Step', `${m.steps}${settling}`],
    ['Filled', `${filled} / ${m.cells.length}`],
    ['Oracle fills', oracleCounts.join(' or ')],
    ['Steps/s', String(stepsPerSec)],
    ['ms/step', msPerStep ? msPerStep.toFixed(2) : '—'],
  ];
  // More than one acceptable target (area and rimcount disagree, or a tie): any one of those sides may stay unfilled.
  if (oracles.length > 1) rows.splice(3, 0, ['Oracle', oracles.length === 2 ? 'either side' : `any of ${oracles.length}`]);
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  if (badge !== shownBadge) $('verdict').innerHTML = shownBadge = badge;
  const slow = currentMap.fixed && msPerStep > SLOW_STEP_MS
    ? ` This board is large (${m.cells.length} cells): it settles at about ${stepsPerSec} steps/s, slower than the slider asks for — nothing is wrong, it is just a lot of cells for scalar JS.`
    : '';
  $('status').innerHTML = (notice || (resetOnEdit
    ? 'Drag to paint walls. Each edit restarts the network from the fresh state.'
    : 'Drag to paint walls. Edits go in live: the network carries on from where it was.')) + slow;
}

function metaLine(): string {
  const parts: string[] = [`version ${weights.version}`];
  if (typeof meta.trainedR === 'number' || Array.isArray(meta.trainedR)) parts.push(`trained at R=${[meta.trainedR].flat().join(', ')}`);
  if (Array.isArray(meta.steps)) parts.push(`${meta.steps[0]}–${meta.steps[1]} steps`);
  if (typeof meta.iterations === 'number') parts.push(`${meta.iterations} iterations`);
  parts.push(meta.pool === true ? 'edit-trained (pool)' : meta.pool === false ? 'fresh starts only' : 'pool not recorded');
  parts.push(`${C} channels, ${weights.hidden} hidden, clamp ${weights.clamp ? `[${weights.clamp.join(', ')}]` : 'none'}`);
  parts.push(`perception: ${weights.perception}`);
  parts.push(`consts: ${weights.consts.join(', ')}`);
  const note = typeof meta.note === 'string' && meta.note ? ` · “${meta.note}”` : '';
  return `Weights: ${parts.join(' · ')}${note}`;
}

/** What the scope select includes, for the legend's one-line explanation. */
function scopeLabel(): string {
  return superScope === 'hidden' ? 'the hidden channels' : superScope === 'all' ? 'every channel' : 'every channel but fill';
}

function legend(): void {
  const sw = (colour: string, label: string, cls = 'sw') => `<span><i class="${cls}" style="background:${colour}"></i>${label}</span>`;
  if (shown === 'super') {
    $('legend').innerHTML = superMode === 'pca'
      ? [`<span>Superimposed: ${scopeLabel()}, PCA's top 3 directions of spread → <b>red</b>/<b>green</b>/<b>blue</b>.</span>`,
        `<span>Refit every ${SUPER_REFIT_STEPS} steps, or on demand.</span>`].join('')
      : [`<span>Superimposed: ${scopeLabel()}, each channel its own hue, brightness = |value|, added together.</span>`].join('');
    return;
  }
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
  if (currentMap.fixed) return; // hexagon only: a ragged field has no borderRing to walk
  const board = makeBoard(m.R);
  const blocked = new Set<number>();
  for (const i of m.cells) {
    if (!m.walls[i]) continue;
    const [q, r] = cellCoords(m.R, i);
    blocked.add(indexOf(board, q, r));
  }
  const loop = randomLoop(board, rng((Math.random() * 2 ** 32) >>> 0), { blocked });
  if (!loop) return noRoom('another loop');
  edit(loop.map((slot) => {
    const [q, r] = coordsOf(board, slot);
    return cellIndex(m.R, q, r);
  }), 1);
}

function addBridge(): void {
  if (currentMap.fixed) return; // hexagon only: same reason as addLoop
  const bridge = randomBridge(m.walls, m.R, rng((Math.random() * 2 ** 32) >>> 0));
  if (!bridge) return noRoom('a bridge');
  edit(bridge, 1);
}

function noRoom(what: string): void {
  notice = `No room for ${what} here: clear some walls or raise the radius.`;
  readoutDirty = true;
}

/** Switch to a map by id (MAPS): rebuilds the model on it, keeping the walls that still fit. */
function setMap(id: string): void {
  currentMap = mapById.get(id) ?? MAPS[0];
  const isHex = !currentMap.fixed;
  $<HTMLSelectElement>('map').value = currentMap.id;
  $('radiusRow').hidden = !isHex;
  $<HTMLButtonElement>('loop').disabled = !isHex;
  $<HTMLButtonElement>('bridge').disabled = !isHex;
  $('mapInfo').textContent = isHex ? '' : `${currentMap.count} cells, fixed size (R = ${currentMap.fixed!.R}) — random loop/bridge are hexagon-only.`;
  newModel();
}

bindSlider('radius', () => newModel());
bindSlider('speed');
$<HTMLSelectElement>('map').addEventListener('change', (ev) => setMap((ev.target as HTMLSelectElement).value));
$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  m.step();
  stateChanged();
});
$('reset').addEventListener('click', () => {
  m.reset();
  editedAt = 0;
  stateChanged();
});
$('loop').addEventListener('click', addLoop);
$('bridge').addEventListener('click', addBridge);
$('clear').addEventListener('click', () => edit(Array.from(m.cells), 0));
$('showFill').addEventListener('click', () => show(null));
$('showSuper').addEventListener('click', () => show('super'));
$<HTMLSelectElement>('superScope').addEventListener('change', (ev) => {
  superScope = (ev.target as HTMLSelectElement).value as SuperScope;
  superBasis = null;
  superFitStep = -1;
  refreshSuperHues();
  if (shown === 'super') {
    legend();
    dirty = true;
  }
});
for (const mode of ['pca', 'hue'] as const) {
  $<HTMLInputElement>(`superMode-${mode}`).addEventListener('change', () => {
    superMode = mode;
    if (shown === 'super') {
      legend();
      dirty = true;
    }
  });
}
$<HTMLInputElement>('superOverlay').addEventListener('change', (ev) => {
  superOverlay = (ev.target as HTMLInputElement).checked;
  if (shown === 'super') dirty = true;
});
$('superRecompute').addEventListener('click', () => {
  fitSuperBasis(true);
  if (shown === 'super') dirty = true;
});
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

// ── Weights from a URL (the training dashboard's Play) ──────────────────────

const params = new URLSearchParams(location.search);
/** ?weights=<url>, relative to the page or absolute: a run's weights JSON (nca/export.py's format). */
const weightsUrl = params.get('weights');
/** ?name=<label>: what to call them (the run's name). */
const weightsName = params.get('name');
/** With its box ticked, the page refetches this often and swaps in weights that changed. */
const AUTO_RELOAD_MS = 120_000;
/** The fetched text now in use, to tell new weights from the same file again. */
let weightsText = '';
let source: { label: string; loaded: Date; checked?: Date; error?: string } = { label: 'bundled weights', loaded: new Date() };
let reloading = false;
let autoTimer = 0;

const why = (e: unknown) => (e instanceof Error ? e.message : String(e));
const clock = (d: Date) => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/** A strand run's weights (nca/strand/export.py) belong on the strand page: the dashboard's Play opens this
 * page for every run, so a strand run's weights land here first and go on, with the same query. */
const STRAND_PAGE = 'strand.html';

async function fetchWeights(url: string): Promise<{ text: string; w: NCAWeights }> {
  const res = await fetch(new URL(url, location.href), { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  const json = JSON.parse(text) as { format?: unknown };
  if (json && json.format === 'hexca-strand') {
    location.replace(new URL(STRAND_PAGE + location.search, location.href).href);
    throw new Error('strand weights: opening the strand page');
  }
  return { text, w: loadWeights(json) };
}

/** `w` becomes the weights the page uses: what follows from them (names, colour range) too. */
function setWeights(w: NCAWeights): void {
  weights = w;
  meta = w.meta;
  C = w.channels;
  [LO, HI] = w.clamp ?? [-1, 1];
  NAMES = Array.from({ length: C }, (_, c) => (c === 0 ? 'wall' : c === 1 ? 'fill' : 'hidden'));
  superBasis = null; // the channel count may just have changed
  superFitStep = -1;
  refreshSuperHues();
}

function showSource(): void {
  // The run's own iteration (the dashboard's) when the export says it, the --init chain's total too if it differs.
  const own = typeof meta.runIteration === 'number' ? meta.runIteration : null;
  const all = typeof meta.iterations === 'number' ? meta.iterations : null;
  const n = own ?? all;
  const it = n === null ? 'iteration not recorded' : `iteration ${n.toLocaleString()}`
    + (own !== null && all !== null && all !== own ? ` (${all.toLocaleString()} with its --init)` : '');
  const el = $('source');
  el.textContent = `${source.label} · ${it} · loaded ${clock(source.loaded)}`
    + (source.checked ? ` · checked ${clock(source.checked)}: unchanged` : '');
  if (source.error) {
    const err = document.createElement('span');
    err.className = 'err';
    err.textContent = source.error;
    el.append(err);
  }
}

/** Before the first model: the URL's weights, else the bundled ones with a notice saying why. */
async function initialWeights(): Promise<void> {
  if (!weightsUrl) return;
  $('weightsGroup').hidden = false;
  if (weightsName) document.title = `${weightsName} · ${document.title}`;
  $('status').textContent = `Loading weights from ${weightsUrl}…`;
  try {
    const { text, w } = await fetchWeights(weightsUrl);
    weightsText = text;
    setWeights(w);
    source = { label: weightsName || weightsUrl, loaded: new Date() };
    // The radius and the edit mode follow these weights, as they do the bundled ones.
    const rs = ([] as unknown[]).concat(meta.trainedR ?? []).filter((v): v is number => typeof v === 'number');
    settings.radius = clampInt(rs.length ? Math.max(...rs) : 8, 4, MAX_R);
    $<HTMLInputElement>('radius').value = $('radiusOut').textContent = String(settings.radius);
    resetOnEdit = resetBox.checked = meta.pool !== true;
  } catch (e) {
    source = { label: 'bundled weights', loaded: new Date(), error: `Could not load ${weightsUrl} (${why(e)}), so these are the bundled weights.` };
  }
  showSource();
}

/** Fetch the URL again and swap the model in: the walls stay, the state starts over. Automatic: only if the file changed. */
async function reloadWeights(auto: boolean): Promise<void> {
  if (!weightsUrl || reloading) return;
  reloading = true;
  const btn = $<HTMLButtonElement>('reloadWeights');
  btn.disabled = true;
  try {
    const { text, w } = await fetchWeights(weightsUrl);
    if (auto && text === weightsText) {
      source.checked = new Date();
    } else {
      weightsText = text;
      setWeights(w);
      tiles.length = 0;
      mini = null;
      $('tiles').replaceChildren();
      buildTiles();
      $('meta').textContent = metaLine();
      newModel();
      show(shown === 'super' ? 'super' : shown !== null && shown < C ? shown : null);
      source = { label: weightsName || weightsUrl, loaded: new Date() };
    }
    source.error = undefined;
  } catch (e) {
    source.error = `Could not reload ${weightsUrl} (${why(e)}): still the weights loaded before.`;
  } finally {
    reloading = false;
    btn.disabled = false;
    showSource();
  }
}

$('reloadWeights').addEventListener('click', () => void reloadWeights(false));
$<HTMLInputElement>('autoReload').addEventListener('change', (ev) => {
  clearInterval(autoTimer);
  if ((ev.target as HTMLInputElement).checked) autoTimer = window.setInterval(() => void reloadWeights(true), AUTO_RELOAD_MS);
});

// For the console and headless checks.
Object.assign(window, {
  hexnca: {
    get model() { return m; },
    get targets() { return oracles; },
    get weights() { return weights; },
    /** Which weights are in use (?weights=): label, load time, last error. */
    get source() { return source; },
    /** Client coordinates of the cell at axial (q, r). */
    point(q: number, r: number): [number, number] {
      const rect = canvas.getBoundingClientRect();
      const [x, y] = centreOf(geom, q, r);
      return [rect.left + x, rect.top + y];
    },
  },
});

/** ?map=<id>: which board to open on (MAPS' ids; falls back to the hexagon). */
const initialMap = params.get('map');
/** ?view=super, else the sticky choice (localStorage): open straight into the superimposed view. */
const initialView: ShownView = params.get('view') === 'super' || loadStoredView() === 'super' ? 'super' : null;

void initialWeights().then(() => {
  $('meta').textContent = metaLine();
  buildTiles();
  setMap(initialMap && mapById.has(initialMap) ? initialMap : 'hex');
  addLoop();
  show(initialView);
  setPlaying(true);
  requestAnimationFrame(frame);
});
