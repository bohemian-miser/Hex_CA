// The strand play page (web/strand.html): tap tiles of one of Spectacle's hex fields, each tap with its own rule;
// a trained strand NCA (src/strand-nca.ts, weights from nca/strand/export.py) grows the strand of every tap, and
// the true strand (src/strand.ts, Spectacle's rule and walker) can be shown thinly underneath.
//
// Weights: ?weights=<url>&name=<label> (the training dashboard's Play), else strand-weights.json beside the page
// (the build copies web/strand-weights.json there), else the GitHub Pages copy. ?map=l2|l3|l4, ?rule=<describeRule>.

import dataJson from './strand-data.json';
import { Board, PAIRS, RuleTable, walk, type Rule, type Strand, type StrandData } from '../src/strand.js';
import { StrandNCA, chordsAt, loadStrandWeights, type StrandWeights, type Tap } from '../src/strand-nca.js';

const data = dataJson as unknown as StrandData;
const table = new RuleTable(data);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

const DEFAULT_WEIGHTS = 'strand-weights.json';
const PAGES_WEIGHTS = 'https://bohemian-miser.github.io/Hex_CA/strand-weights.json';
/** Per frame, no more than this much time stepping (at least one step when one is due). */
const FRAME_MS = 14;
/** Paused once nothing drawn has changed for this many steps (it would only burn CPU). */
const SETTLE_STEPS = 120;
/** Past this many ms a step, the status line says why the board grows slowly. */
const SLOW_MS = 120;
const SQ3 = Math.sqrt(3);
/** Spectacle's tile colours (shared/tiles/colors.ts PALETTE_BRIGHT), by leaf type. */
const TYPE_RGB: Record<string, [number, number, number]> = {
  Delta: [220, 220, 220], Theta: [255, 191, 191], Lambda: [255, 160, 122], Xi: [255, 242, 0], Pi: [135, 206, 250],
  Sigma: [245, 245, 220], Phi: [0, 255, 0], Psi: [0, 255, 255], Gamma: [255, 255, 255],
};
const MAP_LABELS: Record<string, string> = { l2: 'Level 2', l3: 'Level 3', l4: 'Level 4' };
const N_COLOURS = 8;

const params = new URLSearchParams(location.search);
const weightsUrl = params.get('weights');
const weightsName = params.get('name');

// ── State ──────────────────────────────────────────────────────────────────

interface PlayTap {
  id: number;
  tap: Tap;
  colour: number;
  strand: Strand;
  /** trueEdge[i * 6 + d]: edge d of cell i is on the true strand. */
  trueEdge: Uint8Array;
  /** cell -> the true strand's chords there. */
  trueChords: Map<number, [number, number][]>;
  /** The tap's rule's chords on every board position. */
  ruleBits: Int16Array;
  /** Its own model when taps don't share a board. */
  model: StrandNCA | null;
  exact: boolean;
  since: number | null;
  drawn: number;
  missing: number;
  stray: number;
}

let weights: StrandWeights | null = null;
let weightsText = '';
let weightsFrom = '';
let weightsError = '';
const boards = new Map<string, Board>();
let mapId = 'l3';
let board!: Board;
let rule: Rule = table.parse('15·000000000') as Rule;
let taps: PlayTap[] = [];
let nextId = 1;
let sharedModel: StrandNCA | null = null;
let shared = true;
let playing = true;
let speed = 30;
let showTruth = true;
let showTypes = true;
let notice = '';
let lastChange = 0;
let drawnKey = 0;
let dirty = true;
let tapsDirty = true;
/** The tap owning each cell, for strays on a shared board: the nearest tap's strand (BFS over cells). */
let owner = new Int16Array(0);

const why = (e: unknown) => (e instanceof Error ? e.message : String(e));
const fmt = (x: number | undefined | null, d = 2) => (typeof x === 'number' ? x.toFixed(d) : '–');

function boardOf(id: string): Board {
  let b = boards.get(id);
  if (!b) {
    b = new Board(data.boards[id]);
    boards.set(id, b);
  }
  return b;
}

const canShare = () => !!weights && !weights.ruleEverywhere;
const usingShared = () => shared && canShare();
/** Steps the models have run (a shared board's, else the oldest tap's). */
function stepCount(): number {
  if (sharedModel) return sharedModel.steps;
  return taps.reduce((a, t) => Math.max(a, t.model?.steps ?? 0), 0);
}

/** Every model rebuilt from the fresh state, the taps held from step 0. */
function buildModels(): void {
  sharedModel = null;
  for (const t of taps) t.model = null;
  if (!weights) return;
  if (usingShared()) {
    sharedModel = new StrandNCA(weights, board, table);
    sharedModel.setTaps(taps.map((t) => t.tap));
  } else {
    for (const t of taps) {
      t.model = new StrandNCA(weights, board, table);
      t.model.setTaps([t.tap]);
    }
  }
  for (const t of taps) t.since = null;
  lastChange = 0;
  drawnKey = 0;
  dirty = tapsDirty = true;
}

function makeTap(tap: Tap): PlayTap {
  const ex = table.exits(tap.rule, board);
  const strand = walk(ex, board, tap.row, tap.col, tap.d0, tap.d1);
  const trueEdge = new Uint8Array(board.n * 6);
  const trueChords = new Map<number, [number, number][]>();
  for (let k = 0; k < strand.rows.length; k++) {
    const i = board.cellOf[strand.rows[k] * board.w + strand.cols[k]];
    trueEdge[i * 6 + strand.ins[k]] = 1;
    trueEdge[i * 6 + strand.outs[k]] = 1;
    trueChords.set(i, [...(trueChords.get(i) ?? []), [strand.ins[k], strand.outs[k]]]);
  }
  const used = new Set(taps.map((t) => t.colour));
  let colour = 0;
  while (used.has(colour) && colour < N_COLOURS - 1) colour++;
  if (used.has(colour)) colour = (nextId - 1) % N_COLOURS;
  return {
    id: nextId++, tap, colour, strand, trueEdge, trueChords, ruleBits: table.render(tap.rule, board), model: null,
    exact: false, since: null, drawn: 0, missing: 0, stray: 0,
  };
}

/** The taps changed: the shared model takes the new set (its state kept, as a board in play would), or the new
 * tap gets a model of its own. */
function tapsChanged(added: PlayTap | null): void {
  if (weights) {
    if (usingShared()) {
      if (!sharedModel) buildModels();
      else sharedModel.setTaps(taps.map((t) => t.tap));
    } else if (added) {
      added.model = new StrandNCA(weights, board, table);
      added.model.setTaps([added.tap]);
    }
  }
  computeOwners();
  lastChange = stepCount();
  tapsDirty = dirty = true;
  if (taps.length && !playing) setPlaying(true);
}

function computeOwners(): void {
  owner = new Int16Array(board.n).fill(-1);
  const queue: number[] = [];
  taps.forEach((t, k) => {
    for (const i of t.trueChords.keys()) {
      if (owner[i] < 0) {
        owner[i] = k;
        queue.push(i);
      }
    }
  });
  for (let q = 0; q < queue.length; q++) {
    const i = queue[q];
    for (let d = 0; d < 6; d++) {
      const j = board.nbr[i * 6 + d];
      if (j >= 0 && owner[j] < 0) {
        owner[j] = owner[i];
        queue.push(j);
      }
    }
  }
  for (let i = 0; i < board.n; i++) if (owner[i] < 0) owner[i] = 0;
}

/** Add a tap of the current rule at cell (row, col), the chord (d0, d1); a tap already on that cell is replaced
 * (or, the same rule and chord again, taken away). */
function addTap(row: number, col: number, d0: number, d1: number): void {
  const at = taps.findIndex((t) => t.tap.row === row && t.tap.col === col);
  if (at >= 0) {
    const old = taps[at];
    const same = table.describe(old.tap.rule) === table.describe(rule)
      && ((old.tap.d0 === d0 && old.tap.d1 === d1) || (old.tap.d0 === d1 && old.tap.d1 === d0));
    taps.splice(at, 1);
    if (same) {
      notice = 'Tap taken away (the same rule and chord again).';
      tapsChanged(null);
      return;
    }
  }
  if (weights?.ruleEverywhere && taps.length >= 8) {
    notice = 'Eight taps is plenty when each needs a board of its own.';
    return;
  }
  const t = makeTap({ rule: { s: rule.s, digits: rule.digits.slice() }, row, col, d0, d1 });
  taps.push(t);
  notice = '';
  tapsChanged(t);
}

function removeTap(id: number): void {
  taps = taps.filter((t) => t.id !== id);
  tapsChanged(null);
}

/** Board positions by distance from the board's centre. */
function byCentre(): number[] {
  let mx = 0;
  let my = 0;
  for (let i = 0; i < board.n; i++) {
    const [x, y] = unitCentre(board.pos[i]);
    mx += x / board.n;
    my += y / board.n;
  }
  const d = (p: number) => { const [x, y] = unitCentre(p); return (x - mx) ** 2 + (y - my) ** 2; };
  return Array.from(board.pos).sort((a, b) => d(a) - d(b));
}

/** A tap of `r` near the board's centre whose strand is worth watching (lo..hi chords if one is close, else the
 * longest of the nearest few), or null if the rule draws nothing there. */
function centralChoice(r: Rule, lo = 8, hi = 80): { row: number; col: number; d0: number; d1: number; n: number } | null {
  const ex = table.exits(r, board);
  let best: { row: number; col: number; d0: number; d1: number; n: number } | null = null;
  const near = byCentre().slice(0, 48);
  for (const p of near) {
    const row = Math.floor(p / board.w);
    const col = p % board.w;
    for (const [d0, d1] of chordsAt(table, r, board, row, col)) {
      const n = walk(ex, board, row, col, d0, d1).rows.length;
      if (n >= lo && n <= hi) return { row, col, d0, d1, n };
      if (!best || (best.n < lo ? n > best.n : n < best.n)) best = { row, col, d0, d1, n };
    }
  }
  return best;
}

/** The first tap on a fresh board: the current rule near the centre. */
function centralTap(): void {
  const c = centralChoice(rule);
  if (c) addTap(c.row, c.col, c.d0, c.d1);
}

/** With no ?rule=, a random held-out rule whose strand near the centre is a fair length (a few tries). */
function startingRule(): Rule {
  const fallback = table.parse('15·000000000') as Rule;
  for (let k = 0; k < 40; k++) {
    const r = table.sample(Math.random, 'heldout');
    const c = centralChoice(r, 12, 60);
    if (c && c.n >= 12 && c.n <= 60) return r;
  }
  return fallback;
}

// ── Evaluation: what each model draws against the true strands ─────────────

function drawnOf(m: StrandNCA): Uint8Array {
  const out = new Uint8Array(board.n * 6);
  const C = m.C;
  const st = m.state;
  for (let i = 0; i < board.n; i++) for (let d = 0; d < 6; d++) out[i * 6 + d] = st[i * C + 1 + d] > 0.5 ? 1 : 0;
  return out;
}

/** A cheap fingerprint of everything drawn (for the "nothing changed" pause). */
function fingerprint(): number {
  let h = 0x811c9dc5;
  const mix = (m: StrandNCA) => {
    const C = m.C;
    for (let i = 0; i < board.n; i++) {
      let bits = 0;
      for (let d = 0; d < 6; d++) if (m.state[i * C + 1 + d] > 0.5) bits |= 1 << d;
      if (bits) h = Math.imul(h ^ (i * 64 + bits), 0x01000193);
    }
  };
  if (sharedModel) mix(sharedModel);
  else for (const t of taps) if (t.model) mix(t.model);
  return h >>> 0;
}

function countChords(t: PlayTap, P: Uint8Array): number {
  let n = 0;
  for (let k = 0; k < t.strand.rows.length; k++) {
    const i = board.cellOf[t.strand.rows[k] * board.w + t.strand.cols[k]];
    if (P[i * 6 + t.strand.ins[k]] && P[i * 6 + t.strand.outs[k]]) n++;
  }
  return n;
}

/** Each tap's exact / drawn / stray. Its own board: exact = what it draws is its true strand, nothing more
 * (train2.evaluate's exact). A shared board: its strand all drawn, and no stray edge (drawn, on no tap's strand)
 * in the cells nearest its strand — with one tap, the same thing. */
function evaluate(): void {
  if (sharedModel) {
    const P = drawnOf(sharedModel);
    const union = new Uint8Array(board.n * 6);
    for (const t of taps) for (let e = 0; e < union.length; e++) union[e] |= t.trueEdge[e];
    const stray = new Int32Array(Math.max(1, taps.length));
    for (let e = 0; e < P.length; e++) if (P[e] && !union[e]) stray[owner[(e / 6) | 0]]++;
    taps.forEach((t, k) => {
      let missing = 0;
      for (let e = 0; e < P.length; e++) if (t.trueEdge[e] && !P[e]) missing++;
      t.missing = missing;
      t.stray = stray[k] ?? 0;
      t.drawn = countChords(t, P);
    });
  } else {
    for (const t of taps) {
      if (!t.model) continue;
      const P = drawnOf(t.model);
      let missing = 0;
      let stray = 0;
      for (let e = 0; e < P.length; e++) {
        if (t.trueEdge[e] && !P[e]) missing++;
        if (P[e] && !t.trueEdge[e]) stray++;
      }
      t.missing = missing;
      t.stray = stray;
      t.drawn = countChords(t, P);
    }
  }
  for (const t of taps) {
    const steps = t.model ? t.model.steps : stepCount();
    t.exact = t.missing === 0 && t.stray === 0 && steps > 0;
    if (!t.exact) t.since = null;
    else if (t.since === null) t.since = steps;
  }
}

// ── Geometry ───────────────────────────────────────────────────────────────

interface Geom { size: number; ox: number; oy: number }
let geom: Geom = { size: 10, ox: 0, oy: 0 };
let fitSize = 10;
let cssW = 1;
let cssH = 1;

/** A board position's centre with unit cell size, origin at (0, 0). */
function unitCentre(p: number): [number, number] {
  const row = Math.floor(p / board.w);
  const col = p % board.w;
  return [SQ3 * (col + row / 2), 1.5 * row];
}

function centre(p: number): [number, number] {
  const [x, y] = unitCentre(p);
  return [geom.ox + geom.size * x, geom.oy + geom.size * y];
}

/** The midpoint of edge d of a cell centred at (x, y): direction d sits at -60·d degrees. */
function edgeMid(x: number, y: number, d: number, f = 1): [number, number] {
  const a = (-Math.PI / 3) * d;
  const r = geom.size * (SQ3 / 2) * f;
  return [x + r * Math.cos(a), y + r * Math.sin(a)];
}

function fit(): void {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < board.n; i++) {
    const [x, y] = unitCentre(board.pos[i]);
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const pad = 12;
  const size = Math.max(0.5, Math.min((cssW - 2 * pad) / (maxX - minX + SQ3), (cssH - 2 * pad) / (maxY - minY + 2)));
  fitSize = size;
  geom = { size, ox: cssW / 2 - (size * (minX + maxX)) / 2, oy: cssH / 2 - (size * (minY + maxY)) / 2 };
  bgDirty = dirty = true;
}

function resize(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  const refit = Math.abs(geom.size - fitSize) < 1e-9 || cssW <= 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  // keep the view's centre where it was
  geom = { size: geom.size, ox: geom.ox + (w - cssW) / 2, oy: geom.oy + (h - cssH) / 2 };
  cssW = w;
  cssH = h;
  if (refit && board) fit();
  bgDirty = dirty = true;
}

/** The board position under a canvas point, or -1. */
function posAt(x: number, y: number): number {
  const fr = (y - geom.oy) / (1.5 * geom.size);
  const fq = (x - geom.ox) / (SQ3 * geom.size) - fr / 2;
  const fs = -fq - fr;
  let q = Math.round(fq);
  let r = Math.round(fr);
  const s = Math.round(fs);
  const dq = Math.abs(q - fq);
  const dr = Math.abs(r - fr);
  const ds = Math.abs(s - fs);
  if (dq > dr && dq > ds) q = -r - s;
  else if (dr > ds) r = -q - s;
  return board.on(r, q) ? r * board.w + q : -1;
}

function zoomAt(x: number, y: number, f: number): void {
  const size = Math.min(fitSize * 16, Math.max(fitSize, geom.size * f));
  const k = size / geom.size;
  geom = { size, ox: x - (x - geom.ox) * k, oy: y - (y - geom.oy) * k };
  bgDirty = dirty = true;
}

// ── Colours ────────────────────────────────────────────────────────────────

interface Colours { cell: string; edge: string; fg: string; muted: string; stray: string; halo: string; good: string; bad: string; mix: number; taps: string[] }
let colours: Colours;
function readColours(): void {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  colours = {
    cell: v('--cell'), edge: v('--cell-edge'), fg: v('--fg'), muted: v('--muted'), stray: v('--stray'), halo: v('--halo'),
    good: v('--good'), bad: v('--bad'), mix: Number(v('--type-mix')) || 0.5,
    taps: Array.from({ length: N_COLOURS }, (_, k) => v(`--tap${k}`)),
  };
}

function rgbOf(css: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(css);
  if (m) { const n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  const nums = css.match(/[\d.]+/g)?.map(Number) ?? [255, 255, 255];
  return [nums[0], nums[1], nums[2]];
}

// ── Drawing ────────────────────────────────────────────────────────────────

let bg: HTMLCanvasElement | null = null;
let bgDirty = true;

function hexPath(path: Path2D, x: number, y: number, s: number): void {
  for (let k = 0; k < 6; k++) {
    const a = (Math.PI / 180) * (60 * k - 30);
    const px = x + s * Math.cos(a);
    const py = y + s * Math.sin(a);
    if (k === 0) path.moveTo(px, py);
    else path.lineTo(px, py);
  }
  path.closePath();
}

/** The tiles: Spectacle's type colours (or plain), outlines, and each tile's rotation (a dart at local edge 0,
 * as Spectacle's board draws it) once there is room. Cached; redrawn on a view, board or theme change. */
function drawBackground(): void {
  const dpr = window.devicePixelRatio || 1;
  if (!bg) bg = document.createElement('canvas');
  bg.width = canvas.width;
  bg.height = canvas.height;
  const g = bg.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);
  const cell = rgbOf(colours.cell);
  const groups = new Map<string, Path2D>();
  const s = geom.size * 0.985;
  const outline = new Path2D();
  const darts = new Path2D();
  for (let i = 0; i < board.n; i++) {
    const p = board.pos[i];
    const [x, y] = centre(p);
    if (x < -geom.size || y < -geom.size || x > cssW + geom.size || y > cssH + geom.size) continue;
    let fill = colours.cell;
    if (showTypes) {
      const c = TYPE_RGB[table.leafOrder[board.type(p)]] ?? [255, 255, 255];
      const m = colours.mix;
      fill = `rgb(${c.map((v, k) => Math.round(cell[k] + (v - cell[k]) * m)).join(' ')})`;
    }
    let path = groups.get(fill);
    if (!path) groups.set(fill, (path = new Path2D()));
    hexPath(path, x, y, s);
    if (geom.size > 5) hexPath(outline, x, y, s);
    if (showTypes && geom.size >= 9) {
      const d = board.edgeDir(p, 0);
      const [tx, ty] = edgeMid(x, y, d, 0.62);
      const a = (-Math.PI / 3) * d;
      const bx = x + geom.size * 0.12 * Math.cos(a);
      const by = y + geom.size * 0.12 * Math.sin(a);
      const w = geom.size * 0.17;
      darts.moveTo(tx, ty);
      darts.lineTo(bx - w * Math.sin(a), by + w * Math.cos(a));
      darts.lineTo(bx + w * Math.sin(a), by - w * Math.cos(a));
      darts.closePath();
    }
  }
  for (const [fill, path] of groups) {
    g.fillStyle = fill;
    g.fill(path);
  }
  g.strokeStyle = colours.edge;
  g.lineWidth = 0.6;
  g.stroke(outline);
  g.globalAlpha = 0.35;
  g.fillStyle = colours.muted;
  g.fill(darts);
  g.globalAlpha = 1;
  bgDirty = false;
}

/** A model's drawn edges paired up into chords, cell by cell: first the true strands' chords, then any tap rule's
 * chords, then what is left two by two (a lone edge: a stub to the centre). */
function addModelChords(P: Uint8Array, own: PlayTap[], solid: Path2D[], stray: Path2D, strayOf: Path2D[] | null): void {
  for (let i = 0; i < board.n; i++) {
    let mask = 0;
    for (let d = 0; d < 6; d++) if (P[i * 6 + d]) mask |= 1 << d;
    if (!mask) continue;
    const p = board.pos[i];
    const [x, y] = centre(p);
    const line = (path: Path2D, a: number, b: number) => {
      const [ax, ay] = edgeMid(x, y, a);
      path.moveTo(ax, ay);
      if (b < 0) path.lineTo(x, y);
      else path.lineTo(...edgeMid(x, y, b));
    };
    for (const t of own) {
      for (const [a, b] of t.trueChords.get(i) ?? []) {
        if ((mask >> a) & 1 && (mask >> b) & 1) {
          line(solid[taps.indexOf(t)], a, b);
          mask &= ~((1 << a) | (1 << b));
        }
      }
    }
    if (!mask) continue;
    const strayPath = strayOf ? strayOf[taps.indexOf(own[0])] : stray;
    for (const t of own) {
      const bits = t.ruleBits[p];
      for (let k = 0; k < 15 && mask; k++) {
        if (!((bits >> k) & 1)) continue;
        const [a, b] = PAIRS[k];
        if ((mask >> a) & 1 && (mask >> b) & 1) {
          line(strayPath, a, b);
          mask &= ~((1 << a) | (1 << b));
        }
      }
    }
    const left: number[] = [];
    for (let d = 0; d < 6; d++) if ((mask >> d) & 1) left.push(d);
    for (let k = 0; k + 1 < left.length; k += 2) line(strayPath, left[k], left[k + 1]);
    if (left.length % 2) line(strayPath, left[left.length - 1], -1);
  }
}

function draw(): void {
  dirty = false;
  if (bgDirty || !bg) drawBackground();
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bg!, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const s = geom.size;
  // the true strands, thin, underneath
  if (showTruth) {
    ctx.lineWidth = Math.max(1.2, s * 0.11);
    ctx.globalAlpha = 0.8;
    for (const t of taps) {
      const path = new Path2D();
      const st = t.strand;
      for (let k = 0; k < st.rows.length; k++) {
        const [x, y] = centre(st.rows[k] * board.w + st.cols[k]);
        path.moveTo(...edgeMid(x, y, st.ins[k]));
        path.lineTo(...edgeMid(x, y, st.outs[k]));
      }
      ctx.strokeStyle = colours.taps[t.colour];
      ctx.stroke(path);
    }
    ctx.globalAlpha = 1;
  }
  // what the model draws
  const solid = taps.map(() => new Path2D());
  const stray = new Path2D();
  const strayOf = sharedModel ? null : taps.map(() => new Path2D());
  if (sharedModel) addModelChords(drawnOf(sharedModel), taps, solid, stray, null);
  else for (const t of taps) if (t.model) addModelChords(drawnOf(t.model), [t], solid, stray, strayOf);
  const lw = Math.max(1.6, s * 0.26);
  ctx.strokeStyle = colours.halo;
  ctx.lineWidth = lw + Math.max(2, s * 0.14);
  for (const p of solid) ctx.stroke(p);
  ctx.stroke(stray);
  if (strayOf) for (const p of strayOf) ctx.stroke(p);
  ctx.lineWidth = lw;
  taps.forEach((t, k) => {
    ctx.strokeStyle = colours.taps[t.colour];
    ctx.stroke(solid[k]);
  });
  ctx.setLineDash([Math.max(2, s * 0.3), Math.max(2, s * 0.25)]);
  ctx.lineWidth = Math.max(1.2, s * 0.18);
  ctx.strokeStyle = colours.stray;
  ctx.stroke(stray);
  if (strayOf) taps.forEach((t, k) => { ctx.strokeStyle = colours.taps[t.colour]; ctx.stroke(strayOf[k]); });
  ctx.setLineDash([]);
  // the taps: a ring on the tapped tile
  for (const t of taps) {
    const [x, y] = centre(t.tap.row * board.w + t.tap.col);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(4, s * 0.62), 0, 2 * Math.PI);
    ctx.lineWidth = Math.max(3, s * 0.2);
    ctx.strokeStyle = colours.halo;
    ctx.stroke();
    ctx.lineWidth = Math.max(1.5, s * 0.11);
    ctx.strokeStyle = colours.taps[t.colour];
    ctx.stroke();
  }
}

// ── Side panel ─────────────────────────────────────────────────────────────

function ruleTags(r: Rule): string {
  const held = table.heldOut(r);
  return `<span class="tag ${held ? 'held' : 'train'}">${held ? 'held out' : 'training'}</span>`;
}

function describeRuleNow(): void {
  $<HTMLInputElement>('ruleText').value = table.describe(rule);
  const sub = table.subsets[rule.s];
  $('ruleNow').innerHTML = `<span class="swatch" style="background:${colours.taps[nextColour()]}"></span> `
    + `<b>${table.describe(rule)}</b>${ruleTags(rule)}<br><span class="note">edge classes ${sub.edges.join(', ')} carry lines; `
    + `${sub.count.toLocaleString()} rules in subset ${sub.key}</span>`;
}

function nextColour(): number {
  const used = new Set(taps.map((t) => t.colour));
  for (let c = 0; c < N_COLOURS; c++) if (!used.has(c)) return c;
  return taps.length % N_COLOURS;
}

function setRule(r: Rule, why = ''): void {
  rule = { s: r.s, digits: r.digits.slice() };
  $('ruleErr').textContent = why;
  $('ruleErr').className = 'note';
  describeRuleNow();
  const u = new URL(location.href);
  u.searchParams.set('rule', table.describe(rule));
  history.replaceState(null, '', u);
}

const tapRows = new Map<number, { badge: HTMLElement; sub: HTMLElement }>();

function renderTaps(): void {
  const ol = $('taps');
  ol.replaceChildren();
  tapRows.clear();
  for (const t of taps) {
    const li = document.createElement('li');
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = colours.taps[t.colour];
    const what = document.createElement('div');
    what.className = 'what';
    const name = document.createElement('button');
    name.textContent = table.describe(t.tap.rule);
    name.title = 'Use this rule for the next tap';
    name.addEventListener('click', () => setRule(t.tap.rule));
    const tag = document.createElement('span');
    tag.innerHTML = ruleTags(t.tap.rule);
    const sub = document.createElement('div');
    sub.className = 'sub';
    const badge = document.createElement('div');
    badge.className = 'badge';
    what.append(name, tag, sub, badge);
    const x = document.createElement('button');
    x.className = 'x';
    x.textContent = '×';
    x.title = 'Take this tap away';
    x.setAttribute('aria-label', `Take away the tap of ${table.describe(t.tap.rule)}`);
    x.addEventListener('click', () => removeTap(t.id));
    li.append(sw, what, x);
    ol.append(li);
    tapRows.set(t.id, { badge, sub });
  }
  $('tapsEmpty').hidden = taps.length > 0;
  tapsDirty = false;
  updateTapBadges();
  describeRuleNow();
}

function updateTapBadges(): void {
  for (const t of taps) {
    const row = tapRows.get(t.id);
    if (!row) continue;
    const n = t.strand.rows.length;
    const p = t.tap.row * board.w + t.tap.col;
    row.sub.textContent = `${table.leafOrder[board.type(p)]} at (${t.tap.row}, ${t.tap.col}) · true strand ${n} chord${n > 1 ? 's' : ''}, `
      + `${t.strand.closed ? 'a circuit' : 'tail to tail'}`;
    if (t.exact) {
      row.badge.className = 'badge ok';
      row.badge.textContent = `exact ✓ since step ${t.since}`;
    } else {
      row.badge.className = 'badge no';
      row.badge.textContent = `${t.drawn}/${n} chords drawn${t.stray ? ` · ${t.stray} stray edge${t.stray > 1 ? 's' : ''}` : ''}`;
    }
  }
}

function sharedNote(): void {
  const box = $<HTMLInputElement>('shared');
  box.disabled = !canShare();
  box.checked = usingShared();
  $('sharedNote').textContent = !weights ? '' : !canShare()
    ? `These weights carry the rule on every tile (${weights.kind === 'v1' ? "v1's chord planes" : 'the code broadcast'}), so each tap runs on a board of its own.`
    : usingShared()
      ? 'Every tap is an input of one board, so strands can run into each other. The models were never trained on that: this shows what they do.'
      : 'Each tap runs on a board of its own, as in training; the drawings are laid over each other.';
}

let stepsDone = 0;
let stepMs = 0;
let rateFrom = performance.now();
let stepsPerSec = 0;
let msPerStep = 0;

function readout(): void {
  const exact = taps.filter((t) => t.exact).length;
  const rows: [string, string][] = [
    ['Step', stepCount().toLocaleString()],
    ['Steps/s', playing ? String(stepsPerSec) : 'paused'],
    ['ms/step', msPerStep ? msPerStep.toFixed(1) : '–'],
    ['Tiles', `${board.n.toLocaleString()} (${taps.length && !sharedModel ? `${taps.length} boards` : 'one board'})`],
    ['Exact', taps.length ? `${exact}/${taps.length} taps` : '–'],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  const slow = msPerStep > SLOW_MS
    ? ` <span class="warn">${MAP_LABELS[mapId]} has ${board.n.toLocaleString()} tiles: about ${Math.round(msPerStep)} ms a step here (the network runs on every tile, every step, on this browser's main thread), so strands grow slowly.</span>`
    : '';
  $('status').innerHTML = (notice || (taps.length
    ? `Tap a tile to add a strand with <b>${table.describe(rule)}</b>; tap a tapped tile to replace its tap.`
    : `Tap a tile: the strand of <b>${table.describe(rule)}</b> starts at the chord nearest your tap.`)) + slow;
}

function modelLine(): void {
  const el = $('model');
  if (!weights) {
    el.textContent = weightsError || 'Loading weights…';
    return;
  }
  const m = weights.meta;
  const name = weightsName || m.run || weightsFrom;
  const arch = weights.kind === 'v1'
    ? 'v1, the rule pre-rendered as chords on every tile'
    : weights.arch === 'frame'
      ? `option E (every tile in its own frame)${weights.inputs === 'e-bc' ? ' + the rule broadcast to every tile (a diagnostic)' : ''}`
      : `option ${weights.inputs.toUpperCase()} (the grid's frame)${weights.inputs.endsWith('-bc') ? ' + the rule broadcast to every tile (a diagnostic)' : ''}`;
  const ho = m.heldOut;
  const lv = ho?.byLevel ? Object.entries(ho.byLevel).map(([k, v]) => `L${k} ${fmt(v.exact)}`).join(', ') : '';
  const held = ho && typeof ho.exact === 'number'
    ? `held-out exact <b>${fmt(ho.exact)}</b>${lv ? ` (${lv})` : ''}, length-balanced ${fmt(ho.balanced)}`
    : 'held-out exact not recorded';
  const it = typeof m.iteration === 'number'
    ? `${m.file ?? 'checkpoint'} at iteration ${m.iteration.toLocaleString()}${m.prevIterations ? ` (+${m.prevIterations.toLocaleString()} before, from ${m.init ?? 'its --init'})` : ''}`
    : '';
  el.innerHTML = `Weights: <b>${escapeHtml(String(name))}</b> — ${arch}; ${held}${it ? `; ${escapeHtml(it)}` : ''}.`
    + (weightsError ? `<span class="err">${escapeHtml(weightsError)}</span>` : '');
  $('source').textContent = `${weightsFrom} · ${weights.kind} ${weights.arch}, inputs ${weights.inputs}`;
  $('meta').textContent = `${weights.channels} channels (1-6: the edges drawn), ${weights.hidden} hidden, depth ${weights.depth}, `
    + `${weights.nIn} const inputs (${weights.consts.map(([k, n]) => `${k} ${n}`).join(', ')}), clamp `
    + `${weights.clamp ? `[${weights.clamp.join(', ')}]` : 'none'}; a tile's edge is drawn where its channel > 0.5.`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

function mapInfo(): void {
  const b = data.boards[mapId];
  $('mapInfo').textContent = `Spectacle's hex field, level ${b.level}, the ${b.root} patch: ${b.tiles.toLocaleString()} tiles`
    + `${mapId === 'l4' ? ' (the models trained on levels 2 and 3; this one is ~8× the work of level 3 a step)' : ''}.`;
}

function legend(): void {
  $('legend').innerHTML = [
    '<span><i></i>the network\'s strand (in its tap\'s colour)</span>',
    '<span><i class="thin"></i>the true strand</span>',
    '<span><i class="dash"></i>drawn, on no true strand</span>',
  ].join('');
}

// ── Loop ───────────────────────────────────────────────────────────────────

const hasWork = () => !!sharedModel || taps.some((t) => t.model);

function stepAll(): void {
  if (sharedModel) sharedModel.step();
  else for (const t of taps) t.model?.step();
  const key = fingerprint();
  if (key !== drawnKey) {
    drawnKey = key;
    lastChange = stepCount();
  }
}

let last = performance.now();
let acc = 0;
function frame(now: number): void {
  const dt = Math.min(100, now - last);
  last = now;
  if (playing && hasWork()) {
    acc += (dt * speed) / 1000;
    const t0 = performance.now();
    let n = 0;
    while (acc >= 1 && (n === 0 || performance.now() - t0 < FRAME_MS)) {
      stepAll();
      acc -= 1;
      n++;
    }
    if (acc > 1) acc = 1;
    if (n) {
      stepsDone += n;
      stepMs += performance.now() - t0;
      dirty = true;
      if (stepCount() - lastChange >= SETTLE_STEPS) {
        setPlaying(false);
        notice = `Paused: nothing drawn has changed for ${SETTLE_STEPS} steps. Tap a tile, or Run, to carry on.`;
      }
    }
  }
  if (now - rateFrom >= 1000) {
    stepsPerSec = Math.round((stepsDone * 1000) / (now - rateFrom));
    if (stepsDone) msPerStep = stepMs / stepsDone;
    stepsDone = 0;
    stepMs = 0;
    rateFrom = now;
    readout();
  }
  if (dirty && board) {
    evaluate();
    if (tapsDirty) renderTaps();
    else updateTapBadges();
    draw();
    readout();
  }
  requestAnimationFrame(frame);
}

function setPlaying(on: boolean): void {
  playing = on;
  $('play').textContent = playing ? 'Pause' : 'Run';
  if (on) {
    notice = '';
    lastChange = stepCount();
  }
  readout();
}

// ── Board and weights ──────────────────────────────────────────────────────

function setMap(id: string): void {
  mapId = data.boards[id] ? id : 'l3';
  $<HTMLSelectElement>('map').value = mapId;
  board = boardOf(mapId);
  taps = [];
  buildModels();
  cssW = cssH = 1;
  resize();
  fit();
  mapInfo();
  computeOwners();
  centralTap();
  msPerStep = 0;
  const u = new URL(location.href);
  u.searchParams.set('map', mapId);
  history.replaceState(null, '', u);
  tapsDirty = dirty = true;
}

async function fetchWeights(url: string): Promise<{ text: string; w: StrandWeights }> {
  const res = await fetch(new URL(url, location.href), { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  return { text, w: loadStrandWeights(JSON.parse(text)) };
}

/** The URL's weights, else the default ones (beside the page, else the Pages copy), saying why. */
async function loadAnyWeights(): Promise<void> {
  const errors: string[] = [];
  const tries = (weightsUrl ? [weightsUrl] : []).concat([DEFAULT_WEIGHTS, PAGES_WEIGHTS]);
  for (const url of tries) {
    try {
      const { text, w } = await fetchWeights(url);
      weights = w;
      weightsText = text;
      weightsFrom = url;
      weightsError = weightsUrl && url !== weightsUrl
        ? `Could not load ${weightsUrl} (${errors[0]}), so these are the page's default weights.` : '';
      return;
    } catch (e) {
      errors.push(why(e));
    }
  }
  weightsError = `Could not load any weights: ${errors.join('; ')}.`;
}

async function reloadWeights(): Promise<void> {
  const btn = $<HTMLButtonElement>('reloadWeights');
  btn.disabled = true;
  try {
    const { text, w } = await fetchWeights(weightsFrom || DEFAULT_WEIGHTS);
    const changed = text !== weightsText;
    weights = w;
    weightsText = text;
    weightsError = '';
    if (!canShare() && shared && taps.length > 1) notice = 'These weights take one tap per board: each tap now runs on its own.';
    buildModels();
    notice = notice || (changed ? 'New weights loaded: every model starts over.' : 'Weights reloaded (unchanged): every model starts over.');
  } catch (e) {
    notice = `Could not reload ${weightsFrom}: ${why(e)}. Still the weights loaded before.`;
  } finally {
    btn.disabled = false;
    modelLine();
    sharedNote();
    setPlaying(true);
  }
}

// ── Controls ───────────────────────────────────────────────────────────────

let down: { x: number; y: number; ox: number; oy: number; id: number; moved: boolean } | null = null;
canvas.addEventListener('pointerdown', (ev) => {
  canvas.setPointerCapture(ev.pointerId);
  down = { x: ev.offsetX, y: ev.offsetY, ox: geom.ox, oy: geom.oy, id: ev.pointerId, moved: false };
});
canvas.addEventListener('pointermove', (ev) => {
  if (!down || ev.pointerId !== down.id) return;
  const dx = ev.offsetX - down.x;
  const dy = ev.offsetY - down.y;
  if (!down.moved && Math.hypot(dx, dy) < 5) return;
  down.moved = true;
  canvas.classList.add('panning');
  geom = { size: geom.size, ox: down.ox + dx, oy: down.oy + dy };
  bgDirty = dirty = true;
});
canvas.addEventListener('pointerup', (ev) => {
  if (!down || ev.pointerId !== down.id) return;
  const moved = down.moved;
  down = null;
  canvas.classList.remove('panning');
  if (!moved) tapAtPoint(ev.offsetX, ev.offsetY);
});
canvas.addEventListener('pointercancel', () => { down = null; canvas.classList.remove('panning'); });
canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  zoomAt(ev.offsetX, ev.offsetY, Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015)));
}, { passive: false });
$('zoomIn').addEventListener('click', () => zoomAt(cssW / 2, cssH / 2, 1.5));
$('zoomOut').addEventListener('click', () => zoomAt(cssW / 2, cssH / 2, 1 / 1.5));
$('zoomFit').addEventListener('click', () => fit());

/** A tap at a canvas point: the current rule's chord nearest the point, on the tile under it. */
function tapAtPoint(x: number, y: number): void {
  const p = posAt(x, y);
  if (p < 0) return;
  const row = Math.floor(p / board.w);
  const col = p % board.w;
  const chords = chordsAt(table, rule, board, row, col);
  if (!chords.length) {
    notice = `No chord of ${table.describe(rule)} on that tile (a ${table.leafOrder[board.type(p)]}): try another tile or rule.`;
    readout();
    return;
  }
  const [cx, cy] = centre(p);
  let best = chords[0];
  let bestD = Infinity;
  for (const ch of chords) {
    const [ax, ay] = edgeMid(cx, cy, ch[0]);
    const [bx, by] = edgeMid(cx, cy, ch[1]);
    const vx = bx - ax;
    const vy = by - ay;
    const t = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)));
    const d = (ax + vx * t - x) ** 2 + (ay + vy * t - y) ** 2;
    if (d < bestD) {
      bestD = d;
      best = ch;
    }
  }
  addTap(row, col, best[0], best[1]);
}

$('ruleUse').addEventListener('click', () => useRuleText());
$<HTMLInputElement>('ruleText').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') useRuleText(); });
function useRuleText(): void {
  const r = table.parse($<HTMLInputElement>('ruleText').value);
  if (typeof r === 'string') {
    $('ruleErr').textContent = r;
    $('ruleErr').className = 'note err';
    return;
  }
  setRule(r);
}
$('ruleHeld').addEventListener('click', () => setRule(table.sample(Math.random, 'heldout')));
$('ruleTrain').addEventListener('click', () => setRule(table.sample(Math.random, 'train')));
$<HTMLSelectElement>('preset').addEventListener('change', (ev) => {
  const sel = ev.target as HTMLSelectElement;
  const r = table.parse(sel.value);
  if (typeof r !== 'string') setRule(r);
  sel.value = '';
});
$('clearTaps').addEventListener('click', () => {
  taps = [];
  buildModels();
  computeOwners();
  notice = '';
});
$<HTMLInputElement>('shared').addEventListener('change', (ev) => {
  shared = (ev.target as HTMLInputElement).checked;
  buildModels();
  sharedNote();
  setPlaying(true);
});
$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  if (hasWork()) stepAll();
  dirty = true;
});
$('reset').addEventListener('click', () => {
  buildModels();
  notice = 'Every model is back at the fresh state, its taps held from step 0.';
  setPlaying(true);
});
$<HTMLInputElement>('speed').addEventListener('input', (ev) => {
  speed = Number((ev.target as HTMLInputElement).value);
  $('speedOut').textContent = String(speed);
});
$<HTMLInputElement>('truth').addEventListener('change', (ev) => {
  showTruth = (ev.target as HTMLInputElement).checked;
  dirty = true;
});
$<HTMLInputElement>('types').addEventListener('change', (ev) => {
  showTypes = (ev.target as HTMLInputElement).checked;
  bgDirty = dirty = true;
});
$<HTMLSelectElement>('map').addEventListener('change', (ev) => setMap((ev.target as HTMLSelectElement).value));
$('reloadWeights').addEventListener('click', () => void reloadWeights());
document.addEventListener('keydown', (ev) => {
  if (ev.target instanceof HTMLInputElement && ev.target.type === 'text') return;
  if (ev.key === 't' || ev.key === 'T') {
    const box = $<HTMLInputElement>('truth');
    box.checked = !box.checked;
    showTruth = box.checked;
    dirty = true;
  } else if (ev.key === ' ' && !(ev.target instanceof HTMLButtonElement)) {
    ev.preventDefault();
    setPlaying(!playing);
  }
});

new ResizeObserver(() => resize()).observe(canvas);
const restyle = () => {
  readColours();
  bgDirty = dirty = tapsDirty = true;
};
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', restyle);
new MutationObserver(restyle).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

// For the console and headless checks.
Object.assign(window, {
  strandplay: {
    get weights() { return weights; },
    get board() { return board; },
    get taps() { return taps; },
    get shared() { return sharedModel; },
    get steps() { return stepCount(); },
    table,
    /** Client coordinates of a board cell's centre. */
    point(row: number, col: number): [number, number] {
      const rect = canvas.getBoundingClientRect();
      const [x, y] = centre(row * board.w + col);
      return [rect.left + x, rect.top + y];
    },
    /** Each tap's numbers: rule, length, drawn, stray, exact. */
    stats() {
      return taps.map((t) => ({ rule: table.describe(t.tap.rule), row: t.tap.row, col: t.tap.col, length: t.strand.rows.length,
        closed: t.strand.closed, drawn: t.drawn, missing: t.missing, stray: t.stray, exact: t.exact, since: t.since }));
    },
    setRule(text: string) { const r = table.parse(text); if (typeof r !== 'string') setRule(r); return r; },
    run(n: number) { for (let k = 0; k < n; k++) stepAll(); dirty = true; },
  },
});

// ── Start ──────────────────────────────────────────────────────────────────

readColours();
legend();
$<HTMLSelectElement>('map').append(...Object.keys(data.boards).map((id) => new Option(
  `${MAP_LABELS[id] ?? id} (${data.boards[id].tiles.toLocaleString()} tiles)`, id)));
const ruleParam = params.get('rule');
if (ruleParam) {
  const r = table.parse(ruleParam);
  if (typeof r === 'string') {
    describeRuleNow();
    $('ruleErr').textContent = `?rule=${ruleParam}: ${r}`;
    $('ruleErr').className = 'note err';
  } else setRule(r);
} else describeRuleNow();
if (weightsName) document.title = `${weightsName} · ${document.title}`;
void loadAnyWeights().then(() => {
  modelLine();
  const id = params.get('map') ?? 'l3';
  if (!ruleParam) {
    board = boardOf(data.boards[id] ? id : 'l3');
    setRule(startingRule());
  }
  setMap(id);
  sharedNote();
  setPlaying(!!weights);
  requestAnimationFrame(frame);
});
