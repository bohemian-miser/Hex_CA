// The demo page (DESIGN.md §9): draw lines, erase and paint walls, watch the
// fill rule settle, and check the settled state against the oracle.

import { CA } from '../src/engine.js';
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

/** "Step to commit" gives up after this many steps. */
const SEEK_MAX = 20000;
/** Per frame: at most this many steps, and no more than this much time stepping. */
const FRAME_STEPS = 5000;
const FRAME_MS = 14;
/** Heat maps are quantised so a frame fills one path per colour, not one per cell. */
const LEVELS = 32;

type Tool = 'on' | 'erase' | 'wall';
type View = 'state' | 'diff' | 'epoch' | 'channel';
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
  const x = (ev.clientX - rect.left - ox) / size;
  const y = (ev.clientY - rect.top - oy) / size;
  const [q, r] = cubeRound((SQ3 / 3) * x - y / 3, (2 / 3) * y);
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

let size = 10;
let ox = 0;
let oy = 0;
/** Cell centres in CSS pixels, per slot (field slots only). */
let cx = new Float64Array(0);
let cy = new Float64Array(0);

function layout(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const R = field.board.radius;
  size = Math.min((rect.width - 16) / (SQ3 * (2 * R + 1)), (rect.height - 16) / (3 * R + 2));
  ox = rect.width / 2;
  oy = rect.height / 2;
  cx = new Float64Array(field.board.size);
  cy = new Float64Array(field.board.size);
  for (const i of field.topo.cells) {
    const [q, r] = coordsOf(field.board, i);
    cx[i] = ox + size * SQ3 * (q + r / 2);
    cy[i] = oy + size * 1.5 * r;
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

/** a + (b − a)·t for two #rrggbb colours; a when either is not one. */
function mix(a: string, b: string, t: number): string {
  const pa = /^#([0-9a-f]{6})$/i.exec(a);
  const pb = /^#([0-9a-f]{6})$/i.exec(b);
  if (!pa || !pb) return t < 0.5 ? a : b;
  const na = parseInt(pa[1], 16);
  const nb = parseInt(pb[1], 16);
  const ch = (sh: number) => Math.round(((na >> sh) & 255) * (1 - t) + ((nb >> sh) & 255) * t);
  return `rgb(${ch(16)} ${ch(8)} ${ch(0)})`;
}

function heat(t: number): string {
  return isDark() ? `hsl(${200 - 160 * t} 55% ${18 + 40 * t}%)` : `hsl(${200 - 160 * t} 65% ${92 - 45 * t}%)`;
}

function hashed(v: number): string {
  return isDark() ? `hsl(${(v * 137.5) % 360} 45% 42%)` : `hsl(${(v * 137.5) % 360} 60% 72%)`;
}

function draw(): void {
  dirty = false;
  css = getComputedStyle(document.documentElement);
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const col = {
    cell: tok('--cell'), edge: tok('--cell-edge'), line: tok('--line'), wall: tok('--wall'),
    flood: tok('--flood'), accent: tok('--accent'), pulse: tok('--pulse'),
  };
  const stateColour = (v: number) => (v === ON ? col.line : v === WALL ? col.wall : v === FILLED ? col.flood : col.cell);
  const cells = field.topo.cells;
  const cellList = Array.from(cells);
  const p = ca.ch[PAINT];
  const st = ca.ch[STATE];
  const s = size * 0.97;
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
  } else {
    const name = NAMES[shownChannel];
    const v = ca.ch[shownChannel];
    const region = REGION.has(name);
    const shown = (c: number) => !region || p[c] === OFF;
    let lo = Infinity;
    let hi = -Infinity;
    for (const c of cells) {
      if (!shown(c)) continue;
      if (v[c] < lo) lo = v[c];
      if (v[c] > hi) hi = v[c];
    }
    for (const c of cells) {
      if (!shown(c)) {
        fills.add(stateColour(p[c]), c);
        continue;
      }
      let colour: string;
      if (HASHED.has(name)) colour = v[c] === 0 ? col.cell : hashed(v[c]);
      else colour = heat(hi > lo ? Math.round(((v[c] - lo) / (hi - lo)) * LEVELS) / LEVELS : 0);
      fills.add(colour, c);
      if (p[c] !== OFF) inset.add(stateColour(p[c]), c);
    }
  }
  fills.draw();
  inset.draw();
  if (size > 6) {
    ctx.strokeStyle = col.edge;
    ctx.lineWidth = 0.5;
    hexes(cellList, s, () => ctx.stroke());
  }
  readoutDirty = true;
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
  const items: Record<View, string[]> = {
    state: [sw('var(--line)', 'line'), sw('var(--wall)', 'wall'), sw('var(--flood)', 'filled'), sw('var(--cell)', 'off')],
    diff: [sw('var(--pulse)', 'want ≠ state'), sw('var(--flood)', 'filled (dimmed)')],
    epoch: [sw('var(--accent)', 'newest epoch'), sw('var(--cell)', 'older, fading with age')],
    channel: HASHED.has(NAMES[shownChannel])
      ? ['ids hashed to hues', sw('var(--cell)', '0')]
      : [sw(heat(0), 'board minimum'), sw(heat(1), 'maximum')],
  };
  $('legend').innerHTML = items[view].join('');
}

// ── Loop and controls ───────────────────────────────────────────────────────

let last = performance.now();
let acc = 0;
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
for (const v of ['state', 'diff', 'epoch', 'channel'] as const) {
  $<HTMLInputElement>(`view-${v}`).addEventListener('change', () => setView(v));
}
const channelSel = $<HTMLSelectElement>('channelSel');
channelSel.innerHTML = NAMES.map((name, c) => `<option value="${c}"${c === shownChannel ? ' selected' : ''}>${name}</option>`).join('');
channelSel.addEventListener('change', () => {
  shownChannel = Number(channelSel.value);
  setView('channel');
});
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
      return [rect.left + ox + size * SQ3 * (q + r / 2), rect.top + oy + size * 1.5 * r];
    },
  },
});

newField();
legend();
requestAnimationFrame(frame);
