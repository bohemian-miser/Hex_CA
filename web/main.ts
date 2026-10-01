// The demo page: paint and erase walls, watch the automaton settle, and
// check the settled fill against the reference.

import { Automaton, Cell, NONE, filled } from '../src/ca.js';
import { Board, DIRS, coordsOf, hexDistance, indexOf, makeBoard } from '../src/hex.js';
import { expectedFill, randomWalls, rng } from '../src/lines.js';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

let board: Board;
let ca: Automaton;
let rand = rng(Date.now() >>> 0);
let playing = true;
let view: 'fill' | 'fields' = 'fill';
/** The reference answer for the current walls, once the automaton has settled. */
let check: { ok: boolean; want: number } | null = null;
let settledAt: number | null = null;

const settings = { radius: 12, speed: 600 };

function newField(withPicture: boolean): void {
  board = makeBoard(settings.radius);
  ca = new Automaton(board);
  if (withPicture) for (const c of randomWalls(board, rand)) ca.setWall(c, true);
  touched();
  layout();
  draw();
}

/** The walls changed (or the state did): the old check no longer holds. */
function touched(): void {
  check = null;
  settledAt = null;
}

function step(): void {
  ca.step();
  if (ca.changed === 0 && !check) {
    settledAt = ca.generation;
    const want = expectedFill(board, ca.walls());
    const got = ca.filled();
    check = { ok: got.size === want.size && [...got].every((c) => want.has(c)), want: want.size };
  } else if (ca.changed > 0 && check) {
    touched();
  }
}

// ── Painting ────────────────────────────────────────────────────────────────

let painting: boolean | null = null; // true = adding walls, false = erasing
let eraser = false;
let lastCell = -1;

function cellAt(ev: PointerEvent): number {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left - ox) / size;
  const y = (ev.clientY - rect.top - oy) / size;
  const [q, r] = cubeRound((SQ3 / 3) * x - y / 3, (2 / 3) * y);
  return hexDistance(q, r) > board.radius ? -1 : indexOf(board, q, r);
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

function paintTo(cell: number): void {
  if (painting === null || cell < 0) return;
  for (const c of lastCell >= 0 ? hexLine(lastCell, cell) : [cell]) {
    if (board.inside[c] && ca.cells[c].wall !== painting) {
      ca.setWall(c, painting);
      touched();
    }
  }
  lastCell = cell;
  draw();
}

canvas.addEventListener('pointerdown', (ev) => {
  const cell = cellAt(ev);
  if (cell < 0) return;
  canvas.setPointerCapture(ev.pointerId);
  // Paint, unless the eraser is picked or it's the right button.
  painting = !(ev.button === 2 || eraser);
  lastCell = -1;
  paintTo(cell);
});
canvas.addEventListener('pointermove', (ev) => {
  if (painting !== null) paintTo(cellAt(ev));
});
const stop = () => {
  painting = null;
  lastCell = -1;
};
canvas.addEventListener('pointerup', stop);
canvas.addEventListener('pointercancel', stop);
canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

// ── Drawing ─────────────────────────────────────────────────────────────────

let size = 10;
let ox = 0;
let oy = 0;
const SQ3 = Math.sqrt(3);

function layout(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const R = board.radius;
  size = Math.min((rect.width - 16) / (SQ3 * (2 * R + 1)), (rect.height - 16) / (3 * R + 2));
  ox = rect.width / 2;
  oy = rect.height / 2;
}

function centre(i: number): [number, number] {
  const [q, r] = coordsOf(board, i);
  return [ox + size * SQ3 * (q + r / 2), oy + size * 1.5 * r];
}

function hexPath(x: number, y: number, s: number): void {
  ctx.beginPath();
  for (let k = 0; k < 6; k++) {
    const a = (Math.PI / 180) * (60 * k - 30);
    if (k === 0) ctx.moveTo(x + s * Math.cos(a), y + s * Math.sin(a));
    else ctx.lineTo(x + s * Math.cos(a), y + s * Math.sin(a));
  }
  ctx.closePath();
}

function dirVec(d: number): [number, number] {
  const [dq, dr] = DIRS[d];
  const x = SQ3 * (dq + dr / 2);
  const y = 1.5 * dr;
  const len = Math.hypot(x, y);
  return [x / len, y / len];
}

let css: CSSStyleDeclaration;
const tok = (name: string) => css.getPropertyValue(name).trim();

function isDark(): boolean {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function fieldColour(c: Cell): string {
  // Region label → hue; distance to the label's source → lightness.
  if (c.lab === NONE) return isDark() ? 'hsl(0 0% 12%)' : 'hsl(0 0% 88%)';
  const hue = (c.lab * 137.5) % 360;
  const t = Math.min(1, c.d / Math.max(8, board.radius * 2));
  return isDark() ? `hsl(${hue} 45% ${42 - 22 * t}%)` : `hsl(${hue} 60% ${80 - 30 * t}%)`;
}

function draw(): void {
  css = getComputedStyle(document.documentElement);
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const col = {
    cell: tok('--cell'), ring: tok('--cell-ring'), edge: tok('--cell-edge'), wall: tok('--line'),
    flood: tok('--flood'), accent: tok('--accent'), fg: tok('--fg'),
  };
  const s = size * 0.97;
  for (const i of board.cells) {
    const c = ca.cells[i];
    const [x, y] = centre(i);
    const onRing = isRingCell(i);
    let fill: string;
    if (c.wall) fill = col.wall;
    else if (view === 'fields') fill = fieldColour(c);
    else if (filled(c)) fill = col.flood;
    else fill = onRing ? col.ring : col.cell;
    hexPath(x, y, s);
    ctx.fillStyle = fill;
    ctx.fill();
    if (size > 6) {
      ctx.strokeStyle = col.edge;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }
  if (view === 'fields') {
    // The spanning trees: a tick from each cell towards its parent; roots ringed, with their count.
    ctx.strokeStyle = col.fg;
    ctx.fillStyle = col.fg;
    ctx.lineWidth = Math.max(1, size * 0.1);
    ctx.font = `600 ${Math.max(8, size * 0.7)}px ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const i of board.cells) {
      const c = ca.cells[i];
      if (c.wall || c.lab === NONE) continue;
      const [x, y] = centre(i);
      if (c.par >= 0) {
        const [vx, vy] = dirVec(c.par);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + vx * size * 0.9, y + vy * size * 0.9);
        ctx.stroke();
      } else if (c.d === 0) {
        hexPath(x, y, s * 0.8);
        ctx.stroke();
        ctx.fillText(String(c.s), x, y);
      }
    }
  }
  readout();
}

const ringCache = new Map<Board, Set<number>>();
function isRingCell(i: number): boolean {
  let set = ringCache.get(board);
  if (!set) {
    set = new Set(board.cells.filter((c) => {
      for (let d = 0; d < 6; d++) {
        const j = board.neighbours[c * 6 + d];
        if (j < 0 || !board.inside[j]) return true;
      }
      return false;
    }));
    ringCache.set(board, set);
  }
  return set.has(i);
}

function readout(): void {
  // Read straight off the cells: the east corner holds T; roots hold their region's count.
  let T = 0;
  const roots: number[] = [];
  let enclosed = 0;
  for (const i of board.cells) {
    const c = ca.cells[i];
    if (isRingCell(i)) T = Math.max(T, c.tot);
    if (!c.wall && c.lab !== NONE && c.d === 0) roots.push(c.s);
    if (!c.wall && c.lab === NONE) enclosed++;
  }
  const state = check ? `settled at step ${settledAt}` : `settling (${ca.changed} cells changed)`;
  const ref = !check
    ? '<span class="pill wait">wait</span>'
    : check.ok ? '<span class="pill good">matches</span>' : '<span class="pill bad">differs</span>';
  const rows: Array<[string, string]> = [
    ['Step', String(ca.generation)],
    ['State', state],
    ['Open border T', String(T)],
    ['Border regions', roots.length ? roots.sort((a, b) => b - a).join(' · ') : '—'],
    ['Walled-in cells', String(enclosed)],
    ['Filled', check ? `${ca.filled().size} / ${check.want}` : String(ca.filled().size)],
    ['Reference', ref],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ── Loop and controls ───────────────────────────────────────────────────────

let last = performance.now();
let acc = 0;
function frame(now: number): void {
  const dt = Math.min(100, now - last);
  last = now;
  if (playing) {
    acc += (dt * settings.speed) / 1000;
    let n = 0;
    while (acc >= 1 && n < 2000) {
      step();
      acc -= 1;
      n++;
    }
    if (acc > 1) acc = 1;
    if (n) draw();
  }
  requestAnimationFrame(frame);
}

function setPlaying(on: boolean): void {
  playing = on;
  $('play').textContent = playing ? 'Pause' : 'Play';
}

function bindSlider(id: 'radius' | 'speed', onChange?: () => void): void {
  const input = $<HTMLInputElement>(id);
  const out = $(id + 'Out');
  input.addEventListener('input', () => {
    settings[id] = Number(input.value);
    out.textContent = input.value;
    onChange?.();
  });
}

bindSlider('radius', () => newField(false));
bindSlider('speed');
$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  step();
  draw();
});
$('random').addEventListener('click', () => {
  rand = rng((Math.random() * 2 ** 32) >>> 0);
  newField(true);
});
$('clear').addEventListener('click', () => {
  for (const i of board.cells) ca.setWall(i, false);
  touched();
  draw();
});
$('scramble').addEventListener('click', () => {
  // Garbage in every field but the walls: the rule has to recover on its own.
  const { N, P } = ca.limits;
  const int = (n: number) => Math.floor(Math.random() * n);
  for (const i of board.cells) {
    ca.cells[i] = {
      ...ca.cells[i],
      idx: int(P + 1), cnt: int(P + 1), tot: int(P + 1),
      lab: Math.random() < 0.2 ? NONE : int(P), d: int(N + 1),
      par: int(7) - 1, s: int(P + 1), v: Math.random() < 0.5,
    };
  }
  touched();
  draw();
});
for (const tool of ['paint', 'erase'] as const) {
  $<HTMLInputElement>(`tool-${tool}`).addEventListener('change', () => {
    eraser = tool === 'erase';
  });
}
for (const v of ['fill', 'fields'] as const) {
  $<HTMLInputElement>(`view-${v}`).addEventListener('change', () => {
    view = v;
    draw();
  });
}
new ResizeObserver(() => {
  if (!board) return;
  layout();
  draw();
}).observe(canvas);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => draw());
new MutationObserver(() => draw()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

newField(true);
requestAnimationFrame(frame);
