// The demo page: draw lines on the field (or let it draw), run the automaton
// on a canvas, and check every round against the reference fill.

import { Automaton, Event, Foot, Orient, Terrain } from '../src/ca.js';
import { Board, DIRS, coordsOf, hexDistance, indexOf, makeBoard } from '../src/hex.js';
import { Outcome, expectedOutcome, randomLine, randomLoop, randomScribble, rng } from '../src/lines.js';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

let board: Board;
let ca: Automaton;
let rand = rng(Date.now() >>> 0);
/** Cells waiting to be drawn, one per step. */
let queue: number[] = [];
/** Fill of finished rounds: cell → round number (for its colour). */
let history = new Map<number, number>();
let roundNo = 0;
let roundsOk = 0;
let roundsDone = 0;
/** The current line's outcome once it has closed and the board went quiet. */
let verdict: { want: Outcome; ok: boolean } | null = null;
let closedAt: number | null = null;
let playing = true;
let drawing = false;
let autoDraw = true;

const settings = { radius: 14, speed: 30 };

function newField(): void {
  board = makeBoard(settings.radius);
  ca = new Automaton(board);
  queue = [];
  history = new Map();
  roundNo = 0;
  roundsOk = 0;
  roundsDone = 0;
  verdict = null;
  closedAt = null;
  ca.step(); // let the border learn itself
  layout();
  draw();
}

/** Finish the current line (whatever state it is in) so a new one can start. */
function endLine(): void {
  if (!ca.line.length) return;
  const flooded = ca.commit();
  for (const c of flooded) history.set(c, roundNo);
  queue = [];
  verdict = null;
  closedAt = null;
}

function lineIsFinished(): boolean {
  return ca.line.length > 0 && !ca.canExtend;
}

/** Start a new line at `cell` (ending the last one). */
function startLine(cell: number): boolean {
  endLine();
  if (!ca.canPlace(cell)) return false;
  roundNo++;
  queue = [cell];
  return true;
}

function queueDrawing(cells: number[]): void {
  if (!cells.length || !startLine(cells[0])) return;
  queue = cells.slice();
}

function randomDrawing(kind: 'bridge' | 'loop' | 'scribble'): void {
  // Old lines, and the current one (about to become old), block new ones.
  const blocked = new Set<number>();
  for (const i of board.cells) if (ca.cells[i].terrain !== Terrain.Empty) blocked.add(i);
  const cells =
    kind === 'bridge' ? randomLine(board, rand, { blocked, wiggle: 0.6 + rand() * 1.4 })
    : kind === 'loop' ? randomLoop(board, rand, { blocked, wiggle: 0.4 + rand() * 1.2 })
    : randomScribble(board, rand, { blocked });
  if (!cells) {
    // The field is full: start again.
    newField();
    return;
  }
  queueDrawing(cells);
}

// ── Stepping ────────────────────────────────────────────────────────────────

function step(): void {
  if (queue.length > (drawing ? DRAG_LAG : 0) && ca.canExtend) {
    const c = queue.shift()!;
    if (ca.canPlace(c)) ca.place(c);
    else queue = [];
  } else if (!ca.canExtend) {
    queue = [];
  }
  ca.step();
  if (closedAt === null && lineIsFinished()) closedAt = ca.generation;
  if (closedAt !== null && !verdict && ca.changed === 0) {
    const want = expectedOutcome(board, ca.line);
    const got = ca.flooded();
    const ok = want.event === ca.event && got.size === want.fill.size && [...got].every((c) => want.fill.has(c));
    verdict = { want, ok };
    roundsDone++;
    if (ok) roundsOk++;
  }
}

// ── Pointer drawing ─────────────────────────────────────────────────────────

function cellAt(ev: PointerEvent): number {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left - ox) / size;
  const y = (ev.clientY - rect.top - oy) / size;
  // Pixel → fractional axial (pointy-top), then cube rounding.
  const fq = (SQ3 / 3) * x - y / 3;
  const fr = (2 / 3) * y;
  const [q, r] = cubeRound(fq, fr);
  if (hexDistance(q, r) > board.radius) return -1;
  return indexOf(board, q, r);
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

/** The cells on the straight hex line from a to b, excluding a. */
function hexLine(a: number, b: number): number[] {
  const [aq, ar] = coordsOf(board, a);
  const [bq, br] = coordsOf(board, b);
  const n = hexDistance(bq - aq, br - ar);
  const out: number[] = [];
  for (let k = 1; k <= n; k++) {
    const t = k / n;
    const [q, r] = cubeRound(aq + (bq - aq) * t + 1e-6, ar + (br - ar) * t + 1e-6);
    out.push(indexOf(board, q, r));
  }
  return out;
}

/** The cell a drag extends from: the last queued cell, else the head. */
function dragEnd(): number {
  return queue.length ? queue[queue.length - 1] : ca.head;
}

/** Loops shorter than this enclose nothing: a touch that would make one is a wiggle of the hand, not a circuit. */
const MIN_LOOP = 6;
/** While dragging, keep this many cells back from the board so wiggles can still be smoothed out. */
const DRAG_LAG = 2;

function extendTo(cell: number): void {
  const from = dragEnd();
  if (from < 0 || cell === from) return;
  for (const c of hexLine(from, cell)) {
    if (c === dragEnd()) continue;
    // Dragging back over cells not yet drawn takes them back.
    const back = queue.indexOf(c);
    if (back >= 0) {
      queue.length = back + 1;
      continue;
    }
    if (ca.cells[c].terrain !== Terrain.Empty || ca.line.includes(c) || !adjacent(dragEnd(), c)) return;
    if (!smoothTo(c)) return;
    queue.push(c);
  }
}

/**
 * Make room for `c` as the next cell. If it would touch the line just behind
 * its head (a loop too small to hold anything), cut the corner instead: drop
 * the queued cells after the one it touches. Returns false if the touch is to
 * cells already on the board, where there is nothing to cut.
 */
function smoothTo(c: number): boolean {
  for (;;) {
    const seq = [...ca.line, ...queue];
    let near = -1;
    for (let k = seq.length - 3; k >= 0; k--) {
      if (adjacent(seq[k], c)) {
        near = k;
        break;
      }
    }
    if (near < 0 || seq.length - near + 1 >= MIN_LOOP) return true; // no touch, or a real circuit
    if (near < ca.line.length) return false;
    queue.length = near - ca.line.length + 1;
  }
}

function adjacent(a: number, b: number): boolean {
  for (let d = 0; d < 6; d++) if (board.neighbours[a * 6 + d] === b) return true;
  return false;
}

canvas.addEventListener('pointerdown', (ev) => {
  const cell = cellAt(ev);
  if (cell < 0) return;
  setAuto(false);
  if (!playing) setPlaying(true);
  canvas.setPointerCapture(ev.pointerId);
  // Carry on from the head if we're next to it and the line is still open…
  if (ca.canExtend && ca.line.length && (cell === dragEnd() || adjacent(cell, dragEnd()))) {
    drawing = true;
    if (cell !== dragEnd()) extendTo(cell);
    return;
  }
  // …otherwise start a new line here, on free ground.
  if (ca.cells[cell].terrain === Terrain.Empty) drawing = startLine(cell);
});
canvas.addEventListener('pointermove', (ev) => {
  if (!drawing || !ca.canExtend) return;
  const cell = cellAt(ev);
  if (cell >= 0) extendTo(cell);
});
const stopDrawing = () => {
  drawing = false;
};
canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointercancel', stopDrawing);

// ── Drawing the board ───────────────────────────────────────────────────────

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
  // Pointy-top hexagon field: width √3·size·(2R+1), height size·(3R+2).
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
    const px = x + s * Math.cos(a);
    const py = y + s * Math.sin(a);
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

/** Screen-space unit vector of direction d. */
function dirVec(d: number): [number, number] {
  const [dq, dr] = DIRS[d];
  const x = SQ3 * (dq + dr / 2);
  const y = 1.5 * dr;
  const len = Math.hypot(x, y);
  return [x / len, y / len];
}

let css: CSSStyleDeclaration;
const tok = (name: string) => css.getPropertyValue(name).trim();

function matchesDark(): boolean {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function historyColour(n: number): string {
  // Old rounds: the flood hue walked round the wheel, kept soft.
  const hue = (165 + n * 47) % 360;
  return matchesDark() ? `hsl(${hue} 35% 30%)` : `hsl(${hue} 45% 84%)`;
}

function draw(): void {
  css = getComputedStyle(document.documentElement);
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const c = {
    cell: tok('--cell'), ring: tok('--cell-ring'), edge: tok('--cell-edge'), line: tok('--line'),
    wall: tok('--wall'), flood: tok('--flood'), pulse: tok('--pulse'), accent: tok('--accent'), bad: tok('--bad'),
  };
  const s = size * 0.97;
  const thin = size > 6;
  for (const i of board.cells) {
    const cell = ca.cells[i];
    const [x, y] = centre(i);
    let fill = cell.border ? c.ring : c.cell;
    const h = history.get(i);
    if (h !== undefined) fill = historyColour(h);
    if (cell.flood) fill = c.flood;
    if (cell.terrain === Terrain.Wall) fill = cell.flood ? c.flood : c.wall;
    if (cell.terrain === Terrain.Line) fill = c.line;
    hexPath(x, y, s);
    ctx.fillStyle = fill;
    ctx.fill();
    if (thin) {
      ctx.strokeStyle = c.edge;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
    if (cell.terrain === Terrain.Wall && cell.flood) {
      // An old line under the flood: keep it visible.
      hexPath(x, y, s * 0.45);
      ctx.fillStyle = c.wall;
      ctx.fill();
    }
  }
  // Cells queued to be drawn.
  ctx.lineWidth = Math.max(1, size * 0.14);
  ctx.strokeStyle = c.line;
  for (const i of queue) {
    const [x, y] = centre(i);
    hexPath(x, y, s * 0.6);
    ctx.stroke();
  }
  // Ring pulses: a dot pushed towards where it is heading.
  for (const i of board.cells) {
    const p = ca.cells[i].pulse;
    if (!p) continue;
    const [x, y] = centre(i);
    for (let d = 0; d < 6; d++) {
      if (!(p & (1 << d))) continue;
      const [vx, vy] = dirVec(d);
      ctx.beginPath();
      ctx.arc(x - vx * size * 0.3, y - vy * size * 0.3, Math.max(2, size * 0.42), 0, Math.PI * 2);
      ctx.fillStyle = c.pulse;
      ctx.fill();
    }
  }
  // Signals on the line: pass 1 (turn count) amber, pass 2 (fill) in the flood colour; the anchor ringed.
  for (const i of ca.line) {
    const cell = ca.cells[i];
    const [x, y] = centre(i);
    if (cell.sum >= 0 || cell.sweep !== Orient.None) {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, size * 0.5), 0, Math.PI * 2);
      ctx.fillStyle = cell.sum >= 0 ? c.pulse : c.flood;
      ctx.fill();
    }
    if (cell.decided !== Orient.None || (cell.anchorDir >= 0 && ca.event === Event.Circuit && verdict === null)) {
      hexPath(x, y, s * 1.12);
      ctx.lineWidth = Math.max(1.5, size * 0.2);
      ctx.strokeStyle = cell.decided !== Orient.None ? c.accent : c.pulse;
      ctx.stroke();
    }
  }
  // The head: a ring while drawing; the bridge judge's state once it has closed.
  const h = ca.head;
  if (h >= 0) {
    const cell = ca.cells[h];
    const [x, y] = centre(h);
    ctx.lineWidth = Math.max(1.5, size * 0.22);
    if (cell.foot === Foot.Armed || cell.foot === Foot.Listening) {
      hexPath(x, y, s * 1.12);
      ctx.strokeStyle = c.pulse;
      ctx.stroke();
    } else if (cell.foot === Foot.Tie) {
      ctx.strokeStyle = c.bad;
      const k = size * 0.5;
      ctx.beginPath();
      ctx.moveTo(x - k, y - k); ctx.lineTo(x + k, y + k);
      ctx.moveTo(x + k, y - k); ctx.lineTo(x - k, y + k);
      ctx.stroke();
    } else if (cell.foot >= Foot.Decided) {
      const [vx, vy] = dirVec(cell.foot - Foot.Decided);
      ctx.strokeStyle = c.accent;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + vx * size * 1.6, y + vy * size * 1.6);
      ctx.stroke();
      hexPath(x, y, s * 1.12);
      ctx.stroke();
    } else if (ca.canExtend) {
      hexPath(x, y, s * 0.55);
      ctx.fillStyle = c.accent;
      ctx.fill();
    }
  }
  readout();
}

function readout(): void {
  const event = ca.event;
  const head = ca.head;
  let kind = ca.line.length ? 'open line' : '—';
  let detail = '—';
  let phase = ca.line.length ? 'drawing' : 'waiting';
  if (event === Event.Circuit) {
    kind = 'circuit';
    const anchor = ca.line.find((i) => ca.cells[i].decided !== Orient.None);
    const o = anchor === undefined ? Orient.None : ca.cells[anchor].decided;
    const loopLen = anchor === undefined ? null : ca.line.length - ca.line.indexOf(anchor);
        // Pass 1 walks the loop backwards, so the drawing went the other way round.
    const drawn = o === Orient.Ccw ? 'drawn clockwise (sum 6)' : o === Orient.Cw ? 'drawn anticlockwise (sum 2)' : 'impossible sum';
    detail = o === Orient.None ? 'counting turns' : `loop ${loopLen}, ${drawn}`;
    phase = verdict ? 'done' : o === Orient.None ? 'pass 1: count' : 'pass 2: fill';
  } else if (event === Event.Bridge) {
    kind = 'bridge';
    const want = verdict?.want ?? expectedOutcome(board, ca.line);
    if (want.arcs) detail = `arcs ${want.arcs[0].length} · ${want.arcs[1].length}`;
    const foot = ca.cells[head].foot;
    phase = verdict ? 'done' : foot === Foot.Armed || foot === Foot.Listening ? 'echoes' : foot === Foot.Tie ? 'tie' : 'flood';
  }
  const ref = !verdict
    ? '<span class="pill wait">—</span>'
    : verdict.ok ? '<span class="pill good">matches</span>' : '<span class="pill bad">differs</span>';
  const rows: Array<[string, string]> = [
    ['Line', `${ca.line.length} cells`],
    ['Closed as', kind],
    ['Detail', detail],
    ['Phase', phase],
    ['Step', String(ca.generation)],
    ['Flooded', verdict ? `${ca.flooded().size} / ${verdict.want.fill.size}` : String(ca.flooded().size)],
    ['Reference', ref],
    ['Rounds ok', `${roundsOk} / ${roundsDone}`],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  status();
}

function status(): void {
  const el = $('status');
  const event = ca.event;
  let text: string;
  if (!ca.line.length) text = 'Drag on the field to draw a line.';
  else if (ca.canExtend) text = 'Keep going: touch your own line to close a circuit, or reach the edge again to make a bridge.';
  else if (event === Event.Circuit) text = verdict ? 'Circuit filled. Draw again to start a new line.' : 'Circuit closed. Counting the turns round the loop…';
  else if (event === Event.Bridge) {
    if (!verdict) text = 'Bridge made. Timing the echoes round the border…';
    else if (ca.cells[ca.head].foot === Foot.Tie) text = 'Bridge made, but the two arcs tie, so nothing floods.';
    else text = 'Bridge filled. Draw again to start a new line.';
  }
  else text = 'Drag on the field to draw a line.';
  el.textContent = text;
}

// ── Loop and controls ───────────────────────────────────────────────────────

let last = performance.now();
let acc = 0;
let nextAuto = 0;
const kinds: Array<'bridge' | 'loop' | 'scribble'> = ['loop', 'bridge', 'scribble', 'bridge', 'loop'];
let kindAt = 0;

function frame(now: number): void {
  const dt = Math.min(250, now - last);
  last = now;
  if (playing) {
    acc += dt;
    const every = 1000 / settings.speed;
    let n = 0;
    while (acc >= every && n < 50) {
      step();
      acc -= every;
      n++;
    }
    if (n) draw();
    // The demo draws on its own until someone draws.
    if (autoDraw && !queue.length && (verdict || !ca.line.length || (ca.canExtend && !drawing))) {
      if (!nextAuto) nextAuto = now + (verdict ? 1400 : 300);
      if (now >= nextAuto) {
        nextAuto = 0;
        randomDrawing(kinds[kindAt++ % kinds.length]);
        draw();
      }
    } else nextAuto = 0;
  }
  requestAnimationFrame(frame);
}

function setPlaying(on: boolean): void {
  playing = on;
  $('play').textContent = playing ? 'Pause' : 'Play';
}

function setAuto(on: boolean): void {
  autoDraw = on;
  $<HTMLInputElement>('auto').checked = on;
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

bindSlider('radius', () => newField());
bindSlider('speed');
$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  step();
  draw();
});
$('fresh').addEventListener('click', () => {
  rand = rng((Math.random() * 2 ** 32) >>> 0);
  newField();
});
const manual = (kind: 'bridge' | 'loop' | 'scribble') => () => {
  setAuto(false);
  if (!playing) setPlaying(true);
  randomDrawing(kind);
  draw();
};
$('randBridge').addEventListener('click', manual('bridge'));
$('randLoop').addEventListener('click', manual('loop'));
$('randScribble').addEventListener('click', manual('scribble'));
$<HTMLInputElement>('auto').addEventListener('change', (e) => {
  autoDraw = (e.target as HTMLInputElement).checked;
});
// Re-measure whenever the canvas changes size (window, fonts, the status line), or the pointer maps to the wrong cells.
new ResizeObserver(() => {
  if (!board) return;
  layout();
  draw();
}).observe(canvas);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => draw());
new MutationObserver(() => draw()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

newField();
requestAnimationFrame(frame);
