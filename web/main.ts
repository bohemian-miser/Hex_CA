// The demo page: runs the automaton on a canvas and checks each round against
// the reference fill.

import { Automaton, Foot, Terrain } from '../src/ca.js';
import { Board, DIRS, coordsOf, makeBoard } from '../src/hex.js';
import { Expected, expectedFill, randomLine, rng } from '../src/lines.js';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

interface Round {
  line: number[];
  expected: Expected;
  decidedAt: number | null;
  done: boolean;
  ok: boolean | null;
}

let board: Board;
let ca: Automaton;
let rand = rng(Date.now() >>> 0);
let walls = new Set<number>();
/** Fill of finished rounds: cell → round number (for its colour). */
let history = new Map<number, number>();
let round: Round;
let roundNo = 0;
let roundsOk = 0;
let roundsDone = 0;
let playing = true;
let doneAt = 0;

const settings = { radius: 18, wiggle: 1.2, speed: 30, auto: true };

function newField(): void {
  board = makeBoard(settings.radius);
  ca = new Automaton(board, () => Terrain.Empty);
  walls = new Set();
  history = new Map();
  roundNo = 0;
  roundsOk = 0;
  roundsDone = 0;
  if (!startRound()) newField();
  layout();
  draw();
}

function startRound(): boolean {
  const line = randomLine(board, rand, { blocked: walls, wiggle: settings.wiggle });
  if (!line) return false;
  if (roundNo > 0) {
    const flooded = ca.commit(line);
    for (const c of flooded) history.set(c, roundNo);
    for (const c of round.line) walls.add(c);
  } else {
    ca.commit(line);
  }
  roundNo++;
  round = { line, expected: expectedFill(board, line), decidedAt: null, done: false, ok: null };
  return true;
}

function step(): void {
  if (round.done) return;
  const start = ca.generation;
  ca.step();
  const foot = ca.cells[round.line[0]].foot;
  if (round.decidedAt === null && foot >= Foot.Tie) round.decidedAt = ca.generation;
  if (ca.changed === 0 && ca.generation - start === 1) {
    round.done = true;
    const got = ca.flooded();
    const want = round.expected.fill;
    round.ok = got.size === want.size && [...got].every((c) => want.has(c));
    roundsDone++;
    if (round.ok) roundsOk++;
    doneAt = performance.now();
  }
}

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

function historyColour(n: number): string {
  // Old rounds: the flood hue walked round the wheel, kept soft.
  const hue = (165 + n * 47) % 360;
  const dark = matchesDark();
  return dark ? `hsl(${hue} 35% 30%)` : `hsl(${hue} 45% 84%)`;
}

function matchesDark(): boolean {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
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
  // Pulses: a dot pushed towards where it is heading (away from where it came from).
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
  // Feet: ring by state, an arrow once decided.
  for (const f of [round.line[0], round.line[round.line.length - 1]]) {
    const foot = ca.cells[f].foot;
    const [x, y] = centre(f);
    ctx.lineWidth = Math.max(1.5, size * 0.22);
    if (foot === Foot.Armed || foot === Foot.Listening) {
      hexPath(x, y, s * 1.1);
      ctx.strokeStyle = c.pulse;
      ctx.stroke();
    } else if (foot === Foot.Tie) {
      ctx.strokeStyle = c.bad;
      const k = size * 0.5;
      ctx.beginPath();
      ctx.moveTo(x - k, y - k); ctx.lineTo(x + k, y + k);
      ctx.moveTo(x + k, y - k); ctx.lineTo(x - k, y + k);
      ctx.stroke();
    } else if (foot >= Foot.Decided) {
      const [vx, vy] = dirVec(foot - Foot.Decided);
      ctx.strokeStyle = c.accent;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + vx * size * 1.6, y + vy * size * 1.6);
      ctx.stroke();
      hexPath(x, y, s * 1.1);
      ctx.stroke();
    }
  }
  readout();
}

function readout(): void {
  const [a0, a1] = round.expected.arcs;
  const shorter = Math.min(a0.length, a1.length);
  const foot = ca.cells[round.line[0]].foot;
  const phase =
    foot === Foot.None ? 'border' : foot === Foot.Armed ? 'arm' : foot === Foot.Listening ? 'race'
    : round.done ? 'done' : foot === Foot.Tie ? 'tie' : 'flood';
  const verdict = round.ok === null
    ? '<span class="pill wait">running</span>'
    : round.ok ? '<span class="pill good">matches</span>' : '<span class="pill bad">differs</span>';
  const rows: Array<[string, string]> = [
    ['Round', String(roundNo)],
    ['Step', String(ca.generation)],
    ['Phase', phase],
    ['Arcs', `${a0.length} · ${a1.length}`],
    ['Decided', round.decidedAt === null ? `due at ${2 + shorter}` : `step ${round.decidedAt}`],
    ['Flooded', `${ca.flooded().size} / ${round.expected.fill.size}`],
    ['Reference', verdict],
    ['Rounds ok', `${roundsOk} / ${roundsDone}`],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ── Loop and controls ───────────────────────────────────────────────────────

let last = performance.now();
let acc = 0;
function frame(now: number): void {
  const dt = Math.min(250, now - last);
  last = now;
  if (playing) {
    if (round.done) {
      if (settings.auto && now - doneAt > 1200) {
        if (!startRound()) newField();
        draw();
      }
    } else {
      acc += dt;
      const every = 1000 / settings.speed;
      let n = 0;
      while (acc >= every && n < 50) {
        step();
        acc -= every;
        n++;
      }
      if (n) draw();
    }
  }
  requestAnimationFrame(frame);
}

function bindSlider(id: 'radius' | 'wiggle' | 'speed', scale: number, onChange?: () => void): void {
  const input = $<HTMLInputElement>(id);
  const out = $(id + 'Out');
  input.addEventListener('input', () => {
    settings[id] = Number(input.value) * scale;
    out.textContent = String(Math.round(settings[id] * 10) / 10);
    onChange?.();
  });
}

bindSlider('radius', 1, () => newField());
bindSlider('wiggle', 0.1);
bindSlider('speed', 1);
$('play').addEventListener('click', () => {
  playing = !playing;
  $('play').textContent = playing ? 'Pause' : 'Play';
});
$('step').addEventListener('click', () => {
  playing = false;
  $('play').textContent = 'Play';
  step();
  draw();
});
$('finish').addEventListener('click', () => {
  while (!round.done) step();
  draw();
});
$('fresh').addEventListener('click', () => {
  rand = rng((Math.random() * 2 ** 32) >>> 0);
  newField();
});
$('add').addEventListener('click', () => {
  while (!round.done) step();
  if (!startRound()) newField();
  draw();
});
$<HTMLInputElement>('auto').addEventListener('change', (e) => {
  settings.auto = (e.target as HTMLInputElement).checked;
});
window.addEventListener('resize', () => {
  layout();
  draw();
});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => draw());
new MutationObserver(() => draw()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

newField();
requestAnimationFrame(frame);
