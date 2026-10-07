// The hybrid game's page (web/game.html, docs/spectacle-ca-hybrid.md): Spectacle on one of its hex fields, the
// lines a hand-written local CA (src/game/line-ca.ts), each player's area the trained flood fill (src/game/area.ts),
// refereed by src/game/host.ts. Players sit at this screen (select one, tap for them); each has a rule, sticky
// across visits. ?map=l2|l3|l4, ?rules=<describeRule>,<describeRule>,… (one per seat), ?theme=light|dark.

import dataJson from './strand-data.json';
import weightsJson from './nca-weights.json';
import { allStrands, Board, PAIRS, RuleTable, type Rule, type Strand, type StrandData } from '../src/strand.js';
import { loadWeights, type HexNCA } from '../src/nca.js';
import {
  axialAt, bounds, centreOf, cssRgb, drawTiles, edgeMid, fitGeom, hexPath, strandPaths, strokePattern, type Geom,
} from '../src/draw.js';
import { css3, divLevel, fitPca3, hslRgb, hueBlend, LEVELS, mixRgb, pcaColour, pixel, ramps, type Pca3 } from '../src/ramp.js';
import { Game, type GameEvent, type Refusal } from '../src/game/host.js';
import { Bot } from '../src/game/bot.js';

const data = dataJson as unknown as StrandData;
const table = new RuleTable(data);
const weights = loadWeights(weightsJson);
const [CLAMP_LO, CLAMP_HI] = weights.clamp ?? [-1, 1];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('board');
const ctx = canvas.getContext('2d')!;

const MAP_LABELS: Record<string, string> = { l2: 'Level 2', l3: 'Level 3', l4: 'Level 4' };
/** Seats (and colours): the area layer runs one flood per seat. */
const MAX_SEATS = 8;
/** Per frame, the clock's work stops here once one flood step has run (what is left is owed to the next frame). */
const FRAME_MS = 12;
/** The channel tiles redraw at most this often. */
const GRID_MS = 150;
/** The side panel's numbers redraw at most this often. */
const PANEL_MS = 120;
const SETUP_KEY = 'hexca.game.setup';
const SPARK_MS = { hit: 700, closed: 1100, tap: 380, refused: 520 } as const;
const SUPER_REFIT_STEPS = 50;
const SUPER_RGB_STEP = 8;
const params = new URLSearchParams(location.search);

const theme = params.get('theme');
if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;

// ── Setup: seats, knobs, map (sticky across visits) ─────────────────────────

interface Seat { name: string; rule: Rule; bot: boolean }
interface Knobs { bounded: boolean; fuel: number; oneWay: boolean; scoreFill: boolean; speed: number; floodPer: number }
interface Setup { seats: Seat[]; selected: number; map: string; knobs: Knobs; pattern: boolean; types: boolean }

const DEFAULT_KNOBS: Knobs = { bounded: false, fuel: 8, oneWay: false, scoreFill: true, speed: 20, floodPer: 1 };

function loadSetup(): Partial<Setup> | null {
  try {
    const raw = localStorage.getItem(SETUP_KEY);
    if (!raw) return null;
    const j = JSON.parse(raw) as { seats?: { name?: unknown; rule?: unknown; bot?: unknown }[] } & Partial<Setup>;
    const seats: Seat[] = [];
    for (const s of j.seats ?? []) {
      const r = typeof s.rule === 'string' ? table.parse(s.rule) : 'missing';
      if (typeof r === 'string') continue;
      seats.push({ name: typeof s.name === 'string' ? s.name : `Player ${seats.length + 1}`, rule: r, bot: s.bot === true });
    }
    return { ...j, seats };
  } catch {
    return null; // private window, or site data cleared/blocked: just not sticky
  }
}

function saveSetup(): void {
  try {
    localStorage.setItem(SETUP_KEY, JSON.stringify({
      seats: seats.map((s) => ({ name: s.name, rule: table.describe(s.rule), bot: s.bot })),
      selected, map: mapId, knobs, pattern: showPattern, types: showTypes,
    }));
  } catch {
    // ditto
  }
}

const stored = loadSetup();
let seats: Seat[] = stored?.seats ?? [];
let selected = Math.max(0, Math.min(seats.length - 1, Number(stored?.selected ?? 0)));
let mapId = params.get('map') ?? stored?.map ?? 'l3';
if (!data.boards[mapId]) mapId = 'l3';
let knobs: Knobs = { ...DEFAULT_KNOBS, ...(stored?.knobs ?? {}) };
let showPattern = stored?.pattern ?? true;
let showTypes = stored?.types ?? false;

const boards = new Map<string, Board>();
function boardOf(id: string): Board {
  let b = boards.get(id);
  if (!b) boards.set(id, (b = new Board(data.boards[id])));
  return b;
}
let board = boardOf(mapId);
let game!: Game;
/** The bot seats' tappers, by owner. */
let bots = new Map<number, Bot>();
/** Game time (ms): runs while playing. */
let clock = 0;
let playing = true;
let notice = '';

/** A random rule whose lines close a few loops of a fair size on this board (a game wants area to win), not
 * held by `taken`. */
function goodRule(taken: readonly Rule[] = []): Rule {
  const held = new Set(taken.map((r) => table.describe(r)));
  for (let k = 0; k < 80; k++) {
    const r = table.sample(Math.random, Math.random() < 0.5 ? 'train' : 'heldout');
    if (held.has(table.describe(r))) continue;
    const loops = allStrands(table.exits(r, board), board).filter((s) => s.closed && s.rows.length >= 10 && s.rows.length <= 90);
    if (loops.length >= 3) return r;
  }
  for (const text of ['128·010000000', '01346·000001000', '15·000000010', '258·010010000']) {
    const r = table.parse(text) as Rule;
    if (!held.has(table.describe(r))) return r;
  }
  return table.sample(Math.random, 'train');
}

// ?rules=a,b,… replaces the seats' rules (adding seats as needed).
const rulesParam = params.get('rules');
if (rulesParam) {
  rulesParam.split(',').slice(0, MAX_SEATS).forEach((text, i) => {
    const r = table.parse(text);
    if (typeof r === 'string') return;
    if (seats[i]) seats[i].rule = r;
    else seats.push({ name: `Player ${i + 1}`, rule: r, bot: false });
  });
}
if (!seats.length) {
  const first = goodRule();
  seats = [{ name: 'Player 1', rule: first, bot: false }, { name: 'Bot 2', rule: goodRule([first]), bot: true }];
}

/** The game owner of seat i: seats sit in order, so owner i + 1. */
const ownerOf = (i: number) => i + 1;
const seatOf = (owner: number) => owner - 1;

/** A fresh game on the current board: every seat re-joins with its rule (a rule two seats share: the later one
 * gets a new random rule). */
function newGame(): void {
  board = boardOf(mapId);
  game = new Game(board, table, weights, undefined, gameKnobs());
  seats.forEach((s, i) => {
    let owner = game.addPlayer(s.name, s.rule, s.bot);
    if (owner < 0) {
      s.rule = goodRule(seats.map((t) => t.rule));
      owner = game.addPlayer(s.name, s.rule, s.bot);
    }
    if (owner !== ownerOf(i)) throw new Error(`seat ${i} sat as owner ${owner}`);
  });
  bots = new Map(seats.flatMap((s, i) => (s.bot ? [[ownerOf(i), new Bot(game, ownerOf(i), Math.random)] as const] : [])));
  clock = 0;
  sparks.length = 0;
  superBasis = null;
  mini = null;
  ruleStrands = null;
  dirty = bgDirty = panelDirty = gridDirty = tilesDirty = true;
  saveSetup();
}

function gameKnobs() {
  return {
    fuel: knobs.bounded ? knobs.fuel : 0, oneWay: knobs.oneWay, scoreFill: knobs.scoreFill,
    stepMs: 1000 / knobs.speed, floodPerStep: knobs.floodPer,
  };
}

/** Knob changes go in live (the host's, and the line CA's one-way, which only taps read) — except the growth
 * mode: the line CA builds its bounded-growth rule once, so a new mode or chord count starts the board over. */
function applyKnobs(): void {
  const k = gameKnobs();
  if (k.fuel !== game.knobs.fuel) {
    newGame();
    notice = k.fuel ? `Bounded growth: a tap lays at most ${k.fuel} chords each way (tap a stuck end for more). The board starts over.`
      : 'Lines grow until they stop. The board starts over.';
    return;
  }
  Object.assign(game.knobs, k);
  const lk = (game.lines as unknown as { knobs?: Record<string, unknown> }).knobs;
  if (lk) lk.oneWay = k.oneWay;
  saveSetup();
  panelDirty = dirty = true;
}

// ── Camera ───────────────────────────────────────────────────────────────────

let geom: Geom = { size: 10, ox: 0, oy: 0 };
let fitSize = 10;
let cssW = 1;
let cssH = 1;
/** Per board cell, its centre on screen (rebuilt when the camera moves). */
let cx = new Float64Array(0);
let cy = new Float64Array(0);

function boardBounds() {
  return bounds((add) => { for (let i = 0; i < board.n; i++) add(board.pos[i] % board.w, Math.floor(board.pos[i] / board.w)); });
}

function placeCells(): void {
  cx = new Float64Array(board.n);
  cy = new Float64Array(board.n);
  for (let i = 0; i < board.n; i++) [cx[i], cy[i]] = centreOf(geom, board.pos[i] % board.w, Math.floor(board.pos[i] / board.w));
  bgDirty = dirty = true;
}

function fit(): void {
  geom = fitGeom(boardBounds(), cssW, cssH, 24);
  fitSize = geom.size;
  placeCells();
}

function resize(): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  const refit = Math.abs(geom.size - fitSize) < 1e-9 || cssW <= 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  geom = { size: geom.size, ox: geom.ox + (w - cssW) / 2, oy: geom.oy + (h - cssH) / 2 };
  cssW = w;
  cssH = h;
  if (refit) fit();
  else placeCells();
}

function zoomAt(x: number, y: number, f: number): void {
  const size = Math.min(fitSize * 16, Math.max(fitSize, geom.size * f));
  const k = size / geom.size;
  geom = { size, ox: x - (x - geom.ox) * k, oy: y - (y - geom.oy) * k };
  placeCells();
}

// ── Colours ──────────────────────────────────────────────────────────────────

interface Colours {
  cell: string; edge: string; fg: string; muted: string; halo: string; bad: string; wave: string; contest: string;
  flood: string; neg: string; pos: string; mix: number; tintFill: number; tintLine: number; seats: string[];
}
let colours: Colours;
function readColours(): void {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  colours = {
    cell: v('--cell'), edge: v('--cell-edge'), fg: v('--fg'), muted: v('--muted'), halo: v('--halo'), bad: v('--bad'),
    wave: v('--wave'), contest: v('--contest'), flood: v('--flood'), neg: v('--neg'), pos: v('--pos'),
    mix: Number(v('--type-mix')) || 0.3, tintFill: Number(v('--tint-fill')) || 0.3, tintLine: Number(v('--tint-line')) || 0.45,
    seats: Array.from({ length: MAX_SEATS }, (_, k) => v(`--tap${k}`)),
  };
}
const ownerColour = (owner: number) => colours.seats[seatOf(owner) % MAX_SEATS] ?? colours.fg;
const rgba = (css: string, a: number) => `rgb(${cssRgb(css).join(' ')} / ${a})`;

// ── Sparks: what just happened, drawn for a moment ──────────────────────────

interface Spark { kind: keyof typeof SPARK_MS; cell: number; owner: number; at: number }
const sparks: Spark[] = [];

function absorb(events: GameEvent[], now: number): void {
  for (const e of events) {
    if (e.kind === 'refused' && bots.has(e.owner)) continue; // a bot weighing its options
    if (e.kind === 'hit' || e.kind === 'closed' || e.kind === 'tap' || e.kind === 'refused') {
      if (e.cell >= 0) sparks.push({ kind: e.kind, cell: e.cell, owner: e.owner, at: now });
    }
    if (e.kind === 'hit' && e.owner) {
      const name = seats[seatOf(e.owner)]?.name ?? `owner ${e.owner}`;
      notice = `${name}'s line was in a collision: both lines are wiped, a wave at a time.`;
    } else if (e.kind === 'closed') {
      notice = `${seats[seatOf(e.owner)]?.name ?? 'A player'} closed a loop: the flood fills what it encloses.`;
    }
  }
  if (sparks.length > 200) sparks.splice(0, sparks.length - 200);
}

// ── The rule pattern (the selected seat's, faint, under the lines) ─────────

let ruleStrands: { key: string; strands: Strand[] } | null = null;
let patternPaths: { key: string; solid: Path2D; dash: Path2D } | null = null;
function ensurePattern(): { solid: Path2D; dash: Path2D } {
  const rule = seats[selected].rule;
  const key = `${mapId}|${table.describe(rule)}`;
  if (!ruleStrands || ruleStrands.key !== key) ruleStrands = { key, strands: allStrands(table.exits(rule, board), board) };
  const pk = `${key}|${geom.size.toFixed(6)}|${geom.ox.toFixed(3)}|${geom.oy.toFixed(3)}`;
  if (!patternPaths || patternPaths.key !== pk) patternPaths = { key: pk, ...strandPaths(ruleStrands.strands, geom) };
  return patternPaths;
}

// ── Views ──────────────────────────────────────────────────────────────────

/** A channel, as colours per board cell: the line CA's, the area's, or the selected seat's flood network's. */
interface ChannelView { id: string; name: string; colour(i: number): readonly [number, number, number]; range?(): string }
/** What the board shows: 'game' (the default), 'super' (the selected seat's flood, every channel superimposed),
 * or one channel's id, large. */
let shown = 'game';

/** The selected seat's flood network and the board-cell → its slot map, if the area layer exposes them. */
function floodOf(owner: number): { nca: HexNCA; slot: Int32Array } | null {
  const a = game.area as unknown as { ncas?: HexNCA[]; slot?: Int32Array; frame?: { slot: Int32Array } };
  const nca = a.ncas?.[owner - 1];
  const slot = a.frame?.slot ?? a.slot;
  return nca && slot ? { nca, slot } : null;
}

const popcount = (v: number) => { let k = 0; for (; v; v &= v - 1) k++; return k; };

function lineViews(): ChannelView[] {
  const ch = game.lines.ch as typeof game.lines.ch & { fuel?: Int32Array };
  const cell = cssRgb(colours.cell);
  const toward = (css: string, t: number) => mixRgb(colours.cell, css, Math.max(0, Math.min(1, t)));
  const own = (o: number) => (o > 0 ? cssRgb(ownerColour(o)) : o < 0 ? cssRgb(colours.contest) : cell);
  const bits = (a: Int32Array, css: string) => (i: number) => (a[i] ? toward(css, 0.35 + (0.65 * popcount(a[i])) / 6) : cell);
  const terr = game.territory();
  const views: ChannelView[] = [
    { id: 'rule', name: 'rule', colour: (i) => own(game.ruleInfo(ch.rule[i])?.owner ?? 0) },
    { id: 'owner', name: 'owner', colour: (i) => own(ch.owner[i]) },
    { id: 'D', name: 'D: chords', colour: bits(ch.D, colours.fg) },
    { id: 'T', name: 'T: tips', colour: bits(ch.T, colours.pos) },
    { id: 'W', name: 'W: waves', colour: bits(ch.W, colours.wave) },
    { id: 'H', name: 'H: hits', colour: (i) => (ch.H[i] ? cssRgb(colours.bad) : cell) },
    { id: 'closed', name: 'closed', colour: bits(ch.closed, colours.flood) },
  ];
  if (ch.fuel && knobs.bounded) {
    const f = ch.fuel;
    views.push({ id: 'fuel', name: 'fuel', colour: (i) => (f[i] > 0 ? toward(colours.flood, f[i] / Math.max(1, knobs.fuel)) : cell) });
  }
  views.push({ id: 'terr', name: 'territory', colour: (i) => own(terr[i]) });
  for (const p of game.seated()) {
    const fill = game.area.fill(p.owner);
    views.push({ id: `fill${p.owner}`, name: `fill: ${p.name}`, colour: (i) => (fill[i] ? cssRgb(ownerColour(p.owner)) : cell) });
  }
  return views;
}

function floodViews(): ChannelView[] {
  const f = floodOf(ownerOf(selected));
  if (!f) return [];
  const rp = ramps(colours.cell, colours.flood, colours.neg, colours.pos);
  const out: ChannelView[] = [];
  for (let c = 0; c < f.nca.channels; c++) {
    const v = f.nca.channel(c);
    out.push({
      id: `nca${c}`, name: c === 0 ? 'wall' : c === 1 ? 'fill' : 'hidden',
      colour: (i) => rp.div[divLevel(v[f.slot[i]], CLAMP_LO, CLAMP_HI) + LEVELS],
      range: () => {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < board.n; i++) { const x = v[f.slot[i]]; if (x < lo) lo = x; if (x > hi) hi = x; }
        const fmt = (x: number) => (Math.abs(x) < 0.005 ? '0' : x.toFixed(2));
        return `${fmt(lo)} … ${fmt(hi)}`;
      },
    });
  }
  return out;
}

// Superimposed: the selected seat's flood, every (hidden) channel folded into one picture (web/nca.ts's view).
type SuperScope = 'hidden' | 'all' | 'noOut';
let superScope: SuperScope = 'hidden';
let superMode: 'pca' | 'hue' = 'pca';
let superOverlay = true;
let superBasis: Pca3 | null = null;
let superFitAt = -1;
let superFor = 0;

function superChannels(C: number): number[] {
  const out: number[] = [];
  for (let c = 0; c < C; c++) {
    if (superScope === 'hidden' && c < 2) continue;
    if (superScope === 'noOut' && c === 1) continue;
    out.push(c);
  }
  return out;
}

function superColours(): ((i: number) => [number, number, number]) | null {
  const owner = ownerOf(selected);
  const f = floodOf(owner);
  if (!f) return null;
  const sel = superChannels(f.nca.channels);
  const chans = sel.map((c) => f.nca.channel(c));
  const bg = cssRgb(colours.cell);
  if (!sel.length) return () => bg;
  const snap = (x: number) => Math.max(0, Math.min(255, Math.round(x / SUPER_RGB_STEP) * SUPER_RGB_STEP));
  if (superMode === 'pca') {
    if (!superBasis || superFor !== owner || f.nca.steps - superFitAt >= SUPER_REFIT_STEPS || f.nca.steps < superFitAt) {
      const cells = Int32Array.from({ length: board.n }, (_, i) => i);
      superBasis = fitPca3((i, k) => chans[k][f.slot[i]], sel.length, cells, superFor === owner ? superBasis ?? undefined : undefined);
      superFitAt = f.nca.steps;
      superFor = owner;
    }
    const basis = superBasis;
    const row = new Float64Array(sel.length);
    return (i) => {
      for (let k = 0; k < sel.length; k++) row[k] = chans[k][f.slot[i]];
      const [r, g, b] = pcaColour(basis, row);
      return [snap(r), snap(g), snap(b)];
    };
  }
  const hues = sel.map((_, k) => hslRgb((360 * k) / sel.length, 75, 55));
  const w = new Array<number>(sel.length);
  const span = Math.max(1e-6, CLAMP_HI);
  return (i) => {
    for (let k = 0; k < sel.length; k++) w[k] = Math.min(1, Math.abs(chans[k][f.slot[i]]) / span);
    const [r, g, b] = hueBlend(w, hues, bg);
    return [snap(r), snap(g), snap(b)];
  };
}

// ── Drawing ────────────────────────────────────────────────────────────────

let bg: HTMLCanvasElement | null = null;
let bgDirty = true;
let dirty = true;

function drawBackground(): void {
  const dpr = window.devicePixelRatio || 1;
  if (!bg) bg = document.createElement('canvas');
  bg.width = canvas.width;
  bg.height = canvas.height;
  const g = bg.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);
  drawTiles(g, board, table.leafOrder, geom, cssW, cssH,
    { types: showTypes, cell: colours.cell, edge: colours.edge, muted: colours.muted, mix: colours.mix });
  bgDirty = false;
}

/** Cells filled per colour, one path per colour. */
function fillCells(colourOf: (i: number) => string | null, s: number): void {
  const groups = new Map<string, Path2D>();
  for (let i = 0; i < board.n; i++) {
    const c = colourOf(i);
    if (!c) continue;
    let path = groups.get(c);
    if (!path) groups.set(c, (path = new Path2D()));
    hexPath(path, cx[i], cy[i], s);
  }
  for (const [c, path] of groups) {
    ctx.fillStyle = c;
    ctx.fill(path);
  }
}

/** Who holds each tile: a player's colour over the tiles they hold, stronger under their own lines; contested
 * tiles (inside two players' fills) grey. */
function drawTerritory(s: number): void {
  const terr = game.territory();
  const { D, owner } = game.lines.ch;
  const tints = new Map<number, [string, string]>();
  for (const p of game.seated()) tints.set(p.owner, [rgba(ownerColour(p.owner), colours.tintFill), rgba(ownerColour(p.owner), colours.tintLine)]);
  const contest = rgba(colours.contest, 0.4);
  fillCells((i) => {
    const t = terr[i];
    if (t < 0) return contest;
    if (!t) return null;
    const pair = tints.get(t);
    if (!pair) return null;
    return D[i] && owner[i] === t ? pair[1] : pair[0];
  }, s * 0.985);
}

/** Every drawn chord in its owner's colour (closed loops a little bolder), with a halo; tips as dots at the edge
 * they will cross, waves as sparks on the edge they run out of, a hit as a ring. `thin`: over a channel view. */
function drawLines(s: number, thin = false): void {
  const { rule, D, T, W, H, closed } = game.lines.ch;
  const open = new Map<number, Path2D>();
  const loops = new Map<number, Path2D>();
  const pathOf = (m: Map<number, Path2D>, o: number) => { let p = m.get(o); if (!p) m.set(o, (p = new Path2D())); return p; };
  for (let i = 0; i < board.n; i++) {
    const d = D[i];
    if (!d) continue;
    const info = game.ruleInfo(rule[i]);
    if (!info) continue;
    const bitsHere = info.chords[board.pos[i]];
    for (let k = 0; k < PAIRS.length; k++) {
      if (!((bitsHere >> k) & 1)) continue;
      const [a, b] = PAIRS[k];
      if (!((d >> a) & 1) || !((d >> b) & 1)) continue;
      const path = pathOf((closed[i] >> a) & 1 ? loops : open, info.owner);
      path.moveTo(...edgeMid(cx[i], cy[i], a, s));
      path.lineTo(...edgeMid(cx[i], cy[i], b, s));
    }
  }
  const lw = thin ? Math.max(1.2, s * 0.14) : Math.max(1.8, s * 0.24);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = colours.halo;
  ctx.lineWidth = lw + Math.max(2, s * 0.14);
  for (const p of open.values()) ctx.stroke(p);
  ctx.lineWidth = lw * 1.35 + Math.max(2, s * 0.14);
  for (const p of loops.values()) ctx.stroke(p);
  for (const [o, p] of open) { ctx.strokeStyle = ownerColour(o); ctx.lineWidth = lw; ctx.stroke(p); }
  for (const [o, p] of loops) { ctx.strokeStyle = ownerColour(o); ctx.lineWidth = lw * 1.35; ctx.stroke(p); }
  if (thin) return;
  // tips and waves
  const rTip = Math.max(2.2, s * 0.17);
  for (let i = 0; i < board.n; i++) {
    const t = T[i];
    const w = W[i];
    if (!t && !w && !H[i]) continue;
    for (let d = 0; d < 6; d++) {
      if ((t >> d) & 1) {
        const [x, y] = edgeMid(cx[i], cy[i], d, s, 0.92);
        ctx.beginPath();
        ctx.arc(x, y, rTip + 1.5, 0, 2 * Math.PI);
        ctx.fillStyle = colours.halo;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, rTip, 0, 2 * Math.PI);
        ctx.fillStyle = ownerColour(game.lines.ch.owner[i]);
        ctx.fill();
      }
      if ((w >> d) & 1) {
        const [x, y] = edgeMid(cx[i], cy[i], d, s, 0.8);
        ctx.beginPath();
        ctx.arc(x, y, Math.max(2, s * 0.2), 0, 2 * Math.PI);
        ctx.fillStyle = colours.wave;
        ctx.fill();
      }
    }
    if (H[i]) {
      ctx.beginPath();
      ctx.arc(cx[i], cy[i], s * 0.55, 0, 2 * Math.PI);
      ctx.strokeStyle = colours.bad;
      ctx.lineWidth = Math.max(2, s * 0.16);
      ctx.stroke();
    }
  }
}

function drawSparks(s: number, now: number): boolean {
  let live = false;
  for (const sp of sparks) {
    const age = (now - sp.at) / SPARK_MS[sp.kind];
    if (age >= 1 || age < 0) continue;
    live = true;
    const x = cx[sp.cell];
    const y = cy[sp.cell];
    ctx.globalAlpha = 1 - age;
    ctx.lineWidth = Math.max(1.5, s * 0.12);
    if (sp.kind === 'refused') {
      const r = s * 0.4;
      ctx.strokeStyle = colours.bad;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
      ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
      ctx.stroke();
    } else {
      const grow = sp.kind === 'closed' ? 2.6 : sp.kind === 'hit' ? 1.8 : 0.9;
      ctx.strokeStyle = sp.kind === 'hit' ? colours.bad : ownerColour(sp.owner);
      ctx.beginPath();
      ctx.arc(x, y, s * (0.4 + grow * age), 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  if (!live) sparks.length = 0;
  return live;
}

function draw(now: number): void {
  dirty = false;
  if (bgDirty || !bg) drawBackground();
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bg!, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const s = geom.size;
  if (shown === 'game') {
    drawTerritory(s);
    if (showPattern) strokePattern(ctx, ensurePattern(), s, colours.halo, colours.muted);
    drawLines(s);
  } else if (shown === 'super') {
    const colourOf = superColours();
    if (colourOf) fillCells((i) => css3(colourOf(i)), s * 0.985);
    if (superOverlay) drawLines(s, true);
  } else {
    const view = lineViews().concat(floodViews()).find((v) => v.id === shown);
    if (view) fillCells((i) => css3(view.colour(i)), s * 0.985);
    drawLines(s, true);
  }
  if (drawSparks(s, now)) dirty = true;
}

// ── Channel tiles ────────────────────────────────────────────────────────────

interface Tile { id: string; el: HTMLButtonElement; cv: HTMLCanvasElement; g: CanvasRenderingContext2D; name: HTMLElement; range: HTMLElement }
const tiles = new Map<string, Tile>();
let tilesDirty = true;
let gridDirty = true;
let lastGrid = 0;
let mini: { map: Int32Array; img: ImageData; px: Uint32Array; forBoard: Board; w: number; h: number } | null = null;

function syncTiles(views: ChannelView[], box: HTMLElement): void {
  const want = new Set(views.map((v) => v.id));
  for (const [id, t] of tiles) if (t.el.parentElement === box && !want.has(id)) { t.el.remove(); tiles.delete(id); }
  for (const v of views) {
    let t = tiles.get(v.id);
    if (!t) {
      const el = document.createElement('button');
      el.className = 'tile';
      el.type = 'button';
      const cv = document.createElement('canvas');
      const head = document.createElement('span');
      const name = document.createElement('b');
      head.append(name);
      const range = document.createElement('i');
      el.append(cv, head, range);
      el.addEventListener('click', () => show(shown === v.id ? 'game' : v.id));
      t = { id: v.id, el, cv, g: cv.getContext('2d')!, name, range };
      tiles.set(v.id, t);
      mini = null;
    }
    box.append(t.el); // keeps the order
    if (t.name.textContent !== v.name) t.name.textContent = v.name;
    t.el.title = `Show ${v.name} on the board`;
    t.el.setAttribute('aria-pressed', String(shown === v.id));
  }
}

function miniLayout(): NonNullable<typeof mini> | null {
  const first = tiles.values().next().value as Tile | undefined;
  if (!first) return null;
  const rect = first.cv.getBoundingClientRect();
  if (rect.width < 4) return null;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (mini && mini.forBoard === board && mini.w === w && mini.h === h) {
    for (const t of tiles.values()) if (t.cv.width !== w || t.cv.height !== h) { t.cv.width = w; t.cv.height = h; }
    return mini;
  }
  const g = fitGeom(boardBounds(), rect.width, rect.height, 4);
  const map = new Int32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [q, r] = axialAt(g, (x + 0.5) / dpr, (y + 0.5) / dpr);
      map[y * w + x] = board.on(r, q) ? board.cellOf[r * board.w + q] : -1;
    }
  }
  for (const t of tiles.values()) { t.cv.width = w; t.cv.height = h; }
  const img = new ImageData(w, h);
  return (mini = { map, img, px: new Uint32Array(img.data.buffer), forBoard: board, w, h });
}

function drawGrid(): void {
  const lv = lineViews();
  const fv = floodViews();
  if (tilesDirty) {
    syncTiles(lv, $('lineTiles'));
    syncTiles(fv, $('floodTiles'));
    $('floodLabel').textContent = fv.length
      ? `Flood channels of ${seats[selected].name}: the trained network, with their lines as its walls`
      : 'Flood channels: not exposed by this area layer';
    tilesDirty = false;
  }
  const g = miniLayout();
  if (!g) return;
  gridDirty = false;
  const tone = new Uint32Array(board.n);
  for (const v of lv.concat(fv)) {
    const t = tiles.get(v.id);
    if (!t) continue;
    for (let i = 0; i < board.n; i++) tone[i] = pixel(v.colour(i));
    const { map, px } = g;
    for (let p = 0; p < map.length; p++) px[p] = map[p] < 0 ? 0 : tone[map[p]];
    t.g.putImageData(g.img, 0, 0);
    const text = v.range ? v.range() : '';
    if (t.range.textContent !== text) t.range.textContent = text;
  }
}

function show(id: string): void {
  shown = id;
  for (const t of tiles.values()) t.el.setAttribute('aria-pressed', String(t.id === id));
  $('showGame').setAttribute('aria-pressed', String(id === 'game'));
  $('showSuper').setAttribute('aria-pressed', String(id === 'super'));
  $('superGroup').hidden = id !== 'super';
  if (id === 'super') superBasis = null;
  legend();
  dirty = true;
}

// ── Side panel ───────────────────────────────────────────────────────────────

let panelDirty = true;
let lastPanel = 0;
let playersKey = '';

const REASONS: Record<Refusal, (name: string) => string> = {
  'no chord': () => `No chord of this rule on that tile: try a neighbouring tile.`,
  'rival tile': () => `Refused: a rival's line is on that tile, and a line owns its whole tile.`,
  drawn: (n) => `Refused: ${n}'s line already runs on every chord of that tile.`,
  hot: () => `Refused: that tile was wiped this very step. Try again.`,
  heads: (n) => `Refused: ${n} already has ${game.knobs.maxTips} growing tips. Wait for a line to stop.`,
  respawn: (n) => `Refused: ${n} was just in a collision. Wait a moment.`,
  territory: () => `Refused: that tile is inside a rival's area.`,
};

function renderPlayers(): void {
  const ol = $('players');
  const hud = $('hud');
  const scores = game.scores();
  const key = seats.map((s) => `${s.name}|${table.describe(s.rule)}|${s.bot}`).join(',') + `|${selected}|${colours.seats.join()}`;
  if (key !== playersKey) {
    playersKey = key;
    ol.replaceChildren();
    hud.replaceChildren();
    seats.forEach((s, i) => {
      const li = document.createElement('li');
      li.setAttribute('aria-selected', String(i === selected));
      li.title = 'Select: your taps on the board are this player\'s';
      li.innerHTML = `<span class="swatch" style="background:${colours.seats[i]}"></span>`
        + `<div class="what"><b></b>${s.bot ? '<span class="tag">bot</span>' : ''}<div class="sub"></div></div><span class="score"></span>`;
      li.querySelector('b')!.textContent = `${s.name} · ${table.describe(s.rule)}`;
      li.addEventListener('click', () => select(i));
      const x = document.createElement('button');
      x.className = 'x';
      x.textContent = '×';
      x.title = 'Remove this player (the board starts over)';
      x.setAttribute('aria-label', `Remove ${s.name}`);
      x.disabled = seats.length <= 1;
      x.addEventListener('click', (ev) => { ev.stopPropagation(); removeSeat(i); });
      li.append(x);
      ol.append(li);
      const chip = document.createElement('button');
      chip.className = 'chip';
      chip.setAttribute('aria-pressed', String(i === selected));
      chip.title = `${s.name}: ${table.describe(s.rule)} (select)`;
      chip.innerHTML = `<span class="swatch" style="background:${colours.seats[i]}"></span><span class="n"></span><span class="s"></span>`;
      chip.querySelector('.n')!.textContent = s.name;
      chip.addEventListener('click', () => select(i));
      hud.append(chip);
    });
  }
  seats.forEach((s, i) => {
    const owner = ownerOf(i);
    const li = ol.children[i] as HTMLElement | undefined;
    const chip = hud.children[i] as HTMLElement | undefined;
    if (!li || !chip) return;
    const tips = game.lines.countTips(owner);
    const wait = game.respawnLeft(owner);
    const sub = `${game.lines.lineTiles(owner)} line tiles · ${tips}/${game.knobs.maxTips} tips${wait > 0 ? ` · waiting ${(wait / 1000).toFixed(1)} s` : ''}`;
    const subEl = li.querySelector('.sub')!;
    if (subEl.textContent !== sub) subEl.textContent = sub;
    const sc = String(scores[owner] ?? 0);
    const scEl = li.querySelector('.score')!;
    if (scEl.textContent !== sc) scEl.textContent = sc;
    const chipS = chip.querySelector('.s')!;
    if (chipS.textContent !== sc) chipS.textContent = sc;
    chip.classList.toggle('wait', wait > 0);
  });
  $<HTMLButtonElement>('addPlayer').disabled = $<HTMLButtonElement>('addBot').disabled = seats.length >= MAX_SEATS;
}

function describeRule(): void {
  const s = seats[selected];
  $('ruleLabel').textContent = `Rule of ${s.name}`;
  $<HTMLInputElement>('ruleText').value = table.describe(s.rule);
  const sub = table.subsets[s.rule.s];
  $('ruleNow').innerHTML = `<span class="swatch" style="background:${colours.seats[selected]}"></span> <b></b><br>`
    + `<span class="note">edge classes ${sub.edges.join(', ')} carry lines; ${sub.count.toLocaleString()} rules in subset ${sub.key}</span>`;
  $('ruleNow').querySelector('b')!.textContent = table.describe(s.rule);
}

let stepsDone = 0;
let rateFrom = performance.now();
let stepsPerSec = 0;
let floodPerSec = 0;
let floodFrom = 0;
let flooding = 0;
let floodMs = 0;

function readout(): void {
  const rows: [string, string][] = [
    ['Line steps', game.steps.toLocaleString()],
    ['Steps/s', playing ? String(stepsPerSec) : 'paused'],
    ['Flood steps', `${game.floodSteps.toLocaleString()}${game.settling() ? '' : ' (settled)'}`],
    ['Flood steps/s', playing ? String(floodPerSec) : '–'],
    ['ms/flood step', floodMs ? floodMs.toFixed(1) : '–'],
    ['Tiles', board.n.toLocaleString()],
  ];
  $('readout').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  const s = seats[selected];
  const slow = floodMs > 60
    ? ` <span class="warn">The flood is ${Math.round(floodMs)} ms a step here (a network on every tile, per player, on this browser's main thread), so areas fill in slowly.</span>`
    : '';
  $('status').innerHTML = (notice || `Tap a tile to start a line of <b>${escapeHtml(s.name)}</b>'s rule (${table.describe(s.rule)}) at the chord nearest your tap.`) + slow;
}

function legend(): void {
  const items = shown === 'game' || shown === 'super'
    ? [
      '<span><i></i>a line (bolder: a closed loop)</span>',
      `<span><i class="dot" style="background:${colours.seats[selected]}"></i>a growing tip</span>`,
      `<span><i class="dot" style="background:${colours.wave}"></i>a wipe wave</span>`,
      '<span><i class="sw" style="background:var(--contest)"></i>contested</span>',
    ].concat(showPattern && shown === 'game' ? [`<span><i class="thin"></i>${table.describe(seats[selected].rule)}'s pattern</span>`] : [])
    : [`<span>Board: <b>${escapeHtml(tiles.get(shown)?.name.textContent ?? shown)}</b>, the lines thin on top</span>`];
  if (shown === 'super') {
    items.unshift(superMode === 'pca'
      ? `<span>${escapeHtml(seats[selected].name)}'s flood, ${superScope === 'hidden' ? 'hidden channels' : superScope === 'all' ? 'every channel' : 'every channel but fill'}: PCA's top 3 directions → <b>red</b>/<b>green</b>/<b>blue</b>.</span>`
      : `<span>${escapeHtml(seats[selected].name)}'s flood: each channel its own hue, brightness = |value|.</span>`);
  }
  $('legend').innerHTML = items.join('');
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

function mapInfo(): void {
  const b = data.boards[mapId];
  $('mapInfo').textContent = `Spectacle's hex field, level ${b.level}, the ${b.root} patch: ${b.tiles.toLocaleString()} tiles`
    + (mapId === 'l4' ? ' (the flood is slow here on a CPU: areas take a long while to fill).' : '.');
}

// ── Seats ─────────────────────────────────────────────────────────────────

function select(i: number): void {
  selected = Math.max(0, Math.min(seats.length - 1, i));
  describeRule();
  notice = '';
  superBasis = null;
  tilesDirty = gridDirty = panelDirty = dirty = true;
  legend();
  saveSetup();
}

function addSeat(bot: boolean): void {
  if (seats.length >= MAX_SEATS) return;
  const rule = goodRule(seats.map((s) => s.rule));
  const name = `${bot ? 'Bot' : 'Player'} ${seats.length + 1}`;
  const owner = game.addPlayer(name, rule, bot);
  if (owner !== ownerOf(seats.length)) return;
  seats.push({ name, rule, bot });
  if (bot) {
    bots.set(owner, new Bot(game, owner, Math.random));
    playersKey = '';
    panelDirty = tilesDirty = gridDirty = true;
    saveSetup();
  } else select(seats.length - 1);
}

function removeSeat(i: number): void {
  if (seats.length <= 1) return;
  seats.splice(i, 1);
  seats.forEach((s, k) => { if (/^(Player|Bot) \d+$/.test(s.name)) s.name = `${s.bot ? 'Bot' : 'Player'} ${k + 1}`; });
  if (selected >= seats.length) selected = seats.length - 1;
  newGame();
  select(selected);
  notice = 'A player left: the board starts over.';
}

/** The selected seat's rule, if no other seat holds it. Lines already on the board keep the rule they grew with. */
function setSeatRule(r: Rule): void {
  const s = seats[selected];
  const holder = seats.findIndex((t, k) => k !== selected && table.describe(t.rule) === table.describe(r));
  if (holder >= 0) {
    $('ruleErr').textContent = `${seats[holder].name} already plays ${table.describe(r)}: every player's rule is their own.`;
    $('ruleErr').className = 'note err';
    return;
  }
  if (!game.setRule(ownerOf(selected), r)) return;
  s.rule = { s: r.s, digits: r.digits.slice() };
  $('ruleErr').textContent = game.lines.lineTiles(ownerOf(selected))
    ? 'The lines already drawn keep their old rule (another pattern of yours: your new lines collide with them).' : '';
  $('ruleErr').className = 'note';
  describeRule();
  panelDirty = dirty = true;
  legend();
  saveSetup();
}

// ── Loop ─────────────────────────────────────────────────────────────────────

let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(100, now - last);
  last = now;
  if (playing) {
    clock += dt;
    const s0 = game.steps;
    const f0 = game.floodSteps;
    const t0 = performance.now();
    game.tick(clock, FRAME_MS);
    const ran = game.steps - s0;
    const flooded = game.floodSteps - f0;
    if (flooded) {
      flooding += performance.now() - t0;
      floodFrom += flooded;
    }
    stepsDone += ran;
    if (ran || flooded) dirty = panelDirty = gridDirty = true;
    for (const bot of bots.values()) bot.act(clock);
  }
  const events = game.drain();
  if (events.length) {
    absorb(events, now);
    dirty = panelDirty = true;
  }
  if (now - rateFrom >= 1000) {
    stepsPerSec = Math.round((stepsDone * 1000) / (now - rateFrom));
    floodPerSec = Math.round((floodFrom * 1000) / (now - rateFrom));
    if (floodFrom) floodMs = flooding / floodFrom;
    stepsDone = floodFrom = 0;
    flooding = 0;
    rateFrom = now;
    panelDirty = true;
  }
  if (dirty) draw(now);
  if (panelDirty && now - lastPanel >= PANEL_MS) {
    panelDirty = false;
    lastPanel = now;
    renderPlayers();
    readout();
  }
  if ((gridDirty || tilesDirty) && now - lastGrid >= GRID_MS) {
    drawGrid();
    lastGrid = now;
  }
  requestAnimationFrame(frame);
}

function setPlaying(on: boolean): void {
  playing = on;
  $('play').textContent = playing ? 'Pause' : 'Run';
  panelDirty = true;
}

// ── Controls ───────────────────────────────────────────────────────────────

/** A tap at a canvas point, for the selected seat. */
function tapAt(x: number, y: number): void {
  const owner = ownerOf(selected);
  const why = game.tap(owner, (x - geom.ox) / geom.size, (y - geom.oy) / geom.size);
  notice = why ? REASONS[why](seats[selected].name) : '';
  absorb(game.drain(), performance.now());
  dirty = panelDirty = gridDirty = true;
  if (!why && !playing) setPlaying(true);
  readout();
}

let down: { x: number; y: number; ox: number; oy: number; id: number; moved: boolean } | null = null;
canvas.addEventListener('pointerdown', (ev) => {
  canvas.setPointerCapture(ev.pointerId);
  down = { x: ev.offsetX, y: ev.offsetY, ox: geom.ox, oy: geom.oy, id: ev.pointerId, moved: false };
});
canvas.addEventListener('pointermove', (ev) => {
  if (!down || ev.pointerId !== down.id) return;
  const dx = ev.offsetX - down.x;
  const dy = ev.offsetY - down.y;
  if (!down.moved && Math.hypot(dx, dy) < 6) return;
  down.moved = true;
  canvas.classList.add('panning');
  geom = { size: geom.size, ox: down.ox + dx, oy: down.oy + dy };
  placeCells();
});
canvas.addEventListener('pointerup', (ev) => {
  if (!down || ev.pointerId !== down.id) return;
  const moved = down.moved;
  down = null;
  canvas.classList.remove('panning');
  if (!moved) tapAt(ev.offsetX, ev.offsetY);
});
canvas.addEventListener('pointercancel', () => { down = null; canvas.classList.remove('panning'); });
canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  zoomAt(ev.offsetX, ev.offsetY, Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0015)));
}, { passive: false });
$('zoomIn').addEventListener('click', () => zoomAt(cssW / 2, cssH / 2, 1.5));
$('zoomOut').addEventListener('click', () => zoomAt(cssW / 2, cssH / 2, 1 / 1.5));
$('zoomFit').addEventListener('click', () => fit());

$('addPlayer').addEventListener('click', () => addSeat(false));
$('addBot').addEventListener('click', () => addSeat(true));
$('ruleUse').addEventListener('click', () => useRuleText());
$<HTMLInputElement>('ruleText').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') useRuleText(); });
function useRuleText(): void {
  const r = table.parse($<HTMLInputElement>('ruleText').value);
  if (typeof r === 'string') {
    $('ruleErr').textContent = r;
    $('ruleErr').className = 'note err';
    return;
  }
  setSeatRule(r);
}
$('ruleRandom').addEventListener('click', () => setSeatRule(goodRule(seats.map((s) => s.rule))));
$<HTMLSelectElement>('preset').addEventListener('change', (ev) => {
  const sel = ev.target as HTMLSelectElement;
  const r = table.parse(sel.value);
  if (typeof r !== 'string') setSeatRule(r);
  sel.value = '';
});

function bindKnobs(): void {
  const growthFree = $<HTMLInputElement>('growth-free');
  const growthFuel = $<HTMLInputElement>('growth-fuel');
  const fuel = $<HTMLInputElement>('fuel');
  const syncGrowth = () => {
    growthFree.checked = !knobs.bounded;
    growthFuel.checked = knobs.bounded;
    $('fuelRow').hidden = !knobs.bounded;
    fuel.value = String(knobs.fuel);
    $('fuelOut').textContent = String(knobs.fuel);
  };
  syncGrowth();
  growthFree.addEventListener('change', () => { knobs.bounded = false; syncGrowth(); applyKnobs(); tilesDirty = true; });
  growthFuel.addEventListener('change', () => { knobs.bounded = true; syncGrowth(); applyKnobs(); tilesDirty = true; });
  fuel.addEventListener('input', () => { $('fuelOut').textContent = fuel.value; });
  fuel.addEventListener('change', () => { knobs.fuel = Number(fuel.value); syncGrowth(); applyKnobs(); });
  const box = (id: 'oneWay' | 'scoreFill') => {
    const el = $<HTMLInputElement>(id);
    el.checked = knobs[id];
    el.addEventListener('change', () => { knobs[id] = el.checked; applyKnobs(); });
  };
  box('oneWay');
  box('scoreFill');
  const slider = (id: 'speed' | 'floodPer', out: string) => {
    const el = $<HTMLInputElement>(id);
    el.value = String(knobs[id]);
    $(out).textContent = el.value;
    el.addEventListener('input', () => { knobs[id] = Number(el.value); $(out).textContent = el.value; applyKnobs(); });
  };
  slider('speed', 'speedOut');
  slider('floodPer', 'floodPerOut');
  const pattern = $<HTMLInputElement>('pattern');
  pattern.checked = showPattern;
  pattern.addEventListener('change', () => { showPattern = pattern.checked; legend(); saveSetup(); dirty = true; });
  const types = $<HTMLInputElement>('types');
  types.checked = showTypes;
  types.addEventListener('change', () => { showTypes = types.checked; saveSetup(); bgDirty = dirty = true; });
}
bindKnobs();

$<HTMLSelectElement>('map').append(...Object.keys(data.boards).map((id) => new Option(
  `${MAP_LABELS[id] ?? id} (${data.boards[id].tiles.toLocaleString()} tiles)`, id)));
$<HTMLSelectElement>('map').addEventListener('change', (ev) => setMap((ev.target as HTMLSelectElement).value));
function setMap(id: string): void {
  mapId = data.boards[id] ? id : 'l3';
  $<HTMLSelectElement>('map').value = mapId;
  newGame();
  cssW = cssH = 1;
  resize();
  mapInfo();
  notice = '';
  const u = new URL(location.href);
  u.searchParams.set('map', mapId);
  history.replaceState(null, '', u);
}

$('play').addEventListener('click', () => setPlaying(!playing));
$('step').addEventListener('click', () => {
  setPlaying(false);
  clock += game.knobs.stepMs;
  game.tick(clock);
  dirty = panelDirty = gridDirty = true;
});
$('reset').addEventListener('click', () => {
  newGame();
  notice = 'A clear board: the same players and rules.';
  setPlaying(true);
});
$('showGame').addEventListener('click', () => show('game'));
$('showSuper').addEventListener('click', () => show('super'));
$<HTMLSelectElement>('superScope').addEventListener('change', (ev) => {
  superScope = (ev.target as HTMLSelectElement).value as SuperScope;
  superBasis = null;
  legend();
  dirty = true;
});
for (const mode of ['pca', 'hue'] as const) {
  $<HTMLInputElement>(`superMode-${mode}`).addEventListener('change', () => {
    superMode = mode;
    superBasis = null;
    legend();
    dirty = true;
  });
}
$<HTMLInputElement>('superOverlay').addEventListener('change', (ev) => {
  superOverlay = (ev.target as HTMLInputElement).checked;
  dirty = true;
});
$('superRecompute').addEventListener('click', () => { superBasis = null; dirty = true; });

document.addEventListener('keydown', (ev) => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (ev.target instanceof HTMLInputElement && ev.target.type === 'text') return;
  if (ev.key >= '1' && ev.key <= '8' && Number(ev.key) <= seats.length) {
    select(Number(ev.key) - 1);
  } else if (ev.key === 'p' || ev.key === 'P') {
    const box = $<HTMLInputElement>('pattern');
    box.checked = showPattern = !showPattern;
    legend();
    saveSetup();
    dirty = true;
  } else if (ev.key === ' ' && !(ev.target instanceof HTMLButtonElement || ev.target instanceof HTMLSelectElement)) {
    ev.preventDefault();
    setPlaying(!playing);
  }
});

new ResizeObserver(() => { resize(); mini = null; gridDirty = true; }).observe(canvas);
const restyle = () => {
  readColours();
  playersKey = '';
  bgDirty = dirty = panelDirty = gridDirty = true;
  describeRule();
  legend();
};
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', restyle);
new MutationObserver(restyle).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

// For the console and headless checks.
Object.assign(window, {
  hexgame: {
    get game() { return game; },
    get board() { return board; },
    get seats() { return seats; },
    get selected() { return selected; },
    table,
    /** Client coordinates of a board cell's centre (cell: the board's cell index). */
    point(cell: number): [number, number] {
      const rect = canvas.getBoundingClientRect();
      return [rect.left + cx[cell], rect.top + cy[cell]];
    },
    /** Client coordinates of edge d's midpoint of a cell, `f` of the way out from its centre. */
    edgePoint(cell: number, d: number, f = 0.5): [number, number] {
      const rect = canvas.getBoundingClientRect();
      const [x, y] = edgeMid(cx[cell], cy[cell], d, geom.size, f);
      return [rect.left + x, rect.top + y];
    },
    /** Each seat's numbers. */
    stats() {
      const sc = game.scores();
      return seats.map((s, i) => ({ name: s.name, rule: table.describe(s.rule), score: sc[ownerOf(i)], lineTiles: game.lines.lineTiles(ownerOf(i)),
        tips: game.lines.countTips(ownerOf(i)) }));
    },
    get notice() { return notice; },
  },
});

// ── Start ────────────────────────────────────────────────────────────────────

readColours();
setMap(mapId);
select(selected);
show('game');
setPlaying(true);
requestAnimationFrame(frame);
