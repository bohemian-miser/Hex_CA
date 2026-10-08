// The browser half of scripts/area-gl-parity.ts: cpuArea against glArea on strand walls, and GL ms per step.
// Runs on load, synchronously, and writes one JSON line into the page (`RESULT {...}`) for --dump-dom to carry out.

import strandData from '../web/strand-data.json';
import weightsJson from '../web/nca-weights.json';
import { CpuArea, type LineChannels } from '../src/game/area.js';
import { GlArea } from '../src/game/area-gl.js';
import { rng } from '../src/lines.js';
import { loadWeights } from '../src/nca.js';
import { allStrands, Board, DCOL, DROW, RuleTable, type Rule, type StrandData } from '../src/strand.js';

const params = new URLSearchParams(location.search);
const data = strandData as unknown as StrandData;
const table = new RuleTable(data);
const weights = loadWeights(weightsJson);
const out: Record<string, unknown> = {};

function context(): WebGL2RenderingContext {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) throw new Error('no webgl2');
  return gl;
}

/** Lines for `owners` players on a board: per player, 1-3 loops or rim-to-rim claims of one random rule; a strand on a
 * tile another player already holds is skipped (one rule per tile). */
function lines(board: Board, owners: number, rand: () => number): LineChannels & { cells: Int32Array } {
  const ch = { owner: new Int32Array(board.n), D: new Int32Array(board.n) };
  for (let p = 1; p <= owners; p++) {
    for (let tries = 0; tries < 50; tries++) {
      const s = Math.floor(rand() * table.nSub);
      const rule: Rule = { s, digits: table.nOpt[s].map((n) => Math.floor(rand() * n)) };
      const strands = allStrands(table.exits(rule, board), board).filter((st) => {
        if (st.closed) return true;
        const n = st.rows.length;
        const offA = !board.on(st.rows[0] + DROW[st.ins[0]], st.cols[0] + DCOL[st.ins[0]]);
        const offB = !board.on(st.rows[n - 1] + DROW[st.outs[n - 1]], st.cols[n - 1] + DCOL[st.outs[n - 1]]);
        return offA && offB;
      });
      if (!strands.length) continue;
      const k = 1 + Math.floor(rand() * 3);
      for (let j = 0; j < k && strands.length; j++) {
        const st = strands.splice(Math.floor(rand() * strands.length), 1)[0];
        const cells = st.rows.map((r, t) => board.cellOf[r * board.w + st.cols[t]]);
        if (cells.some((c) => ch.owner[c] && ch.owner[c] !== p)) continue;
        cells.forEach((c, t) => { ch.D[c] |= (1 << st.ins[t]) | (1 << st.outs[t]); ch.owner[c] = p; });
      }
      break;
    }
  }
  return { ch, cells: Int32Array.from({ length: board.n }, (_, i) => i) };
}

try {
  const gl = context();
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  out.renderer = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  out.maxDrawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS);

  // Parity: the same lines into both layers, read out every 8 steps up to 16 R.
  const parity: unknown[] = [];
  const plan: [string, number, number][] = [['l2', 8, 2], ['l3', Number(params.get('l3') ?? 3), 2]];
  for (const [name, boardsN, owners] of plan) {
    const board = new Board(data.boards[name]);
    const rand = rng(17);
    for (let b = 0; b < boardsN; b++) {
      const ln = lines(board, owners, rand);
      const cpu = new CpuArea(board, weights, owners);
      const glA = new GlArea(gl, board, weights, owners);
      cpu.update(ln, ln.cells);
      glA.update(ln, ln.cells);
      const { R, S } = cpu.frame;
      const N = S * S;
      let maxCh1 = 0;
      let maxAll = 0;
      let fillDiff = 0;
      let filled = 0;
      for (let s = 8; s <= 16 * R; s += 8) {
        cpu.step(8);
        glA.step(8);
        for (let p = 1; p <= owners; p++) {
          if (!cpu.wallCount[p - 1]) continue;
          const a = cpu.ncas[p - 1].state;
          const g = glA.readState(p);
          for (let i = N; i < 2 * N; i++) maxCh1 = Math.max(maxCh1, Math.abs(a[i] - g[i]));
          if (s === 16 * R || s % 64 === 0) for (let i = 0; i < a.length; i++) maxAll = Math.max(maxAll, Math.abs(a[i] - g[i]));
        }
      }
      for (let p = 1; p <= owners; p++) {
        const fa = cpu.fill(p);
        const fg = glA.fill(p);
        for (let i = 0; i < fa.length; i++) { if (fa[i] !== fg[i]) fillDiff++; filled += fa[i]; }
      }
      const ta = cpu.territory();
      const tg = glA.territory();
      let terrDiff = 0;
      for (let i = 0; i < ta.length; i++) if (ta[i] !== tg[i]) terrDiff++;
      parity.push({ board: name, n: b, steps: 16 * R, owners, walls: Array.from(cpu.wallCount), filled, maxCh1: +maxCh1.toExponential(2),
        maxAll: +maxAll.toExponential(2), fillDiff, terrDiff });
    }
  }
  out.parity = parity;

  // ms per step: one player (and four) with a few lines, after a warm-up; readPixels to wait for the GPU.
  const timing: unknown[] = [];
  for (const name of ['l2', 'l3', 'l4']) {
    const board = new Board(data.boards[name]);
    for (const owners of [1, 4]) {
      const glA = new GlArea(gl, board, weights, owners);
      const ln = lines(board, owners, rng(3));
      glA.update(ln, ln.cells);
      glA.step(4);
      glA.fill(1);
      const n = name === 'l4' ? 40 : 100;
      const t0 = performance.now();
      glA.step(n);
      glA.fill(1);
      const ms = (performance.now() - t0) / n;
      const row: Record<string, unknown> = { board: name, cells: board.n, owners, glMsPerStep: +ms.toFixed(3) };
      if (owners === 1) {
        const cpu = new CpuArea(board, weights, 1);
        cpu.update(ln, ln.cells);
        cpu.step(2);
        const m = name === 'l4' ? 4 : name === 'l3' ? 20 : 100;
        const t1 = performance.now();
        cpu.step(m);
        row.cpuMsPerStep = +((performance.now() - t1) / m).toFixed(2);
      }
      timing.push(row);
    }
  }
  out.timing = timing;
} catch (e) {
  out.error = String(e instanceof Error ? e.stack ?? e.message : e);
}
document.body.textContent = `RESULT ${JSON.stringify(out)}`;
