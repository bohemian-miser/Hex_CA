// The GL half of scripts/area-probe.ts: the same boards and scoring (scripts/area-probe-lib.ts) with the flood run
// by src/game/area-gl.ts, in Chromium. Options and weights come in `window.PROBE` (a script beside the page); one
// line, `RESULT {...}`, goes back out through --dump-dom.

import strandData from '../web/strand-data.json';
import { areaFrame } from '../src/game/area.js';
import { GlArea } from '../src/game/area-gl.js';
import { rng } from '../src/lines.js';
import { loadWeights } from '../src/nca.js';
import { Board, RuleTable, type StrandData } from '../src/strand.js';
import { describeSamples, liveWalls, levelSeed, probe, reportLines, sampleBoards, type Result, type Runner } from './area-probe-lib.js';

interface Options { levels: string[]; counts: Record<string, number>; mults: number[]; seed: number; minFill: number;
  verbose: boolean; live: boolean; weights: { name: string; json: unknown }[] }
const opts = (window as unknown as { PROBE: Options }).PROBE;
const out: { error?: string; renderer?: string; lines: string[]; results: Result[] } = { lines: [], results: [] };
/** The bare flood: no settle rule, so it runs every step the probe asks for (AreaOptions). */
const RAW = { settlePerR: 0, capPerR: 0 };

/** GlArea (one player) as the probe's runner: a load wipes the old walls (the flood resets) and draws the new. */
function glRunner(area: GlArea, board: Board): Runner {
  const ch = { owner: new Int32Array(board.n), D: new Int32Array(board.n) };
  const all = Int32Array.from({ length: board.n }, (_, i) => i);
  const N = area.frame.S ** 2;
  return {
    load(walls) {
      ch.owner.fill(0);
      ch.D.fill(0);
      area.update({ ch }, all);
      for (let i = 0; i < board.n; i++) if (walls[area.frame.slot[i]]) { ch.owner[i] = 1; ch.D[i] = 1; }
      area.update({ ch }, all);
    },
    edit(walls) {
      const changed: number[] = [];
      for (let i = 0; i < board.n; i++) {
        const v = walls[area.frame.slot[i]] ? 1 : 0;
        if (ch.D[i] !== v) { ch.owner[i] = v; ch.D[i] = v; changed.push(i); }
      }
      area.update({ ch }, Int32Array.from(changed));
    },
    get steps() { return area.steps(1); },
    advance(n) { area.step(n); },
    ch1: () => area.readState(1).subarray(N, 2 * N),
  };
}

try {
  const data = strandData as unknown as StrandData;
  const table = new RuleTable(data);
  const gl = document.createElement('canvas').getContext('webgl2');
  if (!gl) throw new Error('no webgl2');
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  out.renderer = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  for (const level of opts.levels) {
    const board = new Board(data.boards[level]);
    const frame = areaFrame(board);
    const samples = sampleBoards(table, board, frame, opts.counts[level], rng(levelSeed(opts.seed, level, opts.minFill)), opts.minFill);
    out.lines.push('', describeSamples(level, board, frame, samples));
    for (const { name, json } of opts.weights) {
      const res = probe(level, name, board, frame, samples, opts.mults, glRunner(new GlArea(gl, board, loadWeights(json), 1, RAW), board),
        opts.live ? liveWalls(level, frame.S) : undefined);
      for (const r of res) out.lines.push(...reportLines(r, opts.verbose));
      out.results.push(...res);
    }
  }
} catch (e) {
  out.error = String(e instanceof Error ? e.stack ?? e.message : e);
}
document.body.textContent = `RESULT ${JSON.stringify(out)}`;
