// B1 of docs/spectacle-ca-hybrid.md (§3.6): how well the trained flood fills strand-shaped walls — the fine-tune
// go/no-go. Per level, n boards of 1-3 loops or rim-to-rim claims of one random rule (scripts/area-probe-lib.ts),
// the flood run from the fresh state in the area layer's frame and read out at mult·R steps, exact against
// targets(); by enclosed size, by kind, and by how even the rim regions are.
//
//   npx tsx scripts/area-probe.ts [--levels l2,l3,l4] [--n 200] [--n-l4 24] [--mults 8,16] [--seed 1]
//                                 [--weights name=path ...] [--json out.json] [--dry] [--nontrivial] [--min-fill N]
//                                 [--verbose] [--gl [--chromium path]]
//
// Default weights: web/nca-weights.json (fb-r6816nt). Every weights set sees the same boards. --gl runs the flood
// with src/game/area-gl.ts in headless Chromium (scripts/area-probe-page.ts) instead of HexNCA in Node: the same
// boards and scoring, for level 4, where the CPU takes seconds a step.

import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { areaFrame } from '../src/game/area.js';
import { rng } from '../src/lines.js';
import { HexNCA, loadWeights } from '../src/nca.js';
import { Board, RuleTable, type StrandData } from '../src/strand.js';
import { describeSamples, levelSeed, probe, reportLines, sampleBoards, type Result, type Runner } from './area-probe-lib.js';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
function args(name: string): string[] {
  const out: string[] = [];
  process.argv.forEach((a, i) => { if (a === `--${name}` && i + 1 < process.argv.length) out.push(process.argv[i + 1]); });
  return out;
}

const levels = arg('levels', 'l2,l3,l4').split(',');
const nDefault = Number(arg('n', '200'));
const mults = arg('mults', '8,16').split(',').map(Number).sort((a, b) => a - b);
const seed = Number(arg('seed', '1'));
const jsonOut = arg('json', '');
const dry = process.argv.includes('--dry'); // sample the boards and print their make-up, run nothing
// Only boards whose primary target fills at least this many cells (--nontrivial: at least 1).
const minFill = process.argv.includes('--nontrivial') ? Math.max(1, Number(arg('min-fill', '1'))) : Number(arg('min-fill', '0'));
const verbose = process.argv.includes('--verbose'); // every miss, region by region
const gl = process.argv.includes('--gl');
const weightArgs = args('weights').length ? args('weights') : ['fb-r6816nt=web/nca-weights.json'];
const weightJson = weightArgs.map((s) => {
  const [name, path] = s.includes('=') ? s.split('=') : [s, s];
  return { name, json: JSON.parse(readFileSync(path, 'utf8')) as unknown };
});
const counts = Object.fromEntries(levels.map((l) => [l, Number(arg(`n-${l}`, l === 'l4' ? '24' : String(nDefault)))]));

const data = JSON.parse(readFileSync('web/strand-data.json', 'utf8')) as StrandData;
const table = new RuleTable(data);

/** HexNCA as the probe's runner. */
function cpuRunner(nca: HexNCA): Runner {
  return {
    load(walls) { for (const s of nca.cells) nca.setWall(s, walls[s] as 0 | 1); nca.reset(); },
    get steps() { return nca.steps; },
    advance(n) { nca.step(n); },
    ch1: () => nca.channel(1),
  };
}

let results: Result[] = [];
if (gl && !dry) {
  // The same probe in Chromium: the page gets the options and weights from a script beside it.
  const res = await build({ entryPoints: ['scripts/area-probe-page.ts'], bundle: true, format: 'iife', write: false, target: 'es2020' });
  const dir = mkdtempSync(join(tmpdir(), 'area-probe-'));
  writeFileSync(join(dir, 'probe.js'), `window.PROBE = ${JSON.stringify({ levels, counts, mults, seed, minFill, verbose, weights: weightJson })};`);
  writeFileSync(join(dir, 'index.html'), '<!doctype html><html><body><script src="probe.js"></script><script>' +
    `${res.outputFiles[0].text.replace(/<\/script/g, '<\\/script')}</script></body></html>`);
  const dom = execFileSync(arg('chromium', process.env.CHROMIUM ?? 'chromium'),
    ['--headless=new', '--no-sandbox', '--dump-dom', '--enable-unsafe-swiftshader', `file://${join(dir, 'index.html')}`],
    { encoding: 'utf8', maxBuffer: 1 << 28, timeout: 6 * 3600_000, stdio: ['ignore', 'pipe', 'ignore'] });
  const m = /RESULT (\{.*\})/.exec(dom);
  if (!m) throw new Error(`no result in the page:\n${dom.slice(0, 2000)}`);
  const out = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')) as
    { error?: string; renderer?: string; lines: string[]; results: Result[] };
  if (out.error) throw new Error(out.error);
  console.log(`GL: ${out.renderer}`);
  for (const line of out.lines) console.log(line);
  results = out.results;
} else {
  for (const level of levels) {
    const board = new Board(data.boards[level]);
    const frame = areaFrame(board);
    const samples = sampleBoards(table, board, frame, counts[level], rng(levelSeed(seed, level, minFill)), minFill);
    console.log(`\n${describeSamples(level, board, frame, samples)}`);
    if (dry) continue;
    for (const { name, json } of weightJson) {
      const res = probe(level, name, board, frame, samples, mults, cpuRunner(new HexNCA(loadWeights(json), frame.R, frame.mask)));
      for (const r of res) for (const line of reportLines(r, verbose)) console.log(line);
      results.push(...res);
    }
  }
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ seed, mults, minFill, gl, results }, null, 1));
