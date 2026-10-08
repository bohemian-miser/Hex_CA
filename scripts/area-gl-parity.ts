// B4 of docs/spectacle-ca-hybrid.md (§4.3): glArea (src/game/area-gl.ts) against cpuArea in a real browser — the same
// strand lines into both, channel 1 compared every 8 steps to 16 R, the fills and territory at the end — and the
// GL layer's ms per step at levels 2-4. Bundles scripts/area-gl-page.ts into a page and runs it in headless Chromium
// (no dependency: --dump-dom carries the result out).
//
//   npx tsx scripts/area-gl-parity.ts [--chromium /usr/bin/chromium] [--swiftshader] [--l3 3]
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const argv = process.argv;
const opt = (name: string, def: string) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : def; };
const exe = opt('chromium', process.env.CHROMIUM ?? 'chromium');
const res = await build({ entryPoints: ['scripts/area-gl-page.ts'], bundle: true, format: 'iife', write: false, target: 'es2020' });
const dir = mkdtempSync(join(tmpdir(), 'area-gl-'));
const page = join(dir, 'index.html');
writeFileSync(page, `<!doctype html><html><body><script>${res.outputFiles[0].text.replace(/<\/script/g, '<\\/script')}</script></body></html>`);
const flags = ['--headless=new', '--no-sandbox', '--dump-dom', '--enable-unsafe-swiftshader'];
if (argv.includes('--swiftshader')) flags.push('--use-angle=swiftshader');
const dom = execFileSync(exe, [...flags, `file://${page}?l3=${opt('l3', '3')}`], { encoding: 'utf8', maxBuffer: 1 << 26, timeout: 1_800_000, stdio: ['ignore', 'pipe', 'ignore'] });
const m = /RESULT (\{.*\})/.exec(dom);
if (!m) throw new Error(`no result in the page:\n${dom.slice(0, 2000)}`);
const out = JSON.parse(m[1].replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"'));
if (out.error) throw new Error(out.error);
console.log(`renderer: ${out.renderer}, MAX_DRAW_BUFFERS ${out.maxDrawBuffers}; glUsable ${out.picker.usable}, ` +
  `bestArea is GL: auto ${out.picker.auto}, force ${out.picker.force}, off ${out.picker.off}`);
console.log('parity (cpuArea vs glArea): board, owners, walls, steps, max |Δ| channel 1 / all channels, fill cells differing (of filled), territory differing');
for (const p of out.parity) console.log(`  ${p.board} #${p.n}: ${p.owners} owners, walls ${p.walls.join('/')}, ${p.steps} steps: ${p.maxCh1} / ${p.maxAll}, fill ${p.fillDiff} (${p.filled}), territory ${p.terrDiff}`);
console.log('the settle rule on both (status@flood steps since the walls changed, CPU / GL):');
for (const r of out.settle) console.log(`  ${r.board} #${r.n}: p1 ${r.p1}; p2 ${r.p2}${r.same1 && r.same2 ? '' : '  <- DIFFERENT'}`);
console.log('ms per step:');
for (const t of out.timing) console.log(`  ${t.board} (${t.cells} cells), ${t.owners} owner${t.owners > 1 ? 's' : ''}: GL ${t.glMsPerStep}${t.cpuMsPerStep !== undefined ? `, CPU ${t.cpuMsPerStep}` : ''}`);
