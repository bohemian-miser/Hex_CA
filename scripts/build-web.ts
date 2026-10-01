// Bundles web/main.ts into web/page.html → dist/hex-ca.html, one self-contained file.
import { build } from 'esbuild';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const out = await build({
  entryPoints: ['web/main.ts'],
  bundle: true,
  format: 'iife',
  minify: true,
  write: false,
  target: 'es2020',
});
const js = out.outputFiles[0].text.replace(/<\/script/g, '<\\/script');
const page = readFileSync('web/page.html', 'utf8').replace('<script>/*APP*/</script>', () => `<script>${js}</script>`);
mkdirSync('dist', { recursive: true });
writeFileSync('dist/hex-ca.html', page);
console.log(`dist/hex-ca.html ${(page.length / 1024).toFixed(1)} KB`);
