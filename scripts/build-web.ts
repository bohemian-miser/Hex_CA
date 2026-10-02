// Bundles web/main.ts into web/page.html, written twice:
//   dist/hex-ca.html  the page body alone (the artifact host wraps it in its own document)
//   dist/index.html   a whole document, for GitHub Pages
import { build } from 'esbuild';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
  '<path d="M16 2l12 7v14l-12 7-12-7V9z" fill="#0f7a66"/>' +
  '<path d="M16 9l6 3.5v7L16 23l-6-3.5v-7z" fill="#3fb59a"/></svg>';

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

// page.html is <title>, links and <style> (the head), then the body.
const cut = page.indexOf('</style>') + '</style>'.length;
const doc = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="One local cellular-automaton rule on a hexagon field: the inside of every drawn loop fills, and so does the smaller side of every line from edge to edge.">
<link rel="icon" href="data:image/svg+xml,${encodeURIComponent(ICON)}">
${page.slice(0, cut)}
</head>
<body>
${page.slice(cut)}
</body>
</html>
`;
writeFileSync('dist/index.html', doc);

for (const f of ['dist/hex-ca.html', 'dist/index.html']) {
  console.log(`${f} ${(readFileSync(f).length / 1024).toFixed(1)} KB`);
}
