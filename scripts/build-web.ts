// Bundles each page's script into its HTML (web/*.html: <title>, links and
// <style> as the head, then the body, then <script>/*APP*/</script>):
//   dist/hex-ca.html  the hand-written CA's page body alone (the artifact host wraps it in its own document)
//   dist/index.html   the hand-written CA as a whole document, for GitHub Pages
//   dist/nca.html     the trained NCA (web/nca.ts, weights bundled in) as a whole document
import { build } from 'esbuild';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
  '<path d="M16 2l12 7v14l-12 7-12-7V9z" fill="#0f7a66"/>' +
  '<path d="M16 9l6 3.5v7L16 23l-6-3.5v-7z" fill="#3fb59a"/></svg>';

/** The page's HTML with its entry point bundled into the script tag. */
async function bundle(html: string, entry: string): Promise<string> {
  const out = await build({
    entryPoints: [entry],
    bundle: true,
    format: 'iife',
    minify: true,
    write: false,
    target: 'es2020',
  });
  const js = out.outputFiles[0].text.replace(/<\/script/g, '<\\/script');
  return readFileSync(html, 'utf8').replace('<script>/*APP*/</script>', () => `<script>${js}</script>`);
}

/** A page as a whole document: its head part (up to the end of <style>) in <head>, the rest in <body>. */
function documentOf(page: string, description: string): string {
  const cut = page.indexOf('</style>') + '</style>'.length;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="${description}">
<link rel="icon" href="data:image/svg+xml,${encodeURIComponent(ICON)}">
${page.slice(0, cut)}
</head>
<body>
${page.slice(cut)}
</body>
</html>
`;
}

mkdirSync('dist', { recursive: true });
const page = await bundle('web/page.html', 'web/main.ts');
writeFileSync('dist/hex-ca.html', page);
writeFileSync('dist/index.html', documentOf(page,
  'One local cellular-automaton rule on a hexagon field: the inside of every drawn loop fills, and so does the smaller side of every line from edge to edge.'));
const nca = await bundle('web/nca.html', 'web/nca.ts');
writeFileSync('dist/nca.html', documentOf(nca,
  'A neural cellular automaton on a hexagon field, trained to fill every region the drawn walls enclose.'));

for (const f of ['dist/hex-ca.html', 'dist/index.html', 'dist/nca.html']) {
  console.log(`${f} ${(readFileSync(f).length / 1024).toFixed(1)} KB`);
}
