// Colour ramp math shared by the fill NCA and strand NCA pages' channel views (web/nca.ts, web/strand.ts):
// a quantised fill ramp (0…1, towards an accent) and a quantised diverging ramp (a clamp's lo…hi, towards
// one of two accents either side of 0), so a frame fills one Path2D per colour, not one per cell. Also the
// "superimposed" view's maths: a quick PCA basis over a set of channels (fitPca3, pcaColour) and an additive
// hue blend (hueBlend) — both pages use these to turn many hidden channels into one picture. Pure, no DOM.

/** Ramps are quantised to this many steps each way: a frame fills one path per colour, not one per cell. */
export const LEVELS = 32;

/** A #rrggbb colour as [r, g, b] (grey when it is not one — a CSS colour the regex doesn't cover). */
export function rgbOf(hex: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function mixRgb(a: string, b: string, t: number): [number, number, number] {
  const pa = rgbOf(a);
  const pb = rgbOf(b);
  return [0, 1, 2].map((k) => Math.round(pa[k] + (pb[k] - pa[k]) * t)) as [number, number, number];
}

export const css3 = ([r, g, b]: readonly number[]): string => `rgb(${r} ${g} ${b})`;

/** [r, g, b] as one ImageData pixel (little-endian RGBA, alpha opaque). */
export const pixel = ([r, g, b]: readonly number[]): number => ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;

/** Fill intensity: a value clamped to 0…1, quantised to 0…LEVELS. */
export function fillLevel(v: number): number {
  return Math.round(Math.max(0, Math.min(1, v)) * LEVELS);
}

/** A value on the diverging map (lo…hi around 0) as a level in −LEVELS…LEVELS (negative: towards the "neg"
 * colour, positive: towards "pos"). */
export function divLevel(v: number, lo: number, hi: number): number {
  const t = v >= 0 ? (hi > 0 ? Math.min(1, v / hi) : 0) : lo < 0 ? -Math.min(1, v / lo) : 0;
  return Math.round(t * LEVELS);
}

export interface Ramps {
  /** fill[k], k in 0…LEVELS. */
  fill: [number, number, number][];
  /** div[k + LEVELS], k in −LEVELS…LEVELS. */
  div: [number, number, number][];
}

/** The colours of fill levels 0…LEVELS (cell → flood) and diverging levels −LEVELS…LEVELS (cell → neg or
 * cell → pos), as RGB triples. */
export function ramps(cell: string, flood: string, neg: string, pos: string): Ramps {
  const fill = Array.from({ length: LEVELS + 1 }, (_, k) => mixRgb(cell, flood, k / LEVELS));
  const div = Array.from({ length: 2 * LEVELS + 1 }, (_, k) => {
    const t = (k - LEVELS) / LEVELS;
    return mixRgb(cell, t < 0 ? neg : pos, Math.abs(t));
  });
  return { fill, div };
}

// ── Superimposed view: many channels, one picture ───────────────────────────
// Two false-colour ways to fold a set of state channels into one RGB value per cell, for the "all hidden
// channels at once" view on both pages: a PCA basis (the default: fitPca3 + pcaColour) and an additive hue
// blend (hueBlend, one fixed hue per channel, weighted by its value). Neither knows about cells, boards or
// models — the caller hands in a value accessor and which channels (by a local 0…n-1 index) to use.

/** An HSL colour (h in degrees, s/l in 0…100) as [r, g, b], 0…255. */
export function hslRgb(h: number, s: number, l: number): [number, number, number] {
  const hh = ((h % 360) + 360) % 360;
  const S = s / 100;
  const L = l / 100;
  const c = (1 - Math.abs(2 * L - 1)) * S;
  const x = c * (1 - Math.abs(((hh / 60) % 2) - 1));
  const m = L - c / 2;
  const [r1, g1, b1] = hh < 60 ? [c, x, 0] : hh < 120 ? [x, c, 0] : hh < 180 ? [0, c, x]
    : hh < 240 ? [0, x, c] : hh < 300 ? [x, 0, c] : [c, 0, x];
  return [Math.round((r1 + m) * 255), Math.round((g1 + m) * 255), Math.round((b1 + m) * 255)];
}

/** A value-weighted average of per-channel hues, lifted back up (an average of many hues greys out) and laid
 * over `bg` by the strongest weight — so a cell where every channel is near zero stays the board's own colour.
 * Each weight should already be 0…1 (e.g. a channel's |value| over its clamp). The same blend web/main.ts's
 * hand-written-CA "spectrum" view uses. */
export function hueBlend(
  weights: readonly number[],
  hues: readonly [number, number, number][],
  bg: readonly [number, number, number],
  lift = 1.8,
): [number, number, number] {
  let sum = 0, top = 0, r = 0, g = 0, b = 0;
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (w <= 0) continue;
    sum += w;
    if (w > top) top = w;
    r += w * hues[i][0];
    g += w * hues[i][1];
    b += w * hues[i][2];
  }
  if (sum === 0) return [bg[0], bg[1], bg[2]];
  r /= sum; g /= sum; b /= sum;
  const mid = (r + g + b) / 3;
  const liftC = (x: number) => mid + (x - mid) * lift;
  const clip = (x: number) => Math.max(0, Math.min(255, Math.round(x)));
  return [
    clip(bg[0] + (liftC(r) - bg[0]) * top),
    clip(bg[1] + (liftC(g) - bg[1]) * top),
    clip(bg[2] + (liftC(b) - bg[2]) * top),
  ];
}

/** A quick PCA basis over a set of channels: up to 3 unit vectors (fewer once the data runs out of variance),
 * the mean they are centred on, and each one's projected range over the cells it was fit on (for scaling a
 * projection to 0…1). `fitPca3`'s channels are a local 0…n-1 index — the caller maps that to its own channel
 * numbering through `value`. */
export interface Pca3 {
  mean: Float64Array;
  /** Up to 3 unit vectors, each as long as `mean`; index 0 drives red, 1 green, 2 blue (pcaColour). */
  vecs: Float64Array[];
  /** vecs[k]'s projected [min, max] over the fitted cells. */
  ranges: [number, number][];
}

const PCA_ITERS = 40;

function matVec(cov: Float64Array, n: number, v: Float64Array, out: Float64Array): void {
  for (let a = 0; a < n; a++) {
    let s = 0;
    const row = a * n;
    for (let b = 0; b < n; b++) s += cov[row + b] * v[b];
    out[a] = s;
  }
}

function norm2(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** The top `k` eigenvectors of a symmetric n×n matrix, by power iteration with deflation — plenty for a false-
 * colour basis recomputed every so often, not meant for serious numerics. `seed[c]`, if given, both starts
 * iteration near the previous basis's c'th vector and fixes the converged one's sign to agree with it (power
 * iteration only finds a vector up to sign), so refitting doesn't flip the colours that vector drives. */
function topEigenvectors(cov0: Float64Array, n: number, k: number, seed?: readonly Float64Array[]): Float64Array[] {
  const cov = Float64Array.from(cov0);
  const vecs: Float64Array[] = [];
  const tmp = new Float64Array(n);
  for (let c = 0; c < k; c++) {
    const v = new Float64Array(n);
    const sv = seed?.[c];
    if (sv && norm2(sv) > 1e-9) v.set(sv);
    else for (let i = 0; i < n; i++) v[i] = Math.sin(1 + i * 12.9898 + c * 78.233);
    let len = norm2(v);
    if (len < 1e-12) break;
    for (let i = 0; i < n; i++) v[i] /= len;
    for (let it = 0; it < PCA_ITERS; it++) {
      matVec(cov, n, v, tmp);
      len = norm2(tmp);
      if (len < 1e-12) break;
      for (let i = 0; i < n; i++) v[i] = tmp[i] / len;
    }
    matVec(cov, n, v, tmp);
    let eigenvalue = 0;
    for (let i = 0; i < n; i++) eigenvalue += v[i] * tmp[i];
    if (eigenvalue <= 1e-9) break; // no variance left worth a colour
    if (sv) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += v[i] * sv[i];
      if (dot < 0) for (let i = 0; i < n; i++) v[i] = -v[i];
    }
    vecs.push(v);
    for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) cov[a * n + b] -= eigenvalue * v[a] * v[b];
  }
  return vecs;
}

/**
 * Fits a top-3 PCA basis of `nChannels` local channels over `cells`: the mean, up to 3 directions of the most
 * variance, and each one's range (for pcaColour's 0…1 scaling). `value(cell, localChannel)` reads one channel
 * of one cell; `prev`, the last basis (if there is one), orients the new vectors to agree with it so a refit
 * doesn't flip the picture's colours. O(cells × channels²): fit this now and then (e.g. every N steps, or on
 * demand), not every frame — projecting a fitted basis (pcaColour) is cheap enough for every frame.
 */
export function fitPca3(
  value: (cell: number, localChannel: number) => number,
  nChannels: number,
  cells: ArrayLike<number>,
  prev?: Pca3,
): Pca3 {
  const n = nChannels;
  const mean = new Float64Array(n);
  const count = Math.max(1, cells.length);
  for (let k = 0; k < cells.length; k++) {
    const i = cells[k];
    for (let c = 0; c < n; c++) mean[c] += value(i, c);
  }
  for (let c = 0; c < n; c++) mean[c] /= count;
  const cov = new Float64Array(n * n);
  const row = new Float64Array(n);
  for (let k = 0; k < cells.length; k++) {
    const i = cells[k];
    for (let c = 0; c < n; c++) row[c] = value(i, c) - mean[c];
    for (let a = 0; a < n; a++) {
      const ra = row[a];
      if (ra === 0) continue;
      for (let b = a; b < n; b++) cov[a * n + b] += ra * row[b];
    }
  }
  for (let a = 0; a < n; a++) {
    for (let b = a; b < n; b++) {
      cov[a * n + b] /= count;
      cov[b * n + a] = cov[a * n + b];
    }
  }
  const vecs = topEigenvectors(cov, n, 3, prev?.vecs);
  const ranges: [number, number][] = vecs.map(() => [Infinity, -Infinity]);
  for (let k = 0; k < cells.length; k++) {
    const i = cells[k];
    for (let c = 0; c < n; c++) row[c] = value(i, c) - mean[c];
    for (let vi = 0; vi < vecs.length; vi++) {
      const vec = vecs[vi];
      let s = 0;
      for (let c = 0; c < n; c++) s += vec[c] * row[c];
      if (s < ranges[vi][0]) ranges[vi][0] = s;
      if (s > ranges[vi][1]) ranges[vi][1] = s;
    }
  }
  return { mean, vecs, ranges };
}

/** One cell's `row` (its selected channels' raw values, same order fitPca3 was given) projected onto `basis`
 * and scaled to RGB, 0…255: vecs[0] → red, vecs[1] → green, vecs[2] → blue. A component the fit never found
 * (not enough variance) stays mid-grey (128). */
export function pcaColour(basis: Pca3, row: ArrayLike<number>): [number, number, number] {
  const out: [number, number, number] = [128, 128, 128];
  for (let k = 0; k < 3; k++) {
    const vec = basis.vecs[k];
    if (!vec) continue;
    let s = 0;
    for (let c = 0; c < vec.length; c++) s += vec[c] * (row[c] - basis.mean[c]);
    const [lo, hi] = basis.ranges[k];
    const t = hi > lo ? Math.max(0, Math.min(1, (s - lo) / (hi - lo))) : 0.5;
    out[k] = Math.round(t * 255);
  }
  return out;
}
