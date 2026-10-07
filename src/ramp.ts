// Colour ramp math shared by the fill NCA and strand NCA pages' channel views (web/nca.ts, web/strand.ts):
// a quantised fill ramp (0…1, towards an accent) and a quantised diverging ramp (a clamp's lo…hi, towards
// one of two accents either side of 0), so a frame fills one Path2D per colour, not one per cell. Pure, no DOM.

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
