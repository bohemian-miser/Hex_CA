// The area layer on WebGL2 (docs/spectacle-ca-hybrid.md §4.3): the same flood as src/nca.ts's HexNCA, one fragment
// per cell, for boards where scalar JS is too slow (level 4: ~250 ms a step a player on the CPU).
//
// Per player, the C state channels live in G = C/4 RGBA32F textures (S×S, the area frame of src/game/area.ts),
// ping-ponged; one fragment shader gathers the 7 taps, runs the H hidden units (ReLU) and the residual add, clamps,
// and writes all G textures at once through multiple render targets (in two or more passes, recomputing the hidden
// layer, where MAX_DRAW_BUFFERS < G). w1 is a (7·G + 3) × H float texture: the 7·G state features, the mask's 7
// taps in two texels, then b1; w2 is G × H. Channel 0 is read from the wall texture (R8), never from the state, so
// a wall edit counts from the next step exactly as HexNCA.setWall's "channel 0 follows at once" does. Off-board
// fragments write zeros. Reads of channel 1 (fill, territory) go through readPixels on group 0.
//
// Supports what the shipped weights use: perception 'taps', consts ['mask'], fireRate 1. Anything else throws, and
// the caller falls back to cpuArea (`bestArea` does).

import type { Board } from '../strand.js';
import type { NCAWeights } from '../nca.js';
import { cpuArea, FloodLayer, type AreaLayer, type AreaOptions } from './area.js';

/** The 7 taps as (drow, dcol), src/nca.ts's order (kernel index k = (drow + 1)·3 + dcol + 1). */
const TAPS: ReadonlyArray<readonly [number, number]> = [[-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0]];
const T = TAPS.length;

const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

/** The step shader for output groups [from, to). */
function fragSource(G: number, H: number, from: number, to: number): string {
  const L: string[] = [];
  L.push('#version 300 es', 'precision highp float;', 'precision highp int;', 'precision highp sampler2D;');
  for (let g = 0; g < G; g++) L.push(`uniform sampler2D uS${g};`);
  L.push('uniform sampler2D uWall;', 'uniform sampler2D uMask;', 'uniform sampler2D uW1;', 'uniform sampler2D uW2;');
  L.push(`uniform vec4 uB2[${G}];`, 'uniform vec2 uClamp;', 'uniform int uS;');
  for (let g = from; g < to; g++) L.push(`layout(location = ${g - from}) out vec4 o${g};`);
  L.push('float onMask(ivec2 q) {',
    '  if (q.x < 0 || q.y < 0 || q.x >= uS || q.y >= uS) return 0.0;',
    '  return texelFetch(uMask, q, 0).r > 0.5 ? 1.0 : 0.0;',
    '}');
  L.push('void main() {', '  ivec2 p = ivec2(gl_FragCoord.xy);');
  L.push(`  if (onMask(p) < 0.5) { ${Array.from({ length: to - from }, (_, k) => `o${from + k} = vec4(0.0);`).join(' ')} return; }`);
  L.push(`  vec4 x[${T * G}];`, '  float m[8];', '  m[7] = 0.0;');
  TAPS.forEach(([dr, dc], t) => {
    L.push(`  { ivec2 q = p + ivec2(${dc}, ${dr}); float on = onMask(q); m[${t}] = on;`);
    L.push('    if (on > 0.5) {');
    for (let g = 0; g < G; g++) L.push(`      x[${t * G + g}] = texelFetch(uS${g}, q, 0);`);
    L.push(`      x[${t * G}].r = texelFetch(uWall, q, 0).r;`);
    L.push(`    } else { ${Array.from({ length: G }, (_, g) => `x[${t * G + g}] = vec4(0.0);`).join(' ')} }`);
    L.push('  }');
  });
  L.push('  vec4 mk0 = vec4(m[0], m[1], m[2], m[3]);', '  vec4 mk1 = vec4(m[4], m[5], m[6], m[7]);');
  for (let g = from; g < to; g++) L.push(`  vec4 a${g} = uB2[${g}];`);
  L.push(`  for (int h = 0; h < ${H}; h++) {`,
    `    float z = texelFetch(uW1, ivec2(${T * G + 2}, h), 0).r;`,
    `    for (int f = 0; f < ${T * G}; f++) z += dot(texelFetch(uW1, ivec2(f, h), 0), x[f]);`,
    `    z += dot(texelFetch(uW1, ivec2(${T * G}, h), 0), mk0) + dot(texelFetch(uW1, ivec2(${T * G + 1}, h), 0), mk1);`,
    '    if (z > 0.0) {');
  for (let g = from; g < to; g++) L.push(`      a${g} += texelFetch(uW2, ivec2(${g}, h), 0) * z;`);
  L.push('    }', '  }');
  for (let g = from; g < to; g++) {
    L.push(`  o${g} = clamp(texelFetch(uS${g}, p, 0) + a${g}, uClamp.x, uClamp.y);`);
    if (g === 0) L.push('  o0.r = texelFetch(uWall, p, 0).r;');
  }
  L.push('}');
  return L.join('\n');
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`area-gl: shader: ${gl.getShaderInfoLog(sh)}`);
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: WebGLShader, fsSrc: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`area-gl: link: ${gl.getProgramInfoLog(prog)}`);
  return prog;
}

function floatTex(gl: WebGL2RenderingContext, w: number, h: number, data: Float32Array | null): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function byteTex(gl: WebGL2RenderingContext, S: number, data: Uint8Array): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, S, S, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  return t;
}

interface Pass { prog: WebGLProgram; from: number; to: number; uS: (WebGLUniformLocation | null)[]; uWall: WebGLUniformLocation | null;
  uMask: WebGLUniformLocation | null; uW1: WebGLUniformLocation | null; uW2: WebGLUniformLocation | null;
  uB2: WebGLUniformLocation | null; uClamp: WebGLUniformLocation | null; uSide: WebGLUniformLocation | null }

/** One player's flood: two sets of G state textures, the walls (as uploaded) and a framebuffer per set and pass. */
interface Player {
  tex: WebGLTexture[][]; // [buffer][group]
  fbo: WebGLFramebuffer[][]; // [buffer][pass]
  readFbo: WebGLFramebuffer[]; // [buffer]: group 0 alone, for readPixels
  cur: number;
  wall: Uint8Array; // S², 255 = wall
  wallTex: WebGLTexture;
  dirty: boolean;
  steps: number;
}

/** True when WebGL2 can run the flood here: float render targets, a hardware renderer (software ones, SwiftShader
 * and the like, are slower than cpuArea). Checked on a throwaway canvas. */
export function glUsable(allowSoftware = false): boolean {
  try {
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
    const gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
    if (!gl || !gl.getExtension('EXT_color_buffer_float')) return false;
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    const name = String(info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return allowSoftware || !/swiftshader|llvmpipe|softpipe|software|basic render/i.test(name);
  } catch {
    return false;
  }
}

export class GlArea extends FloodLayer {
  readonly G: number;
  readonly C: number;
  private readonly passes: Pass[];
  private readonly players: Player[];
  private readonly maskTex: WebGLTexture;
  private readonly w1Tex: WebGLTexture;
  private readonly w2Tex: WebGLTexture;
  private readonly b2: Float32Array;
  private readonly clamp: [number, number];
  private readonly vao: WebGLVertexArrayObject;
  private readonly readBuf: Float32Array;

  constructor(readonly gl: WebGL2RenderingContext, board: Board, w: NCAWeights, owners: number, options: Partial<AreaOptions> = {}) {
    super(board, owners, options);
    if (w.perception !== 'taps' || w.consts.length !== 1 || w.consts[0] !== 'mask' || w.fireRate !== 1) {
      throw new Error('area-gl: only taps-perception weights with the mask const and fireRate 1');
    }
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('area-gl: no EXT_color_buffer_float');
    const C = (this.C = w.channels);
    const H = w.hidden;
    const G = (this.G = Math.ceil(C / 4));
    const K = 1;
    const CK = C + K;
    const S = this.frame.S;
    this.clamp = w.clamp ?? [-3.4e38, 3.4e38];

    // w1: per hidden unit, 7·G state texels (tap t, group g → x[t·G + g]), two mask texels (taps 0-3, 4-6), b1.
    const NF = T * G;
    const w1 = new Float32Array((NF + 3) * H * 4);
    for (let h = 0; h < H; h++) {
      const row = h * (NF + 3) * 4;
      TAPS.forEach(([dr, dc], t) => {
        const k = (dr + 1) * 3 + (dc + 1);
        for (let c = 0; c < C; c++) w1[row + (t * G + (c >> 2)) * 4 + (c & 3)] = w.w1[(h * CK + c) * 9 + k];
        w1[row + (NF + (t >> 2)) * 4 + (t & 3)] = w.w1[(h * CK + C) * 9 + k];
      });
      w1[row + (NF + 2) * 4] = w.b1[h];
    }
    const w2 = new Float32Array(G * H * 4);
    for (let h = 0; h < H; h++) for (let o = 0; o < C; o++) w2[(h * G + (o >> 2)) * 4 + (o & 3)] = w.w2[o * H + h];
    this.b2 = new Float32Array(G * 4);
    this.b2.set(w.b2);

    this.w1Tex = floatTex(gl, NF + 3, H, w1);
    this.w2Tex = floatTex(gl, G, H, w2);
    const mask = new Uint8Array(S * S);
    for (let i = 0; i < mask.length; i++) mask[i] = this.frame.mask[i] ? 255 : 0;
    this.maskTex = byteTex(gl, S, mask);

    const maxDB = Math.min(gl.getParameter(gl.MAX_DRAW_BUFFERS) as number, gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) as number);
    const per = Math.min(G, maxDB);
    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    this.passes = [];
    for (let from = 0; from < G; from += per) {
      const to = Math.min(G, from + per);
      const prog = link(gl, vs, fragSource(G, H, from, to));
      const u = (name: string) => gl.getUniformLocation(prog, name);
      this.passes.push({ prog, from, to, uS: Array.from({ length: G }, (_, g) => u(`uS${g}`)), uWall: u('uWall'), uMask: u('uMask'),
        uW1: u('uW1'), uW2: u('uW2'), uB2: u('uB2'), uClamp: u('uClamp'), uSide: u('uS') });
    }
    this.vao = gl.createVertexArray()!;
    this.readBuf = new Float32Array(S * S * 4);

    this.players = [];
    for (let p = 0; p < owners; p++) {
      const tex = [0, 1].map(() => Array.from({ length: G }, () => floatTex(gl, S, S, null)));
      const fbo = tex.map((set) => this.passes.map((pass) => {
        const f = gl.createFramebuffer()!;
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        for (let g = pass.from; g < pass.to; g++) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + g - pass.from, gl.TEXTURE_2D, set[g], 0);
        gl.drawBuffers(Array.from({ length: pass.to - pass.from }, (_, k) => gl.COLOR_ATTACHMENT0 + k));
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('area-gl: framebuffer incomplete');
        return f;
      }));
      const readFbo = tex.map((set) => {
        const f = gl.createFramebuffer()!;
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, set[0], 0);
        return f;
      });
      const wall = new Uint8Array(S * S);
      this.players.push({ tex, fbo, readFbo, cur: 0, wall, wallTex: byteTex(gl, S, wall), dirty: false, steps: 0 });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    for (const pl of this.players) this.clear(pl);
  }

  /** The fresh state: every channel 0 (channel 0 is read from the walls, so it needs nothing). */
  private clear(pl: Player): void {
    const gl = this.gl;
    const zero = new Float32Array(4);
    for (let b = 0; b < 2; b++) {
      for (let k = 0; k < this.passes.length; k++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, pl.fbo[b][k]);
        for (let a = 0; a < this.passes[k].to - this.passes[k].from; a++) gl.clearBufferfv(gl.COLOR, a, zero);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    pl.cur = 0;
    pl.steps = 0;
  }

  protected setWallAt(p: number, s: number, v: 0 | 1): boolean {
    const pl = this.players[p - 1];
    if ((pl.wall[s] ? 1 : 0) === v) return false;
    pl.wall[s] = v ? 255 : 0;
    pl.dirty = true;
    return true;
  }

  protected resetFlood(p: number): void {
    this.clear(this.players[p - 1]);
  }

  private upload(pl: Player): void {
    if (!pl.dirty) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, pl.wallTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.frame.S, this.frame.S, gl.RED, gl.UNSIGNED_BYTE, pl.wall);
    pl.dirty = false;
  }

  protected runFlood(p: number, n: number): void {
    const gl = this.gl;
    const S = this.frame.S;
    const G = this.G;
    gl.viewport(0, 0, S, S);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(this.vao);
    const pl = this.players[p - 1];
    this.upload(pl);
    for (let s = 0; s < n; s++) {
      const src = pl.tex[pl.cur];
      const dst = 1 - pl.cur;
      for (let k = 0; k < this.passes.length; k++) {
        const pass = this.passes[k];
        gl.useProgram(pass.prog);
        gl.bindFramebuffer(gl.FRAMEBUFFER, pl.fbo[dst][k]);
        for (let g = 0; g < G; g++) {
          gl.activeTexture(gl.TEXTURE0 + g);
          gl.bindTexture(gl.TEXTURE_2D, src[g]);
          gl.uniform1i(pass.uS[g], g);
        }
        const units: [WebGLTexture, WebGLUniformLocation | null][] = [[pl.wallTex, pass.uWall], [this.maskTex, pass.uMask], [this.w1Tex, pass.uW1], [this.w2Tex, pass.uW2]];
        units.forEach(([tex, loc], j) => {
          gl.activeTexture(gl.TEXTURE0 + G + j);
          gl.bindTexture(gl.TEXTURE_2D, tex);
          gl.uniform1i(loc, G + j);
        });
        gl.uniform4fv(pass.uB2, this.b2);
        gl.uniform2f(pass.uClamp, this.clamp[0], this.clamp[1]);
        gl.uniform1i(pass.uSide, S);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      pl.cur = dst;
      pl.steps++;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
  }

  /** Steps the owner's flood has taken since its last reset. */
  steps(owner: number): number {
    return this.players[owner - 1].steps;
  }

  /** The owner's channels 0-3 (group 0) as S² RGBA floats; a GPU sync. */
  private readGroup0(owner: number): Float32Array {
    const gl = this.gl;
    const pl = this.players[owner - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, pl.readFbo[pl.cur]);
    gl.readPixels(0, 0, this.frame.S, this.frame.S, gl.RGBA, gl.FLOAT, this.readBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return this.readBuf;
  }

  /** The owner's whole state, channel-major like HexNCA.state (C × S²; channel 0 = the walls), for parity checks. */
  readState(owner: number): Float32Array {
    const gl = this.gl;
    const pl = this.players[owner - 1];
    const S = this.frame.S;
    const N = S * S;
    const out = new Float32Array(this.C * N);
    const buf = new Float32Array(N * 4);
    const f = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    for (let g = 0; g < this.G; g++) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pl.tex[pl.cur][g], 0);
      gl.readPixels(0, 0, S, S, gl.RGBA, gl.FLOAT, buf);
      for (let j = 0; j < 4 && g * 4 + j < this.C; j++) for (let i = 0; i < N; i++) out[(g * 4 + j) * N + i] = buf[i * 4 + j];
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(f);
    for (let i = 0; i < N; i++) out[i] = pl.wall[i] ? 1 : 0;
    return out;
  }

  protected readFill(p: number): Uint8Array {
    const out = new Uint8Array(this.board.n);
    const buf = this.readGroup0(p);
    const { slot } = this.frame;
    for (let i = 0; i < out.length; i++) out[i] = buf[slot[i] * 4 + 1] > 0.5 ? 1 : 0;
    return out;
  }
}

export function glArea(gl: WebGL2RenderingContext, board: Board, weights: NCAWeights, owners: number,
  options: Partial<AreaOptions> = {}): AreaLayer {
  return new GlArea(gl, board, weights, owners, options);
}

/** glArea on a canvas of its own when `glUsable()`, else cpuArea (or when the GL layer cannot be built). `gl`:
 * 'auto' (the default), 'force' (software renderers too, for checks) or 'off'. */
export function bestArea(board: Board, weights: NCAWeights, owners: number, gl: 'auto' | 'force' | 'off' = 'auto',
  options: Partial<AreaOptions> = {}): AreaLayer {
  if (gl !== 'off' && glUsable(gl === 'force')) {
    try {
      const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas');
      const ctx = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
      if (ctx) return new GlArea(ctx, board, weights, owners, options);
    } catch {
      // fall through to the CPU
    }
  }
  return cpuArea(board, weights, owners, options);
}
