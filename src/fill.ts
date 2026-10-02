// The fill rule (DESIGN.md §2, §3, §6): one uniform, synchronous radius-1
// update over 22 integer channels. The inside of every painted loop fills,
// and so does the smaller side of every edge-to-edge bridge.
//
// In one breath: an EPOCH wave (a max-flood of the host's stamps) makes
// everything computed under an older picture invisible; each OFF region
// names itself by its largest id (LEADER, with a BFS DIST), grows a tree
// towards that leader (RPAR) and counts itself up the tree (CDONE, SUB) and
// back down (RSIZE); RIM and REDGE say whether a wall or a region touches
// the exterior; the host's static tree (gpar) carries the largest edge
// region's size (GMAX) up to the root, whose quiet run (QT, RUN) certifies
// the fixed point and sends COMMIT and M back down; STATE latches WANT only
// while the cell's epoch is the committed one, so no wrong fill is ever
// shown. Every loop over the six taps is a max, min, sum, any, all or "the
// tap whose id is x" — never a tap index — so tap order is irrelevant and
// the code ports to GLSL line for line.

import { CA, ChannelSpec, Rule, TAPS as ENGINE_TAPS, Uniforms } from './engine.js';
import { Field } from './field.js';

/** Picture values (paint) and states. FILLED is a state, never painted. */
export const OFF = 0, ON = 1, FILLED = 2, WALL = 3;

/** Channel indices: positions in `fillRule.channels`, so P[c * TAPS + t] reads channel c at tap t. MAXM is the channel `M`. */
const FIELD = 0, ID = 1, GPAR = 2, AREA = 3, PAINT = 4, STAMP = 5,
  EPOCH = 6, RIM = 7, REDGE = 8, LEADER = 9, DIST = 10, RPAR = 11, CDONE = 12, SUB = 13, RSIZE = 14,
  GMAX = 15, MAXM = 16, QT = 17, RUN = 18, COMMIT = 19, WANT = 20, STATE = 21;

/**
 * `dist` starts (and reads at dead slots) as CAP = N + 1, which only a Field
 * knows; the static spec carries this stand-in and `fillCA` substitutes the
 * field's CAP. It is never read: a dead tap is never free, and a field slot's
 * first step overwrites it.
 */
const DIST_FAR = 0x7fffffff;

const spec = (name: string, kind: ChannelSpec['kind'], bank: number, init = 0, dead = init, watch = false): ChannelSpec =>
  ({ name, kind, bank, init, dead, watch });

/** The table of DESIGN.md §2, in index order. Banks: B0–B3 the GPU's MRT targets, B4 inputs, B5 constants. */
const CHANNELS: readonly ChannelSpec[] = [
  spec('field', 'const', 5),
  spec('id', 'const', 5),
  spec('gpar', 'const', 5),
  spec('area', 'const', 5),
  spec('paint', 'input', 4, OFF, WALL),
  spec('stamp', 'input', 4),
  spec('epoch', 'hidden', 0, 0, 0, true),
  spec('rim', 'hidden', 2, 0, 1, true),
  spec('redge', 'hidden', 2, 0, 0, true),
  spec('leader', 'hidden', 1, 0, 0, true),
  spec('dist', 'hidden', 1, DIST_FAR, DIST_FAR, true),
  spec('rpar', 'hidden', 1, 0, 0, true),
  spec('cdone', 'hidden', 1, 0, 0, true),
  spec('sub', 'hidden', 2, 0, 0, true),
  spec('rsize', 'hidden', 2, 0, 0, true),
  spec('gmax', 'gate', 3),
  spec('M', 'gate', 3),
  spec('qt', 'gate', 3, 0, 1),
  spec('run', 'gate', 3),
  spec('commit', 'gate', 0, -1),
  spec('want', 'out', 0),
  spec('state', 'out', 0, OFF, WALL),
];

/** The watched channels: the closed system whose fixed point the gate certifies. */
const WATCHED = Int32Array.from(CHANNELS.flatMap((s, c) => (s.watch ? [c] : [])));

/**
 * The engine's P stride, as a module constant: V8 folds a local const into
 * `update`'s hot loop but reads an imported binding each time (~10% slower on
 * a boot, 3× under a CommonJS transform).
 */
const TAPS = ENGINE_TAPS;

/** DESIGN.md §3, line for line. P[c * 7 + t]: channel c at tap t (0 = self). */
function update(P: Int32Array, out: Int32Array, u: Uniforms): void {
  const CAP = u.CAP;
  const wallsBound = u.wallsBound === 1;
  const id = P[ID * TAPS];
  const paint = P[PAINT * TAPS];
  const gpar = P[GPAR * TAPS];

  // e = max(epoch, stamp, max_k epoch_k, max_k stamp_k): the newest edit in reach.
  let e = P[EPOCH * TAPS];
  if (P[STAMP * TAPS] > e) e = P[STAMP * TAPS];
  for (let t = 1; t <= 6; t++) {
    if (P[EPOCH * TAPS + t] > e) e = P[EPOCH * TAPS + t];
    if (P[STAMP * TAPS + t] > e) e = P[STAMP * TAPS + t];
  }
  const fresh = e > P[EPOCH * TAPS];

  // Per-tap bits. cur: on this epoch, or dead (a dead tap reads rim = 1);
  // fr: a free (OFF) field cell on this epoch. Taps on an older epoch are invisible.
  let fr = 0;
  let edge = false;
  for (let t = 1; t <= 6; t++) {
    const F = P[FIELD * TAPS + t] === 1;
    if (F && P[EPOCH * TAPS + t] !== e) continue;
    const pk = P[PAINT * TAPS + t];
    if (F && pk === OFF) fr |= 1 << t;
    if (P[RIM * TAPS + t] === 1 || (wallsBound && F && pk === WALL)) edge = true;
  }
  out[EPOCH] = e;
  out[RIM] = paint === WALL && edge ? 1 : 0;

  // The region channels; a blocked cell (ON or WALL) resets them all.
  let redge = 0, leader = 0, dist = CAP, rpar = 0, cdone = 0, sub = 0, rsize = 0;
  if (paint === OFF) {
    redge = edge ? 1 : 0;
    // (leader, dist) = max of {(id, 0)} ∪ {(leader_k, dist_k + 1)}: larger leader, then smaller dist.
    leader = id;
    dist = 0;
    for (let t = 1; t <= 6; t++) {
      if (!((fr >> t) & 1)) continue;
      if (P[REDGE * TAPS + t] === 1) redge = 1;
      const lk = P[LEADER * TAPS + t];
      const dk = P[DIST * TAPS + t] + 1;
      if (lk <= 0 || dk >= CAP) continue;
      if (lk > leader || (lk === leader && dk < dist)) {
        leader = lk;
        dist = dk;
      }
    }
    // rpar: the highest-id free tap one step nearer the leader.
    if (dist > 0) {
      for (let t = 1; t <= 6; t++) {
        if (!((fr >> t) & 1)) continue;
        if (P[LEADER * TAPS + t] === leader && P[DIST * TAPS + t] === dist - 1 && P[ID * TAPS + t] > rpar) rpar = P[ID * TAPS + t];
      }
    }
    // kid(k): a free tap of this leader whose parent is this cell. settled: nothing
    // about the tree moved here, and every free tap agrees on the leader.
    let agree = true;
    let kidsDone = true;
    let sum = P[AREA * TAPS];
    for (let t = 1; t <= 6; t++) {
      if (!((fr >> t) & 1)) continue;
      if (P[LEADER * TAPS + t] !== leader) {
        agree = false;
        continue;
      }
      if (P[RPAR * TAPS + t] === id) {
        if (P[CDONE * TAPS + t] !== 1) kidsDone = false;
        sum += P[SUB * TAPS + t];
      }
    }
    const settled = !fresh && leader === P[LEADER * TAPS] && dist === P[DIST * TAPS] && rpar === P[RPAR * TAPS] && agree;
    if (settled && kidsDone) {
      cdone = 1;
      sub = sum < CAP ? sum : CAP;
    }
    // rsize: the leader's own count, broadcast down the region tree.
    if (dist === 0) rsize = sub;
    else {
      for (let t = 1; t <= 6; t++) {
        if (!((fr >> t) & 1)) continue;
        if (P[ID * TAPS + t] === rpar && P[LEADER * TAPS + t] === leader) rsize = P[RSIZE * TAPS + t];
      }
    }
  }
  out[REDGE] = redge;
  out[LEADER] = leader;
  out[DIST] = dist;
  out[RPAR] = rpar;
  out[CDONE] = cdone;
  out[SUB] = sub;
  out[RSIZE] = rsize;

  // The static tree: the largest edge region in the subtree (lagged, unwatched)
  // and whether the subtree was quiet. skid(k): a field tap whose gpar is this cell.
  let gmax = paint === OFF && redge === 1 ? rsize : 0;
  let kidsQuiet = true;
  for (let t = 1; t <= 6; t++) {
    if (P[FIELD * TAPS + t] !== 1 || P[GPAR * TAPS + t] !== id) continue;
    if (P[GMAX * TAPS + t] > gmax) gmax = P[GMAX * TAPS + t];
    if (P[QT * TAPS + t] !== 1) kidsQuiet = false;
  }
  out[GMAX] = gmax;

  let changed = false;
  for (let i = 0; i < WATCHED.length; i++) {
    const c = WATCHED[i];
    if (out[c] !== P[c * TAPS]) {
      changed = true;
      break;
    }
  }
  const qt = !changed && kidsQuiet;
  out[QT] = qt ? 1 : 0;

  // The gate. The root counts quiet steps and fires at K + 1: the whole tree
  // was quiet at one common step, so the watched system is at its fixed point
  // and the lagged gmax is exactly M. Everyone else copies (commit, M) from
  // its static parent, the tap whose id is gpar.
  let commit = P[COMMIT * TAPS];
  let M = P[MAXM * TAPS];
  if (gpar === 0) {
    const run = qt ? Math.min(P[RUN * TAPS] + 1, u.K + 2) : 0;
    out[RUN] = run;
    if (run === u.K + 1) {
      commit = e;
      M = gmax;
    }
  } else {
    out[RUN] = 0;
    for (let t = 1; t <= 6; t++) {
      if (P[FIELD * TAPS + t] === 1 && P[ID * TAPS + t] === gpar) {
        commit = P[COMMIT * TAPS + t];
        M = P[MAXM * TAPS + t];
      }
    }
  }
  out[COMMIT] = commit;
  out[MAXM] = M;

  const want = paint === OFF && rsize > 0 && (redge === 0 || rsize < M) ? 1 : 0;
  out[WANT] = want;
  out[STATE] = paint === ON ? ON
    : paint === WALL ? WALL
    : commit === e ? (want ? FILLED : OFF)
    : P[STATE * TAPS] === FILLED ? FILLED : OFF;
}

export const fillRule: Rule = { channels: CHANNELS, update };

/** The same rule with `dist`'s init/dead at the field's CAP (DESIGN.md §2). */
function ruleFor(CAP: number): Rule {
  const channels = CHANNELS.map((s) => (s.name === 'dist' ? { ...s, init: CAP, dead: CAP } : s));
  return { channels, update };
}

/** A fill automaton over `field`: constants field/id/gpar/area, uniforms N/CAP/K/wallsBound. */
export function fillCA(field: Field, opts: { wallsBound?: boolean } = {}): CA {
  const { topo, N, CAP, K } = field;
  const area = new Int32Array(topo.size);
  for (const slot of topo.cells) area[slot] = 1;
  const constants = { field: Int32Array.from(topo.isField), id: field.ids, gpar: field.gpar, area };
  return new CA(topo, ruleFor(CAP), { N, CAP, K, wallsBound: opts.wallsBound ? 1 : 0 }, constants);
}

/**
 * Paint `slots` with `v`, stamped with the number of the step that absorbs
 * the edit (generation + 1, DESIGN.md §6's serial contract). Dead slots and
 * slots already painted `v` are skipped.
 */
export function paint(ca: CA, slots: Iterable<number>, v: 0 | 1 | 3): void {
  const stamp = ca.generation + 1;
  const { size, isField } = ca.topo;
  const current = ca.ch[PAINT];
  for (const slot of slots) {
    if (!(slot >= 0 && slot < size) || !isField[slot] || current[slot] === v) continue;
    ca.write('paint', slot, v);
    ca.write('stamp', slot, stamp);
  }
}

/** Re-stamp the root: a new epoch over the same picture (everything recomputes and re-certifies). */
export function bump(ca: CA): void {
  ca.write('stamp', rootOf(ca), ca.generation + 1);
}

/** The field slot with gpar 0. */
function rootOf(ca: CA): number {
  const gpar = ca.ch[GPAR];
  for (const slot of ca.topo.cells) if (gpar[slot] === 0) return slot;
  throw new Error('no root: the gpar constants are missing');
}

/** The drawn state per slot (0 off, 1 on, 2 filled, 3 wall; 3 at dead slots). */
export function stateOf(ca: CA): Uint8Array {
  return Uint8Array.from(ca.ch[STATE]);
}

/**
 * In-range garbage into every hidden, gate and out channel at every field
 * slot (the picture is kept), and every field slot relisted so the garbage
 * gets computed. Serials stay at or below the generation, so the next edit
 * or `bump` outranks them.
 */
export function scramble(ca: CA, rand: () => number): void {
  const { N, CAP, K } = ca.u;
  const g = ca.generation;
  // [channel, max]: a uniform draw from 0..max.
  const ranges: ReadonlyArray<readonly [number, number]> = [
    [EPOCH, g], [RIM, 1], [REDGE, 1], [LEADER, N], [DIST, CAP], [RPAR, N], [CDONE, 1], [SUB, CAP], [RSIZE, CAP],
    [GMAX, CAP], [MAXM, CAP], [QT, 1], [RUN, K + 2], [COMMIT, g], [WANT, 1], [STATE, WALL],
  ];
  const current = ca.ch[PAINT];
  for (const slot of ca.topo.cells) {
    for (const [c, max] of ranges) ca.ch[c][slot] = Math.floor(rand() * (max + 1));
    // A write of the value already there: only to list the slot and its taps.
    ca.write('paint', slot, current[slot]);
  }
}
