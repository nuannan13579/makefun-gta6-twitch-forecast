// Leave-one-out validation for the forecaster in ../main.ts.
//
// Pools every comparable found in the four practice case files (deduped by
// name + day_start). Each usable comparable takes a turn as the "case": its
// hour-1 print is the given, its hours 2..24 are the truth, and the forecast
// is built from all the OTHER comparables with exactly the algorithm main.ts
// runs. Scores are mean pinball loss on the log scale (the contest's rule),
// for this model and for the house baseline formula on the same folds.
//
// Usage: node tools/loo.mjs [practice-input-dir]
// Default input dir: ../../../kits/gta6-twitch-ccu-forecast-01-kit/input
// relative to the repository root. Note: the practice set is four specific
// launches and the comparables are other launches again, so this LOO score
// is an approximation of the final, not a promise.

import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_OPTS, LEVELS, dayCurve, forecastCase } from '../main.ts';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const inputDir = process.argv[2] ?? join(repoRoot, '..', '..', 'kits', 'gta6-twitch-ccu-forecast-01-kit', 'input');

const Z_BASELINE = [-1.645, -0.674, 0, 0.674, 1.645];

function pinball(y, q, tau) {
  const d = Math.log(Math.max(1, y)) - Math.log(q);
  return 2 * Math.max(tau * d, (tau - 1) * d);
}

function scoreRows(rows, truth) {
  let sum = 0;
  let n = 0;
  for (const { hour, q } of rows) {
    const y = truth.get(hour);
    if (y === undefined || !(y > 0)) continue;
    for (let j = 0; j < LEVELS.length; j++) {
      sum += pinball(y, q[j], LEVELS[j]);
      n++;
    }
  }
  return n === 0 ? null : { mean: sum / n, items: n };
}

function baselineRows(p1, hours) {
  return hours.map((h) => ({
    hour: h,
    q: Z_BASELINE.map((z) => Math.round(p1 * Math.exp(z * 0.15 * Math.sqrt(h - 1)) * 100) / 100),
  }));
}

// ---- pool the comparables of all practice cases, deduped ----
const pool = new Map();
for (const file of readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort()) {
  const c = JSON.parse(readFileSync(join(inputDir, file), 'utf8'));
  for (const comp of c.comparables ?? []) {
    const key = `${comp.name}|${comp.day_start}`;
    if (!pool.has(key)) pool.set(key, comp);
  }
}
const comps = [...pool.values()];
console.log(`pooled comparables: ${comps.length}`);

// ---- folds: comparables with a usable release day ----
const folds = [];
for (let i = 0; i < comps.length; i++) {
  const curve = dayCurve(comps[i]);
  if (curve === null) continue;
  let present = 0;
  for (let h = 2; h <= 24; h++) if ((curve.get(h) ?? 0) > 0) present++;
  if (present >= 18) folds.push({ idx: i, curve });
}
console.log(`usable folds: ${folds.length}`);

const hoursList = Array.from({ length: 23 }, (_, i) => i + 2);

function runConfig(opts) {
  let modelSum = 0;
  let baseSum = 0;
  let items = 0;
  for (const fold of folds) {
    const others = comps.filter((_, j) => j !== fold.idx);
    const p1 = fold.curve.get(1);
    const fakeCase = {
      levels: LEVELS,
      known: [{ hour: 1, viewers: p1 }],
      forecast: hoursList.map((hour) => ({ hour })),
      comparables: others,
    };
    const m = scoreRows(forecastCase(fakeCase, opts), fold.curve);
    const b = scoreRows(baselineRows(p1, hoursList), fold.curve);
    if (m === null || b === null) continue;
    modelSum += m.mean * m.items;
    baseSum += b.mean * b.items;
    items += m.items;
  }
  return { model: modelSum / items, baseline: baseSum / items, items };
}

const configs = [
  ['defaults (emp, mult 1, shrink 1, no weight)', { ...DEFAULT_OPTS }],
  ['normal fit', { ...DEFAULT_OPTS, method: 'norm' }],
  ['weight bw 0.8', { ...DEFAULT_OPTS, weightBw: 0.8 }],
  ['weight bw 1.0', { ...DEFAULT_OPTS, weightBw: 1.0 }],
  ['weight bw 1.2', { ...DEFAULT_OPTS, weightBw: 1.2 }],
  ['weight bw 1.5', { ...DEFAULT_OPTS, weightBw: 1.5 }],
  ['weight bw 2.0', { ...DEFAULT_OPTS, weightBw: 2.0 }],
  ['bw 1.0 + spread x0.9', { ...DEFAULT_OPTS, weightBw: 1.0, spreadMult: 0.9 }],
  ['bw 1.0 + spread x1.1', { ...DEFAULT_OPTS, weightBw: 1.0, spreadMult: 1.1 }],
  ['bw 1.0 + shrink 0.95', { ...DEFAULT_OPTS, weightBw: 1.0, shrink: 0.95 }],
  ['bw 1.0 + shrink 0.9', { ...DEFAULT_OPTS, weightBw: 1.0, shrink: 0.9 }],
  ['bw 1.2 + spread x0.9', { ...DEFAULT_OPTS, weightBw: 1.2, spreadMult: 0.9 }],
  ['bw 1.2 + shrink 0.95', { ...DEFAULT_OPTS, weightBw: 1.2, shrink: 0.95 }],
  ['bw 1.0 + norm', { ...DEFAULT_OPTS, weightBw: 1.0, method: 'norm' }],
  ['bw 1.5 + norm', { ...DEFAULT_OPTS, weightBw: 1.5, method: 'norm' }],
  ['bw 0.8 + norm', { ...DEFAULT_OPTS, weightBw: 0.8, method: 'norm' }],
  ['bw 1.2 + norm', { ...DEFAULT_OPTS, weightBw: 1.2, method: 'norm' }],
  ['bw 1.0 + norm + spread x0.9', { ...DEFAULT_OPTS, weightBw: 1.0, method: 'norm', spreadMult: 0.9 }],
  ['bw 1.0 + norm + spread x1.1', { ...DEFAULT_OPTS, weightBw: 1.0, method: 'norm', spreadMult: 1.1 }],
  ['bw 1.0 + norm + shrink 0.95', { ...DEFAULT_OPTS, weightBw: 1.0, method: 'norm', shrink: 0.95 }],
];

let best = null;
for (const [name, opts] of configs) {
  const r = runConfig(opts);
  console.log(`${name.padEnd(44)} model ${r.model.toFixed(6)}  baseline ${r.baseline.toFixed(6)}  items ${r.items}`);
  if (best === null || r.model < best.r.model) best = { name, opts, r };
}
console.log(`\nbest: ${best.name} -> ${best.r.model.toFixed(6)} (baseline ${best.r.baseline.toFixed(6)})`);
console.log(`best opts: ${JSON.stringify(best.opts)}`);

// Per-fold breakdown for the winning config: how often does it beat the
// baseline head-to-head, and where does it lose?
{
  let wins = 0;
  const losses = [];
  for (const fold of folds) {
    const others = comps.filter((_, j) => j !== fold.idx);
    const p1 = fold.curve.get(1);
    const fakeCase = {
      levels: LEVELS,
      known: [{ hour: 1, viewers: p1 }],
      forecast: hoursList.map((hour) => ({ hour })),
      comparables: others,
    };
    const m = scoreRows(forecastCase(fakeCase, best.opts), fold.curve);
    const b = scoreRows(baselineRows(p1, hoursList), fold.curve);
    if (m === null || b === null) continue;
    if (m.mean < b.mean) wins++;
    else losses.push(`${comps[fold.idx].name} model ${m.mean.toFixed(3)} vs base ${b.mean.toFixed(3)}`);
  }
  console.log(`\nper-fold: model beats baseline in ${wins}/${folds.length} folds`);
  for (const line of losses) console.log(`  loses: ${line}`);
}
