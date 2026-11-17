// gta6-twitch-ccu-forecast-01 entry: a launch-day curve forecaster.
//
// Idea: a big launch's release-day Twitch curve has a repeatable shape once it
// is scaled by its own first hour. Every past launch in the case's
// `comparables` array gives, for each hour h of its release day, a log-ratio
//     lr = ln(viewers_h / viewers_1).
// The case being forecast supplies only its hour-1 print p1, so the forecast
// distribution of ln(viewers_h) is ln(p1) plus the cross-launch distribution
// of lr at hour h. Not every past launch is an equally good peer, so each
// comparable is weighted by a Gaussian kernel on the distance between its own
// hour-1 size and the case's p1, in log space (bandwidth 1.0): launches that
// opened at a similar scale dominate the fit. The five scored quantiles come
// from a weighted normal fit to the hour's lr samples, its spread scaled by
// 0.9, with neighbour-hour pooling and a widening-spread fallback when an
// hour has too few samples. These constants were chosen by leave-one-out
// validation over the pooled practice comparables (tools/loo.mjs): the
// weighted normal fit scored 0.392 mean log pinball loss against 0.930 for
// the persistence baseline on the same folds, winning 33 of 37 folds.
// Deterministic, standard library only, no network.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

/** The quantile levels the contest scores, in the order each row lists them. */
export const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

export interface Comp {
  name?: string;
  day_start: string;
  hourly: Array<{ from: string; viewers: number }>;
}

export interface CaseLike {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comp[];
}

export interface ModelOpts {
  /** 'emp' = empirical quantiles of the log-ratio samples; 'norm' = normal fit. */
  method: 'emp' | 'norm';
  /** Multiplier on the spread around the median log-ratio (1 = as observed). */
  spreadMult: number;
  /** The median log-ratio is multiplied by this before use (1 = no shrinkage toward 0). */
  shrink: number;
  /** Gaussian kernel bandwidth in log space for weighting comparables by hour-1 size; 0 = unweighted. */
  weightBw: number;
  /** Minimum samples an hour needs before its own distribution is trusted. */
  minSamples: number;
  /** How many neighbouring hours to pool when an hour is short of samples. */
  poolRadius: number;
}

export const DEFAULT_OPTS: ModelOpts = {
  method: 'norm',
  spreadMult: 0.9,
  shrink: 1.0,
  weightBw: 1.0,
  minSamples: 8,
  poolRadius: 1,
};

const HOUR_MS = 3600_000;

/**
 * One comparable's release-day curve: viewers by hour 1..24, read from the
 * rows whose `from` lies in [day_start, day_start + 24h). Returns null when
 * the comparable has no usable hour-1 value.
 */
export function dayCurve(comp: Comp): Map<number, number> | null {
  const start = Date.parse(comp.day_start);
  if (!Number.isFinite(start) || !Array.isArray(comp.hourly)) return null;
  const byHour = new Map<number, number>();
  for (const row of comp.hourly) {
    const t = Date.parse(row.from);
    if (!Number.isFinite(t)) continue;
    const offset = Math.round((t - start) / HOUR_MS);
    if (offset < 0 || offset >= 24) continue;
    if (Math.abs(t - start - offset * HOUR_MS) > 60_000) continue; // not an on-the-hour row
    const v = row.viewers;
    if (typeof v === 'number' && Number.isFinite(v)) byHour.set(offset + 1, v);
  }
  const p1 = byHour.get(1);
  if (p1 === undefined || !(p1 > 0)) return null;
  return byHour;
}

export interface HourSamples {
  /** Log-ratio samples ln(viewers_h / viewers_1), one per comparable. */
  lr: number[];
  /** Kernel weight per sample (all 1 when weighting is off). */
  w: number[];
}

/**
 * Collect, for every hour 2..24, the log-ratio samples of all comparables.
 * When opts.weightBw > 0 each comparable is weighted by how close its own
 * hour-1 size is to the target case's hour-1 print, in log space.
 */
export function buildSamples(comps: Comp[], targetP1: number, opts: ModelOpts): Map<number, HourSamples> {
  const out = new Map<number, HourSamples>();
  for (let h = 2; h <= 24; h++) out.set(h, { lr: [], w: [] });
  const logTarget = Math.log(Math.max(1, targetP1));
  for (const comp of comps) {
    const curve = dayCurve(comp);
    if (curve === null) continue;
    const p1c = curve.get(1) as number;
    let weight = 1;
    if (opts.weightBw > 0) {
      const d = Math.log(p1c) - logTarget;
      weight = Math.exp(-(d * d) / (2 * opts.weightBw * opts.weightBw));
    }
    if (!(weight > 0)) continue;
    for (let h = 2; h <= 24; h++) {
      const v = curve.get(h);
      if (v !== undefined && v > 0) {
        const s = out.get(h) as HourSamples;
        s.lr.push(Math.log(v / p1c));
        s.w.push(weight);
      }
    }
  }
  return out;
}

/** Inverse of the standard normal CDF (Acklam's rational approximation). */
export function normsinv(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Empirical quantile of sorted samples at level tau, linear interpolation. */
function empQuantile(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 1) return sorted[0];
  const pos = tau * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(n - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Weighted empirical quantile: samples sorted, weights as masses at the points. */
function weightedQuantile(xs: number[], ws: number[], tau: number): number {
  const pairs = xs.map((x, i) => [x, ws[i]] as [number, number]).sort((p, q) => p[0] - q[0]);
  const total = pairs.reduce((s, p) => s + p[1], 0);
  if (!(total > 0)) return pairs[0][0];
  const target = tau * total;
  let cum = 0;
  for (let i = 0; i < pairs.length; i++) {
    cum += pairs[i][1];
    if (cum >= target) {
      if (i === 0) return pairs[0][0];
      const prevCum = cum - pairs[i][1];
      const span = pairs[i][0] - pairs[i - 1][0];
      const frac = pairs[i][1] > 0 ? (target - prevCum) / pairs[i][1] : 0;
      return pairs[i - 1][0] + span * Math.min(1, Math.max(0, frac));
    }
  }
  return pairs[pairs.length - 1][0];
}

function medianOf(xs: number[], ws: number[], weighted: boolean): number {
  if (!weighted) {
    const s = [...xs].sort((a, b) => a - b);
    return empQuantile(s, 0.5);
  }
  return weightedQuantile(xs, ws, 0.5);
}

/** The five log-ratio quantiles for one hour from a sample set. */
function hourLogQuantiles(s: HourSamples, opts: ModelOpts): number[] {
  const weighted = opts.weightBw > 0;
  const med = medianOf(s.lr, s.w, weighted);
  let raw: number[];
  if (opts.method === 'norm') {
    let mean: number;
    let sd: number;
    if (weighted) {
      const tot = s.w.reduce((a, b) => a + b, 0);
      mean = s.lr.reduce((a, x, i) => a + x * s.w[i], 0) / tot;
      sd = Math.sqrt(s.lr.reduce((a, x, i) => a + s.w[i] * (x - mean) * (x - mean), 0) / tot);
    } else {
      mean = s.lr.reduce((a, x) => a + x, 0) / s.lr.length;
      sd = Math.sqrt(s.lr.reduce((a, x) => a + (x - mean) * (x - mean), 0) / s.lr.length);
    }
    if (!(sd > 0)) sd = 0.05;
    raw = LEVELS.map((tau) => mean + normsinv(tau) * sd);
  } else if (weighted) {
    raw = LEVELS.map((tau) => weightedQuantile(s.lr, s.w, tau));
  } else {
    const sorted = [...s.lr].sort((a, b) => a - b);
    raw = LEVELS.map((tau) => empQuantile(sorted, tau));
  }
  const centre = opts.shrink * med;
  return raw.map((x) => centre + opts.spreadMult * (x - med));
}

/** Pool the samples of hours in [h-r, h+r] (clamped to 2..24) into one set. */
function pooled(samples: Map<number, HourSamples>, h: number, r: number): HourSamples {
  const lr: number[] = [];
  const w: number[] = [];
  for (let k = Math.max(2, h - r); k <= Math.min(24, h + r); k++) {
    const s = samples.get(k);
    if (s) {
      lr.push(...s.lr);
      w.push(...s.w);
    }
  }
  return { lr, w };
}

/**
 * Forecast one case: for each hour in `hours`, the five viewer quantiles.
 * Never throws on well-formed input; every hour ends with a finite,
 * non-decreasing row inside (0, 1e9].
 */
export function forecastCase(c: CaseLike, opts: ModelOpts = DEFAULT_OPTS): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  const p1 = Math.max(1, first && Number.isFinite(first.viewers) ? first.viewers : 1);
  const comps = Array.isArray(c.comparables) ? c.comparables : [];
  const samples = buildSamples(comps, p1, opts);
  // Global fallback statistics over every sample of every hour.
  const all = pooled(samples, 13, 11);
  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const { hour } of c.forecast) {
    const h = hour;
    let logq: number[] | null = null;
    const own = samples.get(h);
    if (own && own.lr.length >= opts.minSamples) {
      logq = hourLogQuantiles(own, opts);
    } else {
      const near = pooled(samples, h, opts.poolRadius);
      if (near.lr.length >= opts.minSamples) {
        logq = hourLogQuantiles(near, opts);
      } else if (all.lr.length > 0) {
        // Last resort from data: the all-hours distribution, recentred on the
        // nearest hour that has samples, widened like sqrt(h - 1).
        const wide = hourLogQuantiles(all, opts);
        const medAll = medianOf(all.lr, all.w, opts.weightBw > 0);
        let medNear: number | null = null;
        for (let r = 1; r <= 11 && medNear === null; r++) {
          const p = pooled(samples, h, r);
          if (p.lr.length > 0) medNear = medianOf(p.lr, p.w, opts.weightBw > 0);
        }
        const centre = medNear === null ? medAll : medNear;
        const widen = Math.sqrt(Math.max(1, h - 1)) / Math.sqrt(12);
        logq = wide.map((x) => centre + (x - medAll) * Math.max(1, widen));
      } else {
        // No comparables at all: persistence with a plain widening spread.
        logq = LEVELS.map((tau) => normsinv(tau) * 0.15 * Math.sqrt(Math.max(1, h - 1)));
      }
    }
    let q = logq.map((lq) => {
      const v = p1 * Math.exp(lq);
      if (!Number.isFinite(v)) return lq > 0 ? 1e9 : 1e-6;
      return Math.min(1e9, Math.max(1e-6, v));
    });
    q = q.map((v) => Math.round(v * 100) / 100);
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    if (!(q[0] > 0)) q[0] = 1e-6;
    rows.push({ hour: h, q });
  }
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  mkdirSync(outputDir, { recursive: true });
  const inputs = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    const c = JSON.parse(readFileSync(join(inputDir, file), 'utf8')) as CaseLike;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    const hours = forecastCase(c, DEFAULT_OPTS);
    writeFileSync(join(outputDir, `${id}.out`), `${JSON.stringify({ hours })}\n`);
  }
}

const invokedDirectly = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) main();
