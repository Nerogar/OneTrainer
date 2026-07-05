import type { TimestepDistribution } from "@/types/generated/enums";

const NUM_TRAIN_TIMESTEPS = 1000;
const SAMPLE_COUNT = 200_000;

export interface TimestepDistParams {
  distribution: TimestepDistribution;
  minNoisingStrength: number;
  maxNoisingStrength: number;
  noisingWeight: number;
  noisingBias: number;
  timestepShift: number;
}

/** Box-Muller transform: one sample from N(mean, std). */
function normalSample(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std + mean;
}

/**
 * Marsaglia-Tsang sample from Gamma(shape, scale=1), shape > 0.
 * Used to build a general Beta(alpha, beta) sampler via X/(X+Y) with X~Gamma(alpha), Y~Gamma(beta),
 * mirroring what `torch.distributions.Beta` does internally.
 */
function gammaSample(shape: number): number {
  if (shape < 1) {
    // Boost by 1 and correct with a Uniform(0,1)^(1/shape) factor.
    const u = Math.random();
    return gammaSample(1 + shape) * u ** (1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  for (;;) {
    let x: number;
    let v: number;
    do {
      x = normalSample(0, 1);
      v = (1 + c * x) ** 3;
    } while (v <= 0);

    const u = Math.random();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x ** 2 + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** One sample from Beta(alpha, beta) via the Gamma-ratio construction. */
function betaSample(alpha: number, beta: number): number {
  const x = gammaSample(alpha);
  const y = gammaSample(beta);
  return x / (x + y);
}

/**
 * Compute a 1000-bin histogram of the timestep distribution.
 * Faithful port of ModelSetupNoiseMixin._get_timestep_discrete.
 *
 * Continuous distributions (UNIFORM, LOGIT_NORMAL, HEAVY_TAIL, BETA) are Monte-Carlo
 * sampled. Discrete distributions (COS_MAP, SIGMOID, INVERTED_PARABOLA) return
 * exact probability weights scaled to match the sample count.
 */
export function computeTimestepHistogram(params: TimestepDistParams): Float64Array {
  const {
    distribution,
    minNoisingStrength,
    maxNoisingStrength,
    noisingWeight,
    noisingBias,
    timestepShift: shift,
  } = params;

  const N = NUM_TRAIN_TIMESTEPS;
  const minT = Math.floor(N * minNoisingStrength);
  const maxT = Math.floor(N * maxNoisingStrength);
  const numT = maxT - minT;
  const bins = new Float64Array(N);

  if (numT <= 0) return bins;

  if (
    distribution === "UNIFORM" ||
    distribution === "LOGIT_NORMAL" ||
    distribution === "HEAVY_TAIL" ||
    distribution === "BETA"
  ) {
    // BETA (B-TTDM): Noising Weight -> alpha, Noising Bias -> beta (matches Python's mapping).
    const betaAlpha = Math.max(1e-4, noisingWeight);
    const betaBeta = Math.max(1e-4, noisingBias);

    for (let i = 0; i < SAMPLE_COUNT; i++) {
      let t: number;

      if (distribution === "UNIFORM") {
        t = minT + (maxT - minT) * Math.random();
      } else if (distribution === "LOGIT_NORMAL") {
        const bias = noisingBias;
        const scale = noisingWeight + 1.0;
        const logitNormal = 1 / (1 + Math.exp(-normalSample(bias, scale)));
        t = logitNormal * numT + minT;
      } else if (distribution === "HEAVY_TAIL") {
        const scale = noisingWeight;
        let u = Math.random();
        u = 1.0 - u - scale * (Math.cos((Math.PI / 2.0) * u) ** 2 - 1.0 + u);
        t = u * numT + minT;
      } else {
        // BETA. Mirrors the three branches in ModelSetupNoiseMixin._get_timestep_discrete:
        // beta==1 and alpha==1 use closed-form inverse-CDF sampling; the general case
        // falls back to a full Beta(alpha, beta) sample (Python uses torch.distributions.Beta).
        let u: number;
        if (Math.abs(betaBeta - 1.0) < 1e-5) {
          u = Math.random() ** (1.0 / betaAlpha);
        } else if (Math.abs(betaAlpha - 1.0) < 1e-5) {
          u = 1.0 - Math.random() ** (1.0 / betaBeta);
        } else {
          u = betaSample(betaAlpha, betaBeta);
        }
        t = u * numT + minT;
      }

      // Apply timestep shift (matches Python: N * shift * t / ((shift-1)*t + N))
      t = (N * shift * t) / ((shift - 1) * t + N);
      bins[Math.floor(Math.min(Math.max(t, 0), N - 1))]++;
    }
  } else {
    // Discrete distributions: compute probability weights directly.
    // Equivalent to multinomial sampling with infinite samples.
    const n = numT;

    // Inverse-shifted linspace and its derivative (matches Python exactly)
    const ls = new Float64Array(n);
    const lsDeriv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const x = n > 1 ? i / (n - 1) : 0;
      const denom = shift - shift * x + x; // shift*(1-x) + x
      ls[i] = denom !== 0 ? x / denom : 0;
      const denomD = shift + x - x * shift; // same expression, different variable name in Python
      lsDeriv[i] = denomD !== 0 ? shift / denomD ** 2 : 0;
    }

    const weights = new Float64Array(n);

    if (distribution === "COS_MAP") {
      for (let i = 0; i < n; i++) {
        const v = ls[i];
        const denom = Math.PI - 2.0 * Math.PI * v + 2.0 * Math.PI * v * v;
        weights[i] = denom !== 0 ? (2.0 / denom) * lsDeriv[i] : 0;
      }
    } else if (distribution === "SIGMOID") {
      const bias = noisingBias + 0.5;
      const weight = noisingWeight;
      for (let i = 0; i < n; i++) {
        const v = ls[i];
        // Python applies the shift formula again to already-shifted linspace
        const denom = shift - shift * v + v;
        const w = denom !== 0 ? v / denom : 0;
        weights[i] = (1 / (1 + Math.exp(-weight * (w - bias)))) * lsDeriv[i];
      }
    } else {
      // INVERTED_PARABOLA
      const bias = noisingBias + 0.5;
      const weight = noisingWeight;
      for (let i = 0; i < n; i++) {
        const v = ls[i];
        weights[i] = Math.max(-weight * (v - bias) ** 2 + 2, 0) * lsDeriv[i];
      }
    }

    // Filter NaN / Infinity
    for (let i = 0; i < n; i++) {
      if (!isFinite(weights[i])) weights[i] = 0;
    }

    // Normalize and map to bins (index i → timestep i + minT)
    let total = 0;
    for (let i = 0; i < n; i++) total += weights[i];

    if (total > 0) {
      for (let i = 0; i < n; i++) {
        const bin = i + minT;
        if (bin >= 0 && bin < N) {
          bins[bin] = (weights[i] / total) * SAMPLE_COUNT;
        }
      }
    }
  }

  return bins;
}
