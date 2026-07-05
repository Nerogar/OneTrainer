export function niceStep(range: number, targetTicks: number): number {
  const rough = range / targetTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalized = rough / mag;
  let nice: number;
  if (normalized <= 1.5) nice = 1;
  else if (normalized <= 3) nice = 2;
  else if (normalized <= 7) nice = 5;
  else nice = 10;
  return nice * mag;
}

export function generateTicks(min: number, max: number, targetTicks: number): number[] {
  if (min === max) return [min];
  const step = niceStep(max - min, targetTicks);
  const start = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let t = start; t <= max + step * 0.01; t += step) {
    ticks.push(t);
  }
  return ticks;
}

export function formatValue(v: number): string {
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(1);
  if (Number.isInteger(v)) return v.toString();
  return v.toPrecision(4);
}

export function formatStep(s: number): string {
  if (s >= 1e6) return `${(s / 1e6).toFixed(1)}M`;
  if (s >= 1e3) return `${(s / 1e3).toFixed(1)}k`;
  return s.toString();
}
