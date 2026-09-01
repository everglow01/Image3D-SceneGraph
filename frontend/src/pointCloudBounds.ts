export type CloudBounds = {
  center: [number, number, number];
  radius: number;
};

// ponytail: percentile bounds instead of the exact bounding sphere — a few stray
// SfM points must not inflate framing/frustum scale. Upgrade to a proper
// outlier filter only if percentiles ever prove too coarse.
export function robustCloudBounds(positions: ArrayLike<number>, count: number): CloudBounds {
  if (count <= 0 || positions.length < count * 3) {
    return { center: [0, 0, 0], radius: 1 };
  }
  const axes = [new Float64Array(count), new Float64Array(count), new Float64Array(count)];
  for (let index = 0; index < count; index += 1) {
    axes[0][index] = positions[index * 3];
    axes[1][index] = positions[index * 3 + 1];
    axes[2][index] = positions[index * 3 + 2];
  }
  for (const axis of axes) {
    axis.sort();
  }
  const quantile = (sorted: Float64Array, t: number) =>
    sorted[Math.min(count - 1, Math.max(0, Math.floor(t * (count - 1))))];
  const center = axes.map((sorted) => (quantile(sorted, 0.02) + quantile(sorted, 0.98)) / 2) as [
    number,
    number,
    number
  ];
  const distances = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    const dx = positions[index * 3] - center[0];
    const dy = positions[index * 3 + 1] - center[1];
    const dz = positions[index * 3 + 2] - center[2];
    distances[index] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  distances.sort();
  const radius = quantile(distances, 0.99);
  if (!Number.isFinite(radius) || radius <= 1e-6) {
    return { center, radius: 1 };
  }
  return { center, radius };
}
