export type ResultKind = "salvaged_derivative" | "gaussian_comparison";

export function formatResultKind(value: ResultKind | undefined): string | null {
  if (value === "gaussian_comparison") return "高斯补拟合对比";
  return value === "salvaged_derivative" ? "恢复衍生结果" : null;
}
