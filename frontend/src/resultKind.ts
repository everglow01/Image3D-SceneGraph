export type ResultKind = "salvaged_derivative";

export function formatResultKind(value: ResultKind | undefined): string | null {
  return value === "salvaged_derivative" ? "恢复衍生结果" : null;
}
