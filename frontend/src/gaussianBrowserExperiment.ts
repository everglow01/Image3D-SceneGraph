export type BrowserExperimentLevel = "ply" | "level1" | "level2";

const SOURCE = "/api/jobs/train_validation_final_fit_v1_comparison/assets/variants/mcmc-final-fit/scene.ply";
const ASSETS = "/api/gaussian-browser-experiments/browser-ksplat-20260922-v1";

export function browserExperimentUrl(sourceUrl: string | null, level: BrowserExperimentLevel, search: string): string | null {
  if (sourceUrl !== SOURCE || !new URLSearchParams(search).has("browser-ksplat-ab")) return null;
  return level === "ply" ? sourceUrl : `${ASSETS}/${level}.ksplat`;
}
