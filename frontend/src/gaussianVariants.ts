export type GaussianVariant = {
  id: string;
  label: string;
  scene_splat: string;
  export_metadata: string;
  trainer?: "project" | "mcmc";
  model_stage?: "selection" | "final_fit";
  metric_role?: string;
  psnr?: number;
  ssim?: number;
  display_psnr?: number;
  display_ssim?: number;
  gaussian_count?: number;
  model_sha256?: string;
};

type VariantManifest = {
  result_kind?: string;
  gaussian_variants?: unknown;
  default_gaussian_variant?: string;
  assets: {
    scene_splat?: string;
    gaussian_export_metadata?: string;
    scene_splat_vggt_filtered?: string;
    gaussian_vggt_filtered_export_metadata?: string;
  };
};

function relativePath(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 &&
    !/[\\?#%:]/.test(value) && !value.startsWith("/") &&
    value.split("/").every(part => part !== ".." && part !== "." && part !== "");
}

export function gaussianVariants(manifest: VariantManifest): GaussianVariant[] {
  const rows = manifest.gaussian_variants;
  if (rows === undefined && manifest.result_kind !== "gaussian_comparison") {
    const a = manifest.assets;
    return [
      { id: "original", label: "Original", scene_splat: a.scene_splat, export_metadata: a.gaussian_export_metadata },
      { id: "vggt_filtered", label: "VGGT-filtered", scene_splat: a.scene_splat_vggt_filtered, export_metadata: a.gaussian_vggt_filtered_export_metadata }
    ].filter(row => row.scene_splat && row.export_metadata) as GaussianVariant[];
  }
  if (!Array.isArray(rows) || rows.length === 0 || rows.length > 4) {
    throw new Error("高斯对比列表缺失或无效");
  }
  const ids = new Set<string>();
  for (const row of rows) {
    if (!row || typeof row !== "object" || typeof row.id !== "string" ||
        !/^[a-z0-9_-]+$/.test(row.id) || ids.has(row.id) ||
        typeof row.label !== "string" || !row.label.trim() ||
        !relativePath(row.scene_splat) || !relativePath(row.export_metadata) ||
        !["project", "mcmc"].includes(row.trainer) ||
        !["selection", "final_fit"].includes(row.model_stage) ||
        row.metric_role !== (row.model_stage === "selection"
          ? "held_out_model_selection" : "in_sample_after_train_validation_fit") ||
        ![row.psnr, row.ssim, row.display_psnr, row.display_ssim].every(v => typeof v === "number" && Number.isFinite(v)) ||
        !Number.isSafeInteger(row.gaussian_count) || row.gaussian_count <= 0 ||
        typeof row.model_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(row.model_sha256)) {
      throw new Error("高斯对比模型身份或资产路径无效");
    }
    ids.add(row.id);
  }
  const selected = rows.find(row => row.id === manifest.default_gaussian_variant);
  if (!selected || selected.scene_splat !== manifest.assets.scene_splat ||
      selected.export_metadata !== manifest.assets.gaussian_export_metadata) {
    throw new Error("高斯对比默认模型不一致");
  }
  return rows;
}

export function defaultGaussianVariant(manifest: VariantManifest): string {
  const variants = gaussianVariants(manifest);
  return manifest.default_gaussian_variant ??
    (variants.some(row => row.id === "vggt_filtered") ? "vggt_filtered" : "original");
}
