export type GaussianBrowserAsset = {
  source: string;
  path: string;
  sha256: string;
  bytes: number;
  gaussian_count: number;
  sh_degree: 2;
  compression_level: 2;
};

export type GaussianVariant = {
  id: string;
  label: string;
  scene_splat: string;
  export_metadata: string;
  trainer?: "project" | "mcmc";
  model_stage?: "selection" | "final_fit" | "train_only";
  metric_role?: string;
  psnr?: number;
  ssim?: number;
  display_psnr?: number;
  display_ssim?: number;
  gaussian_count?: number;
  model_sha256?: string;
  browser_asset?: GaussianBrowserAsset;
};

type VariantManifest = {
  result_kind?: string;
  gaussian_variants?: unknown;
  gaussian_browser_assets?: unknown;
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

function withBrowserAssets(rows: GaussianVariant[], manifest: VariantManifest): GaussianVariant[] {
  const assets = manifest.gaussian_browser_assets;
  return rows.map(row => {
    const matches = Array.isArray(assets) && assets.length <= 4
      ? assets.filter(asset => asset?.source === row.scene_splat) : [];
    const asset = matches.length === 1 ? matches[0] : null;
    const valid = asset && relativePath(asset.source) && relativePath(asset.path) &&
      /^lifecycle\/browser\/[a-f0-9]{64}\/scene\.ksplat$/.test(asset.path) &&
      typeof asset.sha256 === "string" && /^[a-f0-9]{64}$/.test(asset.sha256) &&
      Number.isSafeInteger(asset.bytes) && asset.bytes > 0 && asset.bytes <= 1_073_741_824 &&
      Number.isSafeInteger(asset.gaussian_count) && asset.gaussian_count > 0 && asset.gaussian_count <= 3_000_000 &&
      (row.gaussian_count === undefined || row.gaussian_count === asset.gaussian_count) &&
      asset.sh_degree === 2 && asset.compression_level === 2;
    return { ...row, browser_asset: valid ? asset : undefined };
  });
}

export function gaussianVariants(manifest: VariantManifest): GaussianVariant[] {
  const rows = manifest.gaussian_variants;
  if (rows === undefined && manifest.result_kind !== "gaussian_comparison") {
    const a = manifest.assets;
    return withBrowserAssets([
      { id: "original", label: "Original", scene_splat: a.scene_splat, export_metadata: a.gaussian_export_metadata },
      { id: "vggt_filtered", label: "VGGT-filtered", scene_splat: a.scene_splat_vggt_filtered, export_metadata: a.gaussian_vggt_filtered_export_metadata }
    ].filter(row => row.scene_splat && row.export_metadata) as GaussianVariant[], manifest);
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
        !["selection", "final_fit", "train_only"].includes(row.model_stage) ||
        row.metric_role !== (row.model_stage === "selection"
          ? "held_out_model_selection" : row.model_stage === "train_only"
            ? "held_out_after_train_only_control" : "in_sample_after_train_validation_fit") ||
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
  return withBrowserAssets(rows, manifest);
}

export function gaussianVariantMetricLabel(variant: GaussianVariant): string {
  if (variant.model_stage === "train_only") return "Train-only 后 held-out Validation（未参与补拟合；非 Test）";
  return variant.model_stage === "final_fit"
    ? "Train+Validation 拟合集（非 held-out）" : "补拟合前 held-out Validation";
}

export function defaultGaussianVariant(manifest: VariantManifest): string {
  const variants = gaussianVariants(manifest);
  return manifest.default_gaussian_variant ??
    (variants.some(row => row.id === "vggt_filtered") ? "vggt_filtered" : "original");
}
