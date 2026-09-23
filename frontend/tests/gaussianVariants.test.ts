import assert from "node:assert/strict";
import test from "node:test";
import { gaussianVariants, defaultGaussianVariant, gaussianVariantMetricLabel } from "../src/gaussianVariants.ts";
import { formatResultKind } from "../src/resultKind.ts";

function fixture() {
  const variants = ["mcmc", "project"].flatMap(trainer => ["selection", "final_fit"].map(stage => ({
    id: `${trainer}-${stage}`, label: `${trainer} ${stage}`, trainer, model_stage: stage,
    metric_role: stage === "selection" ? "held_out_model_selection" : "in_sample_after_train_validation_fit",
    scene_splat: `${trainer}/${stage}/scene.ply`, export_metadata: `${trainer}/${stage}/export.json`,
    psnr: 25, ssim: 0.8, display_psnr: 25.1, display_ssim: 0.81, gaussian_count: 10, model_sha256: "a".repeat(64)
  })));
  return { result_kind: "gaussian_comparison", gaussian_variants: variants,
    default_gaussian_variant: variants[1].id,
    assets: { scene_splat: variants[1].scene_splat, gaussian_export_metadata: variants[1].export_metadata } };
}

test("four variants keep distinct model and metric identities", () => {
  const m = fixture();
  assert.equal(gaussianVariants(m).length, 4);
  assert.equal(defaultGaussianVariant(m), "mcmc-final_fit");
  assert.equal(formatResultKind("gaussian_comparison"), "高斯补拟合对比");
});

test("invalid comparison cannot silently fall back to Original", () => {
  for (const mutate of [
    (m: ReturnType<typeof fixture>) => { m.gaussian_variants[0].id = m.gaussian_variants[1].id; },
    (m: ReturnType<typeof fixture>) => { m.gaussian_variants[0].scene_splat = "../scene.ply"; },
    (m: ReturnType<typeof fixture>) => { m.gaussian_variants[1].metric_role = "held_out_model_selection"; },
    (m: ReturnType<typeof fixture>) => { m.default_gaussian_variant = "absent"; },
    (m: ReturnType<typeof fixture>) => { m.gaussian_variants[0].psnr = NaN; }
  ]) {
    const m = fixture(); mutate(m); assert.throws(() => gaussianVariants(m));
  }
  assert.throws(() => gaussianVariants({ result_kind: "gaussian_comparison", assets: {} }));
});

test("two Train-only models retain held-out labels and distinct assets", () => {
  const m = fixture();
  m.gaussian_variants = m.gaussian_variants.filter(v => v.model_stage === "final_fit").map(v => ({
    ...v, model_stage: "train_only", metric_role: "held_out_after_train_only_control"
  }));
  const variants = gaussianVariants(m);
  assert.equal(variants.length, 2);
  assert.notEqual(variants[0].scene_splat, variants[1].scene_splat);
  assert.match(gaussianVariantMetricLabel(variants[0]), /Train-only 后 held-out Validation/);
  assert.match(gaussianVariantMetricLabel(variants[0]), /非 Test/);
  assert.match(gaussianVariantMetricLabel(gaussianVariants(fixture())[1]), /非 held-out/);
  for (const wrongRole of ["held_out_model_selection", "in_sample_after_train_validation_fit"]) {
    m.gaussian_variants[0].metric_role = wrongRole;
    assert.throws(() => gaussianVariants(m));
  }
});

test("browser assets bind to their own variant without replacing source identity", () => {
  const m = fixture();
  const row = m.gaussian_variants[1];
  const asset = { source: row.scene_splat, path: `lifecycle/browser/${"b".repeat(64)}/scene.ksplat`,
    sha256: "c".repeat(64), bytes: 100, gaussian_count: 10, sh_degree: 2, compression_level: 2 };
  const variants = gaussianVariants({ ...m, gaussian_browser_assets: [asset] });
  assert.equal(variants[0].browser_asset, undefined);
  assert.deepEqual(variants[1].browser_asset, asset);
  assert.equal(variants[1].scene_splat, row.scene_splat);
  assert.equal(m.gaussian_variants[1].scene_splat, row.scene_splat);
  for (const invalid of [
    { ...asset, source: "other.ply" }, { ...asset, path: "../scene.ksplat" },
    { ...asset, path: "https://example.com/scene.ksplat" }, { ...asset, sha256: "bad" },
    { ...asset, gaussian_count: 9 }, { ...asset, sh_degree: 3 }, { ...asset, compression_level: 1 },
    { ...asset, bytes: 0 }, { ...asset, bytes: 2 ** 40 }
  ]) {
    assert.equal(gaussianVariants({ ...m, gaussian_browser_assets: [invalid] })[1].browser_asset, undefined);
  }
  assert.equal(gaussianVariants({ ...m, gaussian_browser_assets: [asset, asset] })[1].browser_asset, undefined);
  const historical = gaussianVariants({ assets: { scene_splat: row.scene_splat, gaussian_export_metadata: row.export_metadata },
    gaussian_browser_assets: [asset] });
  assert.deepEqual(historical[0].browser_asset, asset);
});

test("historical Original and VGGT default selection remain compatible", () => {
  const m = { assets: { scene_splat: "scene.ply", gaussian_export_metadata: "export.json",
    scene_splat_vggt_filtered: "filtered.ply", gaussian_vggt_filtered_export_metadata: "filtered.json" } };
  assert.deepEqual(gaussianVariants(m).map(v => v.id), ["original", "vggt_filtered"]);
  assert.equal(defaultGaussianVariant(m), "vggt_filtered");
});
