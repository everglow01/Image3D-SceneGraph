import assert from 'node:assert/strict';

export const SPARK_VERSION = '2.2.0';
export const SPARK_PROFILE = 'fixed_camera_spark_v1';
export const SPARK_SETTINGS = Object.freeze({
  autoUpdate: false,
  enableLod: false,
  enableDriveLod: false,
  enableLodFetching: false,
  accumExtSplats: true,
  sortRadial: false,
  minSortIntervalMs: 0,
  preBlurAmount: 0.3,
  blurAmount: 0,
  focalAdjustment: 1,
  minAlpha: 1 / 255,
  maxStdDev: 3,
  maxPixelRadius: 1e10,
});

export function validateSparkAudit(audit) {
  assert.equal(audit.profile, 'frozen_render_consistency_v1', 'expected frozen native audit');
  assert.equal(audit.status, 'completed', 'native audit is not complete');
  assert.equal(audit.split, 'validation', 'only frozen Validation cameras are allowed');
  assert.equal(audit.test_rgb, 'not_loaded', 'Test must remain excluded');
  assert.equal(audit.source_unchanged, true, 'native source integrity did not pass');
  assert.equal(audit.sh_degree, 3, 'this prototype requires an SH3 source');
  assert(Number.isSafeInteger(audit.model_count) && audit.model_count > 0, 'invalid model count');
  assert.match(audit.source_sha256?.ply ?? '', /^[a-f0-9]{64}$/, 'missing source PLY hash');
  assert.deepEqual(audit.background, [0, 0, 0], 'prototype requires the frozen black background');
  assert(Number.isFinite(audit.near) && Number.isFinite(audit.far) &&
    audit.near > 0 && audit.far > audit.near, 'invalid clipping planes');
  assert(Array.isArray(audit.views) && audit.views.length > 0, 'no frozen cameras');
  const ids = new Set();
  for (const view of audit.views) {
    assert(typeof view.image_id === 'string' && /^[\w.-]+$/.test(view.image_id), 'invalid image ID');
    assert(!ids.has(view.image_id), 'duplicate image ID');
    ids.add(view.image_id);
    assert(Number.isSafeInteger(view.width) && view.width > 0 &&
      Number.isSafeInteger(view.height) && view.height > 0, 'invalid image size');
    for (const [matrix, size] of [[view.intrinsic, 3], [view.camera_from_normalized, 4]]) {
      assert(Array.isArray(matrix) && matrix.length === size && matrix.every(row =>
        Array.isArray(row) && row.length === size && row.every(Number.isFinite)), 'invalid camera matrix');
    }
    assert(view.intrinsic[0][0] > 0 && view.intrinsic[1][1] > 0, 'invalid focal length');
    assert.equal(view.intrinsic[0][1], 0, 'Spark projection audit does not support skew');
    assert.equal(view.intrinsic[1][0], 0, 'invalid intrinsic matrix');
    assert.deepEqual(view.intrinsic[2], [0, 0, 1], 'invalid intrinsic matrix');
    assert.deepEqual(view.camera_from_normalized[3], [0, 0, 0, 1], 'invalid homogeneous transform');
  }
  assert.deepEqual([...ids].sort(), [...(audit.image_ids ?? [])].sort(), 'camera ID set mismatch');
}

export function sparkFixture() {
  const fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
    ...Array.from({length: 45}, (_, i) => `f_rest_${i}`), 'opacity',
    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const positions = [[0.3, 0.2], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0, 0.5], [0.5, 0.5]];
  const header = ['ply', 'format binary_little_endian 1.0', `element vertex ${positions.length}`,
    ...fields.map(field => `property float ${field}`), 'end_header', ''].join('\n');
  const rows = positions.map(([x, y]) => {
    const row = Buffer.alloc(fields.length * 4);
    const values = {x, y, z: 2, opacity: Math.log(0.95 / 0.05), rot_0: 1,
      scale_0: Math.log(0.04), scale_1: Math.log(0.04), scale_2: Math.log(0.04),
      f_dc_0: -0.1 / 0.28209479177387814, f_dc_1: -0.1 / 0.28209479177387814,
      f_dc_2: -0.1 / 0.28209479177387814,
      // Blue channel, coefficient l=3,m=0, in channel-major Graphdeco layout.
      f_rest_40: 0.5};
    fields.forEach((field, i) => row.writeFloatLE(values[field] ?? 0, i * 4));
    return row;
  });
  return Buffer.concat([Buffer.from(header), ...rows]);
}

export function checkSparkSelfTest(captures) {
  const find = degree => captures.find(row => row.requested_sh === degree && row.image_id === 'fixture');
  const [zero, two, three] = [0, 2, 3].map(find);
  assert(zero && two && three, 'SH0/SH2/SH3 fixture captures are required');
  for (const row of [zero, two, three]) {
    assert.equal(row.effective_sh, row.requested_sh, 'SH degree silently downgraded');
    assert.equal(row.count, 6, 'fixture count changed');
    assert.equal(row.gl_error, 0, 'WebGL error in fixture');
  }
  assert(zero.reference_probe_pixel[0] > 50 && zero.reference_probe_pixel[0] < 150,
    'fixture projection or linear RGB level is incorrect');
  assert(zero.reference_mirrored_pixel.slice(0, 3).every(value => value < 30), 'camera Y axis was mirrored');
  assert(zero.reference_probe_pixel.slice(0, 3).every((value, i) =>
    Math.abs(value - two.reference_probe_pixel[i]) <= 2), 'zero SH1/SH2 coefficients changed fixture');
  assert(Math.abs(three.reference_probe_pixel[2] - two.reference_probe_pixel[2]) > 10,
    'nonzero SH3 did not affect rendered fixture');
  return {status: 'passed', sh3_blue_delta: three.reference_probe_pixel[2] - two.reference_probe_pixel[2]};
}

export function sparkPage({count, near, far}) {
  return `<!doctype html><meta charset="utf-8"><style>body{margin:0;background:black}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/three.js","three/addons/postprocessing/Pass.js":"/Pass.js"}}</script>
<script type="module">
import * as THREE from '/three.js';
import {SparkRenderer, SplatMesh} from '/spark.js';
const expected = ${JSON.stringify({count, near, far})};
const settings = ${JSON.stringify(SPARK_SETTINGS)};
window.loadScene = async (_view, degree) => {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera();
  const renderer = new THREE.WebGLRenderer({antialias:false, preserveDrawingBuffer:true});
  window.renderer = renderer;
  renderer.setPixelRatio(1);
  renderer.setClearColor(0, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  document.body.appendChild(renderer.domElement);
  const spark = new SparkRenderer({renderer, ...settings});
  const mesh = new SplatMesh({url:'/scene.ply', extSplats:true, lod:false, enableLod:false});
  window.disposeAudit = () => {
    mesh.dispose(); spark.dispose(); scene.clear();
    renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
  };
  scene.add(spark, mesh);
  const loadStart = performance.now();
  await mesh.initialized;
  const loadMs = performance.now() - loadStart;
  if (mesh.numSplats !== expected.count || mesh.extSplats?.numSplats !== expected.count)
    throw Error('source splat count mismatch');
  const sourceSh = mesh.extSplats.getNumSh();
  if (sourceSh !== 3) throw Error('source SH3 coefficients missing');
  mesh.maxSh = degree;
  mesh.updateGenerator();
  window.drawView = async view => {
    renderer.setSize(view.width, view.height);
    camera.near = expected.near; camera.far = expected.far;
    const [fx, skew, cx] = view.intrinsic[0], [, fy, cy] = view.intrinsic[1];
    const w = view.width, h = view.height, n = camera.near, f = camera.far;
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(h / (2 * fy)));
    camera.aspect = w / h;
    camera.projectionMatrix.set(2*fx/w, -2*skew/w, 1-2*cx/w, 0, 0, 2*fy/h, 2*cy/h-1, 0,
      0, 0, -(f+n)/(f-n), -2*f*n/(f-n), 0, 0, -1, 0);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    const cv = new THREE.Matrix4().set(...view.camera_from_normalized.flat());
    const glPose = new THREE.Matrix4().makeScale(1,-1,-1).multiply(cv);
    camera.matrix.copy(glPose.clone().invert());
    camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
    camera.updateMatrixWorld(true); scene.updateMatrixWorld(true);
    renderer.getDrawingBufferSize(spark.renderSize);
    const updateStart = performance.now();
    await spark.update({scene, camera});
    if (spark.sorting || spark.sortDirty || spark.current.mappingVersion !== spark.display.mappingVersion)
      throw Error('fixed-camera sort did not complete');
    if (spark.current.numSplats !== expected.count || spark.display.numSplats !== expected.count ||
        spark.enableLod || mesh.context.enableLod.value)
      throw Error('accumulator count changed or LoD enabled');
    const effectiveSh = Math.min(sourceSh, mesh.extSplats.maxSh);
    if (effectiveSh !== degree) throw Error('SH degree silently downgraded');
    const updateMs = performance.now() - updateStart;
    const renderStart = performance.now();
    renderer.render(scene, camera);
    const gl = renderer.getContext(), pixels = new Uint8Array(w*h*4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const renderReadbackMs = performance.now() - renderStart;
    if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw Error('WebGL context/error failure');
    if (!pixels.some((value, i) => i%4 < 3 && value > 0)) throw Error('blank frame');
    const pixel = (x,y) => x < w && y < h ? Array.from(pixels.slice(((h-1-y)*w+x)*4,((h-1-y)*w+x)*4+4)) : null;
    const first = mesh.extSplats.getSplat(0);
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return {png:renderer.domElement.toDataURL('image/png').split(',')[1], runtime:{
      implementation:'@sparkjsdev/spark', version:'${SPARK_VERSION}', requested_sh:degree,
      source_sh:sourceSh, effective_sh:effectiveSh, count:mesh.numSplats,
      accumulator_count:spark.display.numSplats, active_count:spark.activeSplats,
      sort_running:spark.sorting, sort_dirty:spark.sortDirty, lod_enabled:false,
      source_encoding:'ExtSplats', accumulator_encoding:'ExtSplats', lossless:false,
      first_center:first.center.toArray(), first_color:first.color.toArray(),
      first_opacity:first.opacity, first_scales:first.scales.toArray(), first_quaternion:first.quaternion.toArray(),
      projected_first_center:first.center.clone().project(camera).toArray(),
      settings, width:gl.drawingBufferWidth, height:gl.drawingBufferHeight, gl_error:0,
      output_color_space:renderer.outputColorSpace, tone_mapping:renderer.toneMapping,
      encode_linear:spark.uniforms.encodeLinear.value,
      renderer_pixel_ratio:renderer.getPixelRatio(), device_pixel_ratio:window.devicePixelRatio,
      gpu:ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      near:n, far:f, projection:camera.projectionMatrix.toArray(), world_to_camera:camera.matrixWorldInverse.toArray(),
      load_ms:loadMs, update_sort_ms:updateMs, render_readback_ms:renderReadbackMs,
      performance_scope:'single diagnostic capture including readback; not FPS benchmark',
      reference_probe_pixel:pixel(79,74), reference_mirrored_pixel:pixel(79,h-1-74), corner:pixel(0,0)
    }};
  };
};
window.auditReady = true;
</script>`;
}
