import assert from 'node:assert/strict';
import {mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {test} from 'node:test';
import {runInNewContext} from 'node:vm';
import * as THREE from '../frontend/node_modules/three/build/three.module.js';
import {SPARK_SETTINGS, SPARK_VERSION, validateSparkAudit, sparkFixture, sparkPage,
  checkSparkSelfTest} from '../scripts/gaussian_spark_audit.mjs';

const camera = () => ({image_id:'317', width:128, height:128,
  intrinsic:[[100,0,64],[0,100,64],[0,0,1]],
  camera_from_normalized:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]});
const audit = () => ({profile:'frozen_render_consistency_v1', status:'completed', split:'validation',
  test_rgb:'not_loaded', source_unchanged:true, sh_degree:3, model_count:6,
  source_sha256:{ply:'a'.repeat(64)}, background:[0,0,0], near:0.01, far:1e10,
  image_ids:['317'], views:[camera()]});

test('accepts a frozen Validation camera audit', () => {
  assert.doesNotThrow(() => validateSparkAudit(audit()));
});

test('rejects failed/unfrozen audits, Test, missing SH and altered sources', () => {
  for (const change of [{profile:'other'}, {status:'failed'}, {split:'test'}, {test_rgb:'loaded'},
    {source_unchanged:false}, {sh_degree:2}, {model_count:0}, {model_count:1.5},
    {source_sha256:{}}, {background:[1,1,1]}, {near:0}, {far:0.001}, {views:[]}, {image_ids:['other']}]) {
    assert.throws(() => validateSparkAudit({...audit(), ...change}));
  }
});

test('rejects malformed, skewed, duplicate and path-like cameras', () => {
  for (const edit of [view => {view.width = -1;}, view => {view.intrinsic[0][0] = NaN;},
    view => {view.intrinsic[0][1] = 0.1;}, view => {view.intrinsic[2] = [0,1,1];},
    view => {view.camera_from_normalized[3][0] = 1;}, view => {view.image_id = '../escape';}]) {
    const input = audit(); edit(input.views[0]);
    assert.throws(() => validateSparkAudit(input));
  }
  const input = audit(); input.views.push(camera());
  assert.throws(() => validateSparkAudit(input), /duplicate/);
});

test('fixture has real nonzero SH3 and no lower-order coefficients', () => {
  const bytes = sparkFixture();
  const offset = bytes.indexOf('end_header\n') + 'end_header\n'.length;
  const header = bytes.subarray(0, offset).toString();
  const fields = header.split('\n').filter(line => line.startsWith('property ')).map(line => line.split(' ')[2]);
  assert.match(header, /format binary_little_endian 1.0/);
  assert.match(header, /element vertex 6/);
  assert.equal(fields.length, 62);
  assert.equal(bytes.length - offset, 6 * 62 * 4);
  for (let row = 0; row < 6; row++) {
    for (let i = 0; i < 45; i++) {
      const value = bytes.readFloatLE(offset + (row * 62 + fields.indexOf(`f_rest_${i}`)) * 4);
      assert.equal(value, i === 40 ? 0.5 : 0);
    }
  }
});

const captures = () => [0,2,3].map(degree => ({image_id:'fixture', requested_sh:degree,
  effective_sh:degree, count:6, gl_error:0, reference_probe_pixel:[90,90,degree === 3 ? 150 : 90,255],
  reference_mirrored_pixel:[0,0,0,255]}));

test('SH self-test requires measured SH3 response, not just a reported degree', () => {
  assert.equal(checkSparkSelfTest(captures()).status, 'passed');
  for (const edit of [rows => rows.pop(), rows => {rows[2].effective_sh = 2;},
    rows => {rows[2].reference_probe_pixel[2] = 90;}, rows => {rows[1].reference_probe_pixel[0] = 120;},
    rows => {rows[0].reference_probe_pixel[0] = 200;}, rows => {rows[0].reference_mirrored_pixel[0] = 100;},
    rows => {rows[0].count = 5;}, rows => {rows[0].gl_error = 1282;}]) {
    const rows = captures(); edit(rows);
    assert.throws(() => checkSparkSelfTest(rows));
  }
});

test('controlled profile disables LoD and uses extended source and accumulator encoding', () => {
  assert.equal(SPARK_SETTINGS.enableLod, false);
  assert.equal(SPARK_SETTINGS.enableDriveLod, false);
  assert.equal(SPARK_SETTINGS.enableLodFetching, false);
  assert.equal(SPARK_SETTINGS.autoUpdate, false);
  assert.equal(SPARK_SETTINGS.accumExtSplats, true);
  assert.equal(SPARK_SETTINGS.sortRadial, false);
  assert.equal(SPARK_SETTINGS.preBlurAmount, 0.3);
  assert.equal(SPARK_SETTINGS.blurAmount, 0);
  const page = sparkPage({count:6, near:0.01, far:1e10});
  const code = page.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
  const checked = spawnSync(process.execPath, ['--input-type=module', '--check'], {input:code, encoding:'utf8'});
  assert.equal(checked.status, 0, checked.stderr);
  assert.match(code, /extSplats:true, lod:false, enableLod:false/);
  assert.match(code, /await spark.update/);
  assert.match(code, /spark.sorting \|\| spark.sortDirty/);
  assert.match(code, /spark.display.numSplats !== expected.count/);
  assert.match(code, /mesh.dispose\(\); spark.dispose\(\)/);
  assert.doesNotMatch(code, /https?:\/\//);
});

test('browser adapter applies CV projection and fails closed on incomplete mocked renders', async () => {
  async function load(overrides = {}) {
    let width = 0, height = 0, disposed = 0;
    const gl = {RGBA:1, UNSIGNED_BYTE:2, NO_ERROR:0, RENDERER:3,
      get drawingBufferWidth() { return width; }, get drawingBufferHeight() { return height; },
      readPixels(_x,_y,_w,_h,_format,_type,pixels) { pixels.fill(overrides.blank ? 0 : 90); },
      getError:() => overrides.glError ?? 0, isContextLost:() => false,
      getExtension:() => null, getParameter:() => 'mock; not a GPU measurement'};
    class Renderer {
      constructor() { this.domElement = {toDataURL:() => 'data:image/png;base64,AA==', remove(){}}; }
      setPixelRatio() {} setClearColor() {} render() {} dispose() {} forceContextLoss() {}
      setSize(w,h) { width=w; height=h; }
      getDrawingBufferSize(size) { size.set(width,height); }
      getPixelRatio() { return 1; } getContext() { return gl; }
    }
    class Mesh extends THREE.Object3D {
      constructor() {
        super(); this.numSplats = overrides.sourceCount ?? 6; this.initialized = Promise.resolve();
        this.context = {enableLod:{value:false}};
        this.extSplats = {numSplats:this.numSplats, maxSh:3, getNumSh:() => 3,
          getSplat:() => ({center:new THREE.Vector3(0.3,0.2,2), color:new THREE.Color(0.4,0.4,0.4),
            scales:new THREE.Vector3(0.04,0.04,0.04), opacity:0.95, quaternion:new THREE.Quaternion()})};
      }
      updateGenerator() {} dispose() { disposed++; }
    }
    class Spark extends THREE.Object3D {
      constructor(settings) {
        super(); Object.assign(this,settings); this.renderSize=new THREE.Vector2();
        this.current = this.display = {numSplats:6, mappingVersion:1};
        this.uniforms = {encodeLinear:{value:false}}; this.activeSplats=6;
      }
      async update({scene}) {
        const mesh = scene.children.find(item => item instanceof Mesh);
        mesh.extSplats.maxSh = overrides.effectiveSh ?? mesh.maxSh;
        this.sorting = overrides.sorting ?? false; this.sortDirty = false;
        this.display.numSplats = overrides.accumulatorCount ?? 6;
      }
      dispose() { disposed++; }
    }
    const window = {devicePixelRatio:1};
    const code = sparkPage({count:6, near:0.01, far:1e10})
      .match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/^import .*;$/gm, '');
    runInNewContext(code, {THREE:{...THREE, WebGLRenderer:Renderer}, SparkRenderer:Spark, SplatMesh:Mesh,
      window, document:{body:{appendChild(){}}}, performance});
    return {window, disposed:() => disposed};
  }
  const success = await load();
  await success.window.loadScene(camera(), 3);
  const frame = await success.window.drawView(camera());
  assert.equal(frame.runtime.effective_sh, 3);
  assert.equal(frame.runtime.lossless, false);
  assert.equal(frame.runtime.accumulator_count, 6);
  assert(Math.abs(frame.runtime.projected_first_center[0] - 0.234375) < 1e-9);
  assert(Math.abs(frame.runtime.projected_first_center[1] + 0.15625) < 1e-9);
  success.window.disposeAudit(); assert.equal(success.disposed(), 2);
  const truncated = await load({sourceCount:5});
  await assert.rejects(truncated.window.loadScene(camera(),3), /count mismatch/);
  for (const [overrides, error] of [[{sorting:true}, /sort did not complete/],
    [{accumulatorCount:5}, /accumulator count/], [{effectiveSh:2}, /silently downgraded/],
    [{glError:1282}, /WebGL/], [{blank:true}, /blank frame/]]) {
    const failure = await load(overrides); await failure.window.loadScene(camera(),3);
    await assert.rejects(failure.window.drawView(camera()), error);
  }
});

test('Spark stays pinned and opt-in; the page defaults to the existing library', async () => {
  const pkg = JSON.parse(await readFile(new URL('../frontend/package.json', import.meta.url)));
  assert.equal(pkg.dependencies['@sparkjsdev/spark'], SPARK_VERSION);
  assert.equal(pkg.devDependencies['@sparkjsdev/spark'], undefined);
  const viewer = await readFile(new URL('../frontend/src/GaussianSplatViewer.tsx', import.meta.url), 'utf8');
  assert.match(viewer, /@mkkellogg\/gaussian-splats-3d/);
  assert.match(viewer, /useState<RendererKind>\("legacy"\)/);
  assert.match(viewer, /await import\("\.\/SparkPageViewer"\)/);
});

test('CLI rejects wrong PLY hashes and invalid selections before launching Chrome', async () => {
  const base = process.env.CLAUDE_JOB_DIR ? join(process.env.CLAUDE_JOB_DIR, 'tmp') : tmpdir();
  const root = await mkdtemp(join(base, 'spark-integrity-test-'));
  const input = join(root, 'audit.json'), ply = join(root, 'scene.ply');
  await writeFile(input, JSON.stringify(audit()));
  await writeFile(ply, sparkFixture());
  for (const [name, flags, error] of [['hash', [], /PLY hash mismatch/],
    ['ids', ['--image-ids','missing'], /unknown requested camera/],
    ['probe', ['--sh-probe','missing'], /SH probe must be among/],
    ['mode', ['--modes','product'], /only supports controlled/]]) {
    const output = join(root,name);
    const run = spawnSync(process.execPath, ['scripts/audit_gaussian_browser.mjs', '--renderer', 'spark',
      '--audit', input, '--ply', ply, '--output', output, '--chrome', join(root,'must-not-launch'), ...flags],
    {cwd:resolve(import.meta.dirname, '..'), encoding:'utf8'});
    assert.equal(run.status, 1, run.stderr);
    const result = JSON.parse(await readFile(join(output,'spark.json'), 'utf8'));
    assert.match(result.error, error);
    assert.equal(result.chrome_log, '');
    assert.equal(result.captures.length, 0);
  }
});

test('CLI rejects Test before opening PLY or launching Chrome and preserves existing outputs', async () => {
  const base = process.env.CLAUDE_JOB_DIR ? join(process.env.CLAUDE_JOB_DIR, 'tmp') : tmpdir();
  const root = await mkdtemp(join(base, 'spark-audit-test-'));
  const input = join(root, 'audit.json'), output = join(root, 'result');
  await writeFile(input, JSON.stringify({...audit(), split:'test'}));
  const command = ['scripts/audit_gaussian_browser.mjs', '--renderer', 'spark', '--audit', input,
    '--ply', join(root, 'absent.ply'), '--output', output, '--chrome', join(root, 'must-not-launch')];
  const run = spawnSync(process.execPath, command, {cwd:resolve(import.meta.dirname, '..'), encoding:'utf8'});
  assert.equal(run.status, 1, run.stderr);
  const before = await readFile(join(output, 'spark.json'), 'utf8');
  const result = JSON.parse(before);
  assert.equal(result.status, 'failed');
  assert.match(result.error, /only frozen Validation cameras/);
  assert.equal(result.chrome_log, '');
  assert.equal(result.captures.length, 0);
  const again = spawnSync(process.execPath, command, {cwd:resolve(import.meta.dirname, '..'), encoding:'utf8'});
  assert.notEqual(again.status, 0);
  assert.match(again.stderr, /EEXIST/);
  assert.equal(await readFile(join(output, 'spark.json'), 'utf8'), before);
});
