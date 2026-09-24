import * as THREE from '/three.js';
import { SparkRenderer, SplatMesh, SplatFileType } from '/spark.js';
import { captureP0Source, p0FullMask, verifyP0EncodedRow } from '/src/localGaussianP0Source.ts';
import { P0AlphaMask } from '/src/localGaussianP0Mask.ts';

const check = (v, message) => { if (!v) throw Error(message); };
const p95 = values => [...values].sort((a, b) => a - b)[Math.ceil(values.length * 0.95) - 1];
function difference(a, b) {
  let max = 0, pixelsOver8 = 0;
  for (let i = 0; i < a.length; i += 4) {
    const delta = Math.max(...[0, 1, 2].map(k => Math.abs(a[i + k] - b[i + k])));
    max = Math.max(max, delta); if (delta > 8) pixelsOver8++;
  }
  return { max, pixelsOver8 };
}
const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), v => v.toString(16).padStart(2, '0')).join('');
export async function run() {
  const protocol = await (await fetch('/protocol.json')).json();
  const report = window.realReport = { status: 'running', protocol, stages: [], identity: {}, selections: [], crossViews: [], images: [], resources: [], errors: [] };
  let renderer, spark, mesh, mask, worker;
  const references = new Set();
  const stage = name => { report.stages.push(name); console.info('P0阶段', name); };
  try {
    const [w, h] = protocol.viewport;
    renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1); renderer.setSize(w, h); renderer.setClearColor(0, 1); renderer.toneMapping = THREE.NoToneMapping;
    document.getElementById('probe').append(renderer.domElement);
    const gl = renderer.getContext(), debug = gl.getExtension('WEBGL_debug_renderer_info');
    report.gpu = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : '';
    check(/NVIDIA.*4060/.test(report.gpu) && !/SwiftShader|llvmpipe/i.test(report.gpu), '错误GPU');
    let shaderError = false; renderer.debug.onShaderError = () => { shaderError = true; };
    const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
    spark = new SparkRenderer({ renderer, autoUpdate: false, enableLod: false, enableDriveLod: false, enableLodFetching: false,
      accumExtSplats: true, sortRadial: false, minSortIntervalMs: 0, preBlurAmount: 0.3, blurAmount: 0,
      focalAdjustment: 1, minAlpha: 1 / 255, maxStdDev: 3, maxPixelRadius: 1e10 });
    scene.add(spark);
    const load = async path => {
      stage(`load:${path.split('?')[0]}`);
      const response = await fetch(path); check(response.ok, `读取失败：${path}`);
      const result = new SplatMesh({ stream: response.body, fileType: SplatFileType.PLY, extSplats: true, lod: false, enableLod: false });
      references.add(result); await result.initialized;
      result.maxSh = 3; result.extSplats.setMaxSh(3); result.updateGenerator(); return result;
    };
    const disposeReference = ref => { scene.remove(ref); ref.dispose(); references.delete(ref); };
    const start = performance.now(); mesh = await load('/scene.ply'); references.delete(mesh); scene.add(mesh);
    report.loadMs = performance.now() - start;
    check(mesh.numSplats === protocol.count && mesh.extSplats.getNumSh() === 3, '数量／SH不一致');
    const attributeHashes = async () => {
      const arrays = [...mesh.extSplats.extArrays, ...['sh1', 'sh2', 'sh3a', 'sh3b'].map(k => mesh.extSplats.extra[k])];
      const result = []; for (const a of arrays) result.push(await hash(new Uint8Array(a.buffer, a.byteOffset, a.byteLength))); return result;
    };
    const pose = view => {
      const n = 0.01, f = 1000, fx = view.intrinsic[0][0] * w / view.width, fy = view.intrinsic[1][1] * h / view.height;
      const cx = view.intrinsic[0][2] * w / view.width, cy = view.intrinsic[1][2] * h / view.height;
      camera.near = n; camera.far = f;
      camera.projectionMatrix.set(2 * fx / w, 0, 1 - 2 * cx / w, 0, 0, 2 * fy / h, 2 * cy / h - 1, 0, 0, 0, -(f + n) / (f - n), -2 * f * n / (f - n), 0, 0, -1, 0);
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
      new THREE.Matrix4().makeScale(1, -1, -1).multiply(new THREE.Matrix4().set(...view.camera_from_normalized.flat())).invert()
        .decompose(camera.position, camera.quaternion, camera.scale);
      camera.updateMatrixWorld(true); scene.updateMatrixWorld(true);
    };
    const draw = async (capture = false) => {
      scene.updateMatrixWorld(true); camera.updateMatrixWorld(true); renderer.getDrawingBufferSize(spark.renderSize);
      await spark.update({ scene, camera }); check(!spark.sorting && !spark.sortDirty, '排序未完成');
      renderer.render(scene, camera);
      check(!shaderError && !gl.isContextLost() && gl.getError() === gl.NO_ERROR, 'GPU错误');
      if (capture) { const pixels = new Uint8Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels); check(gl.getError() === gl.NO_ERROR, '读回失败'); return pixels; }
    };
    const screenshot = name => report.images.push({ name, data: renderer.domElement.toDataURL('image/png') });
    pose(protocol.views[0]); stage('initial-draw'); await draw();
    stage('hash-before'); const beforeHashes = await attributeHashes(); stage('capture');
    const extraction = performance.now(), source = await captureP0Source(mesh, protocol.sha256); report.extractMs = performance.now() - extraction;
    stage('source-centers'); const raw = await fetch('/centers.bin'); check(raw.ok, '原始中心流失败');
    let carry = new Uint8Array(0), id = 0;
    for await (const chunk of raw.body) {
      const bytes = new Uint8Array(carry.length + chunk.length); bytes.set(carry); bytes.set(chunk, carry.length);
      const complete = Math.floor(bytes.length / 12) * 12, v = new DataView(bytes.buffer);
      for (let offset = 0; offset < complete; offset += 12, id++) {
        check(id < source.count, '原行过多');
        for (let k = 0; k < 3; k++) check(Number.isFinite(source.geometry[id * 11 + k]) && v.getFloat32(offset + k * 4, true) === source.geometry[id * 11 + k], `源中心错位：${id}`);
      }
      carry = bytes.slice(complete);
    }
    check(!carry.length && id === source.count, '原行数量不一致');
    report.identity.rowsChecked = id; report.identity.duplicateRows = [];
    // Independent one-row PLY references retain the original source ID, even when encodings collide.
    for (const { ids } of protocol.duplicateCenters) for (const sourceId of ids) {
      const reference = await load(`/rows.ply?ids=${sourceId}`);
      stage(`duplicate-gpu:${sourceId}`); verifyP0EncodedRow(mesh.extSplats, sourceId, reference.extSplats);
      const { center, scales } = reference.extSplats.getSplat(0), distance = Math.max(scales.x, scales.y, scales.z, 0.00001) * 30;
      mask = new P0AlphaMask(mesh); const only = new Uint8Array(Math.ceil(source.count / 8)); only[sourceId >> 3] |= 1 << (sourceId & 7); mask.update(only);
      renderer.setClearColor(0xffffff, 1); const views = [];
      for (const angle of [0, 0.35]) {
        camera.fov = 50; camera.aspect = w / h; camera.near = distance / 100; camera.far = distance * 100; camera.updateProjectionMatrix();
        camera.position.copy(center).add(new THREE.Vector3(Math.sin(angle) * distance, 0, Math.cos(angle) * distance)); camera.lookAt(center);
        const isolated = await draw(true); check(isolated.some((v, i) => i % 4 !== 3 && v < 250), '单行参照不可见，不能通过空图');
        scene.remove(mesh); scene.add(reference); const expected = await draw(true);
        scene.remove(reference); scene.add(mesh);
        const delta = difference(isolated, expected); check(delta.max <= protocol.pixelTolerance, `GPU源索引不符：${sourceId}`); views.push({ angle, difference: delta });
      }
      mask.dispose(); mask = undefined; renderer.setClearColor(0, 1); pose(protocol.views[0]); await draw();
      verifyP0EncodedRow(mesh.extSplats, sourceId, reference.extSplats);
      report.identity.duplicateRows.push({ sourceId, exactEncoding: true, views }); disposeReference(reference);
    }
    report.identity.status = 'passed_source_index_representation'; report.stages.push('identity');
    worker = new Worker('/src/gaussianSelection.worker.ts', { type: 'module' });
    const call = (message, transfer = []) => new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(Error('Worker超时')), 6000);
      worker.onerror = event => { clearTimeout(timer); reject(Error(event.message)); };
      worker.onmessage = event => { clearTimeout(timer); resolve(event.data); }; worker.postMessage(message, transfer);
    });
    check((await call({ type: 'source', source, modelGeneration: 1 }, [source.geometry.buffer])).type === 'ready', 'Worker未就绪');
    let sequence = 0; const crossSelections = [], baselines = [];
    for (const [viewIndex, view] of protocol.views.entries()) {
      pose(view); const baseline = await draw(true), noise = difference(baseline, await draw(true)); check(noise.max <= protocol.pixelTolerance, '真实基线不稳定');
      baselines.push(baseline); screenshot(`baseline-${view.image_id}`);
      mask = new P0AlphaMask(mesh); check(difference(baseline, await draw(true)).max <= protocol.pixelTolerance, '全保留改变画面');
      for (const [cx, cy] of protocol.roiCenters) {
        const n = protocol.roiSize / 2;
        const request = { sourceSha256: protocol.sha256, modelGeneration: 1, cameraGeneration: viewIndex, sequence: ++sequence, width: w, height: h,
          modelToView: new THREE.Matrix4().multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld).elements,
          projection: camera.projectionMatrix.elements.slice(), polygon: [[cx - n, cy - n], [cx + n, cy - n], [cx + n, cy + n], [cx - n, cy + n]], visible: p0FullMask(mesh.numSplats), layerTolerance: 0.02 };
        const began = performance.now(), response = await call({ type: 'select', request });
        check(response.type === 'result', `选择失败：${JSON.stringify(response)}`);
        const result = response.result;
        check(result.sourceSha256 === request.sourceSha256 && result.modelGeneration === 1 && result.cameraGeneration === viewIndex && result.sequence === sequence, '过期选集');
        const record = { image_id: view.image_id, center: [cx, cy], wallMs: performance.now() - began, elapsedMs: result.elapsedMs, selectedCount: result.selectedCount };
        report.selections.push(record); check(record.wallMs <= protocol.maxSelectionMs, '选择超过硬预算');
        if (result.selectedCount) {
          const visible = p0FullMask(mesh.numSplats), ids = [];
          for (let id = 0; id < mesh.numSplats; id++) if (result.selected[id >> 3] & (1 << (id & 7))) { visible[id >> 3] &= ~(1 << (id & 7)); ids.push(id); }
          const started = performance.now(); mask.update(visible); const hidden = await draw(true); record.hideMs = performance.now() - started;
          check(difference(baseline, hidden).max > 1, '非空选择没有可见响应');
          mask.update(p0FullMask(mesh.numSplats)); record.restore = difference(baseline, await draw(true)); check(record.restore.max <= protocol.pixelTolerance, '撤销不同');
          if (viewIndex === protocol.crossViewSelectionView && crossSelections.length < protocol.crossViewSelectionLimit) crossSelections.push({ ids, visible, center: [cx, cy] });
        }
      }
      mask.dispose(); mask = undefined; await draw();
    }
    report.selectionP95Ms = p95(report.selections.map(r => r.wallMs));
    check(report.selectionP95Ms <= protocol.targetP95Ms, '选择端到端P95未达标');
    check(report.selections.filter(r => r.hideMs).every(r => r.hideMs <= protocol.maxMaskResponseMs), '隐藏响应未达标');
    check(crossSelections.length === protocol.crossViewSelectionLimit, '缺少跨视角样本');
    for (const [index, selection] of crossSelections.entries()) {
      const reference = await load(`/retained.ply?ids=${selection.ids.join(',')}`);
      check(reference.numSplats === mesh.numSplats - selection.ids.length, '独立保留行数量不符');
      mask = new P0AlphaMask(mesh); mask.update(selection.visible);
      for (const [viewIndex, view] of protocol.views.entries()) {
        pose(view); const hidden = await draw(true); screenshot(`cross-${index}-${view.image_id}-hidden`);
        scene.remove(mesh); scene.add(reference); const expected = await draw(true); scene.remove(reference); scene.add(mesh);
        const delta = difference(hidden, expected), changed = difference(hidden, baselines[viewIndex]);
        check(delta.max <= protocol.pixelTolerance, '真实跨视角mask与原行保留模型不符');
        report.crossViews.push({ selection: index, center: selection.center, selectedCount: selection.ids.length, image_id: view.image_id, referenceDifference: delta, visualChange: changed });
      }
      mask.update(p0FullMask(mesh.numSplats)); for (const [viewIndex, view] of protocol.views.entries()) { pose(view); check(difference(baselines[viewIndex], await draw(true)).max <= protocol.pixelTolerance, '跨视角撤销失败'); }
      mask.dispose(); mask = undefined; disposeReference(reference); await draw();
    }
    report.stages.push('cross-view');
    const resource = label => report.resources.push({ label, timeMs: performance.now(), jsHeapUsed: performance.memory?.usedJSHeapSize ?? null,
      textures: renderer.info.memory.textures, geometries: renderer.info.memory.geometries, programs: renderer.info.programs.length, contextLost: gl.isContextLost() });
    const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
    const motion = async () => {
      pose(protocol.views[0]); const orientation = camera.quaternion.clone(), axis = new THREE.Vector3(0, 1, 0), frames = []; let previous;
      for (let i = -15; i < protocol.motionFrames; i++) {
        const timestamp = await frame(); camera.quaternion.copy(orientation).multiply(new THREE.Quaternion().setFromAxisAngle(axis, protocol.motionYawRadians * Math.sin(i / 20))); await draw();
        if (i > 0) frames.push(timestamp - previous); previous = timestamp;
      }
      return { p95Ms: p95(frames), frames };
    };
    report.navigation = { runs: [] };
    for (const mode of ['baseline', 'edit', 'edit', 'baseline']) {
      if (mode === 'edit') mask = new P0AlphaMask(mesh);
      const result = await motion(); report.navigation.runs.push({ mode, ...result });
      if (mask) { mask.dispose(); mask = undefined; await draw(); }
    }
    const baseP95 = p95(report.navigation.runs.filter(r => r.mode === 'baseline').flatMap(r => r.frames));
    const editP95 = p95(report.navigation.runs.filter(r => r.mode === 'edit').flatMap(r => r.frames));
    Object.assign(report.navigation, { baselineP95Ms: baseP95, editP95Ms: editP95, overheadRatio: editP95 / baseP95,
      targetPassed: editP95 <= protocol.maxFrameP95Ms && editP95 / baseP95 <= protocol.maxEditOverheadRatio, scope: 'P5性能目标的短时诊断，非产品入口验收' });
    pose(protocol.views[0]); await draw(); mask = new P0AlphaMask(mesh); await draw(); resource('soak-start');
    const soakStart = performance.now(); let cycles = 0, lastSample = soakStart;
    while (performance.now() - soakStart < protocol.soakMs || cycles < protocol.soakHideRestoreCycles) {
      if (cycles < protocol.soakHideRestoreCycles) {
        mask.update(crossSelections[cycles % crossSelections.length].visible); await draw();
        mask.update(p0FullMask(mesh.numSplats)); check(difference(baselines[0], await draw(true)).max <= protocol.pixelTolerance, '连续撤销失败'); cycles++;
      } else { await frame(); await draw(); }
      if (performance.now() - lastSample >= 10000) { resource(`soak-${cycles}`); stage(`soak:${Math.round(performance.now() - soakStart)}`); lastSample = performance.now(); }
    }
    resource('soak-end'); report.soak = { durationMs: performance.now() - soakStart, hideRestoreCycles: cycles, scope: '两分钟技术检查；不替代P5二十分钟产品验收' };
    check(report.resources.every(r => !r.contextLost && r.textures === report.resources[0].textures && r.geometries === report.resources[0].geometries), 'GPU资源计数增长／context丢失');
    mask.dispose(); mask = undefined; await draw();
    const afterHashes = await attributeHashes(); check(beforeHashes.every((v, i) => v === afterHashes[i]), '修改了原始几何／SH数组');
    report.attributesUnchanged = true; report.attributeHashes = beforeHashes; report.stages.push('short-soak'); report.status = 'passed';
  } catch (error) { report.status = 'failed'; report.errors.push(String(error)); }
  finally {
    worker?.terminate(); mask?.dispose(); references.forEach(ref => ref.dispose()); mesh?.dispose(); spark?.dispose(); renderer?.dispose(); renderer?.forceContextLoss(); renderer?.domElement.remove();
  }
  return report;
}
