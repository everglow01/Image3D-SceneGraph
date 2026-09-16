import assert from 'node:assert/strict';

export const MOTION_PROTOCOL = Object.freeze({frames:240, warmup:60, trials:3,
  translation:0.01, yaw_degrees:3, sample_every:20, thumbnail_width:320,
  path:'camera-local sinusoidal roundtrip, one pose per animation frame; not constant wall-clock speed'});

export function assertHardware(renderer) {
  assert(typeof renderer === 'string' && /NVIDIA.*L2/i.test(renderer) &&
    !/swiftshader|llvmpipe|software/i.test(renderer), 'NVIDIA L2 hardware rendering required; no software fallback');
}

// Serialized into the audit page; it uses only browser globals and the pinned audit hook.
export async function motionTrial(view, protocol, captureImages) {
  const {THREE, renderer, camera, step, settle} = window.auditMotion;
  await window.drawView(view);
  const gl = renderer.getContext();
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const device = debug && gl.getParameter(debug.UNMASKED_RENDERER_WEBGL);
  if (!device || !/NVIDIA.*L2/i.test(device) || /swiftshader|llvmpipe|software/i.test(device))
    throw Error('hardware renderer lost');
  const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  const origin = camera.position.clone(), rotation = camera.quaternion.clone();
  const axis = new THREE.Vector3(1,0,0).applyQuaternion(rotation);
  const localY = new THREE.Vector3(0,1,0);
  const times = [], cpu = [], queries = [], gpu = [], samples = [];
  let last, disjoint = false;
  const thumb = document.createElement('canvas');
  thumb.width = protocol.thumbnail_width;
  thumb.height = Math.round(view.height * thumb.width / view.width);
  const ctx = thumb.getContext('2d');
  const image = () => {
    ctx.drawImage(renderer.domElement,0,0,thumb.width,thumb.height);
    return thumb.toDataURL('image/png').split(',')[1];
  };
  const pose = i => {
    const a = Math.sin(2*Math.PI*i/(protocol.frames-1));
    camera.position.copy(origin).addScaledVector(axis,protocol.translation*a);
    camera.quaternion.copy(rotation).multiply(new THREE.Quaternion().setFromAxisAngle(localY,
      protocol.yaw_degrees*Math.PI/180*a));
    camera.updateMatrixWorld(true);
  };
  const poll = () => {
    if (timer && gl.getParameter(timer.GPU_DISJOINT_EXT)) disjoint = true;
    for (const item of queries) if (!item.done && gl.getQueryParameter(item.q,gl.QUERY_RESULT_AVAILABLE)) {
      gpu.push({frame:item.frame,ms:gl.getQueryParameter(item.q,gl.QUERY_RESULT)/1e6});
      gl.deleteQuery(item.q);item.done=true;
    }
  };
  for (let i=-protocol.warmup;i<protocol.frames;i++) {
    const now = await new Promise(requestAnimationFrame);
    if (i>=0 && last!==undefined) times.push(now-last);
    last=now;
    pose(Math.max(0,i));
    const q=timer && i>=0 && !captureImages ? gl.createQuery() : null;
    const start=performance.now();
    if(q)gl.beginQuery(timer.TIME_ELAPSED_EXT,q);
    step();
    if(q){gl.endQuery(timer.TIME_ELAPSED_EXT);queries.push({q,frame:i,done:false});}
    if(i>=0)cpu.push(performance.now()-start);
    poll();
    if(gl.isContextLost() || gl.getError()!==gl.NO_ERROR)throw Error('motion WebGL/context error');
    if(captureImages && i>=0 && (i%protocol.sample_every===0 || i===protocol.frames-1)) {
      samples.push({frame:i,position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),png:image()});
    }
  }
  await settle();
  for(let i=0;i<120 && queries.some(q=>!q.done);i++) {await new Promise(requestAnimationFrame);poll();}
  const missing=queries.filter(q=>!q.done).length;
  for(const item of queries)if(!item.done)gl.deleteQuery(item.q);
  if(captureImages) {
    for(const sample of samples){
      camera.position.fromArray(sample.position);camera.quaternion.fromArray(sample.quaternion);camera.updateMatrixWorld(true);
      await settle();
      sample.settled_png=image();
    }
  }
  if(disjoint)gpu.length=0;
  return {device,protocol,capture_images:captureImages,raf_ms:times,cpu_submission_ms:cpu,
    gpu_queries:gpu,gpu_disjoint:disjoint,gpu_timer_available:!!timer,gpu_queries_missing:missing,samples,
    scope:'headless rAF pacing and GPU command time, no presentation/display latency; capture pass excluded from performance'};
}
