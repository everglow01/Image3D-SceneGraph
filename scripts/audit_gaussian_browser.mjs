#!/usr/bin/env node
// Local-only fixed-camera audit. No model training, external services, or new dependencies.
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {readFile, writeFile, mkdir, mkdtemp, stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const selfTest = args.includes('--self-test');
const output = resolve(option('--output', 'outputs/analysis/browser-render-audit'));
await mkdir(output, {recursive: false});
const profile = await mkdtemp(join(tmpdir(), 'image3d-render-audit-'));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const events = [];
const result = {schema_version: 1, status: 'running', self_test: selfTest,
  profile: 'fixed_camera_browser_v1', test_rgb: 'not_loaded', captures: [], events,
  note: 'Pinned library harness, not a deployed product UI screenshot. Fixed camera, no upright rotation or overlays.'};
let browser, server, socket;
let chromeLog='';
let timeout;

function fixture() {
  const fields = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2',
    ...Array.from({length: 45}, (_, i) => `f_rest_${i}`), 'opacity',
    'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];
  const header = ['ply','format binary_little_endian 1.0','element vertex 6',
    ...fields.map(f => `property float ${f}`), 'end_header',''].join('\n');
  const values = new Float32Array(62);
  values.set([0.3,0.2,2], 0);
  values.set([0.5 / 0.28209479177387814,-0.5 / 0.28209479177387814,-0.5 / 0.28209479177387814],6);
  values[54] = Math.log(0.95 / 0.05);
  values.set([Math.log(0.04),Math.log(0.04),Math.log(0.04),1,0,0,0],55);
  // Viewer 0.4.7 sets the indexed draw range to visible-splat count: keep at least six visible fixtures.
  const rows=[[0.3,0.2],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0,0.5],[0.5,0.5]].map(([x,y])=>{
    const row=values.slice();row[0]=x;row[1]=y;return Buffer.from(row.buffer);
  });
  return Buffer.concat([Buffer.from(header),...rows]);
}

const page = `<!doctype html><meta charset="utf-8"><style>body{margin:0;background:black}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"/three.js"}}</script>
<script type="module">
import * as THREE from '/three.js';
import * as GS from '/splats.js';
window.loadScene = async (view, degree, mode) => {
  document.body.style.width=view.width+'px';document.body.style.height=view.height+'px';
  const camera = new THREE.PerspectiveCamera();
  const renderer = new THREE.WebGLRenderer({antialias:false, preserveDrawingBuffer:true});
  renderer.setPixelRatio(1); renderer.setSize(view.width,view.height);
  renderer.setClearColor(0,1); document.body.appendChild(renderer.domElement);
  const viewer = new GS.Viewer({rootElement:document.body, camera, renderer,
    useBuiltInControls:false, selfDrivenMode:false, sharedMemoryForWorkers:false,
    sphericalHarmonicsDegree:degree, ignoreDevicePixelRatio:true, integerBasedSort:false,
    renderMode:GS.RenderMode.OnChange});
  window.viewer = viewer; window.camera = camera; window.renderer = renderer;
  const threshold = mode==='product' ? Math.floor(0.005*255) : 0;
  await viewer.addSplatScene('/scene.ply', {showLoadingUI:false, progressiveLoad:true,
    splatAlphaRemovalThreshold:threshold});
  const material=viewer.getSplatMesh().material;
  if(mode==='product') {
    const anchor=['float opacity = exp(-0.5 * A) * vColor.a;','float opa = vColor.a;'].find(s=>material.fragmentShader.includes(s));
    if(!anchor) throw Error('product alpha shader anchor missing');
    material.fragmentShader=material.fragmentShader.replace(anchor,'if(vColor.a<0.005) discard;\\n'+anchor);
    material.needsUpdate=true;
  }
  window.drawView = async view => {
    document.body.style.width=view.width+'px';document.body.style.height=view.height+'px';
    renderer.setSize(view.width,view.height);
    camera.near=mode==='product'?0.1:0.01; camera.far=mode==='product'?1000:1e10;
    const [fx,skew,cx]=view.intrinsic[0], [,fy,cy]=view.intrinsic[1];
    const w=view.width,h=view.height,n=camera.near,f=camera.far;
    camera.projectionMatrix.set(2*fx/w,-2*skew/w,1-2*cx/w,0, 0,2*fy/h,2*cy/h-1,0,
      0,0,-(f+n)/(f-n),-2*f*n/(f-n), 0,0,-1,0);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    const cv=new THREE.Matrix4().set(...view.camera_from_normalized.flat());
    const gl=new THREE.Matrix4().makeScale(1,-1,-1).multiply(cv);
    camera.matrix.copy(gl.clone().invert()); camera.matrix.decompose(camera.position,camera.quaternion,camera.scale);
    camera.updateMatrixWorld(true);
    if(viewer.sortPromise) await viewer.sortPromise;
    viewer.updateSplatMesh();
    await viewer.runSplatSort(true,true); if(viewer.sortPromise) await viewer.sortPromise;
    if(viewer.sortRunning || viewer.splatSortCount!==viewer.splatRenderCount) {
      throw Error('full fixed-camera sort did not complete');
    }
    viewer.updateSplatMesh();
    if(!viewer.splatRenderReady || material.uniforms.focal.value.x<=0 || material.uniforms.viewport.value.x!==view.width) {
      throw Error('splat projection uniforms not ready for fixed-camera capture');
    }
    viewer.splatMesh.material.uniforms.fadeInComplete.value=1;
    viewer.forceRenderNextFrame(); viewer.render();
    const glContext=renderer.getContext(), pixels=new Uint8Array(w*h*4);
    glContext.readPixels(0,0,w,h,glContext.RGBA,glContext.UNSIGNED_BYTE,pixels);
    if(!pixels.some((value,index)=>index%4<3 && value>0)) throw Error('blank frame; excluded from quality comparisons');
    const pixel=(x,y)=>Array.from(pixels.slice(((h-1-y)*w+x)*4,((h-1-y)*w+x)*4+4));
    const extension=glContext.getExtension('WEBGL_debug_renderer_info');
    const firstColor=new THREE.Vector4(),firstCenter=new THREE.Vector3();
    viewer.splatMesh.getSplatColor(0,firstColor);viewer.splatMesh.getSplatCenter(0,firstCenter);
    return {png:renderer.domElement.toDataURL('image/png').split(',')[1],
      runtime:{requested_sh:degree,effective_sh:viewer.splatMesh.minSphericalHarmonicsDegree,
        render_ready:viewer.splatRenderReady,first_color:firstColor.toArray(),first_center:firstCenter.toArray(),
        projected_first_center:firstCenter.clone().project(camera).toArray(),
        instance_count:viewer.splatMesh.geometry.instanceCount,render_info:renderer.info.render,
        gl_error:glContext.getError(),
        covariances_sample:Array.from(material.uniforms.covariancesTexture.value.image.data.slice(0,8)),
        shader_uniforms:Object.fromEntries(Object.entries(material.uniforms).filter(([k,v])=>typeof v.value==='number'||k==='focal'||k==='viewport').map(([k,v])=>[k,v.value])),
        count:viewer.splatMesh.getSplatCount(),render_count:viewer.splatRenderCount,sort_running:viewer.sortRunning,
        sorted_count:viewer.splatSortCount,
        width:glContext.drawingBufferWidth,height:glContext.drawingBufferHeight,
        device_pixel_ratio:window.devicePixelRatio,renderer_pixel_ratio:renderer.getPixelRatio(),
        output_color_space:renderer.outputColorSpace,tone_mapping:renderer.toneMapping,
        near:camera.near,far:camera.far,kernel_2d_size:viewer.kernel2DSize,
        gpu:extension?glContext.getParameter(extension.UNMASKED_RENDERER_WEBGL):glContext.getParameter(glContext.RENDERER),
        contains_sh3_calculation:material.vertexShader.includes('SH_C3'),
        alpha_loader_threshold:threshold,alpha_shader_threshold:mode==='product'?0.005:0,
        in_memory_compression:viewer.inMemoryCompressionLevel,
        reference_probe_pixel:pixel(79,74),reference_mirrored_pixel:pixel(79,h-1-74),corner:pixel(0,0),
        projection:camera.projectionMatrix.toArray(),world_to_camera:camera.matrixWorldInverse.toArray()}};
  };
  return true;
};
window.auditReady=true;
</script>`;

try {
  const audit = selfTest ? {source_sha256:{},views:[{image_id:'fixture',width:128,height:128,
    intrinsic:[[100,0,64],[0,100,64],[0,0,1]],camera_from_normalized:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]} :
    JSON.parse(await readFile(resolve(option('--audit')), 'utf8'));
  const ply = selfTest ? null : resolve(option('--ply'));
  if (!selfTest) {
    const hash=createHash('sha256');
    for await(const chunk of createReadStream(ply)) hash.update(chunk);
    result.ply_sha256=hash.digest('hex');
    assert.equal(result.ply_sha256,audit.source_sha256.ply,'PLY hash mismatch');
  }
  const files = new Map([
    ['/splats.js', join(repo,'frontend/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js')],
    ['/three.js',join(repo,'frontend/node_modules/three/build/three.module.js')],
    ['/three.core.js',join(repo,'frontend/node_modules/three/build/three.core.js')],
  ]);
  result.library_sha256=createHash('sha256').update(await readFile(files.get('/splats.js'))).digest('hex');
  server=createServer(async(req,res)=>{
    try {
      const path=new URL(req.url,'http://localhost').pathname;
      if(path==='/'){res.setHeader('Content-Type','text/html');res.end(page);return;}
      if(path==='/scene.ply' && selfTest){res.end(fixture());return;}
      const target=path==='/scene.ply'?ply:files.get(path);
      if(!target){res.writeHead(404);res.end();return;}
      res.setHeader('Content-Length',(await stat(target)).size);
      res.setHeader('Content-Type',path.endsWith('.js')?'text/javascript':'application/octet-stream');
      const stream=createReadStream(target);stream.on('error',()=>res.destroy());stream.pipe(res);
    } catch(e){res.writeHead(500);res.end(String(e));}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const port=server.address().port;
  browser=spawn(option('--chrome','/usr/bin/google-chrome'),['--headless=new','--no-first-run',
    '--no-default-browser-check','--disable-dev-shm-usage','--remote-debugging-port=0',
    '--enable-unsafe-swiftshader',...(args.includes('--software')?['--use-angle=swiftshader']:[]),
    `--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
  result.requested_software_webgl=args.includes('--software');
  browser.stderr.on('data',chunk=>chromeLog+=chunk.toString());
  browser.on('error',error=>events.push({chrome_spawn_error:String(error)}));
  timeout=setTimeout(()=>{events.push({error:'20-minute browser audit timeout'});browser.kill();},20*60*1000);
  let devtools;
  for(let i=0;i<100;i++){
    try {devtools=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break;} catch {}
    if(browser.exitCode!==null)throw Error('Chrome exited: '+chromeLog.slice(-4000));
    await pause(100);
  }
  if(!devtools)throw Error('Chrome DevTools did not start: '+chromeLog.slice(-4000));
  const targets=await (await fetch(`http://127.0.0.1:${devtools}/json/list`)).json();
  socket=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let next=0;const pending=new Map();
  socket.onclose=()=>{for(const p of pending.values())p.reject(Error('Chrome DevTools connection closed'));pending.clear();};
  socket.onmessage=event=>{
    const message=JSON.parse(event.data);
    if(message.id){const p=pending.get(message.id);pending.delete(message.id);if(message.error)p.reject(Error(JSON.stringify(message.error)));else p.resolve(message.result);}
    else if(message.method==='Runtime.exceptionThrown'||message.method==='Log.entryAdded'||message.method==='Runtime.consoleAPICalled')events.push(message);
  };
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{
    const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result?.value;
  };
  await call('Runtime.enable');await call('Log.enable');await call('Page.enable');
  const modes=selfTest?[0,2,3].map(degree=>({degree,mode:'controlled'})):[{degree:3,mode:'product'}, {degree:3,mode:'controlled'},
    {degree:0,mode:'controlled',probe:true},{degree:2,mode:'controlled',probe:true}];
  for(const mode of modes){
    const requestedIds=option('--image-ids',null)?.split(',');
    const selected=mode.probe?audit.views.filter(v=>v.image_id===option('--sh-probe','317')):
      audit.views.filter(v=>!requestedIds || requestedIds.includes(v.image_id));
    assert(selected.length>0,'no selected camera descriptors');
    try {
      const url=`http://127.0.0.1:${port}/?mode=${mode.mode}&degree=${mode.degree}`;
      await call('Page.navigate',{url});
      let ready=false;
      for(let i=0;i<100;i++){try{ready=await evaluate(`window.location.href===${JSON.stringify(url)} && window.auditReady === true`);}catch{}if(ready)break;await pause(100);}
      if(!ready)throw Error('diagnostic page did not initialize');
      await evaluate(`window.loadScene(${JSON.stringify(selected[0])},${mode.degree},${JSON.stringify(mode.mode)})`);
      for(const view of selected){
        const capture=await evaluate(`window.drawView(${JSON.stringify(view)})`);
        const name=`browser-${mode.mode}-sh${mode.degree}-${view.image_id}`;
        await writeFile(join(output,name+'.png'),Buffer.from(capture.png,'base64'),{flag:'wx'});
        result.captures.push({name,image_id:view.image_id,...capture.runtime});
        if(selfTest && mode.degree===0){assert(capture.runtime.reference_probe_pixel[0]>150,'expected red Gaussian not at projected CV pixel');
          assert(capture.runtime.reference_mirrored_pixel[0]<30,'camera Y axis was mirrored');}
        await writeFile(join(output,'browser.json'),JSON.stringify(result,null,2)+'\n');
      }
    } catch(error){result.captures.push({...mode,status:'failed',error:String(error)});if(selfTest)throw error;}
    finally {
      try {
        await evaluate('(async()=>{await window.viewer?.dispose();window.renderer?.dispose();window.renderer?.forceContextLoss();window.viewer=null;window.renderer=null;return true;})()');
        await call('HeapProfiler.collectGarbage');
      } catch(error){events.push({cleanup_error:String(error)});}
    }
  }
  result.status=result.captures.some(c=>c.status==='failed')?'partial':'completed';
  if(result.status==='partial')process.exitCode=2;
  result.chrome_log=chromeLog.slice(-12000);
} catch(error){result.status='failed';result.error=String(error);process.exitCode=1;}
finally {
  clearTimeout(timeout);socket?.close();browser?.kill();server?.closeAllConnections();server?.close();
  result.chrome_profile=profile;
  result.chrome_log=chromeLog.slice(-12000);
  await writeFile(join(output,'browser.json'),JSON.stringify(result,null,2)+'\n');
}
console.log(JSON.stringify({status:result.status,output,captures:result.captures.map(c=>({name:c.name,status:c.status,count:c.count,effective_sh:c.effective_sh,error:c.error}))}));
