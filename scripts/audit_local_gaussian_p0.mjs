import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, open, stat } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { spawn, execFile } from 'node:child_process';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from '../frontend/node_modules/typescript/lib/typescript.js';

import { readP0String } from './gaussian_local_p0_cdp.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [outputArg, mode, protocolArg] = process.argv.slice(2);
if (!outputArg || !['synthetic', 'occlusion', 'real'].includes(mode) || (mode === 'real' && !protocolArg)) {
  throw Error('用法：node scripts/audit_local_gaussian_p0.mjs 新输出目录 synthetic|occlusion|real [冻结协议JSON]');
}
const output = resolve(outputArg), real = mode === 'real';
const protocol = real ? JSON.parse(await readFile(resolve(protocolArg), 'utf8')) : null;
const sourceFile = real ? resolve(repo, protocol.path) : null;
const fileHash = async () => { const h = createHash('sha256'); for await (const b of createReadStream(sourceFile)) h.update(b); return h.digest('hex'); };
let header;
if (real) {
  if (await fileHash() !== protocol.sha256 || (await stat(sourceFile)).size !== protocol.bytes ||
      protocol.bytes !== protocol.headerBytes + protocol.count * 248) throw Error('源文件身份／布局不匹配');
  const f = await open(sourceFile, 'r'); try { header = Buffer.alloc(protocol.headerBytes); await f.read(header, 0, header.length, 0); } finally { await f.close(); }
  if (!header.toString().includes(`element vertex ${protocol.count}\n`) || !header.toString().endsWith('end_header\n')) throw Error('PLY头不匹配');
}
await mkdir(output);
const profile = join(output, 'chrome-profile'); await mkdir(profile);
const report = { status: 'running', mode, events: [] };
const files = new Map([
  ['/three.js', 'frontend/node_modules/three/build/three.module.js'], ['/three.core.js', 'frontend/node_modules/three/build/three.core.js'],
  ['/spark.js', 'frontend/node_modules/@sparkjsdev/spark/dist/spark.module.js'], ['/Pass.js', 'frontend/node_modules/three/examples/jsm/postprocessing/Pass.js'],
  ['/real-probe.js', 'scripts/local_gaussian_p0_browser.mjs'],
  ...['localGaussianP0Probe', 'localGaussianP0OcclusionProbe', 'localGaussianP0Source', 'localGaussianP0Mask', 'localGaussianP0Selection', 'localGaussianP0Worker', 'gaussianSelection.worker']
    .map(n => [`/src/${n}.ts`, `frontend/src/${n}.ts`])
]);
const rewrite = s => s.replace(/from\s*(["'])three\1/g, "from '/three.js'").replace(/from\s*(["'])@sparkjsdev\/spark\1/g, "from '/spark.js'")
  .replace(/from\s*(["'])three\/addons\/postprocessing\/Pass.js\1/g, "from '/Pass.js'");
const server = createServer(async (req, res) => {
  try {
    if (req.method !== 'GET') { res.writeHead(405).end(); return; }
    const url = new URL(req.url, 'http://localhost'), path = url.pathname;
    if (path === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><meta charset="utf-8"><title>P0 验收</title><div id="probe"></div>'); return; }
    if (real && path === '/protocol.json') { res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(protocol)); return; }
    if (real && path === '/scene.ply') { res.setHeader('Content-Length', protocol.bytes); await pipeline(createReadStream(sourceFile), res); return; }
    if (real && (path === '/rows.ply' || path === '/retained.ply')) {
      const rawIds = url.searchParams.get('ids');
      if (!rawIds || !/^\d+(,\d+)*$/.test(rawIds)) throw Error('原行列表不合法');
      const ids = rawIds.split(',').map(Number), selected = new Set(ids);
      if (ids.length > 2048 || selected.size !== ids.length || ids.some(id => !Number.isSafeInteger(id) || id < 0 || id >= protocol.count)) throw Error('原行列表越界');
      const count = path === '/rows.ply' ? ids.length : protocol.count - ids.length;
      const newHeader = Buffer.from(header.toString().replace(`element vertex ${protocol.count}`, `element vertex ${count}`));
      res.setHeader('Content-Length', newHeader.length + count * 248); res.write(newHeader);
      if (path === '/rows.ply') {
        const f = await open(sourceFile, 'r'); try { for (const id of ids) { const row = Buffer.alloc(248); await f.read(row, 0, 248, protocol.headerBytes + id * 248); res.write(row); } } finally { await f.close(); }
        res.end(); return;
      }
      let id = 0, carry = Buffer.alloc(0);
      await pipeline(createReadStream(sourceFile, { start: protocol.headerBytes, highWaterMark: 248 * 4096 }), async function* (input) {
        for await (const chunk of input) {
          const bytes = Buffer.concat([carry, chunk]), n = Math.floor(bytes.length / 248), out = Buffer.allocUnsafe(n * 248); let kept = 0;
          for (let row = 0; row < n; row++, id++) if (!selected.has(id)) { bytes.copy(out, kept * 248, row * 248, (row + 1) * 248); kept++; }
          carry = bytes.subarray(n * 248); yield out.subarray(0, kept * 248);
        }
        if (carry.length || id !== protocol.count) throw Error('原行流截断');
      }, res); return;
    }
    if (real && path === '/centers.bin') {
      res.setHeader('Content-Length', protocol.count * 12); let carry = Buffer.alloc(0), count = 0;
      await pipeline(createReadStream(sourceFile, { start: protocol.headerBytes, highWaterMark: 248 * 4096 }), async function* (input) {
        for await (const chunk of input) {
          const bytes = Buffer.concat([carry, chunk]), n = Math.floor(bytes.length / 248), out = Buffer.allocUnsafe(n * 12);
          for (let row = 0; row < n; row++) bytes.copy(out, row * 12, row * 248, row * 248 + 12);
          count += n; carry = bytes.subarray(n * 248); yield out;
        }
        if (carry.length || count !== protocol.count) throw Error('中心流截断');
      }, res); return;
    }
    if (!files.has(path)) { res.writeHead(404).end(); return; }
    let text = await readFile(join(repo, files.get(path)), 'utf8');
    if (path.endsWith('.ts')) text = ts.transpileModule(text, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
    res.setHeader('Content-Type', 'text/javascript'); res.end(rewrite(text));
  } catch (e) { if (!res.headersSent) res.writeHead(500); res.end(String(e)); }
});
let browser, socket, timer, stageTimer, resourceTimer, chromeLog = '';
report.deviceMemorySamples = [];
const sampleMemory = () => execFile('nvidia-smi', ['--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], { timeout: 5000 }, async (error, stdout) => {
  const mem = await readFile('/proc/meminfo', 'utf8');
  report.deviceMemorySamples.push({ time: new Date().toISOString(), gpuMiB: error ? null : stdout.trim(), availableRamKiB: Number(mem.match(/MemAvailable:\s+(\d+)/)?.[1]), scope: '整机采样含其他进程，不是本浏览器精确峰值' });
});
const pause = ms => new Promise(r => setTimeout(r, ms));
try {
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  sampleMemory(); resourceTimer = setInterval(sampleMemory, 10000);
  browser = spawn('/usr/bin/google-chrome-stable', ['--enable-precise-memory-info', '--headless=new', '--no-first-run', '--no-default-browser-check', '--ozone-platform=x11', '--use-angle=vulkan', '--disable-software-rasterizer', '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'],
    { env: { ...process.env, __NV_PRIME_RENDER_OFFLOAD: '1', __GLX_VENDOR_LIBRARY_NAME: 'nvidia', __VK_LAYER_NV_optimus: 'NVIDIA_only' }, stdio: ['ignore', 'ignore', 'pipe'] });
  browser.stderr.on('data', b => { chromeLog += b; });
  timer = setTimeout(() => browser.kill(), real ? 600000 : 180000);
  let port;
  for (let i = 0; i < 100; i++) { try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; break; } catch {} if (browser.exitCode !== null) throw Error('Chrome退出'); await pause(100); }
  if (!port) throw Error('缺少DevTools端口');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let serial = 0; const pending = new Map();
  socket.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id) { const p = pending.get(m.id); pending.delete(m.id); if (m.error) p.reject(Error(JSON.stringify(m.error))); else p.resolve(m.result); }
    else {
      report.events.push(m);
      if (m.method === 'Runtime.consoleAPICalled' && m.params.args[0]?.value === 'P0阶段') {
        report.lastStage = m.params.args[1].value; console.log(`P0阶段：${report.lastStage}`);
        clearTimeout(stageTimer); stageTimer = setTimeout(() => { report.timeoutStage = report.lastStage; browser.kill(); }, 60000);
      }
    }
  };
  socket.onclose = event => { report.socketClose = { code: event.code, reason: event.reason }; for (const p of pending.values()) p.reject(Error(`DevTools关闭：${event.code} ${event.reason}`)); pending.clear(); };
  const call = (method, params = {}) => new Promise((resolve, reject) => { const id = ++serial; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
  await call('Runtime.enable'); await call('Log.enable'); await call('Page.enable');
  await call('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` });
  for (let i = 0; i < 100; i++) { const r = await call('Runtime.evaluate', { expression: 'document.readyState+":"+!!document.getElementById("probe")', returnByValue: true }); if (r.result?.value === 'complete:true') break; await pause(100); }
  const expressions = { real: '(await import("/real-probe.js")).run()', synthetic: '(await import("/src/localGaussianP0Probe.ts")).runLocalGaussianP0Probe(document.getElementById("probe"))',
    occlusion: '(await import("/src/localGaussianP0OcclusionProbe.ts")).runLocalGaussianP0OcclusionProbe(document.getElementById("probe"))' };
  const start = performance.now();
  const expression = `(async()=>{const r=window.p0AuditResult=await ${expressions[mode]};window.p0AuditJson=JSON.stringify({...r,images:r.images?.map(({name})=>({name}))});return window.p0AuditJson.length;})()`;
  const result = await call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  report.elapsedMs = performance.now() - start;
  if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
  report.probe = JSON.parse(await readP0String(call, 'window.p0AuditJson')); report.status = report.probe.status;
  for (const [index, image] of (report.probe.images ?? []).entries()) {
    if (!/^[a-z0-9-]+$/.test(image.name)) throw Error('截图名不合法');
    const data = await readP0String(call, `window.p0AuditResult.images[${index}].data`);
    if (!data.startsWith('data:image/png;base64,')) throw Error('截图格式不合法');
    await writeFile(join(output, `${image.name}.png`), Buffer.from(data.split(',')[1], 'base64'), { flag: 'wx' });
  }
  if (report.status !== 'passed') process.exitCode = 1;
} catch (error) { report.status = 'failed'; report.error = String(error); process.exitCode = 1; }
finally {
  clearTimeout(timer); clearTimeout(stageTimer); clearInterval(resourceTimer); socket?.close();
  if (browser && browser.exitCode === null) { browser.kill(); await new Promise(r => { browser.once('exit', r); setTimeout(r, 5000); }); }
  server.closeAllConnections(); await new Promise(r => server.close(r));
  if (real) { report.sourceAfterSha256 = await fileHash(); report.sourceUnchanged = report.sourceAfterSha256 === protocol.sha256; if (!report.sourceUnchanged) { report.status = 'integrity_failed'; process.exitCode = 1; } }
  report.chromeLog = chromeLog;
  await writeFile(join(output, 'result.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ status: report.status, output, error: report.error }));
}
