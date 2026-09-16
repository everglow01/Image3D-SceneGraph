import assert from 'node:assert/strict';
import {test} from 'node:test';
import {spawnSync} from 'node:child_process';
import {assertHardware, MOTION_PROTOCOL, motionTrial} from '../scripts/gaussian_browser_motion.mjs';

test('hardware guard rejects software fallback and wrong device', () => {
  assert.doesNotThrow(() => assertHardware('ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA L2), NVIDIA)'));
  for (const name of [null,'SwiftShader','ANGLE (NVIDIA L2 software)','NVIDIA RTX','llvmpipe'])
    assert.throws(() => assertHardware(name));
});

test('frozen closed-loop protocol keeps capture pass outside performance', () => {
  assert.equal(MOTION_PROTOCOL.frames,240);
  assert.equal(MOTION_PROTOCOL.trials,3);
  assert.equal(MOTION_PROTOCOL.translation,0.01);
  const source=motionTrial.toString();
  assert.match(source,/!captureImages/);
  assert.match(source,/GPU_DISJOINT_EXT/);
  assert.match(source,/await settle\(\)/);
  assert.equal(spawnSync(process.execPath,['--input-type=module','--check'],{input:`(${source});`}).status,0);
});

test('camera selection is deterministic, disjoint and independent of quality scores', () => {
  const code = `import importlib.util
s=importlib.util.spec_from_file_location('freeze','scripts/freeze_spark_acceptance.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d={'splits':{'validation':[str(i) for i in range(1,2000)],'test':['3000']}}
a=m.select_views(d)
assert len(a)==len(set(a))==12 and not set(a)&m.EXCLUDED
assert a[0]=='1' and a[-1]=='1999'
d['splits']['validation'].reverse()
assert m.select_views(d)==a
d['splits']['test']=['1']
try: m.select_views(d)
except ValueError: pass
else: raise AssertionError('Test overlap accepted')
`;
  const result=spawnSync('python3',['-B','-c',code],{encoding:'utf8'});
  assert.equal(result.status,0,result.stderr);
});
