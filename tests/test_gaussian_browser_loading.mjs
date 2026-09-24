import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import {runInNewContext} from 'node:vm';
import {AbortablePromise} from '../frontend/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js';

const source = readFileSync(new URL('../scripts/audit_gaussian_browser.mjs', import.meta.url), 'utf8');
const start = source.indexOf('const page = `');
const page = runInNewContext(source.slice(start, source.indexOf('\ntry {', start)) + '\npage;', {selfTest: false});
const code = page.split('<script type="module">')[1].split('</script>')[0].replace(/^import .*;$/gm, '');

for (const phase of ['download', 'progressive build']) {
  test(`legacy audit reports ${phase} rejection without waiting for browser timeout`, {timeout: 1000}, async () => {
    const failure = new Error(`${phase} failed`);
    class Viewer {
      getSplatMesh() { return {onSplatTreeReady() {}}; }
      addSplatScene() {
        if (phase === 'progressive build') {
          this.splatSceneDownloadAndBuildPromise = new AbortablePromise((_resolve, reject) => reject(failure));
        }
        return new AbortablePromise((resolve, reject) => phase === 'download' ? reject(failure) : resolve());
      }
    }
    class Renderer {
      domElement = {};
      setPixelRatio() {} setSize() {} setClearColor() {}
    }
    const window = {};
    runInNewContext(code, {window, document: {body: {style: {}, appendChild() {}}},
      THREE: {PerspectiveCamera: class {}, WebGLRenderer: Renderer},
      GS: {Viewer, RenderMode: {OnChange: 1}}});
    await assert.rejects(window.loadScene({width: 128, height: 128}, 3, 'controlled'), error => error === failure);
  });
}
