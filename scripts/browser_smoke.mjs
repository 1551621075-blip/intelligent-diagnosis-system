#!/usr/bin/env node
// Run: node scripts/browser_smoke.mjs
// Requires local Node >= 20, .venv Python dependencies, and installed Google Chrome.
// All runtime data/profile files are temporary. No downloads, mocks, or token logs.
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile, readdir, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const python = join(repo, '.venv/bin/python');
const temp = await mkdtemp(join(tmpdir(), 'diagnosis-browser-smoke-'));
let stage = 'bootstrap', backend, chrome, cdp, credentials = [], ready;
let failed = false, failedCheck;
const evidence = [];
class SmokeFailure extends Error {}
const requireThat = (condition, label) => { if (!condition) throw new SmokeFailure(label); };
const deadline = Date.now() + 180_000;
const childEnvironment = {
  PATH: '/usr/bin:/bin', HOME: temp, TMPDIR: temp,
  PYTHONPATH: join(repo, 'src'), PYTHONDONTWRITEBYTECODE: '1',
  OMP_NUM_THREADS: '1', OPENBLAS_NUM_THREADS: '1', MPLCONFIGDIR: join(temp, 'mpl'),
};
const bootstrap = String.raw`
import json, logging, socket, sys
from pathlib import Path
import uvicorn
from intelligent_diagnosis.api import create_app
from intelligent_diagnosis.ml import train_models
from intelligent_diagnosis.synthetic import make_request
logging.disable(logging.CRITICAL)
root = Path(sys.argv[1])
app = create_app(root / "runtime", root / "models")
store, service = app.state.store, app.state.service
tokens = {role: store.provision_user("smoke-" + role, role)
          for role in ("operator", "expert", "admin")}
metadata = train_models(root / "models", n_devices=96)
service.register_model(metadata, store.authenticate(tokens["admin"]))
requests = [make_request("battery_depleted", seed=912,
            device_id="smoke-device-" + str(i), request_id="browser-diagnosis-" + str(i)
            ).model_dump(mode="json") for i in range(2)]
# This local-only inspection route reads real persisted state; never supplies service results.
@app.get("/__smoke__/state", include_in_schema=False)
def snapshot():
    with store.connect() as db:
        return {
            "services": {r["service_id"]: json.loads(r["payload"])
                         for r in db.execute("SELECT service_id,payload FROM services")},
            "assignments": {r["diagnosis_id"]: json.loads(r["payload"])
                            for r in db.execute("SELECT diagnosis_id,payload FROM assignments")},
            "settings": {r["key"]: r["value"] for r in db.execute("SELECT * FROM settings")},
        }
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
sock.listen(128)
# Captured anonymous parent pipe only, never forwarded to terminal or a log file.
print(json.dumps({"port": sock.getsockname()[1], "tokens": tokens,
                  "version": metadata["version"], "requests": requests}), flush=True)
uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", log_config=None,
               access_log=False, log_level="critical")).run(sockets=[sock])
`;

async function until(check, label, timeout = 12_000) {
  const end = Math.min(deadline, Date.now() + timeout);
  while (Date.now() < end) {
    if (await check()) return;
    await delay(50);
  }
  throw new Error(label);
}

class ChromePipe {
  constructor(child) {
    this.child = child; this.next = 0; this.pending = new Map(); this.listeners = [];
    let buffered = '';
    child.stdio[4].on('data', chunk => {
      buffered += chunk.toString();
      let index;
      while ((index = buffered.indexOf('\0')) >= 0) {
        const packet = JSON.parse(buffered.slice(0, index));
        buffered = buffered.slice(index + 1);
        if (packet.id) {
          const pending = this.pending.get(packet.id);
          if (!pending) continue;
          this.pending.delete(packet.id); clearTimeout(pending.timer);
          if (packet.error) pending.reject(new Error('Chrome protocol failure'));
          else pending.resolve(packet.result);
        } else this.listeners.forEach(listener => listener(packet));
      }
    });
    child.stdio[3].on('error', () => {});
  }
  send(method, params = {}, sessionId) {
    return new Promise((resolve, reject) => {
      const id = ++this.next;
      const timer = setTimeout(() => {
        this.pending.delete(id); reject(new Error('Chrome protocol timeout'));
      }, 12_000);
      this.pending.set(id, {resolve, reject, timer});
      this.child.stdio[3].write(JSON.stringify({id, method, params, sessionId}) + '\0');
    });
  }
  close() {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer); pending.reject(new Error('Chrome closed'));
    }
    this.pending.clear();
  }
}

async function stop(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise(resolve => child.once('exit', resolve));
  child.kill('SIGTERM');
  const timer = setTimeout(() => child.kill('SIGKILL'), 3_000);
  await exited; clearTimeout(timer);
}

async function scanForCredentials(directory) {
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) await scanForCredentials(path);
    else if (entry.isFile()) {
      const bytes = await readFile(path);
      requireThat(!credentials.some(value => bytes.includes(Buffer.from(value)) ||
        bytes.includes(Buffer.from(value, 'utf16le'))), 'credential persisted to temporary disk');
    }
  }
}

// A hard stop also covers bootstrap/training failures and guarantees child cleanup.
const watchdog = setTimeout(() => {
  failed = true;
  backend?.kill('SIGTERM'); chrome?.kill('SIGTERM');
}, 180_000);
try {
  stage = 'real synthetic training and server startup';
  backend = spawn(python, ['-u', '-c', bootstrap, temp], {
    cwd: temp, env: childEnvironment, stdio: ['ignore', 'pipe', 'ignore'],
  });
  ready = await new Promise((resolve, reject) => {
    let data = '';
    const timer = setTimeout(() => reject(new Error('bootstrap timeout')), 90_000);
    backend.once('error', () => { clearTimeout(timer); reject(new Error('Python unavailable')); });
    backend.once('exit', () => { clearTimeout(timer); reject(new Error('bootstrap exited')); });
    backend.stdout.on('data', chunk => {
      data += chunk.toString();
      if (data.includes('\n')) {
        clearTimeout(timer);
        try { resolve(JSON.parse(data.slice(0, data.indexOf('\n')))); }
        catch { reject(new Error('bootstrap did not return readiness')); }
        data = '';
      }
    });
  });
  credentials = Object.values(ready.tokens);
  const origin = `http://127.0.0.1:${ready.port}`;
  await until(async () => {
    try { return (await fetch(origin + '/health')).ok; } catch { return false; }
  }, 'server not ready');
  stage = 'headless browser launch';
  await mkdir(join(temp, 'profile'));
  chrome = spawn(chromePath, [
    '--headless=new', '--remote-debugging-pipe', `--user-data-dir=${join(temp, 'profile')}`,
    '--no-first-run', '--no-default-browser-check', '--disable-background-networking',
    '--disable-component-update', '--disable-sync', '--disable-extensions',
    '--disable-default-apps', '--disable-domain-reliability',
    '--disable-client-side-phishing-detection', '--disable-breakpad', '--disable-crash-reporter',
    '--password-store=basic', '--use-mock-keychain', '--disable-features=OptimizationHints,MediaRouter',
    '--proxy-server=http://127.0.0.1:9', '--proxy-bypass-list=127.0.0.1',
    '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1', 'about:blank',
  ], {cwd: temp, env: childEnvironment, stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe']});
  chrome.on('error', () => {});
  cdp = new ChromePipe(chrome);
  const version = await cdp.send('Browser.getVersion');
  const {targetId} = await cdp.send('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await cdp.send('Target.attachToTarget', {targetId, flatten: true});
  const send = (method, params) => cdp.send(method, params, sessionId);
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    requireThat(!result.exceptionDetails, 'browser evaluation failed');
    return result.result.value;
  };
  const q = JSON.stringify;
  const requests = [];
  const requestMap = new Map();
  let browserExceptions = 0, blockedRequests = 0, failedRequests = 0, badAuth = 0, leakedUrl = 0;
  cdp.listeners.push(packet => {
    if (packet.sessionId !== sessionId) return;
    const p = packet.params;
    if (packet.method === 'Runtime.exceptionThrown') browserExceptions++;
    if (packet.method === 'Fetch.requestPaused') {
      const allowed = p.request.url.startsWith(origin + '/');
      if (!allowed) blockedRequests++;
      send(allowed ? 'Fetch.continueRequest' : 'Fetch.failRequest', allowed ?
        {requestId: p.requestId} : {requestId: p.requestId, errorReason: 'BlockedByClient'}).catch(() => {});
    }
    if (packet.method === 'Network.requestWillBeSent') {
      if (credentials.some(value => p.request.url.includes(value))) leakedUrl++;
      const url = new URL(p.request.url);
      const headers = p.request.headers;
      const authorization = headers.Authorization || headers.authorization;
      if (url.pathname.startsWith('/api/') && !credentials.some(value => authorization === `Bearer ${value}`)) badAuth++;
      // Never retain or log request headers (which contain the bearer credential).
      const row = {path: url.pathname, method: p.request.method, body: p.request.postData};
      requestMap.set(p.requestId, row); requests.push(row);
    }
    if (packet.method === 'Network.responseReceived') {
      const row = requestMap.get(p.requestId);
      if (row) row.status = p.response.status;
    }
    if (packet.method === 'Network.loadingFinished') {
      const row = requestMap.get(p.requestId);
      if (row?.path.startsWith('/api/')) send('Network.getResponseBody', {requestId: p.requestId})
        .then(response => { try { row.response = JSON.parse(response.body); } catch {} row.done = true; })
        .catch(() => { row.done = true; });
    }
    if (packet.method === 'Network.loadingFailed') failedRequests++;
  });
  await send('Runtime.enable'); await send('Page.enable'); await send('Network.enable');
  await send('Fetch.enable', {patterns: [{urlPattern: '*', requestStage: 'Request'}]});
  await send('Emulation.setDeviceMetricsOverride', {width: 1280, height: 900, deviceScaleFactor: 1, mobile: false});
  await send('Page.navigate', {url: origin + '/'});
  await until(() => evaluate('!!document.getElementById("login-form") && document.readyState === "complete"'), 'page load');
  const fill = async (id, value) => evaluate(`document.getElementById(${q(id)}).value=${q(value)}; true`);
  const click = async selector => {
    const point = await evaluate(`(() => {const e=document.querySelector(${q(selector)}); if(!e || e.disabled || !e.checkVisibility()) return null; e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
    requireThat(point, 'control not visible or enabled');
    await send('Input.dispatchMouseEvent', {type: 'mousePressed', button: 'left', clickCount: 1, ...point});
    await send('Input.dispatchMouseEvent', {type: 'mouseReleased', button: 'left', clickCount: 1, ...point});
  };
  const action = async (selector, path, status = 200) => {
    const start = requests.length;
    await click(selector);
    let row;
    await until(() => {
      row = requests.slice(start).find(item => item.path === path && item.done);
      return !!row;
    }, 'form request not completed');
    requireThat(row.status === status, 'unexpected real API response status');
    await until(() => evaluate(`!document.querySelector(${q(selector)}).disabled`), 'form still pending');
    return row;
  };
  const memoryOnly = async () => {
    const state = await evaluate(`(async () => ({local:localStorage.length,session:sessionStorage.length,
      cookies:document.cookie,dbs:(await indexedDB.databases()).length,caches:(await caches.keys()).length,
      input:document.getElementById('token-input').value, clean:!${q(credentials)}.some(v =>
      document.documentElement.outerHTML.includes(v) || location.href.includes(v) || window.name.includes(v))}))()`);
    requireThat(state.local === 0 && state.session === 0 && state.cookies === '' && state.dbs === 0 &&
      state.caches === 0 && state.input === '' && state.clean, 'credential storage or DOM leak');
    requireThat((await send('Network.getAllCookies')).cookies.length === 0, 'browser cookies not empty');
  };
  const login = async role => {
    await fill('token-input', ready.tokens[role]);
    await action('#login-form button[type=submit]', '/api/me');
    await until(() => evaluate(`document.getElementById('identity').textContent === ${q(`smoke-${role} / ${role}`)} && !document.getElementById('workspace').hidden`), 'identity not rendered');
    await until(() => evaluate('!document.getElementById("login-form").querySelector("button[type=submit]").disabled'), 'login pending');
    await memoryOnly();
  };
  const snapshot = async () => (await fetch(origin + '/__smoke__/state')).json();
  const text = id => evaluate(`document.getElementById(${q(id)}).textContent`);
  const assertTextPayload = async id => {
    requireThat((await text(id)).includes(payload), 'literal payload missing');
    requireThat(await evaluate(`!document.getElementById(${q(id)}).querySelector('img,script,svg') && !window.__smokeExecuted`), 'payload interpreted as markup');
  };
  const payload = '<img src=/__smoke_xss__ onerror=window.__smokeExecuted=1>';
  evidence.push(`Actual ${version.product}; application HTML, JavaScript and stylesheet loaded`);

  stage = 'admin model activation form';
  await login('admin');
  await fill('model-version', ready.version);
  const activation = await action('#activate-form button', '/api/models/activate');
  requireThat(JSON.parse(activation.body).version === ready.version, 'activation shape');
  requireThat((await snapshot()).settings.active_model === ready.version, 'model not activated');
  evidence.push('Real LightGBM training (96 synthetic devices), registration and browser-admin activation');

  stage = 'operator diagnosis forms';
  await login('operator');
  requireThat(await evaluate(`!document.querySelector('#close-form').checkVisibility() && !document.querySelector('#rollback-form').checkVisibility() && document.getElementById('knowledge-json').readOnly`), 'operator role controls');
  await fill('example-class', 'battery_depleted');
  await action('#load-example', '/api/examples/battery_depleted');
  requireThat(JSON.parse(await evaluate(`document.getElementById('request-json').value`)).samples.length > 0, 'real example not loaded');
  const diagnoses = [];
  for (const request of ready.requests) {
    await fill('request-json', JSON.stringify(request));
    const row = await action('#diagnosis-form button', '/api/diagnoses');
    const result = row.response;
    requireThat(result.prediction?.class_id === 'battery_depleted' && result.dispatch_candidates.length > 0 &&
      result.model_version === ready.version && result.evidence.length > 0, 'real diagnosis missing evidence or dispatch');
    requireThat(JSON.parse(await text('result-json')).diagnosis_id === result.diagnosis_id, 'diagnosis not rendered');
    diagnoses.push(result);
  }
  evidence.push('Operator example loader and two real model/evidence/dispatch diagnosis submissions');

  stage = 'expert assignment close forms and resources';
  await login('expert');
  requireThat(await evaluate(`!document.querySelector('#rollback-form').checkVisibility()`), 'expert sees admin rollback');
  for (const [index, outcome] of ['cancelled', 'completed'].entries()) {
    const diagnosis = diagnoses[index];
    const selector = `#diagnosis-list button[data-smoke-selected]`;
    await evaluate(`(() => { for(const e of document.querySelectorAll('#diagnosis-list button')) {e.removeAttribute('data-smoke-selected'); if(e.textContent.includes(${q(diagnosis.diagnosis_id)})) e.setAttribute('data-smoke-selected','');} })()`);
    await action(selector, `/api/diagnoses/${diagnosis.diagnosis_id}`);
    const serviceId = diagnosis.dispatch_candidates[0].service_id;
    const part = diagnosis.attribution.required_part;
    requireThat(part, 'fixture does not reserve a part');
    const before = (await snapshot()).services[serviceId];
    await fill('service-id', serviceId);
    const assigned = await action('#assign-submit', `/api/diagnoses/${diagnosis.diagnosis_id}/assign`);
    requireThat(assigned.response.status === 'synthetic_assignment_confirmed' && assigned.response.repair_completed === false, 'assignment result');
    const reserved = (await snapshot()).services[serviceId];
    requireThat(reserved.active_jobs === before.active_jobs + 1 && reserved.parts[part] === before.parts[part] - 1, 'resources not reserved');
    await fill('close-outcome', outcome); await fill('close-note', payload);
    const closed = await action('#close-form button', `/api/diagnoses/${diagnosis.diagnosis_id}/assignment/close`);
    const body = JSON.parse(closed.body);
    requireThat(Object.keys(body).sort().join(',') === 'action_id,note,outcome' && body.outcome === outcome && body.note === payload && !!body.action_id, 'close request shape');
    requireThat(closed.response.status === `synthetic_${outcome}` && closed.response.repair_completed === (outcome === 'completed'), 'close response');
    await assertTextPayload('operation-result');
    const final = await snapshot();
    requireThat(final.services[serviceId].active_jobs === before.active_jobs && final.services[serviceId].parts[part] === before.parts[part] - (outcome === 'completed' ? 1 : 0), 'close resource accounting');
    requireThat(final.assignments[diagnosis.diagnosis_id].status === `synthetic_${outcome}`, 'close not persisted');
    await action('#close-form button', `/api/diagnoses/${diagnosis.diagnosis_id}/assignment/close`);
    requireThat(JSON.stringify((await snapshot()).services) === JSON.stringify(final.services), 'close retry double released');
    await fill('close-id', `browser-terminal-${index}`);
    const terminal = await action('#close-form button', `/api/diagnoses/${diagnosis.diagnosis_id}/assignment/close`, 409);
    requireThat(terminal.response.error.code === 'assignment_terminal' && (await text('notice')).includes('assignment_terminal'), 'terminal error not rendered');
  }
  await action('#refresh-audit', '/api/audit');
  await assertTextPayload('audit-list');
  evidence.push('Expert close: cancel restores slot + part; complete restores slot only; retries idempotent; terminal 409 displayed');
  evidence.push('HTML-shaped close notes rendered literally in real operation responses and audit, without markup execution');

  stage = 'admin controlled rollback error';
  await login('admin');
  const rollbackVersion = 'sim-lgbm-' + '0'.repeat(20);
  await fill('rollback-version', rollbackVersion); await fill('rollback-id', 'browser-rollback');
  await fill('rollback-reason', 'Synthetic browser verification of unapproved history');
  const rollback = await action('#rollback-form button', '/api/models/rollback', 409);
  const rollbackBody = JSON.parse(rollback.body);
  requireThat(Object.keys(rollbackBody).sort().join(',') === 'action_id,reason,version' &&
    rollbackBody.version === rollbackVersion && rollbackBody.action_id === 'browser-rollback' &&
    rollbackBody.reason === 'Synthetic browser verification of unapproved history', 'rollback request shape');
  requireThat(rollback.response.error.code === 'rollback_unapproved' && (await text('notice')).includes('rollback_unapproved'), 'real rollback error not rendered');
  requireThat((await snapshot()).settings.active_model === ready.version, 'failed rollback changed active model');
  evidence.push('Admin rollback sends exact schema; real rollback_unapproved HTTP 409 rendered; active model unchanged');

  stage = 'logout refresh memory-only verification';
  await memoryOnly();
  await click('#logout');
  requireThat(await evaluate(`document.getElementById('workspace').hidden && document.getElementById('identity').textContent === '尚未登录' && document.getElementById('operation-result').textContent === '' && document.getElementById('diagnosis-list').children.length === 0`), 'logout did not clear identity/data');
  await memoryOnly();
  const afterLogout = requests.length;
  await click('#load-example').then(() => { throw new Error('hidden control unexpectedly clickable'); }, () => {});
  requireThat(requests.length === afterLogout, 'logout sent authenticated request');
  await login('admin');
  const beforeReload = requests.length;
  await send('Page.reload', {ignoreCache: true});
  await until(() => evaluate(`document.readyState === 'complete' && document.getElementById('workspace')?.hidden && document.getElementById('identity').textContent === '尚未登录'`), 'reload retained session');
  await memoryOnly();
  requireThat(!requests.slice(beforeReload).some(row => row.path.startsWith('/api/')), 'reload automatically authenticated');
  requireThat(browserExceptions === 0 && blockedRequests === 0 && failedRequests === 0 && badAuth === 0 && leakedUrl === 0, 'browser runtime/network/auth check failed');
  requireThat(requests.some(row => row.path === '/static/app.js' && row.status === 200) && requests.some(row => row.path === '/static/style.css' && row.status === 200), 'static UI assets not loaded');
  evidence.push('All three roles authenticated; credentials only in Authorization; storage/cookies/IndexedDB/caches empty; logout and reload clear session');
  evidence.push('No JavaScript exceptions, failed requests, external page requests, or credential-bearing URLs');
  stage = 'temporary file credential scan';
  await stop(chrome); cdp.close(); await stop(backend);
  await scanForCredentials(temp);
  evidence.push('Post-shutdown temporary runtime/profile scan found no raw credentials (UTF-8/UTF-16)');
} catch (error) {
  // Only our static assertion labels are safe; never expose raw errors/packets/headers.
  failed = true;
  if (error instanceof SmokeFailure) failedCheck = error.message;
} finally {
  clearTimeout(watchdog);
  await stop(chrome); cdp?.close(); await stop(backend);
  await rm(temp, {recursive: true, force: true, maxRetries: 10, retryDelay: 100});
  credentials.fill('');
  if (ready) ready.tokens = {};
}
console.log(JSON.stringify({status: failed ? 'FAIL' : 'PASS', ...(failed ? {stage, failedCheck} : {}), evidence,
  limitations: ['Successful historical model rollback not tested; the runner targets rollback_unapproved rejection.',
    'Desktop Chrome smoke only; no pixel comparison, telemetry, knowledge publishing, or exhaustive security proof.'],
  cleanup: 'Local server/browser stopped; temporary runtime and fresh Chrome profile removed.'}, null, 2));
process.exitCode = failed ? 1 : 0;
