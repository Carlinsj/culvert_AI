import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {fileURLToPath} from 'node:url';

const config=JSON.parse(readFileSync(new URL('../package.json',import.meta.url),'utf8'));
function assertSafeDev(){
  assert.equal(config.scripts.dev,'bash scripts/serve_web.sh');
  assert.equal(config.scripts.start,config.scripts.dev,'Normal npm commands must open the same research workspace');
  for(const hook of ['predev','postdev','prestart','poststart'])assert.equal(config.scripts[hook],undefined,'Startup must not invoke generation/training hooks');
}
test('both normal commands start one research workspace without generation hooks',assertSafeDev);
test('shared launcher uses the project Python and canonical research entry point',()=>{
  const launcher=readFileSync(new URL('./serve_web.sh',import.meta.url),'utf8');
  assert.match(launcher,/PYTHONDONTWRITEBYTECODE=1/);
  assert.match(launcher,/exec .*python\.sh.*serve_research\.py.*"\$@"/);
  assert.doesNotMatch(launcher,/http\.server|run_actual|train_model|predict:actual/);
});
for(const command of ['dev','start'])test(`npm ${command} forwards arguments to the normal research server`,()=>{
  assertSafeDev(); // Check lifecycle safety before executing it, even under regression.
  const result=spawnSync('npm',['run',command,'--','--help'],{
    cwd:new URL('..',import.meta.url),encoding:'utf8',timeout:30000,
  });
  assert.equal(result.status,0,result.stderr||result.error?.message);
  assert.match(result.stdout,/serve_research\.py/);
  assert.match(result.stdout,/--port PORT/);
  assert.doesNotMatch(result.stdout,/staging only|ensure_actual_predictions|Running real Census|train_model/);
});
test('direct launcher works outside the repository working directory',()=>{
  assertSafeDev();
  const result=spawnSync('bash',[fileURLToPath(new URL('./serve_web.sh',import.meta.url)),'--help'],{
    cwd:tmpdir(),encoding:'utf8',timeout:30000,
  });
  assert.equal(result.status,0,result.stderr||result.error?.message);
  assert.match(result.stdout,/Culvert AI local research workspace/);
});
test('research UI has no separate comparison-server link or staging branding',()=>{
  const html=readFileSync(new URL('../staging/feature_web/index.html',import.meta.url),'utf8');
  assert.match(html,/RESEARCH WORKSPACE/);
  assert.match(html,/summer 2027, subject to validation/);
  assert.doesNotMatch(html,/href="http:\/\/127\.0\.0\.1:809[345]|STAGING|staging-only/);
  assert.match(html,/id="policy"/);assert.match(html,/id="grouping"/);
  assert.match(html,/Recommendations require field verification\./);
});
