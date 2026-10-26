import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {displayedSites, exportSites, measurement} from '../staging/web/core.js';

const types=['crossing_supported','stream_supported','terrain_supported','road_corridor_fallback'];
const data={model:{version:'hydrology-sites-v2-RF-experimental',candidate_version:'hybrid-sites-v1',ranking_policy:'dynamic_route_z',calibrated:false},sites:{features:Array.from({length:60},(_,i)=>({type:'Feature',geometry:{type:'Point',coordinates:[-74+i/10000,42]},properties:{global_rank:60-i,candidate_id:`c${i}`,physical_site_id:`s${i}`,evidence_type:types[i%4],member_candidate_ids:JSON.stringify([`c${i}`]),model_score:i/100}}))}};

test('hybrid export preserves evidence, actual geometry and fixed policy',()=>{
  const result=JSON.parse(JSON.stringify(exportSites(data,'25')));
  assert.equal(result.features.length,25);
  assert.deepEqual(result.features,displayedSites(data,'25'));
  assert.equal(result.model.ranking_policy,'dynamic_route_z');
  assert.equal(result.model.calibrated,false);
  assert.equal(new Set(result.features.map(f=>f.properties.evidence_type)).size,4);
  assert.match(result.score_interpretation,/not a probability/);
  assert.equal(result.field_verification_notice,'Recommendations require field verification.');
});
test('changing display budget preserves global ranks and never promotes evidence classes',()=>{
  assert.equal(displayedSites(data,'25')[0].properties.global_rank,1);
  assert.equal(displayedSites(data,'50')[49].properties.global_rank,50);
  assert.equal(displayedSites(data,'all').length,60);
  assert.equal(displayedSites(data,'25')[0].properties.evidence_type,'road_corridor_fallback');
});
test('unavailable values stay distinct from valid zero',()=>{
  assert.equal(measurement(null),'Unavailable');
  assert.equal(measurement(0,' m'),'0 m');
});
test('all four diagnostic controls are present and unchecked by default',()=>{
  const html=fs.readFileSync(new URL('../staging/hybrid_web/index.html',import.meta.url),'utf8');
  for(const id of ['candidates','drainage','crossings','consolidated']) {
    const input=html.match(new RegExp(`<input[^>]*id="${id}"[^>]*>`))[0];
    assert.ok(!input.includes('checked'));
  }
  assert.match(html,/Fallback retains search coverage/);
  assert.match(html,/not confirmed culverts/);
});
