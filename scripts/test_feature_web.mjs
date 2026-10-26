import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {distanceMeters,formattedDistance,RulerState,nearestVisible,endpoint,evidenceLines,exportedSites,shownSites,roadOptions,SUPPORTED_AREAS,roadLabel,oneToOneMatches} from '../staging/feature_web/core.js';

const feature=(id,lng,lat,extra={})=>({type:'Feature',geometry:{type:'Point',coordinates:[lng,lat]},properties:{candidate_id:id,physical_site_id:id,global_rank:1,...extra}});
test('geographic distance matches Leaflet Earth radius and units',()=>{
  assert.ok(Math.abs(distanceMeters({lat:0,lng:0},{lat:1,lng:0})-111194.92664455874)<1e-6);
  assert.ok(Math.abs(distanceMeters({lat:60,lng:0},{lat:60,lng:1})-55596.934)<.01);
  assert.equal(distanceMeters({lat:42,lng:-74},{lat:42,lng:-74}),0);
  assert.equal(formattedDistance(.3048,'ft'),'1 ft');
});
test('Pin A/B placement,independent drag,live distance,and no feature mutation',()=>{
  const f=feature('a',-74,42),original=JSON.stringify(f),r=new RulerState();
  assert.equal(r.place({lat:0,lng:0}),false);r.start(endpoint(f));r.place({lat:42.001,lng:-74});
  const a={...r.pins[0]},before=r.distance;r.set(1,{lat:42.002,lng:-74});
  assert.deepEqual(r.pins[0],a);assert.ok(r.distance>before);assert.equal(JSON.stringify(f),original);
  r.set(0,{lat:42.001,lng:-74});assert.ok(r.distance<before*1.01);
});
test('reset,clear,exit,and invalid coordinate handling',()=>{
  const r=new RulerState();r.start({lat:0,lng:0});r.place({lat:1,lng:0});r.exit();assert.equal(r.active,false);assert.ok(r.distance>0);
  r.clear();assert.equal(r.distance,null);r.reset();assert.equal(r.active,true);assert.deepEqual(r.pins,[null,null]);
  assert.throws(()=>r.set(0,{lat:100,lng:0}));assert.throws(()=>r.set(0,{lat:NaN,lng:0}));
});
test('screen-space snapping uses visible sites only and detaches outside radius',()=>{
  const points=[feature('b',.1,.1),feature('a',.1,.1)],project=p=>({x:p.lng*100,y:p.lat*100});
  assert.equal(nearestVisible({lat:.11,lng:.11},points,project).siteId,'a');
  assert.equal(nearestVisible({lat:9,lng:9},points,project),null);assert.equal(nearestVisible({lat:0,lng:0},[],project),null);
});
test('weak and lower-priority explanations do not claim strong physical support',()=>{
  const lines=evidenceLines({global_rank:200,originated_at_sampling_anchor:true,proposal_kind:'internal_road_probe'},{}).join(' ');
  assert.match(lines,/Physical support is limited/);assert.match(lines,/Lower Priority/);assert.doesNotMatch(lines,/strong inspection/);
});
test('GeoJSON export keeps exact sites,scores,rank and provenance without backfill',()=>{
  const f=feature('a',-74,42,{model_score:.7,localization_evidence:'["profile_local_peak"]'});
  const data={mode:'feature',model:{version:'experimental-v2'},sites:{features:[f]}};
  const original=JSON.stringify(data),out=exportedSites(data,25);
  assert.equal(out.features.length,1);assert.deepEqual(out.features[0],f);assert.equal(JSON.stringify(data),original);
  assert.equal(out.field_verification_notice,'Recommendations require field verification.');
});
test('UI layer separation and ruler mouse/touch/keyboard integration contract',()=>{
  const html=fs.readFileSync('staging/feature_web/index.html','utf8');
  assert.equal((html.match(/data-layer=/g)||[]).length,7);assert.equal((html.match(/data-layer="[^"]+" checked/g)||[]).length,1);
  assert.match(html,/Feature-driven sites — Experimental/);assert.match(html,/Raw road-sampling anchors/);
  const ruler=fs.readFileSync('staging/feature_web/ruler.js','utf8'),app=fs.readFileSync('staging/feature_web/app.js','utf8');
  assert.match(ruler,/\.on\('drag'/);assert.match(ruler,/Escape/);assert.match(ruler,/map\.distance/);assert.match(ruler,/draggable:state.active/);
  assert.match(app,/if\(!ruler\.siteClick\(feature\)\)detail/);assert.match(app,/if\(ruler.active\)/);
});
test('ruler and raw/grouped comparison entry points are discoverable without changing default layers',()=>{
  const html=fs.readFileSync('staging/feature_web/index.html','utf8');
  assert.match(html,/id="measureMap"[^>]*>↔ Measure A → B/);
  assert.match(html,/1 · Place marker A/);assert.match(html,/2 · Place marker B/);
  assert.match(html,/id="grouping"/);assert.match(html,/value="legacy20"/);assert.match(html,/value="identity20"/);
  assert.match(html,/Primary map representation/);assert.match(html,/Raw candidates — Diagnostic only/);
  const app=fs.readFileSync('staging/feature_web/app.js','utf8');
  assert.match(app,/data.raw_candidates.features/);assert.match(app,/member_evidence/);assert.match(app,/grouping:\$\('grouping'\).value/);
});
test('main sidebar stays simple while research controls remain collapsed',()=>{
  const html=fs.readFileSync('staging/feature_web/index.html','utf8');
  const main=html.split('<details id="advanced"')[0];
  assert.match(main,/Explore nearby sites/);assert.match(main,/id="areaSelect"/);assert.match(html,/id="roadFilter"/);
  for(const id of ['mode','policy','grouping','latitude','layers','representation'])assert.doesNotMatch(main,new RegExp('id="'+id+'"'));
  assert.doesNotMatch(html,/<details id="advanced"[^>]*\bopen\b/);
  assert.match(html,/Show sites/);assert.match(html,/Recommendations require field verification/);
});
test('physical discovery and inventory remain explicit advanced comparisons',()=>{
  const html=fs.readFileSync('staging/feature_web/index.html','utf8');
  assert.match(html,/value="physical_discovery"/);
  assert.match(html,/id="inventoryLayer"/);
  assert.doesNotMatch(html,/id="inventoryLayer"[^>]*checked/);
  const app=fs.readFileSync('staging/feature_web/app.js','utf8');
  assert.match(app,/\/api\/research\/inventory/);
  assert.match(app,/oneToOneMatches\(predictions,actuals,20\)/);
  assert.match(app,/revision!==inventoryRevision/);
  assert.match(app,/\$\('inventoryLayer'\).onchange=renderInventory/);
});
test('dedicated comparison and performance workflows expose the required controls',()=>{
  const html=fs.readFileSync('staging/feature_web/index.html','utf8');
  for(const name of ['Predictions','Actual vs Predicted','Model Performance'])assert.match(html,new RegExp('>'+name+'<'));
  for(const id of ['comparisonLocation','comparisonRoute','comparisonLatitude','comparisonLongitude','comparisonModel','comparisonBudget','comparisonTolerance','showRecorded','showComparisonPredictions','fitRecorded','nextMissed'])assert.match(html,new RegExp('id="'+id+'"'));
  for(const value of ['rock','beaver','region6','region8','route','coordinates','all'])assert.match(html,new RegExp('value="'+value+'"'));
  const comparison=fs.readFileSync('staging/feature_web/comparison.js','utf8');
  assert.match(comparison,/visual|clusterFeatures/);assert.match(comparison,/Unmatched \/ unverified prediction/);
  assert.match(comparison,/Full population is not a Top-K result/);
  assert.match(comparison,/v5_baseline_preserving/);
  assert.match(comparison,/v6_compact_physical/);
});
test('actual-vs-predicted defaults,legend,and budgets are explicit',()=>{
  const html=fs.readFileSync('staging/feature_web/index.html','utf8');
  assert.match(html,/id="mapLegend"/);assert.match(html,/Matched actual/);assert.match(html,/Missed actual/);
  assert.match(html,/Unmatched \/ unverified/);assert.match(html,/Match ≤20 m/);
  assert.match(html,/<option selected>250<\/option>/);assert.match(html,/<option>500<\/option>/);assert.match(html,/<option>1000<\/option>/);
  assert.match(html,/Full list — diagnostic/);assert.match(html,/id="performancePanel"/);
});
test('active matches are maximum-cardinality one-to-one and preserve inputs',()=>{
  const predictions=[feature('p0',0,0,{x_m:0,y_m:0}),feature('p1',0,0,{x_m:10,y_m:0})];
  const actuals=[feature('a0',0,0,{x_m:9,y_m:0}),feature('a1',0,0,{x_m:19,y_m:0})];
  const before=JSON.stringify([predictions,actuals]),matches=oneToOneMatches(predictions,actuals,10);
  assert.equal(matches.length,2);assert.deepEqual(matches.map(m=>m.actualIndex).sort(),[0,1]);
  assert.equal(JSON.stringify([predictions,actuals]),before);
});
test('multiple supported road areas have stable valid navigation coordinates',()=>{
  assert.equal(SUPPORTED_AREAS.length,4);assert.equal(new Set(SUPPORTED_AREAS.map(a=>a.id)).size,4);
  for(const name of ['Rock Ledge Ln','Beech Hill Rd','Braden Rd'])assert.ok(SUPPORTED_AREAS.some(a=>a.name===name));
  for(const a of SUPPORTED_AREAS){assert.ok(Number.isFinite(a.latitude)&&a.latitude>41&&a.latitude<43);assert.ok(a.longitude>-76&&a.longitude<-73);}
});
test('opaque road IDs stay selectable without appearing as road names',()=>{
  assert.equal(roadLabel('110450787854'),'Unnamed road');assert.equal(roadLabel('US Hwy 209'),'US Hwy 209');
  const options=roadOptions({analysis:{roads:{features:[{properties:{id:'11',name:'11'}},{properties:{id:'22',name:'22'}}]}},sites:{features:[]}});
  assert.deepEqual(options.map(o=>o.id),['11','22']);assert.deepEqual(options.map(o=>o.name),['Unnamed road 1','Unnamed road 2']);
});
test('road filter applies before display cap and retains original ranks/scores/geometry',()=>{
  const a=feature('a',-74,42,{route_id:'A',road_name:'Road A',global_rank:1,model_score:.9});
  const b=feature('b',-74.1,42.1,{route_id:'B',road_name:'Road B',global_rank:200,model_score:.3});
  const c=feature('c',-74.2,42.2,{route_id:'B',road_name:'Road B',global_rank:300,model_score:.2});
  const data={sites:{features:[c,a,b]},analysis:{roads:{features:[{properties:{id:'zero',name:'No proposals road'}}]}}};
  const before=JSON.stringify(data);
  assert.deepEqual(shownSites(data,1,'B'),[b]);assert.deepEqual(shownSites(data,'all','B'),[b,c]);
  assert.equal(shownSites(data,'all','missing').length,0);
  const exported=exportedSites(data,1,new Map(),'B');assert.deepEqual(exported.features,[b]);assert.equal(exported.road_filter,'B');
  assert.equal(JSON.stringify(data),before);
  const options=roadOptions(data);assert.equal(options.find(r=>r.id==='B').count,2);assert.equal(options.find(r=>r.id==='No proposals road').count,0);
});
test('area overview, explicit apply and cancelled-request guards remain wired',()=>{
  const app=fs.readFileSync('staging/feature_web/app.js','utf8');
  assert.match(app,/areaOverview=L.featureGroup/);assert.match(app,/SUPPORTED_AREAS/);
  assert.match(app,/new AbortController/);assert.match(app,/analysisController\?\.abort/);
  assert.match(app,/error.name==='AbortError'/);assert.match(app,/number!==requestNumber/);
  assert.doesNotMatch(app,/\$\('mode'\).onchange=analyze/);
  assert.doesNotMatch(app,/\$\('grouping'\).onchange=analyze/);
  assert.match(app,/browseAreas\(\);\s*$/);
  assert.match(app,/map.on\('resize',\(\)=>\{if\(browsing\)fitAreaOverview\(\)/);
});
