import test from 'node:test';
import assert from 'node:assert/strict';
import {displayedSites, measurement, exportSites} from '../staging/web/core.js';
const data={model:{version:'experimental'},sites:{features:Array.from({length:55},(_,i)=>({type:'Feature',properties:{global_rank:55-i,candidate_id:String(i)},geometry:{type:'Point',coordinates:[1,2]}}))}};
test('same budget for map/list/export; stable ranking',()=>{const sites=displayedSites(data,'25');assert.equal(sites.length,25);assert.equal(sites[0].properties.global_rank,1);assert.deepEqual(exportSites(data,'25').features,sites);assert.equal(displayedSites(data,'all').length,55);});
test('unavailable does not become invented zero',()=>{assert.equal(measurement(null),'Unavailable');assert.equal(measurement(undefined),'Unavailable');assert.equal(measurement(0,' m'),'0 m');});
test('export warning and uncalibrated experimental metadata',()=>{const result=exportSites(data,'25');assert.equal(result.experimental,true);assert.match(result.score_interpretation,/not a probability/);assert.equal(result.field_verification_notice,'Recommendations require field verification.');assert.equal(result.features[0].geometry.coordinates[0],1);});
