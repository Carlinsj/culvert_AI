export const WARNING='Recommendations require field verification.';
// Existing supported-area coordinates; navigation only, never model features.
export const SUPPORTED_AREAS=Object.freeze([
  {id:'beaver',name:'Beaver Kill area',latitude:42.07647,longitude:-74.207172},
  {id:'rock',name:'Rock Ledge Ln',latitude:41.880519,longitude:-74.110507},
  {id:'beech',name:'Beech Hill Rd',latitude:42.020639,longitude:-74.765347},
  {id:'braden',name:'Braden Rd',latitude:41.719523,longitude:-74.519277},
]);
export function roadLabel(name){const value=String(name??'').trim();return !value||/^\d+$/.test(value)?'Unnamed road':value;}
export function roadOptions(data){
  const options=new Map();
  for(const f of data?.analysis?.roads?.features||[]){const p=f.properties,key=String(p.name||p.id||'');if(key)options.set(key,{id:key,name:p.name||'Unnamed road',count:0});}
  for(const f of data?.sites?.features||[]){const p=f.properties,key=String(p.route_id??'');if(!key)continue;if(!options.has(key))options.set(key,{id:key,name:p.road_name||key,count:0});options.get(key).count++;}
  for(const option of options.values())option.name=roadLabel(option.name);
  const sorted=[...options.values()].sort((a,b)=>a.name.localeCompare(b.name)||a.id.localeCompare(b.id));
  let unnamed=0;for(const option of sorted)if(option.name==='Unnamed road')option.name+=` ${++unnamed}`;
  return sorted;
}
export function distanceMeters(a,b) {
  const rad=Math.PI/180,dy=(b.lat-a.lat)*rad,dx=(b.lng-a.lng)*rad;
  const h=Math.sin(dy/2)**2+Math.cos(a.lat*rad)*Math.cos(b.lat*rad)*Math.sin(dx/2)**2;
  return 6371000*2*Math.atan2(Math.sqrt(Math.min(1,h)),Math.sqrt(Math.max(0,1-h)));
}
export function formattedDistance(m,unit='m') {return `${(unit==='ft'?m/0.3048:m).toLocaleString(undefined,{maximumFractionDigits:2})} ${unit}`;}
export function measurement(v,suffix='') {return v===null||v===undefined||!Number.isFinite(Number(v))?'Not available':Number(v).toLocaleString(undefined,{maximumFractionDigits:3})+suffix;}
export function priority(rank) {return rank<=25?'High Priority':rank<=100?'Medium Priority':'Lower Priority';}
export function priorityColor(rank) {return rank<=25?'#cb513d':rank<=100?'#d69435':'#578575';}
export function evidenceSummary(p) {const labels={physical_crossing:'Drainage crossing',profile_local_peak:'Road-profile depression',flow_valley_local_peak:'Flow / valley maximum',ml_local_peak:'Experimental score peak'};let keys=[];try{keys=JSON.parse(p.localization_evidence||'[]');}catch{}return keys.map(k=>labels[k]||k).join(' · ')||'Limited physical evidence';}
export function shownSites(data,budget,road='') {const all=[...(data?.sites?.features||[])].filter(f=>!road||String(f.properties.route_id)===road).sort((a,b)=>a.properties.global_rank-b.properties.global_rank||a.properties.candidate_id.localeCompare(b.properties.candidate_id));return budget==='all'?all:all.slice(0,Number(budget));}
export function endpoint(feature) {const [lng,lat]=feature.geometry.coordinates;return {lat,lng,siteId:feature.properties.physical_site_id||feature.properties.candidate_id};}
export function featureDistanceMeters(a,b){
  const ap=a?.properties||{},bp=b?.properties||{};
  if([ap.x_m,ap.y_m,bp.x_m,bp.y_m].every(Number.isFinite))return Math.hypot(ap.x_m-bp.x_m,ap.y_m-bp.y_m);
  return distanceMeters(endpoint(a),endpoint(b));
}
export function oneToOneMatches(predictions,actuals,tolerance=20){
  const edges=predictions.map(prediction=>actuals.map((actual,index)=>[index,featureDistanceMeters(prediction,actual)]).filter(([,d])=>d<=tolerance).sort((a,b)=>a[0]-b[0]));
  const assigned=new Map();
  function augment(predictionIndex,seen){for(const [actualIndex] of edges[predictionIndex]){if(seen.has(actualIndex))continue;seen.add(actualIndex);if(!assigned.has(actualIndex)||augment(assigned.get(actualIndex),seen)){assigned.set(actualIndex,predictionIndex);return true;}}return false;}
  predictions.forEach((_,index)=>augment(index,new Set()));
  return [...assigned].map(([actualIndex,predictionIndex])=>({actualIndex,predictionIndex,distance_m:featureDistanceMeters(predictions[predictionIndex],actuals[actualIndex])})).sort((a,b)=>a.predictionIndex-b.predictionIndex||a.actualIndex-b.actualIndex);
}
export function nearestVisible(point,features,project,radius=14) {
  const target=project(point);let best=null,bestD=radius;
  for(const f of features) {const p=endpoint(f),q=project(p),d=Math.hypot(target.x-q.x,target.y-q.y);if(d<bestD||(d===bestD&&p.siteId<(best?.siteId||'\uffff'))) {best=p;bestD=d;}}
  return best;
}
export class RulerState {
  constructor(){this.active=false;this.pins=[null,null];this.unit='m';this.snap=true;}
  start(from=null){this.active=true;if(from)this.pins=[{...from},null];}
  place(point){if(!this.active)return false;this.set(this.pins[0]?1:0,point);return true;}
  set(index,point){if(!Number.isFinite(point.lat)||!Number.isFinite(point.lng)||Math.abs(point.lat)>90||Math.abs(point.lng)>180)throw Error('Use valid endpoint coordinates.');this.pins[index]={lat:point.lat,lng:point.lng,siteId:point.siteId||null};}
  reset(){this.pins=[null,null];this.active=true;}
  clear(){this.pins=[null,null];}
  exit(){this.active=false;}
  get distance(){return this.pins.every(Boolean)?distanceMeters(...this.pins):null;}
}
export function evidenceLines(p,model) {
  const lines=[];
  if(p.originated_at_sampling_anchor)lines.push('Coordinate originated at a regular road-sampling anchor;not an independent geometric detection.');
  if(p.physical_crossing)lines.push(`Two-sided road/drainage crossing from ${p.proposal_kind}. A crossing does not confirm a culvert.`);
  if(p.profile_local_peak)lines.push(`Local road-profile depression maximum: ${measurement(p.route_elevation_dip_60m,' m')};compared along this road within60 m.`);
  if(p.flow_valley_local_peak)lines.push(`Local upstream-area maximum with concave terrain: ${measurement(p.upstream_area_m2,' m²')};negative TPI ${measurement(p.negative_tpi_9x9_m,' m')}.`);
  if(!p.physical_crossing&&!p.profile_local_peak&&!p.flow_valley_local_peak)lines.push('Physical support is limited. This comparison-mode site may be only an ordinary road sample.');
  if(p.ml_local_peak)lines.push('Compatible experimental RF score is a local maximum. This alone does not admit a feature-driven proposal.');
  if(p.global_rank)lines.push(`${priority(p.global_rank)} is relative queue position,not confidence or a probability.`);
  lines.push(`Score source:${model?.version||'Not available'};fixed policy:${model?.ranking_policy||'Not available'}. Frozenv1 is not used.`);
  return lines;
}
export function exportedSites(data,budget,notes=new Map(),road='') {
  return {type:'FeatureCollection',experimental:true,mode:data?.mode,localization_policy:data?.localization_policy,
    model:data?.model,grouping:data?.grouping,grouping_notice:data?.grouping_notice,field_verification_notice:WARNING,score_interpretation:'Uncalibrated ranking score,not a probability.',
    road_filter:road||null,features:shownSites(data,budget,road),local_staging_field_notes:[...notes.values()]};
}
