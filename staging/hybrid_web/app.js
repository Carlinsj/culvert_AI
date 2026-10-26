import {displayedSites, measurement, exportSites} from './core.js';

const $ = id => document.getElementById(id);
const svg = $('map-svg'), ns = 'http://www.w3.org/2000/svg';
const colors = {crossing_supported:'#174f3e',stream_supported:'#347caf',terrain_supported:'#a77a2c',road_corridor_fallback:'#717b78'};
let data = null, center = null, dragging = null;

function element(tag, attributes, parent = svg) {
  const node = document.createElementNS(ns,tag);
  for (const [key,value] of Object.entries(attributes)) node.setAttribute(key,value);
  parent.append(node);
  return node;
}
function project([lon,lat]) {
  return [(lon-center.longitude)*Math.cos(center.latitude*Math.PI/180)*111320,-(lat-center.latitude)*111320];
}
function path(geometry,color,width) {
  if (geometry.type==='MultiLineString') {
    for (const coords of geometry.coordinates) path({type:'LineString',coordinates:coords},color,width);
  } else if (geometry.type==='LineString') {
    element('path',{d:geometry.coordinates.map((c,i)=>(i?'L':'M')+project(c).join(',')).join(' '),fill:'none',stroke:color,'stroke-width':width,'vector-effect':'non-scaling-stroke'});
  }
}
function screenScale() {return svg.viewBox.baseVal.width/Math.min(svg.clientWidth,svg.clientHeight);}
function marker(feature,color,radius,interactive=false) {
  const [x,y]=project(feature.geometry.coordinates);
  const node=element('circle',{cx:x,cy:y,r:radius*screenScale(),'data-radius-px':radius,fill:color,stroke:interactive?'#fff':'none','stroke-width':2,'vector-effect':'non-scaling-stroke'});
  if (!interactive) return;
  node.setAttribute('role','button'); node.setAttribute('tabindex','0');
  node.setAttribute('aria-label',`Proposed site rank ${feature.properties.global_rank}`);
  node.onclick=()=>detail(feature);
  node.onkeydown=e=>{if(e.key==='Enter')detail(feature);};
  element('title',{},node).textContent=`${feature.properties.evidence_type} · rank ${feature.properties.global_rank}`;
}
function diagnosticPoints(features,color,width) {
  element('path',{d:features.map(f=>{const [x,y]=project(f.geometry.coordinates);return `M${x},${y}h.1`;}).join(' '),stroke:color,'stroke-width':width,'stroke-linecap':'round','vector-effect':'non-scaling-stroke',opacity:.5});
}
function render() {
  if (!data) return;
  svg.replaceChildren();
  for (const road of data.analysis.roads.features) path(road.geometry,'#bdc7bc',2);
  if ($('drainage').checked) {
    for (const water of data.hydrology.mapped.features) path(water.geometry,'#3c83b2',3);
    for (const water of data.hydrology.derived.features) path(water.geometry,'#60ada3',1.4);
  }
  if ($('candidates').checked) diagnosticPoints(data.candidates.features,'#bd9677',3);
  if ($('consolidated').checked) diagnosticPoints(data.sites.features,'#46504b',5);
  if ($('crossings').checked) for(const site of data.hydrology.intersections.features) marker(site,'#cbad57',3);
  const shown=displayedSites(data,$('budget').value);
  if ($('recommended').checked) for(const site of shown.toReversed()) marker(site,colors[site.properties.evidence_type]||'#717b78',6,true);
  $('results').replaceChildren();
  for (const site of shown) {
    const p=site.properties, button=document.createElement('button');
    button.className='result'; button.style.borderLeft=`5px solid ${colors[p.evidence_type]}`;
    button.textContent=`${p.global_rank}. ${p.road_name} · ${p.evidence_type.replaceAll('_',' ')} · ${measurement(p.model_score)}`;
    button.onclick=()=>detail(site); $('results').append(button);
  }
  const counts=data.diagnostics;
  $('status').textContent=`${shown.length} displayed / ${data.site_count} proposed sites · ${counts.raw_candidates} raw candidates · ${counts.merged_candidates} merged proposals · ${counts.crossing_proposals} crossings. Map, list and export share the display budget. Fallbacks remain unverified.`;
}
function detail(feature) {
  const p=feature.properties,panel=$('detail'); panel.hidden=false; panel.replaceChildren();
  const close=document.createElement('button');close.textContent='Close';close.onclick=()=>panel.hidden=true;panel.append(close);
  const heading=document.createElement('h2');heading.textContent=`#${p.global_rank} · ${p.road_name}`;panel.append(heading);
  const explanation=document.createElement('p');explanation.textContent=p.selection_reason;panel.append(explanation);
  const dl=document.createElement('dl');
  for(const [label,value] of [['Site evidence',p.evidence_type],['Representative evidence',p.representative_evidence_type],['Raw RF score',measurement(p.model_score)],['Ordering priority',measurement(p.priority)],['Fixed policy',p.ranking_policy],['Sources',p.proposal_sources],['Flow accumulation',measurement(p.flow_accumulation_cells,' cells')],['Upstream area',measurement(p.upstream_area_m2,' m²')],['Road depression',measurement(p.route_elevation_dip_60m,' m')],['Slope',measurement(p.slope_degrees,'°')],['Mapped stream distance',measurement(p.distance_to_nearest_stream_m,' m')],['Member proposals',p.member_count],['Missing measurements',p.missing_feature_count],['Model',p.model_version],['Site ID',p.physical_site_id]]) {
    const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=String(value??'Unavailable');dl.append(dt,dd);
  }
  panel.append(dl);const warning=document.createElement('p');warning.textContent='Recommendations require field verification. Neither evidence class nor RF score confirms a culvert.';panel.append(warning);
}
async function analyze() {
  const buttons=['analyze','locate','supported'].map($);buttons.forEach(b=>b.disabled=true);$('export').disabled=true;
  $('status').textContent='Measuring high-recall proposals and terrain… No training or production changes.';
  try {
    const response=await fetch('/api/analysis/nearby',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:{latitude:Number($('latitude').value),longitude:Number($('longitude').value)},radius_m:Number($('radius').value),policy:$('policy').value})});
    const payload=await response.json();if(!response.ok)throw Error(payload.error);
    data=payload;center=data.analysis;const radius=center.radius_m;
    svg.setAttribute('viewBox',`${-radius} ${-radius} ${radius*2} ${radius*2}`);$('detail').hidden=true;render();$('export').disabled=false;
  } catch(error) {
    data=null;svg.replaceChildren();$('results').replaceChildren();$('detail').hidden=true;
    $('status').textContent=error.message+' Choose the supported Beaver Kill area to return within scope.';
  } finally {buttons.forEach(b=>b.disabled=false);}
}
$('analyze').onclick=analyze;
$('supported').onclick=()=>{$('latitude').value=42.07647;$('longitude').value=-74.207172;analyze();};
$('locate').onclick=()=>{
  if(!navigator.geolocation){$('status').textContent='Location unavailable. Enter coordinates or use the supported-area button.';return;}
  navigator.geolocation.getCurrentPosition(p=>{$('latitude').value=p.coords.latitude;$('longitude').value=p.coords.longitude;analyze();},()=>{$('status').textContent='Location not available. Enter coordinates or explore a supported area.';});
};
for(const id of ['budget','recommended','candidates','drainage','crossings','consolidated'])$(id).onchange=render;
$('export').onclick=()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(exportSites(data,$('budget').value),null,2)],{type:'application/geo+json'}));
  const link=document.createElement('a');link.href=url;link.download='hybrid-proposed-inspection-sites.geojson';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
function resizeMarkers(){for(const node of svg.querySelectorAll('circle[data-radius-px]'))node.setAttribute('r',Number(node.dataset.radiusPx)*screenScale());}
window.addEventListener('resize',resizeMarkers);
svg.addEventListener('wheel',event=>{
  if(!data)return;event.preventDefault();const b=svg.viewBox.baseVal,scale=event.deltaY>0?1.2:1/1.2;
  svg.setAttribute('viewBox',`${b.x+b.width*(1-scale)/2} ${b.y+b.height*(1-scale)/2} ${b.width*scale} ${b.height*scale}`);resizeMarkers();
},{passive:false});
svg.addEventListener('pointerdown',e=>{if(e.target.closest('[role="button"]'))return;dragging=[e.clientX,e.clientY];svg.setPointerCapture(e.pointerId);});
svg.addEventListener('pointermove',e=>{
  if(!dragging)return;const b=svg.viewBox.baseVal,rect=svg.getBoundingClientRect(),scale=Math.max(b.width/rect.width,b.height/rect.height);
  svg.setAttribute('viewBox',`${b.x-(e.clientX-dragging[0])*scale} ${b.y-(e.clientY-dragging[1])*scale} ${b.width} ${b.height}`);dragging=[e.clientX,e.clientY];
});
svg.addEventListener('pointerup',()=>dragging=null);svg.addEventListener('pointercancel',()=>dragging=null);
