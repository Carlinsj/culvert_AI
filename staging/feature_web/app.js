import {WARNING,shownSites,endpoint,priority,priorityColor,measurement,evidenceLines,exportedSites,formattedDistance,SUPPORTED_AREAS,roadOptions,roadLabel,oneToOneMatches} from './core.js';
import {attachRuler} from './ruler.js';
import {setupComparison} from './comparison.js';

const $=id=>document.getElementById(id),L=window.L;
if(!L)throw Error('Leaflet failed to load. Check the network connection required for the research map.');
const map=L.map('map',{preferCanvas:true}).setView([42.07647,-74.207172],14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
L.control.scale({imperial:true,metric:true}).addTo(map);
const roads=L.layerGroup().addTo(map),area=L.layerGroup().addTo(map),spacing=L.layerGroup().addTo(map);
const areaOverview=L.featureGroup();
const groups=new Map([...document.querySelectorAll('[data-layer]')].map(input=>[input.dataset.layer,L.layerGroup()]));
const notes=new Map();let data=null,visible=[],requestNumber=0,analysisController=null,browsing=false,activeWorkspace='predictions';
const inventory=L.layerGroup(),matchLines=L.layerGroup();let inventoryData=null,inventoryRequest=null,inventoryRevision=0,benchmark=null;
const recommendationMarkers=new Map();
const activeMatchDetails=new Map();
const pinIcon=(state='predicted')=>L.divIcon({className:'prediction-marker',html:`<span class="prediction-pin ${state}" aria-hidden="true"><i></i></span>`,iconSize:[24,32],iconAnchor:[12,30],popupAnchor:[0,-29]});
const pct=value=>`${(100*value).toFixed(1)}%`;
function updatePerformance(predictions,actuals,matches){
  const panel=$('performanceMetrics');panel.replaceChildren();
  const budget=$('budget').value,full=budget==='all',matched=matches.length,distances=matches.map(m=>m.distance_m).sort((a,b)=>a-b);
  const median=distances.length?distances[Math.floor((distances.length-1)/2)]:null;
  const grid=document.createElement('dl');grid.className='metric-grid';
  const metrics=[['Model',data?.model?.version||'Not available'],['List',full?'Full diagnostic list':'Ranked prefix'],['Active budget',full?`${predictions.length} (full)`:predictions.length],['Actual supported',actuals.length],['Predictions',predictions.length],['Matched / missed',`${matched} / ${Math.max(0,actuals.length-matched)}`],['Recall',actuals.length?pct(matched/actuals.length):'Not available'],['Matches / 1,000',predictions.length?(1000*matched/predictions.length).toFixed(1):'Not available'],['Median distance',median===null?'Not available':`${median.toFixed(1)} m`]];
  for(const [label,value] of metrics)grid.append(text('dt',label),text('dd',String(value)));
  panel.append(grid);
  const key=full?'13065':budget,comparison=benchmark?.budgets?.[key];
  if(comparison){const base=comparison.existing_baseline_rf,random=comparison.random_road_sampling;panel.append(text('p',`847-record benchmark at ${Number(key).toLocaleString()} stops: baseline RF ${base.matches} matches (${pct(base.recall)}); random-road mean ${random.mean_matches.toFixed(1)} (${random.p025_matches.toFixed(0)}–${random.p975_matches.toFixed(0)} seeded 95% interval, ${random.trials} trials).`,'benchmark-note'));}
  else panel.append(text('p','No equal-budget county benchmark for this display size.','benchmark-note'));
}
async function renderInventory(){
  const revision=++inventoryRevision;
  inventory.clearLayers();matchLines.clearLayers();recommendationMarkers.forEach(marker=>marker.setIcon(pinIcon('predicted')));
  if(activeWorkspace!=='predictions'||!$('inventoryLayer').checked||!data||browsing){map.removeLayer(inventory);map.removeLayer(matchLines);return;}
  try{
    if(!inventoryData){
      inventoryRequest??=fetch('/api/research/inventory').then(async response=>{const result=await response.json();if(!response.ok)throw Error(result.error);return result;}).finally(()=>{inventoryRequest=null;});
      inventoryData=await inventoryRequest;
    }
    if(revision!==inventoryRevision||!$('inventoryLayer').checked||browsing)return;
    const center={lat:data.analysis.latitude,lng:data.analysis.longitude};
    const local=inventoryData.features.filter(f=>map.distance(endpoint(f),center)<=data.analysis.radius_m);
    const actuals=local.filter(f=>f.properties.query_success===true);
    const predictions=shownSites(data,$('budget').value,$('roadFilter').value),matches=oneToOneMatches(predictions,actuals,20);
    const byActual=new Map(matches.map(m=>[m.actualIndex,m])),byPrediction=new Map(matches.map(m=>[m.predictionIndex,m]));
    activeMatchDetails.clear();
    predictions.forEach((prediction,index)=>{const match=byPrediction.get(index),nearest=actuals.reduce((best,actual)=>Math.min(best,map.distance(endpoint(prediction),endpoint(actual))),Infinity);activeMatchDetails.set(prediction.properties.physical_site_id||prediction.properties.candidate_id,{matched:Boolean(match),distance_m:match?.distance_m??(Number.isFinite(nearest)?nearest:null),actual:match?actuals[match.actualIndex]:null});});
    for(const [index,f] of actuals.entries()){
      const point=endpoint(f),match=byActual.get(index),p=f.properties,popup=document.createElement('div');
      popup.append(text('strong',`Recorded culvert · workbook row ${p.source_excel_row}`),text('p',`${p.Region} · ${p.Route}`));
      popup.append(text('p',match?`One-to-one match: ${formattedDistance(match.distance_m,ruler.unit)}`:'No active-budget prediction paired within 20 m.'));
      popup.append(text('small','Retrospective recorded location; not independent validation.'));
      L.circleMarker(point,{radius:7,color:'#fff',weight:2,fillColor:match?'#22863a':'#d64545',fillOpacity:1,bubblingMouseEvents:false}).bindPopup(popup).addTo(inventory);
      if(match){const prediction=predictions[match.predictionIndex];L.polyline([point,endpoint(prediction)],{color:'#6f5b8e',weight:2,dashArray:'5 5',interactive:false}).addTo(matchLines);}
    }
    recommendationMarkers.forEach((marker,id)=>{const index=predictions.findIndex(f=>(f.properties.physical_site_id||f.properties.candidate_id)===id);if(index>=0)marker.setIcon(pinIcon(byPrediction.has(index)?'matched':'unmatched'));});
    inventory.addTo(map);matchLines.addTo(map);updatePerformance(predictions,actuals,matches);
  }catch(error){$('diagnostics').textContent=`Inventory overlay unavailable: ${error.message}`;}
}
const ruler=attachRuler(map,L,()=>visible,active=>{
  const save=$('saveNote');if(save)save.disabled=active;
},()=>renderSpacing());
function text(tag,value,cls=''){const node=document.createElement(tag);node.textContent=value;if(cls)node.className=cls;return node;}
function layerEnabled(key){return document.querySelector(`[data-layer="${key}"]`).checked;}
function onRoad(features){const road=$('roadFilter').value;if(!road)return features;const ids=new Set((data?.analysis?.roads?.features||[]).filter(f=>String(f.properties.name||f.properties.id)===road).map(f=>String(f.properties.id)));return features.filter(f=>String(f.properties.route_id??f.properties.road_name??f.properties.name??f.properties.id)===road||ids.has(String(f.properties.road_id)));}
function updateRoadFilter(){
  $('roadFilter').replaceChildren();const all=text('option','All roads');all.value='';$('roadFilter').append(all);
  for(const road of roadOptions(data)){const option=text('option',`${road.name} (${road.count})`);option.value=road.id;$('roadFilter').append(option);}
}
function chooseArea(choice){
  $('areaSelect').value=choice.id;$('latitude').value=choice.latitude;$('longitude').value=choice.longitude;analyze();
}
function fitAreaOverview(){if(areaOverview.getLayers().length)map.fitBounds(areaOverview.getBounds().pad(.2),{padding:[60,55],maxZoom:11});}
function browseAreas(){
  inventory.clearLayers();matchLines.clearLayers();map.removeLayer(inventory);map.removeLayer(matchLines);recommendationMarkers.clear();
  ++requestNumber;analysisController?.abort();browsing=true;visible=[];area.clearLayers();roads.clearLayers();spacing.clearLayers();
  for(const group of groups.values())group.clearLayers();$('detail').hidden=true;
  $('analyze').disabled=false;$('export').hidden=true;$('export').disabled=true;$('siteFilters').hidden=true;
  $('listTitle').textContent='Road areas';$('listHint').textContent='Choose a place on the map or below.';
  $('results').replaceChildren();areaOverview.clearLayers();areaOverview.addTo(map);
  for(const choice of SUPPORTED_AREAS){
    const button=text('button','','result area-choice');button.append(text('strong',choice.name),text('span','Explore this road area →','result-evidence'));button.onclick=()=>chooseArea(choice);$('results').append(button);
    const marker=L.circleMarker([choice.latitude,choice.longitude],{radius:9,color:'#fff',weight:3,fillColor:'#174f3d',fillOpacity:1,bubblingMouseEvents:false});
    marker.bindTooltip(choice.name,{permanent:true,direction:'top',className:'area-tooltip'});marker.on('click',()=>{if(!ruler.active)chooseArea(choice);});marker.addTo(areaOverview);
  }
  fitAreaOverview();
  $('status').classList.remove('error');$('status').textContent=`${SUPPORTED_AREAS.length} supported areas to explore.`;
  $('mapSummary').textContent='Choose a road area';
}
function interact(marker,feature){marker.on('click',event=>{if(event.originalEvent)L.DomEvent.stopPropagation(event.originalEvent);if(!ruler.siteClick(feature))detail(feature);});}
function diagnostic(key,features,color,radius,extra={}){
  features=onRoad(features);
  const group=groups.get(key);group.clearLayers();
  if(!layerEnabled(key)){map.removeLayer(group);return;}
  for(const feature of features){const p=feature.properties;
    const marker=L.circleMarker(endpoint(feature),{radius,color,fillColor:color,weight:1.7,fillOpacity:.35,bubblingMouseEvents:false,...extra});
    marker.bindTooltip(`${key==='anchors'?'Computational sample — not a detection':key.replaceAll('_',' ')} · ${p.candidate_id||p.physical_site_id}`);
    interact(marker,feature);group.addLayer(marker);visible.push(feature);
  }
  group.addTo(map);
}
function render(){
  if(activeWorkspace!=='predictions'||!data||browsing)return;
  visible=[];
  diagnostic('anchors',data.anchors.features,'#7c8582',2,{fillOpacity:.4});
  diagnostic('mapped_crossings',data.mapped_crossings.features,'#08a6b8',4,{fillOpacity:.8});
  diagnostic('dem_crossings',data.dem_crossings.features,'#226db3',4,{dashArray:'3 2'});
  diagnostic('proposals',data.proposals.features,'#8055ae',5,{fillOpacity:0});
  diagnostic('consolidated',data.consolidated.features,'#2c3b35',7,{fillOpacity:0});
  const hydro=groups.get('hydrology');hydro.clearLayers();
  if(layerEnabled('hydrology')){
    hydro.addLayer(L.geoJSON(data.hydrology.mapped,{style:{color:'#08a6b8',weight:2}}));
    hydro.addLayer(L.geoJSON(data.hydrology.derived,{style:{color:'#226db3',weight:1,dashArray:'5 3'}}));hydro.addTo(map);
  }else map.removeLayer(hydro);
  const shown=shownSites(data,$('budget').value,$('roadFilter').value),group=groups.get('recommendations');group.clearLayers();recommendationMarkers.clear();
  const rawView=$('representation').value==='raw';
  if(layerEnabled('recommendations')){
    for(const f of rawView?onRoad(data.raw_candidates.features):shown){const p=f.properties,m=rawView?L.circleMarker(endpoint(f),{radius:3,color:'#737f78',weight:1,fillColor:'#84918a',fillOpacity:.5,bubblingMouseEvents:false}):L.marker(endpoint(f),{icon:pinIcon('predicted'),keyboard:true,riseOnHover:true});m.bindTooltip(rawView?`Raw computational candidate · ${p.candidate_id} · not a culvert detection`:`#${p.global_rank} · ${priority(p.global_rank)} · Unverified group (${p.member_count||1} candidates)`);interact(m,f);group.addLayer(m);visible.push(f);if(!rawView)recommendationMarkers.set(p.physical_site_id||p.candidate_id,m);}group.addTo(map);
  }else map.removeLayer(group);
  $('results').replaceChildren();
  for(const f of shown){const p=f.properties,b=text('button','','result');b.setAttribute('aria-label',`#${p.global_rank} · ${roadLabel(p.road_name||p.route_id)} · ${priority(p.global_rank)}`);
    b.style.borderLeftColor=priorityColor(p.global_rank);b.append(text('strong',roadLabel(p.road_name||p.route_id)));
    const meta=text('span','','result-meta');meta.append(text('span',priority(p.global_rank)),text('span',`#${p.global_rank}`));b.append(meta,text('span',p.physical_crossing?'Drainage crossing':p.profile_local_peak?'Road low point':p.flow_valley_local_peak?'Flow concentration':'Limited physical evidence','result-evidence'));
    b.onclick=()=>{if(!ruler.siteClick(f)){detail(f);map.panTo(endpoint(f));}};$('results').append(b);}
  if(!shown.length)$('results').append(text('p','No potential sites on this road with the current analysis. Try All roads or another area. This does not mean there are no culverts.','empty-state'));
  const counts=data.diagnostics;
  const total=shownSites(data,'all',$('roadFilter').value).length;
  $('status').classList.remove('error');$('status').textContent=$('roadFilter').value?`${shown.length} of ${total} sites on this road.`:`${shown.length} of ${data.site_count} sites · ${roadOptions(data).length} roads in this area.`;
  $('diagnostics').textContent=`${counts.admitted_proposals} admitted proposals; ${counts.sampling_anchors} raw anchors. ${counts.no_proposal_roads.length} roads without a proposal. No backfilling or calibrated probability.`;
  $('mapSummary').textContent=rawView?`Raw diagnostic · ${onRoad(data.raw_candidates.features).length} samples · not detections`:`${roadLabel($('roadFilter').value||'All roads')} · ${shown.length} potential sites`;
  $('export').hidden=false;$('siteFilters').hidden=false;$('listTitle').textContent='Potential sites';$('listHint').textContent='Priority is relative rank, not confidence.';
  renderSpacing();renderInventory();
}
function renderSpacing(){
  spacing.clearLayers();if(!data||browsing||!$('spacing').checked)return;
  const features=onRoad(data.spacing[$('spacingSource').value].features);
  for(const f of features){const p=f.properties,short=formattedDistance(p.distance_m,ruler.unit),label=`${short} · ${p.measurement}`;
    const coords=f.geometry.coordinates,pixels=coords.map(([lng,lat])=>map.latLngToContainerPoint([lat,lng]));
    const screenLength=pixels.slice(1).reduce((sum,b,i)=>sum+Math.hypot(b.x-pixels[i].x,b.y-pixels[i].y),0);
    const layer=L.geoJSON(f,{style:{color:'#a98922',weight:2,dashArray:'5 5'},onEachFeature:(_,line)=>{
      line.bindTooltip(short,{permanent:map.getZoom()>=17&&screenLength>=Math.max(36,short.length*6+12),className:'spacing-label',direction:'center'});
      line.on('click',e=>{if(!ruler.active)line.bindPopup(label).openPopup();});
    }});spacing.addLayer(layer);
  }
}
function detail(feature){
  const p=feature.properties,panel=$('detail');panel.hidden=false;panel.replaceChildren();
  const close=text('button','×','close');close.setAttribute('aria-label','Close site details');close.onclick=()=>panel.hidden=true;panel.append(close);
  const heading=text('h2',p.global_rank?`#${p.global_rank} · ${roadLabel(p.road_name||p.route_id)}`:'Diagnostic location');heading.id='detailTitle';panel.append(heading);
  const inspectionLine=text('p',p.global_rank?`${priority(p.global_rank)} · ${notes.has(p.physical_site_id)?'Local note recorded':'Not inspected'}`:'Diagnostic evidence,not an inspection recommendation');panel.append(inspectionLine);
  const summary=p.physical_crossing?'A drainage crossing that may be worth a field check.':p.profile_local_peak?'A local low point in the road profile.':p.flow_valley_local_peak?'A terrain location where flow may concentrate.':'A road sample with limited physical support.';
  panel.append(text('p',summary),text('p','Potential location, not a confirmed culvert.'),text('p',`${feature.geometry.coordinates[1].toFixed(6)}, ${feature.geometry.coordinates[0].toFixed(6)}`));
  const technical=document.createElement('details');technical.append(text('summary','Technical details'));for(const line of evidenceLines(p,data.model))technical.append(text('p',line));const dl=document.createElement('dl');
  const [longitude,latitude]=feature.geometry.coordinates;
  const match=activeMatchDetails.get(p.physical_site_id||p.candidate_id);
  for(const [label,value] of [['Site type',p.site_type||p.proposal_kind],['Proposal source',p.proposal_kind],['Evidence type',p.evidence_type],
    ['Inventory status',match?(match.matched?'Matched actual within 20 m':'Unmatched / unverified'):'Not available'],['Matched / nearest distance',match&&match.distance_m!==null?measurement(match.distance_m,' m'):'Not available'],
    ['Representative evidence',p.localization_evidence],['Member evidence',p.member_localization_evidence||p.localization_evidence],
    ['Latitude',latitude.toFixed(7)],['Longitude',longitude.toFixed(7)],['Road / branch',`${p.road_id} / ${p.road_part_index}`],
    ['Chainage',measurement(p.road_chainage_m,' m')],['RF ranking score',measurement(p.model_score)],['Priority',measurement(p.priority)],
    ['Flow accumulation',measurement(p.flow_accumulation_cells,' cells')],['Upstream area',measurement(p.upstream_area_m2,' m²')],
    ['Road depression',measurement(p.route_elevation_dip_60m,' m')],['Mapped drainage distance',measurement(p.distance_to_nearest_stream_m,' m')],
    ['Negative TPI (9px)',measurement(p.negative_tpi_9x9_m,' m')],['Valley depth (9px)',measurement(p.valley_depth_9x9_m,' m')],
    ['Missing RF inputs',p.missing_feature_count],['Across-road elevation','Not available; not an existing compatible measurement'],
    ['Members',p.member_count||1],['Grouping policy',p.grouping_policy||'Ungrouped diagnostic'],['Representative selection',p.representative_selection||'Original candidate coordinate; no relocation'],['Selection policy',p.selection_policy||p.localization_policy||'High-recall comparison'],['Site ID',p.physical_site_id||p.candidate_id]]){
    dl.append(text('dt',label),text('dd',String(value??'Not available')));
  }technical.append(dl);panel.append(technical);
  if(p.member_evidence){
    const members=document.createElement('details');members.className='evidence-members';members.append(text('summary',`Inspect underlying candidate evidence (${p.member_count||1})`));
    const list=document.createElement('ul');
    for(const member of JSON.parse(p.member_evidence)){const coordinate=member.latitude!==undefined?`${member.latitude.toFixed(7)}, ${member.longitude.toFixed(7)}`:`${member.x.toFixed(3)}, ${member.y.toFixed(3)} (${member.crs})`;
      list.append(text('li',`${member.candidate_id} · ${member.source} · ${coordinate} · RF score ${measurement(member.model_score)} · drainage identity ${member.channel_key||'Not available'}${member.measurements?' · '+JSON.stringify(member.measurements):''}`));}
    members.append(list);panel.append(members);
  }
  const measure=text('button','Measure from this site','action');measure.onclick=()=>{ruler.from(feature);panel.hidden=true;};panel.append(measure);
  panel.append(text('p',WARNING));
  if(p.physical_site_id){
    const form=document.createElement('form');form.id='fieldForm';
    form.innerHTML='<h3>Local research note</h3><p>Saved only in this browser session. Not used for model training.</p><label>Observed status<select name="status"><option value="uncertain">Uncertain / not verified</option><option value="confirmed_culvert">Culvert observed</option><option value="no_culvert">No culvert observed</option></select></label><label>Notes<textarea name="notes" rows="2"></textarea></label><button id="saveNote" type="submit">Save local note</button><p id="noteStatus" role="status"></p>';
    form.onsubmit=e=>{e.preventDefault();if(ruler.active){$('noteStatus').textContent='Exit measurement before saving a note.';return;}
      notes.set(p.physical_site_id,{physical_site_id:p.physical_site_id,...Object.fromEntries(new FormData(form)),local_staging_only:true});inspectionLine.textContent=`${priority(p.global_rank)} · Local research note recorded`;$('noteStatus').textContent='Saved in this browser session only. Prediction and ranking unchanged.';};
    panel.append(form);$('saveNote').disabled=ruler.active;
  }
}
async function analyze(){
  const number=++requestNumber;
  analysisController?.abort();analysisController=new AbortController();browsing=false;map.removeLayer(areaOverview);areaOverview.clearLayers();
  $('analyze').disabled=true;$('export').disabled=true;$('status').classList.remove('error');$('status').textContent='Finding potential sites…';
  try{
    const latitude=$('latitude').value,longitude=$('longitude').value;
    if(!latitude.trim()||!longitude.trim())throw Error('Enter both latitude and longitude.');
    const response=await fetch('/api/analysis/nearby',{method:'POST',signal:analysisController.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({location:{latitude:Number(latitude),longitude:Number(longitude)},radius_m:Number($('radius').value),policy:$('policy').value,mode:$('mode').value,grouping:$('grouping').value})});
    const result=await response.json();if(number!==requestNumber)return;if(!response.ok)throw Error(result.error);
    data=result;updateRoadFilter();$('detail').hidden=true;roads.clearLayers();area.clearLayers();
    roads.addLayer(L.geoJSON(data.analysis.roads,{style:{color:'#9aab99',weight:2}}));
    L.circle([data.analysis.latitude,data.analysis.longitude],{radius:data.analysis.radius_m,color:'#47725c',weight:1,fillOpacity:.025,interactive:false}).addTo(area);
    map.fitBounds(area.getBounds?.()||L.latLng(data.analysis.latitude,data.analysis.longitude).toBounds(data.analysis.radius_m*2),{padding:[35,35]});
    render();$('export').disabled=false;
  }catch(error){if(number!==requestNumber||error.name==='AbortError')return;data=null;visible=[];roads.clearLayers();area.clearLayers();spacing.clearLayers();for(const group of groups.values())group.clearLayers();$('results').replaceChildren();$('detail').hidden=true;$('status').classList.add('error');$('status').textContent=error.message+' Select Explore other road areas to choose a supported place.';$('mapSummary').textContent='Choose a supported road area';$('export').hidden=true;}
  finally{if(number===requestNumber)$('analyze').disabled=false;}
}
$('analyze').onclick=analyze;
$('supported').onclick=browseAreas;
$('areaSelect').onchange=()=>{const choice=SUPPORTED_AREAS.find(a=>a.id===$('areaSelect').value);if(choice){$('latitude').value=choice.latitude;$('longitude').value=choice.longitude;}else{$('advanced').open=true;$('latitude').closest('details').open=true;$('latitude').focus();}$('status').textContent='Select Show sites to search this area.';};
$('locate').onclick=()=>{if(!navigator.geolocation){$('status').textContent='Location unavailable. Choose a road area instead.';return;}navigator.geolocation.getCurrentPosition(p=>{$('latitude').value=p.coords.latitude;$('longitude').value=p.coords.longitude;$('areaSelect').value='custom';analyze();},()=>{$('status').textContent='Location unavailable or permission denied. Choose a road area instead.';},{timeout:12000});};
for(const input of document.querySelectorAll('[data-layer]'))input.onchange=render;
$('inventoryLayer').onchange=renderInventory;
$('budget').onchange=render;$('spacing').onchange=renderSpacing;$('spacingSource').onchange=renderSpacing;
$('representation').onchange=render;
$('roadFilter').onchange=()=>{render();$('detail').hidden=true;const own=shownSites(data,'all',$('roadFilter').value);if(own.length){const bounds=L.latLngBounds(own.map(f=>endpoint(f)));map.fitBounds(bounds.pad(.2),{padding:[45,45],maxZoom:16});}};
map.on('zoomend',renderSpacing);
map.on('resize',()=>{if(browsing)fitAreaOverview();});
for(const id of ['mode','policy','grouping','radius','latitude','longitude'])$(id).onchange=()=>{if(id==='latitude'||id==='longitude')$('areaSelect').value='custom';$('status').textContent='Settings changed. Select Show sites to apply.';};
$('export').onclick=()=>{if(!data||browsing)return;const payload=exportedSites(data,$('budget').value,notes,$('roadFilter').value);const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/geo+json'}));const a=document.createElement('a');a.href=url;a.download='experimental-feature-sites.geojson';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
const comparison=setupComparison(map,L);
function setWorkspace(name){
  activeWorkspace=name;
  for(const button of document.querySelectorAll('[data-workspace]'))button.setAttribute('aria-pressed',String(button.dataset.workspace===name));
  for(const panel of document.querySelectorAll('.prediction-workspace'))panel.hidden=name!=='predictions'||panel.id==='performancePanel';
  $('comparisonWorkspace').hidden=name!=='comparison';$('modelPerformanceWorkspace').hidden=name!=='performance';
  if(name==='predictions'){comparison.hide();$('locate').hidden=false;$('measure').hidden=false;if(data&&!browsing)render();else browseAreas();}
  else{++requestNumber;analysisController?.abort();inventory.clearLayers();matchLines.clearLayers();map.removeLayer(inventory);map.removeLayer(matchLines);map.removeLayer(areaOverview);roads.clearLayers();area.clearLayers();spacing.clearLayers();for(const group of groups.values())group.clearLayers();$('detail').hidden=true;$('locate').hidden=true;$('measure').hidden=name==='performance';if(name==='comparison')comparison.show();else{comparison.hide();comparison.showPerformance();$('mapSummary').textContent='Model performance · retrospective 847-record benchmark';}}
}
for(const button of document.querySelectorAll('[data-workspace]'))button.onclick=()=>setWorkspace(button.dataset.workspace);
fetch('/api/research/anti-saturation').then(async response=>{const result=await response.json();if(response.ok){benchmark=result;if(data&&!browsing)renderInventory();}}).catch(()=>{});
browseAreas();
