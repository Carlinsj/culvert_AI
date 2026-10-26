import {RulerState,endpoint,nearestVisible,formattedDistance} from './core.js';

export function attachRuler(map,L,visible,onActive,onUnit) {
  const $=id=>document.getElementById(id),state=new RulerState();
  const group=L.layerGroup().addTo(map),pins=[null,null];let line=null,target=null;
  map.createPane('rulerPins');map.getPane('rulerPins').style.zIndex=680;
  function snapped(latlng) {return state.snap?nearestVisible(latlng,visible(),p=>map.latLngToContainerPoint(p))||{...latlng,siteId:null}:{...latlng,siteId:null};}
  function status() {
    $('measure').setAttribute('aria-pressed',String(state.active));
    $('measureMap').setAttribute('aria-pressed',String(state.active));
    $('measureMap').textContent=state.active?'↔ Measuring · Exit':'↔ Measure A → B';
    $('placePinA').setAttribute('aria-pressed',String(state.active&&(target===0||(!state.pins[0]&&target===null))));
    $('placePinB').setAttribute('aria-pressed',String(Boolean(state.active&&(target===1||(state.pins[0]&&target===null)))));
    $('map').classList.toggle('measuring',state.active);
    $('rulerHint').textContent=state.active?(target!==null?`Tap the map to place marker ${target?'B':'A'}.`:state.pins[0]?(state.pins[1]?'Drag either pin or choose which marker to replace.':'Tap the map to place your second marker B.'):'Tap the map to place your first marker A.'):'Measurement paused. Exit does not change any site.';
    $('rulerPanel').hidden=!state.active&&!state.pins.some(Boolean);
    const m=state.distance===null?null:map.distance(L.latLng(state.pins[0]),L.latLng(state.pins[1]));
    $('distance').textContent=m===null?'Place both pins':formattedDistance(m,state.unit);
    $('distance').dataset.meters=m===null?'':String(m);
    for(let i=0;i<2;i++) {
      const letter=i?'B':'A',p=state.pins[i];
      $('pin'+letter+'Snap').textContent=p?.siteId?`Snapped to ${p.siteId}`:'Free geographic endpoint';
      for(const [field,value] of [['Lat',p?.lat],['Lng',p?.lng]]) {
        const input=$('pin'+letter+field);
        if(document.activeElement!==input)input.value=value===undefined?'':value.toFixed(7);
      }
    }
    onActive(state.active);
  }
  function draw() {
    for(let i=0;i<2;i++) {
      const point=state.pins[i],letter=i?'B':'A';
      if(!point) {if(pins[i])group.removeLayer(pins[i]);pins[i]=null;continue;}
      if(!pins[i]) {
        pins[i]=L.marker(point,{draggable:state.active,pane:'rulerPins',keyboard:true,
          title:`Measurement Pin ${letter}`,alt:`Measurement Pin ${letter}`,bubblingMouseEvents:false,
          icon:L.divIcon({className:`ruler-pin pin-${letter.toLowerCase()}`,html:`<span><b>${letter}</b></span>`,iconSize:[34,42],iconAnchor:[17,38]})}).addTo(group);
        pins[i].on('drag',()=>{state.set(i,snapped(pins[i].getLatLng()));draw();});
        pins[i].on('dragend',()=>{state.set(i,snapped(pins[i].getLatLng()));draw();});
        const el=pins[i].getElement();el.setAttribute('aria-label',`Measurement Pin ${letter};drag or use arrow keys`);
        el.addEventListener('keydown',e=>{
          if(!state.active||!['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key))return;
          e.preventDefault();e.stopPropagation();const p={...state.pins[i],siteId:null},step=e.shiftKey?10:1;
          if(e.key==='ArrowUp')p.lat+=step/111195;if(e.key==='ArrowDown')p.lat-=step/111195;
          if(e.key==='ArrowRight')p.lng+=step/(111195*Math.cos(p.lat*Math.PI/180));
          if(e.key==='ArrowLeft')p.lng-=step/(111195*Math.cos(p.lat*Math.PI/180));
          state.set(i,p);draw();
        });
      }
      pins[i].setLatLng(point);
      if(state.active)pins[i].dragging.enable();else pins[i].dragging.disable();
      pins[i].getElement()?.classList.toggle('snapped',Boolean(point.siteId));
    }
    if(state.pins.every(Boolean)) {
      if(!line)line=L.polyline(state.pins,{color:'#6641a8',weight:4,dashArray:'8 6',interactive:false}).addTo(group);
      else line.setLatLngs(state.pins);
    } else if(line){group.removeLayer(line);line=null;}
    if(state.active)map.doubleClickZoom.disable();else map.doubleClickZoom.enable();
    status();
  }
  function place(point){if(target!==null&&state.active){state.set(target,point);target=null;draw();}else if(state.place(point))draw();}
  map.on('click',event=>{if(state.active)place(snapped(event.latlng));});
  $('measure').onclick=$('measureMap').onclick=()=>{target=null;if(state.active)state.exit();else state.start();draw();};
  $('placePinA').onclick=()=>{state.active=true;target=0;draw();};
  $('placePinB').onclick=()=>{state.active=true;target=1;draw();};
  $('rulerReset').onclick=()=>{target=null;state.reset();draw();};
  $('rulerClear').onclick=()=>{target=null;state.clear();draw();};
  $('rulerExit').onclick=()=>{state.exit();draw();};
  $('rulerUnits').onchange=e=>{state.unit=e.target.value;draw();onUnit(state.unit);};
  $('rulerSnap').onchange=e=>{state.snap=e.target.checked;if(!state.snap)for(const p of state.pins)if(p)p.siteId=null;draw();};
  for(let i=0;i<2;i++) {
    const letter=i?'B':'A';
    $('setPin'+letter).onclick=()=>{
      try {
        const lat=$('pin'+letter+'Lat').value,lng=$('pin'+letter+'Lng').value;
        if(lat.trim()===''||lng.trim()==='')throw Error('Enter both endpoint coordinates.');
        state.active=true;state.set(i,{lat:Number(lat),lng:Number(lng)});draw();
      } catch(e){$('rulerHint').textContent=e.message;}
    };
  }
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&state.active){e.preventDefault();state.exit();draw();}});
  return {get active(){return state.active;},get unit(){return state.unit;},
    siteClick(feature){if(!state.active)return false;place(endpoint(feature));return true;},
    from(feature){target=null;state.start(endpoint(feature));draw();$('rulerPanel').focus();}};
}
