// Test the actual ruler event adapter with isolated Leaflet/DOM interfaces.
// Real Leaflet mouse dragging is additionally verified in the browser.
import test from 'node:test';
import assert from 'node:assert/strict';
import {attachRuler} from '../staging/feature_web/ruler.js';
import {distanceMeters} from '../staging/feature_web/core.js';

function fixture(features=[]){
  const elements=new Map(),keys={},markers=[],groups=[];let active=false,unit=null;
  function element(id){if(!elements.has(id))elements.set(id,{id,value:'',hidden:true,textContent:'',dataset:{},attributes:{},style:{},
    classes:new Set(),classList:{toggle(name,on){if(on)this.owner.classes.add(name);else this.owner.classes.delete(name);}},
    setAttribute(name,value){this.attributes[name]=value;},addEventListener(name,fn){this[name]=fn;},focus(){}});
    const e=elements.get(id);e.classList.owner=e;return e;}
  const doc={getElementById:element,activeElement:null,addEventListener:(name,fn)=>keys[name]=fn};
  global.document=doc;
  const map={events:{},panes:{},doubleClickZoom:{enabled:true,enable(){this.enabled=true;},disable(){this.enabled=false;}},
    createPane(name){this.panes[name]={style:{}};},getPane(name){return this.panes[name];},
    on(name,fn){this.events[name]=fn;},distance:distanceMeters,latLngToContainerPoint:p=>({x:p.lng*100,y:p.lat*100})};
  const L={latLng:p=>p,divIcon:options=>options,
    layerGroup(){const g={layers:new Set(),addTo(){return this;},removeLayer(layer){this.layers.delete(layer);}};groups.push(g);return g;},
    marker(point,options){const m={point:{...point},options,events:{},element:element('marker'+markers.length),
      dragging:{enabled:false,enable(){this.enabled=true;},disable(){this.enabled=false;}},
      addTo(group){group.layers.add(this);return this;},on(name,fn){this.events[name]=fn;return this;},
      getLatLng(){return {...this.point};},setLatLng(p){this.point={...p};return this;},getElement(){return this.element;}};
      markers.push(m);return m;},
    polyline(points){return {points,addTo(g){g.layers.add(this);return this;},setLatLngs(p){this.points=p;}};}};
  const ruler=attachRuler(map,L,()=>features,value=>active=value,value=>unit=value);
  return {elements,element,map,markers,groups,keys,ruler,get active(){return active;},get unit(){return unit;},
    click(lat,lng,type='click'){map.events.click({latlng:{lat,lng},originalEvent:{type}});},
    value(){return Number(element('distance').dataset.meters);}};
}
const site={type:'Feature',geometry:{type:'Point',coordinates:[1,1]},properties:{candidate_id:'site-A'}};

test('adapter map clicks place two independent geographic markers and live line',()=>{
  const f=fixture();f.element('measure').onclick();f.click(1,1);f.click(1.01,1);
  assert.equal(f.markers.length,2);assert.ok(f.value()>1100);assert.equal(f.groups[0].layers.size,3);
  const originalB=f.markers[1].getLatLng(),before=f.value();
  f.markers[0].setLatLng({lat:1.002,lng:1});f.markers[0].events.drag();
  assert.ok(f.value()<before);assert.deepEqual(f.markers[1].getLatLng(),originalB);
  const afterA=f.value();f.markers[1].setLatLng({lat:1.02,lng:1});f.markers[1].events.drag();assert.ok(f.value()>afterA);
});
test('visible map ruler entry and explicit A/B replacement controls work',()=>{
  const f=fixture();f.element('measureMap').onclick();f.click(42,-74);f.click(42.001,-74);
  assert.equal(f.markers.length,2);assert.equal(f.element('measureMap').attributes['aria-pressed'],'true');
  const b=f.markers[1].getLatLng();f.element('placePinA').onclick();f.click(42.0005,-74);
  assert.equal(f.markers[0].getLatLng().lat,42.0005);assert.deepEqual(f.markers[1].getLatLng(),b);
  const a=f.markers[0].getLatLng();f.element('placePinB').onclick();f.click(42.002,-74);
  assert.deepEqual(f.markers[0].getLatLng(),a);assert.equal(f.markers[1].getLatLng().lat,42.002);
  f.element('measureMap').onclick();assert.equal(f.active,false);
});
test('same touch-generated click/drag event path updates geographic measurement',()=>{
  const f=fixture();f.element('measure').onclick();f.click(42,-74,'touchend');f.click(42.01,-74,'touchend');
  const before=f.value();f.markers[1].setLatLng({lat:42.02,lng:-74});f.markers[1].events.drag({originalEvent:{type:'touchmove'}});
  assert.ok(f.value()>before);assert.equal(f.markers[1].dragging.enabled,true);
});
test('candidate endpoint snapping and drag detachment preserve source geometry',()=>{
  const original=JSON.stringify(site),f=fixture([site]);f.ruler.from(site);f.click(1.1,1.1);
  assert.match(f.element('pinASnap').textContent,/Snapped to site-A/);
  f.markers[0].setLatLng({lat:4,lng:4});f.markers[0].events.drag();
  assert.equal(f.element('pinASnap').textContent,'Free geographic endpoint');assert.equal(JSON.stringify(site),original);
});
test('marker clicks are consumed only while measuring',()=>{
  const f=fixture([site]);assert.equal(f.ruler.siteClick(site),false);
  f.element('measure').onclick();assert.equal(f.ruler.siteClick(site),true);
  assert.equal(f.markers.length,1);assert.match(f.element('pinASnap').textContent,/site-A/);
});
test('reset,clear,Escape,units and arrow-key controls use actual DOM adapter',()=>{
  const f=fixture();f.element('measure').onclick();f.click(1,1);f.click(1.01,1);
  const meters=f.value();f.element('rulerUnits').onchange({target:{value:'ft'}});
  assert.ok(Math.abs(f.value()-meters)<1e-8);assert.match(f.element('distance').textContent,/ft$/);
  f.markers[0].element.keydown({key:'ArrowUp',shiftKey:false,preventDefault(){},stopPropagation(){}});
  assert.ok(f.value()<meters);
  f.keys.keydown({key:'Escape',preventDefault(){}});assert.equal(f.active,false);assert.equal(f.markers[0].dragging.enabled,false);
  assert.equal(f.map.doubleClickZoom.enabled,true);
  f.element('rulerReset').onclick();assert.equal(f.active,true);assert.equal(f.groups[0].layers.size,0);
  f.click(1,1);f.element('rulerClear').onclick();assert.equal(f.groups[0].layers.size,0);
});
