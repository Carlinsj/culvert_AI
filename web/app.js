import { explanationLines, exportGeoJson, priorityLabel, rowsToCsv, safeExportRows } from "./product-core.js";

const DEFAULT_CENTER = [41.79, -74.32];
const state = { map: null, modelReady: false, center: null, pendingCenter: null, result: null, limit: 25, selectedId: null, observations: new Map(), markers: new Map(), roadLayer: null, siteLayer: null, locationMarker: null, radiusCircle: null, supportedLayer: null, analysisController: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
let locationRequest = 0;

initialize();

async function initialize() {
  initializeMap(); wireControls();
  await Promise.all([loadModelStatus(), loadObservations()]);
  if (state.modelReady) requestLocation({ automatic: true });
}

function initializeMap() {
  if (!window.L) { document.body.classList.add("map-unavailable"); return; }
  state.map = L.map("map", { zoomControl: true }).setView(DEFAULT_CENTER, 10);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(state.map);
  state.map.on("click", (event) => chooseMapPoint(event.latlng.lat, event.latlng.lng));
}

function wireControls() {
  $("#useLocation").addEventListener("click", () => requestLocation({ automatic: false }));
  $("#locationButton").addEventListener("click", () => requestLocation({ automatic: false }));
  $("#retryLocation").addEventListener("click", () => requestLocation({ automatic: false }));
  $("#showSupportedAreas").addEventListener("click", showSupportedAreas);
  $("#browseSupportedAreas").addEventListener("click", showSupportedAreas);
  $("#supportedLocation").addEventListener("click", () => requestLocation({ automatic: false }));
  $("#chooseOnMap").addEventListener("click", beginMapSelection);
  $("#analyzeHere").addEventListener("click", () => state.pendingCenter && analyzeNearby(state.pendingCenter));
  $("#recenterButton").addEventListener("click", () => state.center && state.map?.flyTo([state.center.latitude, state.center.longitude], 13));
  $("#radiusSelect").addEventListener("change", () => state.center && analyzeNearby(state.center));
  $("#collapsePanel").addEventListener("click", collapsePanel);
  $("#openPanel").addEventListener("click", openPanel);
  $("#homeButton").addEventListener("click", () => { closeOverlays(); openPanel(); state.center && state.map?.flyTo([state.center.latitude, state.center.longitude], 13); });
  $("#coordinateSearch").addEventListener("click", searchCoordinates);
  $("#coordinateInput").addEventListener("keydown", (event) => event.key === "Enter" && searchCoordinates());
  $$("[data-limit]").forEach((button) => button.addEventListener("click", () => { state.limit = button.dataset.limit === "all" ? "all" : Number(button.dataset.limit); $$('[data-limit]').forEach((item) => item.classList.toggle("active", item === button)); renderResultsList(); }));
  $("#closeDetails").addEventListener("click", () => $("#detailSheet").classList.add("hidden"));
  $$("[data-view]").forEach((button) => button.addEventListener("click", () => showView(button.dataset.view)));
  $$(".close-overlay").forEach((button) => button.addEventListener("click", closeOverlays));
  $("#fieldSiteSelect").addEventListener("change", updateFieldSite);
  $("#fieldForm").addEventListener("submit", saveFieldResult);
  $("#exportMenu").addEventListener("click", () => $("#exportPopover").classList.toggle("hidden"));
  $("#exportCsv").addEventListener("click", exportCsv);
  $("#exportGeoJson").addEventListener("click", exportGeoJSON);
}

async function loadModelStatus() {
  const pill = $("#modelPill");
  try {
    const response = await fetch("/api/model-status"); const data = await response.json();
    state.modelReady = Boolean(response.ok && data.ready); pill.className = `model-pill ${state.modelReady ? "ready" : "error"}`; pill.textContent = state.modelReady ? "Ready to explore" : "Analysis unavailable";
    if (!state.modelReady) showError(data.message || "Culvert AI could not verify the prediction model.", data.technical);
  } catch (error) { pill.className = "model-pill error"; pill.textContent = "Analysis unavailable"; showError("Culvert AI could not verify the prediction model.", error.message); }
}

async function loadObservations() {
  try { const response = await fetch("/api/observations"); if (!response.ok) return; const data = await response.json(); for (const feature of data.features || []) { const id = feature.properties?.physical_site_id || feature.properties?.candidate_id; if (id) state.observations.set(String(id), feature.properties); } } catch { /* Field storage does not block mapping. */ }
}

function requestLocation({ automatic }) {
  const request = ++locationRequest;
  if (!state.modelReady) return;
  if (!navigator.geolocation) { showLocationUnavailable("This browser does not provide location access."); return; }
  showLoading("Finding your location…", "Your browser may ask for permission.");
  navigator.geolocation.getCurrentPosition(
    (position) => { if (request === locationRequest) analyzeNearby({ latitude: position.coords.latitude, longitude: position.coords.longitude, source: "device" }); },
    (error) => {
      if (request !== locationRequest) return;
      if (automatic) { showWelcome("Location was not shared. Use the button below or choose a point on the map."); return; }
      showLocationUnavailable(error.code === 1 ? "Location permission was not granted." : "We could not determine your current location.");
    },
    { enableHighAccuracy: true, timeout: 12000, maximumAge: 60000 },
  );
}

function beginMapSelection() { collapsePanel(); $("#analyzeHere").classList.add("hidden"); state.map?.getContainer().classList.add("selecting-location"); }

function chooseMapPoint(latitude, longitude) {
  state.pendingCenter = { latitude, longitude, source: "map" }; placeLocationMarker(state.pendingCenter, false); $("#analyzeHere").classList.remove("hidden"); state.map?.getContainer().classList.remove("selecting-location");
}

function searchCoordinates() {
  const values = $("#coordinateInput").value.split(",").map((value) => Number(value.trim()));
  if (values.length !== 2 || !Number.isFinite(values[0]) || !Number.isFinite(values[1]) || Math.abs(values[0]) > 90 || Math.abs(values[1]) > 180) { showError("Enter coordinates as latitude, longitude."); return; }
  analyzeNearby({ latitude: values[0], longitude: values[1], source: "search" });
}

async function analyzeNearby(center) {
  locationRequest++;
  state.analysisController?.abort();
  const controller = new AbortController(); state.analysisController = controller;
  state.supportedLayer?.remove(); state.supportedLayer = null;
  state.result = null; state.selectedId = null; state.roadLayer?.remove(); state.siteLayer?.remove(); state.markers.clear(); $("#detailSheet").classList.add("hidden");
  if (!state.modelReady) return; state.center = center; state.pendingCenter = null; $("#analyzeHere").classList.add("hidden"); openPanel(); showLoading("Finding nearby potential culverts", "Analyzing roads, terrain, and drainage around this location."); placeLocationMarker(center, true);
  const radius = Number($("#radiusSelect").value || 4800);
  try {
    const response = await fetch("/api/analysis/nearby", { method: "POST", signal: controller.signal, headers: { "content-type": "application/json" }, body: JSON.stringify({ latitude: center.latitude, longitude: center.longitude, radius_m: radius }) }); const data = await response.json();
    if (controller.signal.aborted) return;
    if (!response.ok) throw apiError(data); state.result = data; renderMapData(); renderResults();
  } catch (error) { if (!controller.signal.aborted) showError(error.message, error.technical, error.code); }
}

async function showSupportedAreas() {
  locationRequest++;
  state.pendingCenter = null; $("#analyzeHere").classList.add("hidden"); $("#detailSheet").classList.add("hidden");
  state.analysisController?.abort(); closeOverlays(); openPanel(); showState("supported");
  $("#supportedMessage").textContent = "Choose a road area covered by our current road and terrain data.";
  $("#supportedAreaList").replaceChildren();
  try {
    const response = await fetch("/data/supported-areas.json");
    if (!response.ok) throw new Error("Supported areas could not be loaded. Please try again.");
    const data = await response.json();
    if ($("#supportedState").classList.contains("hidden")) return;
    if (!data.areas?.length) throw new Error("No supported areas are available right now.");
    state.supportedLayer?.remove();
    if (state.map) state.supportedLayer = L.featureGroup().addTo(state.map);
    for (const area of data.areas) {
      const explore = () => { $("#radiusSelect").value = String(area.radius_m); $("#coordinateInput").value = `${area.latitude.toFixed(5)}, ${area.longitude.toFixed(5)}`; analyzeNearby({ latitude: area.latitude, longitude: area.longitude, source: "supported" }); };
      const button = document.createElement("button"); button.className = "primary-action"; button.textContent = `Explore ${area.name}`; button.addEventListener("click", explore); $("#supportedAreaList").append(button);
      if (state.map) L.circleMarker([area.latitude, area.longitude], { radius: 10, color: "#fff", weight: 3, fillColor: "#174d3b", fillOpacity: 1, bubblingMouseEvents: false }).bindTooltip(escapeHtml(area.name), { permanent: true, direction: "top" }).on("click", explore).addTo(state.supportedLayer);
    }
    if (state.map) { state.locationMarker?.remove(); state.radiusCircle?.remove(); state.map.fitBounds(state.supportedLayer.getBounds().pad(.25), { paddingTopLeft: window.innerWidth > 760 ? [430, 100] : [30, 100], paddingBottomRight: [30, window.innerWidth > 760 ? 40 : 320], maxZoom: 11 }); }
  } catch (error) { $("#supportedMessage").textContent = error.message; }
}

function renderMapData() {
  if (!state.map || !state.result) return;
  if (state.roadLayer) state.roadLayer.remove(); if (state.siteLayer) state.siteLayer.remove(); state.markers.clear();
  state.roadLayer = L.geoJSON(state.result.analysis.roads, { style: { color: "#315f52", weight: 4, opacity: .72 } }).addTo(state.map);
  state.siteLayer = L.layerGroup().addTo(state.map);
  for (const feature of state.result.sites.features) { const p = feature.properties; const id = String(p.physical_site_id); const priority = priorityLabel(p.global_rank); const color = priority.startsWith("High") ? "#cb513d" : priority.startsWith("Medium") ? "#d69435" : "#578575"; const inspected = state.observations.has(id); const marker = L.circleMarker([p.latitude, p.longitude], { radius: 7, color: inspected ? "#174d3b" : "#fff", weight: inspected ? 3 : 2, fillColor: color, fillOpacity: .97 }).addTo(state.siteLayer).bindTooltip(`#${p.global_rank} · ${escapeHtml(p.road_name || p.route_id || "Road")}`); marker.on("click", () => selectSite(id)); state.markers.set(id, marker); }
  placeLocationMarker(state.center, true); state.map.fitBounds(state.radiusCircle.getBounds(), { padding: [40, 40] });
}

function placeLocationMarker(center, includeRadius) {
  if (!state.map) return; if (state.locationMarker) state.locationMarker.remove(); state.locationMarker = L.circleMarker([center.latitude, center.longitude], { radius: 9, color: "#fff", weight: 3, fillColor: "#2d73b9", fillOpacity: 1 }).addTo(state.map).bindTooltip(center.source === "device" ? "Your location" : "Selected location");
  if (state.radiusCircle) state.radiusCircle.remove(); if (includeRadius) state.radiusCircle = L.circle([center.latitude, center.longitude], { radius: Number($("#radiusSelect").value || 4800), color: "#2d73b9", weight: 1, dashArray: "5 7", fillColor: "#2d73b9", fillOpacity: .035 }).addTo(state.map);
}

function renderResults() {
  showState("results"); const miles = state.result.analysis.radius_m / 1609.344; $("#areaTitle").textContent = state.center.source === "device" ? "Around your location" : "Selected map area"; $("#areaMeta").textContent = `${miles.toFixed(miles < 2 ? 1 : 0)} mi radius · ${state.result.analysis.feature_count.toLocaleString()} road features`; populateFieldSites(); renderResultsList();
}

function renderResultsList() {
  if (!state.result) return; const all = state.result.sites.features; const visible = state.limit === "all" ? all : all.filter((feature) => Number(feature.properties.global_rank) <= state.limit); $("#resultCount").textContent = `${visible.length.toLocaleString()} of ${all.length.toLocaleString()} ranked sites shown`;
  $("#siteList").innerHTML = visible.length ? visible.map(siteRow).join("") : '<div class="welcome-state"><p>No potential sites were returned for this view.</p></div>';
  $$(".site-row").forEach((button) => button.addEventListener("click", () => selectSite(button.dataset.site)));
}

function siteRow(feature) {
  const p = feature.properties; const priority = priorityLabel(p.global_rank); const klass = priority.startsWith("High") ? "" : priority.startsWith("Medium") ? "medium" : "lower"; const distance = distanceMeters(state.center, { latitude: p.latitude, longitude: p.longitude }); const observation = state.observations.get(String(p.physical_site_id));
  return `<button class="site-row ${state.selectedId === p.physical_site_id ? "selected" : ""}" data-site="${escapeHtml(p.physical_site_id)}"><span class="rank-dot ${klass}">${p.global_rank}</span><span><strong>${escapeHtml(displayRoad(p))}</strong><small>${Number(p.latitude).toFixed(5)}, ${Number(p.longitude).toFixed(5)}</small></span><span class="row-meta">${priority.replace(" Priority", "")}<span>${formatDistance(distance)}${observation ? ` · ${statusLabel(observation.status)}` : ""}</span></span></button>`;
}

function selectSite(id) {
  const feature = state.result?.sites.features.find((item) => String(item.properties.physical_site_id) === String(id)); if (!feature) return; state.selectedId = String(id); renderResultsList(); renderDetails(feature); const p = feature.properties; state.map?.flyTo([p.latitude, p.longitude], Math.max(state.map.getZoom(), 16));
}

function renderDetails(feature) {
  const p = feature.properties; const priority = priorityLabel(p.global_rank); const observation = state.observations.get(String(p.physical_site_id)); const sheet = $("#detailSheet"); sheet.classList.remove("hidden");
  $("#detailContent").innerHTML = `<p class="eyebrow">RANKED SITE #${p.global_rank}</p><h2>${escapeHtml(displayRoad(p))}</h2><span class="detail-priority">${priority}</span><div class="detail-grid"><div><span>Distance</span>${formatDistance(distanceMeters(state.center, { latitude: p.latitude, longitude: p.longitude }))}</div><div><span>Inspection</span>${observation ? statusLabel(observation.status) : "Not inspected"}</div><div><span>Latitude</span>${Number(p.latitude).toFixed(6)}</div><div><span>Longitude</span>${Number(p.longitude).toFixed(6)}</div></div><h3>Why inspect here?</h3><ul class="why">${explanationLines(p).map((line) => `<li>${escapeHtml(line)}</li>`).join("")}</ul><p><small>These are feature-supported summaries, not confirmation that a culvert exists.</small></p><button class="primary-action" id="recordInspection">Record field check</button><details><summary>Technical details</summary><div class="technical-list">FLOW-C score: ${formatNumber(p.flow_c_score)}<br>ROUTE-Z score: ${formatNumber(p.route_z_score)}<br>Global rank: ${p.global_rank}<br>Route-local rank: ${p.route_local_rank ?? "—"}<br>Candidate: ${escapeHtml(p.candidate_id)}<br>Physical site: ${escapeHtml(p.physical_site_id)}<br>Model: FLOW-C Frozen v1<br>Analysis CRS: EPSG:32618</div></details>`;
  $("#recordInspection").addEventListener("click", () => { $("#fieldSiteSelect").value = p.physical_site_id; updateFieldSite(); showView("field"); });
}

function showLoading(title, text) { showState("loading"); $("#loadingState strong").textContent = title; $("#loadingState p").textContent = text; }
function showWelcome(message = "See ranked potential culvert locations on roads nearby. Your location is used only to select the area to analyze.") { showState("welcome"); $("#welcomeState p").textContent = message; }
function showLocationUnavailable(message) { showError(message, "Choose a point on the map or enter coordinates to continue."); }
function showError(message, technical = "", code = "") { showState("error"); const coverage = ["outside_coverage", "partial_coverage", "no_nearby_roads"].includes(code); $("#errorTitle").textContent = message; $("#errorText").textContent = coverage ? "Explore a supported area to find potential culvert sites." : message.includes("coordinate") ? "Use the format 41.75, -74.10." : "You can also choose a point directly on the map."; $("#showSupportedAreas").classList.toggle("hidden", !coverage); const details = $("#errorDetails"); details.classList.toggle("hidden", !technical); details.querySelector("pre").textContent = technical || ""; }
function showState(name) { for (const id of ["welcomeState", "loadingState", "resultsState", "errorState", "supportedState"]) $(`#${id}`).classList.add("hidden"); $(`#${name}State`).classList.remove("hidden"); }

function collapsePanel() { $("#nearbyPanel").classList.add("hidden"); $("#openPanel").classList.remove("hidden"); $("#openPanelCount").textContent = state.result ? `(${state.result.site_count})` : ""; }
function openPanel() { $("#nearbyPanel").classList.remove("hidden"); $("#openPanel").classList.add("hidden"); }
function showView(name) { closeOverlays(); if (name === "field" && !state.result?.sites.features.length) { openPanel(); if ($("#errorState").classList.contains("hidden")) showWelcome("Find sites on Map first, then open Field checks to record what you find."); return; } $$(".header-link").forEach((button) => button.classList.toggle("active", button.dataset.view === name)); if (name === "field") $("#fieldView").classList.remove("hidden"); if (name === "about") $("#aboutView").classList.remove("hidden"); }
function closeOverlays() { $("#fieldView").classList.add("hidden"); $("#aboutView").classList.add("hidden"); $$(".header-link").forEach((button) => button.classList.toggle("active", button.dataset.view === "map")); }

function populateFieldSites() { const select = $("#fieldSiteSelect"); select.innerHTML = state.result.sites.features.map((feature) => `<option value="${escapeHtml(feature.properties.physical_site_id)}">#${feature.properties.global_rank} · ${escapeHtml(displayRoad(feature.properties))}</option>`).join(""); updateFieldSite(); }
function updateFieldSite() { if (!state.result) return; const select = $("#fieldSiteSelect"); const feature = state.result.sites.features.find((item) => item.properties.physical_site_id === select.value) || state.result.sites.features[0]; if (!feature) return; const p = feature.properties; select.value = p.physical_site_id; $("#fieldSiteSummary").innerHTML = `<strong>#${p.global_rank} · ${escapeHtml(displayRoad(p))}</strong><br>${Number(p.latitude).toFixed(6)}, ${Number(p.longitude).toFixed(6)} · ${priorityLabel(p.global_rank)}`; const form = $("#fieldForm"); form.elements.latitude.value = p.latitude; form.elements.longitude.value = p.longitude; }
async function saveFieldResult(event) { event.preventDefault(); const form = event.currentTarget; const feature = state.result.sites.features.find((item) => item.properties.physical_site_id === $("#fieldSiteSelect").value); const p = feature.properties; const payload = { ...Object.fromEntries(new FormData(form)), observation_id: `inspection_${p.physical_site_id}`, physical_site_id: p.physical_site_id, candidate_id: p.candidate_id, road_name: displayRoad(p), predicted_latitude: p.latitude, predicted_longitude: p.longitude, priority_rank: p.global_rank, priority_bucket: priorityLabel(p.global_rank), source: "culvert_ai_field_mode" }; $("#fieldMessage").textContent = "Saving…"; try { const response = await fetch("/api/observations", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) }); const data = await response.json(); if (!response.ok) throw new Error(data.error || "Field result could not be saved."); state.observations.set(String(p.physical_site_id), payload); $("#fieldMessage").textContent = "Field result saved."; renderResultsList(); } catch (error) { $("#fieldMessage").textContent = error.message; } }

function exportCsv() { if (!state.result) return; download("culvert-ai-nearby.csv", rowsToCsv(safeExportRows(state.result.sites.features, state.observations)), "text/csv"); $("#exportPopover").classList.add("hidden"); }
function exportGeoJSON() { if (!state.result) return; download("culvert-ai-nearby.geojson", JSON.stringify(exportGeoJson(state.result.sites.features, state.observations), null, 2), "application/geo+json"); $("#exportPopover").classList.add("hidden"); }
function displayRoad(p) { return p.road_name || p.route_id || "Unnamed road"; }
function apiError(data) { const error = new Error(data.error || "Culvert AI could not complete this analysis."); error.technical = data.technical || ""; error.code = data.code || ""; return error; }
function statusLabel(status) { return ({ confirmed_culvert: "Culvert found", no_culvert: "No culvert", uncertain: "Uncertain" })[status] || "Inspected"; }
function formatNumber(value) { const number = Number(value); return Number.isFinite(number) ? number.toFixed(4) : "—"; }
function formatDistance(meters) { return meters < 1000 ? `${Math.round(meters)} m` : `${(meters / 1609.344).toFixed(1)} mi`; }
function distanceMeters(a, b) { if (!a) return 0; const radius = 6371000; const toRad = (value) => value * Math.PI / 180; const dLat = toRad(b.latitude - a.latitude); const dLon = toRad(b.longitude - a.longitude); const lat1 = toRad(a.latitude); const lat2 = toRad(b.latitude); const value = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2; return 2 * radius * Math.asin(Math.sqrt(value)); }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }
function download(filename, content, type) { const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([content], { type })); link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); }
