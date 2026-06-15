const DATA_URLS = ["/api/findings", "data/findings.geojson"];
const SUMMARY_URLS = ["/api/summary", "data/summary.json"];
const HEALTH_URL = "/api/health";
const OBSERVATIONS_URL = "/api/observations";
const LOCAL_OBSERVATIONS_KEY = "culvert-ai-field-observations";

const OBSERVATION_STATUSES = {
  confirmed_culvert: "Confirmed culvert",
  no_culvert: "No culvert found",
  uncertain: "Needs another look",
};

const state = {
  features: [],
  filtered: [],
  observations: [],
  markers: new Map(),
  selectedId: null,
  layer: null,
  knownLayer: null,
  observationLayer: null,
  map: null,
  placingPoint: false,
};

const els = {
  total: document.querySelector("#total-count"),
  visible: document.querySelector("#visible-count"),
  maxScore: document.querySelector("#max-score"),
  list: document.querySelector("#candidate-list"),
  template: document.querySelector("#candidate-template"),
  search: document.querySelector("#search-input"),
  score: document.querySelector("#score-range"),
  scoreOutput: document.querySelector("#score-output"),
  buckets: [...document.querySelectorAll('input[name="bucket"]')],
  showKnown: document.querySelector("#show-known"),
  placePoint: document.querySelector("#place-point"),
  toggleFilters: document.querySelector("#toggle-filters"),
  filterPanel: document.querySelector("#filter-panel"),
  detail: document.querySelector("#detail-panel"),
  fitVisible: document.querySelector("#fit-visible"),
  backendStatus: document.querySelector("#backend-status"),
};

init();

async function init() {
  if (!window.L) {
    showLoadError("Leaflet did not load. Check the map plugin network connection.");
    return;
  }

  setupMap();
  bindControls();
  updateBackendStatus();

  try {
    const [geojson, summary, observations] = await Promise.all([
      fetchFirst(DATA_URLS),
      fetchFirst(SUMMARY_URLS),
      fetchObservations(),
    ]);
    state.features = geojson.features.map(normalizeFeature);
    state.filtered = state.features;
    state.observations = observations;
    renderSummary(summary);
    render();
  } catch (error) {
    showLoadError(`Could not load web/data/findings.geojson. Run culvert-ai export-web first.`);
    console.error(error);
  }
}

function setupMap() {
  state.map = L.map("map", {
    zoomControl: true,
    preferCanvas: true,
  }).setView([41.73, -74.03], 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 20,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(state.map);

  state.layer = L.layerGroup().addTo(state.map);
  state.knownLayer = L.layerGroup().addTo(state.map);
  state.observationLayer = L.layerGroup().addTo(state.map);
  state.map.on("click", handleMapClick);
  requestAnimationFrame(() => state.map.invalidateSize());
}

function bindControls() {
  els.search.addEventListener("input", render);
  els.score.addEventListener("input", () => {
    els.scoreOutput.value = els.score.value;
    render();
  });
  els.buckets.forEach((input) => input.addEventListener("change", render));
  els.showKnown?.addEventListener("change", render);
  els.placePoint?.addEventListener("click", togglePlacePointMode);
  els.toggleFilters?.addEventListener("click", toggleFilterPanel);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setFilterPanelOpen(false);
    }
  });
  document.addEventListener("click", (event) => {
    if (!els.filterPanel || els.filterPanel.hidden) return;
    if (els.filterPanel.contains(event.target) || els.toggleFilters?.contains(event.target)) return;
    setFilterPanelOpen(false);
  });
  els.fitVisible.addEventListener("click", fitVisibleMarkers);
}

function toggleFilterPanel() {
  setFilterPanelOpen(Boolean(els.filterPanel?.hidden));
}

function setFilterPanelOpen(open) {
  if (!els.filterPanel || !els.toggleFilters) return;
  els.filterPanel.hidden = !open;
  els.toggleFilters.setAttribute("aria-expanded", String(open));
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchFirst(urls) {
  let lastError;
  for (const url of urls) {
    try {
      return await fetchJson(url);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

async function fetchObservations() {
  let remote = [];
  try {
    const collection = await fetchJson(OBSERVATIONS_URL);
    remote = Array.isArray(collection.features) ? collection.features : [];
  } catch {
    remote = [];
  }

  return mergeObservations([...remote, ...loadLocalObservations()]);
}

function loadLocalObservations() {
  try {
    const raw = window.localStorage?.getItem(LOCAL_OBSERVATIONS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed.features) ? parsed.features : [];
  } catch {
    return [];
  }
}

function storeLocalObservation(feature) {
  try {
    const current = loadLocalObservations();
    const merged = mergeObservations([...current, feature]);
    window.localStorage?.setItem(
      LOCAL_OBSERVATIONS_KEY,
      JSON.stringify({ type: "FeatureCollection", features: merged }),
    );
  } catch {
    // Local storage is only a fallback when the dev server has not been restarted.
  }
}

function mergeObservations(features) {
  const byId = new Map();
  features.map(normalizeObservationFeature).filter(Boolean).forEach((feature) => {
    byId.set(feature.properties.observation_id, feature);
  });
  return [...byId.values()].sort((a, b) =>
    String(b.properties.observed_at || "").localeCompare(String(a.properties.observed_at || "")),
  );
}

async function updateBackendStatus() {
  if (!els.backendStatus) return;

  try {
    const health = await fetchJson(HEALTH_URL);
    const dataReady = health.data?.findings?.exists && health.data?.summary?.exists;
    const pythonReady = health.python?.ready;
    els.backendStatus.textContent = dataReady && pythonReady ? "Backend ready" : "Backend needs setup";
  } catch {
    els.backendStatus.textContent = "Static preview";
  }
}

function normalizeFeature(feature) {
  const props = feature.properties || {};
  const coordinates = feature.geometry?.coordinates || [props.longitude, props.latitude];
  const normalizedProps = {
    ...props,
    longitude: Number(props.longitude ?? coordinates[0]),
    latitude: Number(props.latitude ?? coordinates[1]),
    score: Number(props.discovery_score ?? props.culvert_likelihood_score ?? props.culvert_probability ?? 0),
    rank: Number(props.discovery_rank ?? props.priority_rank ?? 999999),
    bucket: String(props.priority_bucket ?? "low"),
    knownFieldMatch: isTruthy(props.is_known_field_match) || isTruthy(props.is_culvert),
  };
  normalizedProps.searchText = buildSearchText(normalizedProps);

  return {
    ...feature,
    properties: normalizedProps,
  };
}

function renderSummary(summary) {
  els.total.textContent = summary.discovery_candidates ?? summary.rows ?? state.features.length;
  const score = Number(summary.max_score ?? 0);
  els.maxScore.textContent = Number.isFinite(score) ? Math.round(score) : "0";
}

function render() {
  const query = normalizeSearchText(els.search.value);
  const routeTokens = routeSearchTokens(els.search.value);
  const minScore = Number(els.score.value);
  const buckets = new Set(els.buckets.filter((input) => input.checked).map((input) => input.value));
  const showKnown = Boolean(els.showKnown?.checked);

  state.filtered = state.features
    .filter((feature) => showKnown || !feature.properties.knownFieldMatch)
    .filter((feature) => feature.properties.score >= minScore)
    .filter((feature) => buckets.has(feature.properties.bucket))
    .filter((feature) => {
      if (!query) return true;
      const props = feature.properties;
      if (props.searchText.includes(query)) return true;
      if (routeTokens.length) {
        return routeTokens.some((token) => props.routeTokens.includes(token));
      }
      return false;
    })
    .sort((a, b) => a.properties.rank - b.properties.rank);

  updateVisibleCount();
  renderMarkers();
  renderList();

  if (!state.selectedId && state.filtered.length) {
    selectFeature(state.filtered[0].properties.candidate_id, { pan: false });
  } else if (state.selectedId && !state.filtered.some((feature) => idOf(feature) === state.selectedId)) {
    if (state.filtered.length) {
      selectFeature(state.filtered[0].properties.candidate_id, { pan: false });
    } else {
      clearDetail();
    }
  }
}

function renderMarkers() {
  state.layer.clearLayers();
  state.knownLayer.clearLayers();
  state.markers.clear();

  state.filtered.filter((feature) => !feature.properties.knownFieldMatch).forEach((feature) => {
    const props = feature.properties;
    if (!Number.isFinite(props.latitude) || !Number.isFinite(props.longitude)) return;

    const marker = L.marker([props.latitude, props.longitude], {
      icon: markerIcon(props),
      riseOnHover: true,
    });
    marker.bindPopup(popupHtml(props));
    marker.on("click", () => selectFeature(idOf(feature), { pan: false }));
    marker.addTo(state.layer);
    state.markers.set(idOf(feature), marker);
  });

  knownFeatures().forEach((feature) => {
    const props = feature.properties;
    if (!Number.isFinite(props.latitude) || !Number.isFinite(props.longitude)) return;

    const marker = L.marker([props.latitude, props.longitude], {
      icon: knownMarkerIcon(props),
      riseOnHover: true,
      zIndexOffset: 700,
    });
    marker.bindTooltip(knownCulvertLabel(props), {
      permanent: true,
      direction: "top",
      offset: [0, -18],
      className: "known-culvert-label",
    });
    marker.bindPopup(popupHtml(props));
    marker.on("click", () => selectFeature(idOf(feature), { pan: false }));
    marker.addTo(state.knownLayer);
    state.markers.set(idOf(feature), marker);
  });

  renderObservationMarkers();
}

function renderList() {
  els.list.replaceChildren();
  const fragment = document.createDocumentFragment();

  state.filtered.forEach((feature) => {
    const props = feature.properties;
    const item = els.template.content.firstElementChild.cloneNode(true);
    const button = item.querySelector(".candidate-button");
    button.classList.toggle("active", idOf(feature) === state.selectedId);
    button.addEventListener("click", () => selectFeature(idOf(feature), { pan: true }));
    item.querySelector(".rank").textContent = props.knownFieldMatch ? "K" : props.rank;
    item.querySelector(".candidate-title").textContent = props.road_name || "Unnamed road";
    item.querySelector(".candidate-subtitle").textContent = listSubtitle(props);
    const pill = item.querySelector(".score-pill");
    pill.textContent = props.knownFieldMatch ? "Known" : Math.round(props.score);
    pill.classList.add(props.knownFieldMatch ? "known-score" : `bucket-${props.bucket}`);
    fragment.append(item);
  });

  els.list.append(fragment);
}

function selectFeature(candidateId, options = { pan: true }) {
  const feature = state.features.find((item) => idOf(item) === candidateId);
  if (!feature) return;
  state.selectedId = candidateId;
  renderDetail(feature);
  renderList();

  const marker = state.markers.get(candidateId);
  const latLng = featureLatLng(feature);
  if (options.pan && latLng) {
    state.map.setView(latLng, Math.max(state.map.getZoom(), 15), { animate: true });
  }

  if (marker) {
    marker.openPopup();
  }
}

function renderDetail(feature) {
  const props = feature.properties;
  const title = props.knownFieldMatch
    ? `Known culvert: ${knownCulvertLabel(props)}`
    : `Rank ${formatValue(props.rank)}: ${props.road_name || "Unnamed road"}`;
  els.detail.innerHTML = `
    <h3>${escapeHtml(title)}</h3>
    <p>${escapeHtml(props.evidence_summary || "No evidence summary available.")}</p>
    <div class="detail-grid">
      ${detailCell("Discovery score", Math.round(props.score))}
      ${detailCell("Status", discoveryStatusLabel(props))}
      ${detailCell("Model rank", formatScorePartFrom100(props.model_rank_score))}
      ${detailCell("Model probability", formatPercent(props.culvert_probability))}
      ${detailCell("GIS evidence", formatScorePartFrom100(props.evidence_score ?? props.culvert_likelihood_score))}
      ${detailCell("Priority", labelBucket(props.bucket))}
      ${detailCell("Drainage", props.stream_name || "Unknown")}
      ${detailCell("Crossing angle", formatNumber(props.crossing_angle_degrees, "deg"))}
      ${detailCell("Road-drainage", formatScorePart(props.road_stream_proximity_score))}
      ${detailCell("Drainage strength", formatScorePart(props.drainage_strength_score))}
      ${detailCell("Crossing geometry", formatScorePart(props.crossing_geometry_score))}
      ${detailCell("Road context", formatScorePart(props.road_context_score))}
      ${detailCell("Valley position", formatScorePart(props.valley_position_score))}
      ${detailCell("Terrain break", formatScorePart(props.terrain_break_score))}
      ${detailCell("Field report", formatScorePart(props.field_report_support_score))}
      ${detailCell("Field match", props.knownFieldMatch ? "yes" : "no")}
      ${detailCell("Report date", props.nearest_field_report_date || props.field_report_date || "n/a")}
      ${detailCell("Culvert ID", props.nearest_field_report_culvert_id || "n/a")}
      ${detailCell("Field distance", formatNumber(props.dist_to_known_culvert_m, "m"))}
      ${detailCell("Valley depth", formatNumber(props.valley_depth_9x9_m, "m"))}
      ${detailCell("Slope", formatNumber(props.slope_degrees, "deg"))}
      ${detailCell("Lat", formatNumber(props.latitude, ""))}
      ${detailCell("Lon", formatNumber(props.longitude, ""))}
    </div>
    <div class="actions">
      ${props.google_earth_url ? `<a href="${escapeAttr(props.google_earth_url)}" target="_blank" rel="noreferrer">Open Google Earth</a>` : ""}
      <a href="https://www.google.com/maps/search/?api=1&query=${props.latitude},${props.longitude}" target="_blank" rel="noreferrer">Open Google Maps</a>
    </div>
    ${fieldFeedbackHtml("Field verification")}
  `;
  bindFeedbackActions((status, notes) => saveObservationForFeature(feature, status, notes));
}

function clearDetail() {
  state.selectedId = null;
  els.detail.innerHTML = `<div class="empty-state">Select a ranked location.</div>`;
}

function listSubtitle(props) {
  if (props.knownFieldMatch) {
    const date = props.nearest_field_report_date || props.field_report_date;
    return date ? `field report match · ${date}` : "field report match";
  }
  const stream = props.stream_name && props.stream_name !== "route sample" ? props.stream_name : props.source;
  return stream ? `undiscovered candidate · ${stream}` : "undiscovered candidate";
}

function fitVisibleMarkers() {
  const latLngs = [
    ...state.filtered.filter((feature) => !feature.properties.knownFieldMatch).map(featureLatLng),
    ...knownFeatures().map(featureLatLng),
    ...state.observations.map(observationLatLng),
  ].filter(Boolean);

  if (!latLngs.length) return;
  state.map.fitBounds(latLngs, { padding: [28, 28], maxZoom: 15 });
}

function knownFeatures() {
  return state.features.filter((feature) => feature.properties.knownFieldMatch);
}

function featureLatLng(feature) {
  const lat = Number(feature.properties.latitude);
  const lon = Number(feature.properties.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return [lat, lon];
}

function markerIcon(props) {
  const bucket = props.bucket || "low";
  const html = `<span class="marker-dot bucket-${bucket}">${Math.round(props.score)}</span>`;
  return L.divIcon({
    className: `priority-marker${props.knownFieldMatch ? " known-marker" : ""}`,
    html,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -16],
  });
}

function knownMarkerIcon(props) {
  const label = knownCulvertShortLabel(props);
  return L.divIcon({
    className: "known-culvert-marker",
    html: `<span class="known-marker-dot">${escapeHtml(label)}</span>`,
    iconSize: [38, 38],
    iconAnchor: [19, 19],
    popupAnchor: [0, -18],
  });
}

function renderObservationMarkers() {
  if (!state.observationLayer) return;
  state.observationLayer.clearLayers();

  state.observations.forEach((feature) => {
    const latLng = observationLatLng(feature);
    if (!latLng) return;

    const marker = L.marker(latLng, {
      icon: observationIcon(feature.properties.status),
      riseOnHover: true,
      zIndexOffset: 900,
    });
    marker.bindPopup(observationPopupHtml(feature.properties));
    marker.addTo(state.observationLayer);
  });
}

function observationIcon(status) {
  const normalized = observationStatus(status);
  const text = normalized === "confirmed_culvert" ? "OK" : normalized === "no_culvert" ? "NO" : "?";
  return L.divIcon({
    className: `observation-marker observation-${normalized}`,
    html: `<span class="observation-dot">${text}</span>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -16],
  });
}

function popupHtml(props) {
  if (props.knownFieldMatch) {
    return `
      <div class="popup-title">Known culvert: ${escapeHtml(knownCulvertLabel(props))}</div>
      <div class="popup-meta">${escapeHtml(props.road_name || "Unnamed road")}</div>
      <div class="popup-meta">${escapeHtml(props.stream_name || "Unnamed drainage")}</div>
      <div class="popup-meta">Report ${escapeHtml(props.nearest_field_report_date || props.field_report_date || "n/a")}</div>
    `;
  }

  return `
    <div class="popup-title">Rank ${formatValue(props.rank)}: ${escapeHtml(props.road_name || "Unnamed road")}</div>
    <div class="popup-meta">${escapeHtml(props.stream_name || "Unnamed drainage")}</div>
    <div class="popup-meta">Discovery ${Math.round(props.score)} · ${labelBucket(props.bucket)}</div>
  `;
}

function knownCulvertLabel(props) {
  return firstPresent([
    props.nearest_field_report_culvert_id,
    props.culvert_id,
    props.nearest_field_report_route,
    props.road_name,
    "Known culvert",
  ]);
}

function knownCulvertShortLabel(props) {
  const label = knownCulvertLabel(props);
  if (/^sc\d+/i.test(label)) return label.replace(/^sc/i, "SC").slice(0, 6);
  return "K";
}

function fieldFeedbackHtml(title) {
  return `
    <section class="field-feedback" aria-label="${escapeAttr(title)}">
      <h4>${escapeHtml(title)}</h4>
      <label for="field-notes">Field notes</label>
      <textarea id="field-notes" rows="3" placeholder="Optional notes from inspection"></textarea>
      <div class="feedback-buttons">
        <button type="button" data-feedback-status="confirmed_culvert">Confirm culvert</button>
        <button type="button" data-feedback-status="no_culvert">Deny culvert</button>
        <button type="button" data-feedback-status="uncertain">Uncertain</button>
      </div>
      <p id="feedback-status" class="feedback-status"></p>
    </section>
  `;
}

function bindFeedbackActions(saveHandler) {
  const statusOutput = els.detail.querySelector("#feedback-status");
  const notesInput = els.detail.querySelector("#field-notes");
  els.detail.querySelectorAll("[data-feedback-status]").forEach((button) => {
    button.addEventListener("click", async () => {
      const status = button.dataset.feedbackStatus;
      const notes = notesInput?.value || "";
      button.disabled = true;
      if (statusOutput) statusOutput.textContent = "Saving field observation...";
      try {
        const result = await saveHandler(status, notes);
        const storage =
          result.storage === "file"
            ? "Saved to data/processed/field_observations.geojson."
            : "Saved in this browser until the server is restarted.";
        if (statusOutput) {
          statusOutput.textContent = `${statusLabel(status)}. ${storage}`;
        }
      } catch (error) {
        if (statusOutput) {
          statusOutput.textContent = `Could not save observation: ${error.message || error}`;
        }
      } finally {
        button.disabled = false;
      }
    });
  });
}

async function saveObservationForFeature(feature, status, notes) {
  const latLng = featureLatLng(feature);
  if (!latLng) throw new Error("Selected feature has no coordinate.");
  const props = feature.properties;
  return saveObservation({
    status,
    notes,
    latitude: latLng[0],
    longitude: latLng[1],
    candidate_id: idOf(feature),
    road_name: props.road_name || "",
    stream_name: props.stream_name || "",
    source: props.knownFieldMatch ? "known_culvert_review" : "prediction_review",
    prediction_score: props.score,
    priority_rank: props.rank,
    priority_bucket: props.bucket,
  });
}

async function saveObservationAtPoint(latLng, status, notes) {
  return saveObservation({
    status,
    notes,
    latitude: latLng.lat,
    longitude: latLng.lng,
    candidate_id: "",
    road_name: "",
    stream_name: "",
    source: "manual_field_point",
  });
}

async function saveObservation(payload) {
  const feature = observationFeatureFromPayload(payload);

  try {
    const response = await fetch(OBSERVATIONS_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(feature.properties),
    });
    if (!response.ok) {
      throw new Error(`server returned ${response.status}`);
    }
    const saved = await response.json();
    const savedFeature = normalizeObservationFeature(saved.feature || feature);
    addObservation(savedFeature);
    return { feature: savedFeature, storage: "file" };
  } catch {
    storeLocalObservation(feature);
    addObservation(feature);
    return { feature, storage: "browser" };
  }
}

function addObservation(feature) {
  const normalized = normalizeObservationFeature(feature);
  if (!normalized) return;
  const index = state.observations.findIndex(
    (item) => item.properties.observation_id === normalized.properties.observation_id,
  );
  if (index >= 0) {
    state.observations[index] = normalized;
  } else {
    state.observations.unshift(normalized);
  }
  renderObservationMarkers();
  updateVisibleCount();
}

function updateVisibleCount() {
  const visiblePredictions = state.filtered.filter((feature) => !feature.properties.knownFieldMatch);
  els.visible.textContent = visiblePredictions.length + knownFeatures().length + state.observations.length;
}

function observationFeatureFromPayload(payload) {
  const latitude = Number(payload.latitude);
  const longitude = Number(payload.longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    throw new Error("Observation needs a valid latitude and longitude.");
  }

  return {
    type: "Feature",
    properties: {
      observation_id: payload.observation_id || makeObservationId(),
      observed_at: payload.observed_at || new Date().toISOString(),
      status: observationStatus(payload.status),
      candidate_id: payload.candidate_id || "",
      road_name: payload.road_name || "",
      stream_name: payload.stream_name || "",
      notes: payload.notes || "",
      source: payload.source || "field_review",
      prediction_score: numberOrNull(payload.prediction_score),
      priority_rank: numberOrNull(payload.priority_rank),
      priority_bucket: payload.priority_bucket || "",
      latitude,
      longitude,
    },
    geometry: {
      type: "Point",
      coordinates: [longitude, latitude],
    },
  };
}

function normalizeObservationFeature(feature) {
  if (!feature || feature.geometry?.type !== "Point") return null;
  const props = feature.properties || {};
  const coordinates = feature.geometry.coordinates || [props.longitude, props.latitude];
  const longitude = Number(props.longitude ?? coordinates[0]);
  const latitude = Number(props.latitude ?? coordinates[1]);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;

  return observationFeatureFromPayload({
    ...props,
    observation_id: props.observation_id,
    observed_at: props.observed_at,
    latitude,
    longitude,
  });
}

function observationLatLng(feature) {
  const props = feature.properties || {};
  const lat = Number(props.latitude);
  const lon = Number(props.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return [lat, lon];
}

function observationPopupHtml(props) {
  const title = statusLabel(props.status);
  const road = props.road_name || props.candidate_id || "Manual field point";
  return `
    <div class="popup-title">${escapeHtml(title)}</div>
    <div class="popup-meta">${escapeHtml(road)}</div>
    <div class="popup-meta">${escapeHtml(props.observed_at || "")}</div>
    ${props.notes ? `<div class="popup-meta">${escapeHtml(props.notes)}</div>` : ""}
  `;
}

function togglePlacePointMode() {
  state.placingPoint = !state.placingPoint;
  updatePlacePointButton();
  if (state.placingPoint) {
    state.selectedId = null;
    renderList();
    els.detail.innerHTML = `
      <h3>Place field point</h3>
      <p>Click the map where you inspected or want to record a field observation.</p>
    `;
  }
}

function updatePlacePointButton() {
  if (!els.placePoint) return;
  els.placePoint.setAttribute("aria-pressed", String(state.placingPoint));
  els.placePoint.classList.toggle("active", state.placingPoint);
  document.body.classList.toggle("placing-point", state.placingPoint);
}

function handleMapClick(event) {
  if (!state.placingPoint) return;
  state.placingPoint = false;
  updatePlacePointButton();
  renderManualPointDetail(event.latlng);
}

function renderManualPointDetail(latLng) {
  state.selectedId = null;
  renderList();
  els.detail.innerHTML = `
    <h3>New field point</h3>
    <p>Lat ${formatNumber(latLng.lat, "")}, Lon ${formatNumber(latLng.lng, "")}</p>
    ${fieldFeedbackHtml("Record observation")}
  `;
  bindFeedbackActions((status, notes) => saveObservationAtPoint(latLng, status, notes));
}

function observationStatus(status) {
  return Object.hasOwn(OBSERVATION_STATUSES, status) ? status : "uncertain";
}

function statusLabel(status) {
  return OBSERVATION_STATUSES[observationStatus(status)];
}

function makeObservationId() {
  return `obs_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function firstPresent(values) {
  const value = values.find((item) => String(item || "").trim());
  return String(value || "n/a").trim();
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function detailCell(label, value) {
  return `<div><label>${escapeHtml(label)}</label><span>${escapeHtml(formatValue(value))}</span></div>`;
}

function labelBucket(bucket) {
  return String(bucket || "unknown").replace("_", " ");
}

function formatNumber(value, suffix) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  const decimals = suffix ? 1 : 5;
  return `${number.toFixed(decimals)}${suffix ? ` ${suffix}` : ""}`;
}

function formatScorePart(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return `${Math.round(number * 100)}/100`;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  if (number > 0 && number < 0.005) return "<1%";
  return `${Math.round(number * 100)}%`;
}

function formatScorePartFrom100(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "n/a";
  return `${Math.round(number)}/100`;
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  return String(value);
}

function discoveryStatusLabel(props) {
  if (props.knownFieldMatch || props.discovery_status === "known_field_match") {
    return "known field match";
  }
  return "new target";
}

function isTruthy(value) {
  if (value === true) return true;
  if (typeof value === "number") return value === 1;
  return ["1", "true", "yes"].includes(String(value || "").toLowerCase());
}

function idOf(feature) {
  return String(feature.properties.candidate_id || feature.properties.priority_rank);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function showLoadError(message) {
  els.detail.innerHTML = `<div class="load-error">${escapeHtml(message)}</div>`;
}

function buildSearchText(props) {
  const routeTokens = routeSearchTokens(
    [props.road_name, props.road_id, props.road_highway].filter(Boolean).join(" "),
  );
  props.routeTokens = routeTokens;
  return normalizeSearchText(
    [
      props.road_name,
      props.stream_name,
      props.evidence_summary,
      props.discovery_status,
      props.candidate_id,
      props.nearest_field_report_culvert_id,
      props.nearest_field_report_route,
      props.nearest_field_report_date,
      props.road_id,
      props.stream_id,
      props.road_highway,
      routeTokens.join(" "),
    ].join(" "),
  );
}

function normalizeSearchText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\bstate\s+(route|rte|rt)\b/g, "route")
    .replace(/\bcounty\s+(road|route|rte|rt)\b/g, "county road")
    .replace(/\bco\s+(road|route|rte|rd|rt)\b/g, "county road")
    .replace(/\binterstate\b/g, "i")
    .replace(/\broute\b|\brte\b|\brt\b/g, "route")
    .replace(/\bny\s*-?\s*(\d+[a-z]?)\b/g, "route $1")
    .replace(/\bnys\s*-?\s*(\d+[a-z]?)\b/g, "route $1")
    .replace(/\bus\s*-?\s*(\d+[a-z]?)\b/g, "route $1")
    .replace(/\bi\s*-?\s*(\d+[a-z]?)\b/g, "i $1")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function routeSearchTokens(value) {
  const text = String(value || "").toLowerCase();
  const tokens = new Set();
  const patterns = [
    /\b(?:state\s+)?(?:route|rte|rt)\s*-?\s*(\d+[a-z]?)\b/g,
    /\bny\s*-?\s*(\d+[a-z]?)\b/g,
    /\bnys\s*-?\s*(\d+[a-z]?)\b/g,
    /\bus\s*-?\s*(\d+[a-z]?)\b/g,
    /\b(?:county\s+(?:road|route|rte|rt)|co\s+(?:road|route|rte|rd|rt))\s*-?\s*(\d+[a-z]?)\b/g,
    /\b(?:interstate|i)\s*-?\s*(\d+[a-z]?)\b/g,
  ];

  patterns.forEach((pattern) => {
    for (const match of text.matchAll(pattern)) {
      tokens.add(`route-${match[1]}`);
    }
  });

  return [...tokens];
}
