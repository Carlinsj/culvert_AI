const DATA_URLS = ["/api/findings", "data/findings.geojson"];
const SUMMARY_URLS = ["/api/summary", "data/summary.json"];
const MODEL_SUMMARY_URLS = ["data/model_summary.json"];
const HEALTH_URL = "/api/health";
const OBSERVATIONS_URL = "/api/observations";
const LOCAL_OBSERVATIONS_KEY = "culvert-ai-field-observations";

const OBSERVATION_STATUSES = {
  confirmed_culvert: "Confirmed culvert",
  no_culvert: "No culvert found",
  uncertain: "Needs another look",
};
const MOBILE_VIEWPORT_QUERY = "(max-width: 820px)";
const NEARBY_FOCUS_LIMIT = 12;
const FIELD_CONTEXT_RADIUS_M = 100;
const SELECTED_POINT_ZOOM = 16;
const SELECTION_POPUP_DELAY_MS = 460;
const LOCATION_FOCUS_ZOOM = 16;
const LOCATION_MIN_MOVE_M = 4;
const LOCATION_LIST_THROTTLE_MS = 900;

const state = {
  features: [],
  filtered: [],
  observations: [],
  markers: new Map(),
  observationMarkers: new Map(),
  selectedId: null,
  selectedObservationId: null,
  listView: "ranked",
  layer: null,
  knownLayer: null,
  observationLayer: null,
  locationLayer: null,
  map: null,
  placingPoint: false,
  locationWatchId: null,
  userLocation: null,
  locationMarker: null,
  locationAccuracyCircle: null,
  lastLocationListRenderAt: 0,
  shouldFocusLocationOnNextUpdate: false,
  modelSummary: null,
};

const els = {
  total: document.querySelector("#total-count"),
  visible: document.querySelector("#visible-count"),
  maxScore: document.querySelector("#max-score"),
  fieldLabels: document.querySelector("#field-label-count"),
  modelQuality: document.querySelector("#model-quality"),
  list: document.querySelector("#candidate-list"),
  listHeading: document.querySelector("#list-heading"),
  listViewButtons: [...document.querySelectorAll("[data-list-view]")],
  abuTab: document.querySelector("#abu-tab"),
  template: document.querySelector("#candidate-template"),
  search: document.querySelector("#search-input"),
  score: document.querySelector("#score-range"),
  scoreOutput: document.querySelector("#score-output"),
  buckets: [...document.querySelectorAll('input[name="bucket"]')],
  showKnown: document.querySelector("#show-known"),
  placePoint: document.querySelector("#place-point"),
  mobileAddPoint: document.querySelector("#mobile-add-point"),
  toggleFilters: document.querySelector("#toggle-filters"),
  filterPanel: document.querySelector("#filter-panel"),
  detail: document.querySelector("#detail-panel"),
  detailModal: document.querySelector("#detail-modal"),
  detailModalBody: document.querySelector("#detail-modal-body"),
  detailModalTitle: document.querySelector("#detail-modal-title"),
  detailModalClose: document.querySelector("#detail-modal-close"),
  fitVisible: document.querySelector("#fit-visible"),
  backendStatus: document.querySelector("#backend-status"),
  sidebar: document.querySelector("#mobile-sidebar"),
  drawerBackdrop: document.querySelector("#drawer-backdrop"),
  mobileMenuToggle: document.querySelector("#mobile-menu-toggle"),
  mobileSidebarClose: document.querySelector("#mobile-sidebar-close"),
  locateMe: document.querySelector("#locate-me"),
  locationStatus: document.querySelector("#location-status"),
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
    const [geojson, summary, observations, modelSummary] = await Promise.all([
      fetchFirst(DATA_URLS),
      fetchFirst(SUMMARY_URLS),
      fetchObservations(),
      fetchFirstOptional(MODEL_SUMMARY_URLS),
    ]);
    state.modelSummary = modelSummary;
    applyDashboardData(geojson, summary, observations);
    syncLocalObservationsToServer();
  } catch (error) {
    showLoadError(`Could not load web/data/findings.geojson. Run culvert-ai export-web first.`);
    console.error(error);
  }
}

function applyDashboardData(geojson, summary, observations, options = {}) {
  const selectedId = options.preserveSelection ? state.selectedId : null;
  const selectedObservationId = options.preserveSelection ? state.selectedObservationId : null;
  const observationFeatures = Array.isArray(observations) ? observations : observations?.features || [];
  state.features = (geojson.features || []).map(normalizeFeature);
  state.filtered = state.features;
  state.observations = mergeObservations([...observationFeatures, ...loadLocalObservations()]);
  state.selectedId = selectedId && state.features.some((feature) => idOf(feature) === selectedId) ? selectedId : null;
  state.selectedObservationId =
    selectedObservationId && state.observations.some((feature) => observationIdOf(feature) === selectedObservationId)
      ? selectedObservationId
      : null;
  renderSummary(summary);
  render();
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
  state.locationLayer = L.layerGroup().addTo(state.map);
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
  els.listViewButtons.forEach((button) => {
    button.addEventListener("click", () => setListView(button.dataset.listView || "ranked"));
  });
  els.placePoint?.addEventListener("click", togglePlacePointMode);
  els.mobileAddPoint?.addEventListener("click", togglePlacePointMode);
  els.toggleFilters?.addEventListener("click", toggleFilterPanel);
  els.mobileMenuToggle?.addEventListener("click", () => setMobileDrawerOpen(true));
  els.mobileSidebarClose?.addEventListener("click", () => setMobileDrawerOpen(false));
  els.drawerBackdrop?.addEventListener("click", () => setMobileDrawerOpen(false));
  els.locateMe?.addEventListener("click", toggleLocationTracking);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setFilterPanelOpen(false);
      setMobileDrawerOpen(false);
    }
  });
  document.addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-observation-delete]");
    if (deleteButton) {
      event.preventDefault();
      deleteObservationById(deleteButton.dataset.observationDelete, deleteButton);
      return;
    }

    if (!els.filterPanel || els.filterPanel.hidden) return;
    if (
      els.filterPanel.contains(event.target) ||
      els.toggleFilters?.contains(event.target) ||
      els.mobileMenuToggle?.contains(event.target) ||
      els.sidebar?.contains(event.target)
    ) {
      return;
    }
    setFilterPanelOpen(false);
  });
  els.fitVisible.addEventListener("click", fitVisibleMarkers);
  window.addEventListener("resize", () => state.map?.invalidateSize());
  els.detailModal?.addEventListener("click", (event) => {
    if (event.target === els.detailModal) {
      els.detailModal.close();
    }
  });
}

function toggleFilterPanel() {
  setFilterPanelOpen(Boolean(els.filterPanel?.hidden));
}

function setFilterPanelOpen(open) {
  if (!els.filterPanel || !els.toggleFilters) return;
  els.filterPanel.hidden = !open;
  els.toggleFilters.setAttribute("aria-expanded", String(open));
}

function setMobileDrawerOpen(open) {
  document.body.classList.toggle("mobile-drawer-open", open);
  els.mobileMenuToggle?.setAttribute("aria-expanded", String(open));
  if (els.drawerBackdrop) {
    els.drawerBackdrop.hidden = !open;
  }
  window.requestAnimationFrame(() => state.map?.invalidateSize());
}

function setListView(view) {
  state.listView = view === "abu" ? "abu" : "ranked";
  if (state.listView === "abu") {
    setFilterPanelOpen(false);
  }
  updateListViewControls();
  renderList();
}

function isMobileViewport() {
  return window.matchMedia(MOBILE_VIEWPORT_QUERY).matches;
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

async function fetchFirstOptional(urls) {
  try {
    return await fetchFirst(urls);
  } catch {
    return null;
  }
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

function removeLocalObservation(observationId) {
  try {
    const current = loadLocalObservations();
    const filtered = current.filter((feature) => feature.properties?.observation_id !== observationId);
    window.localStorage?.setItem(
      LOCAL_OBSERVATIONS_KEY,
      JSON.stringify({ type: "FeatureCollection", features: filtered }),
    );
  } catch {
    // Local storage is only a fallback when the dev server has not been restarted.
  }
}

function clearLocalObservations() {
  try {
    window.localStorage?.removeItem(LOCAL_OBSERVATIONS_KEY);
  } catch {
    // Local storage is a best-effort offline recovery cache.
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
  if (els.fieldLabels) {
    els.fieldLabels.textContent = summary.known_field_matches ?? 0;
  }
  renderModelQuality();
}

function renderModelQuality() {
  if (!els.modelQuality) return;
  const summary = state.modelSummary;
  if (!summary?.available) {
    els.modelQuality.textContent = "Model quality unavailable";
    return;
  }

  const modelName = formatModelName(summary.selected_model);
  const spatialAp = formatMetric(summary.spatial_holdout_average_precision);
  const precisionAt10 = formatMetric(summary.spatial_holdout_top10_precision);
  const labels = Number(summary.training_points ?? summary.positive_labels ?? 0);
  els.modelQuality.textContent =
    `${modelName} · spatial AP ${spatialAp} · P@10 ${precisionAt10} · ${labels} QC labels`;
}

function formatModelName(value) {
  return String(value || "model")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatMetric(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "n/a";
}

function render() {
  const query = normalizeSearchText(els.search.value);
  const routeTokens = routeSearchTokens(els.search.value);
  const minScore = Number(els.score.value);
  const buckets = new Set(els.buckets.filter((input) => input.checked).map((input) => input.value));
  const showKnown = Boolean(els.showKnown?.checked);
  updateFeatureDistances();

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
    .sort(compareFeaturesForList);

  updateVisibleCount();
  renderMarkers();
  renderList();

  const shouldAutoSelect = !isMobileViewport();
  if (!state.selectedId && state.filtered.length && shouldAutoSelect) {
    selectFeature(state.filtered[0].properties.candidate_id, { pan: false });
  } else if (state.selectedId && !state.filtered.some((feature) => idOf(feature) === state.selectedId)) {
    if (state.filtered.length && shouldAutoSelect) {
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
    marker.bindPopup(popupHtml(props), selectedPopupOptions());
    marker.on("click", () => selectFeature(idOf(feature), { pan: true }));
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
    marker.bindPopup(popupHtml(props), selectedPopupOptions());
    marker.on("click", () => selectFeature(idOf(feature), { pan: true }));
    marker.addTo(state.knownLayer);
    state.markers.set(idOf(feature), marker);
  });

  renderObservationMarkers();
  updateSelectedMarkerClass();
}

function renderList() {
  els.list.replaceChildren();
  updateListViewControls();

  if (state.listView === "abu") {
    renderAbuList();
    return;
  }

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

function renderAbuList() {
  const fragment = document.createDocumentFragment();
  const observations = abuObservations();

  if (!observations.length) {
    const empty = document.createElement("li");
    empty.className = "list-empty";
    empty.textContent = "No ABU points saved yet.";
    fragment.append(empty);
    els.list.append(fragment);
    return;
  }

  observations.forEach((feature, index) => {
    const props = feature.properties || {};
    const observationId = observationIdOf(feature);
    const item = document.createElement("li");
    item.className = "abu-list-item";
    item.innerHTML = `
      <div class="abu-card">
        <button type="button" class="abu-select${observationId === state.selectedObservationId ? " active" : ""}">
          <span class="rank">ABU</span>
          <span class="candidate-main">
            <strong class="candidate-title">${escapeHtml(observationTitle(props))} ${escapeHtml(String(index + 1))}</strong>
            <span class="candidate-subtitle">${escapeHtml(observationSubtitle(props))}</span>
          </span>
        </button>
        <button type="button" class="abu-delete" data-observation-delete="${escapeAttr(observationId)}">Delete</button>
      </div>
    `;
    item.querySelector(".abu-select")?.addEventListener("click", () => selectObservation(observationId, { pan: true }));
    fragment.append(item);
  });

  els.list.append(fragment);
}

function updateListViewControls() {
  if (els.listHeading) {
    els.listHeading.textContent = state.listView === "abu" ? "ABU points" : "Ranked locations";
  }
  els.listViewButtons.forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.listView === state.listView));
  });
  if (els.abuTab) {
    const count = abuObservations().length;
    els.abuTab.textContent = count ? `ABU (${count})` : "ABU";
  }
}

function selectFeature(candidateId, options = { pan: true }) {
  const feature = state.features.find((item) => idOf(item) === candidateId);
  if (!feature) return;
  setMobileDrawerOpen(false);
  state.shouldFocusLocationOnNextUpdate = false;
  state.selectedId = candidateId;
  state.selectedObservationId = null;
  renderDetail(feature);
  renderList();
  updateSelectedMarkerClass();

  const marker = state.markers.get(candidateId);
  const latLng = featureLatLng(feature);
  const shouldCenter = options.pan !== false && latLng;
  if (shouldCenter) {
    centerMapOnPoint(latLng);
  }

  if (marker && options.openPopup !== false && !isMobileViewport()) {
    openSelectedPopup(marker, shouldCenter);
  }

  if (options.pan !== false) {
    window.requestAnimationFrame(scrollDetailIntoViewOnMobile);
  }
}

function selectObservation(observationId, options = { pan: true }) {
  const feature = state.observations.find((item) => observationIdOf(item) === observationId);
  if (!feature) return;
  setMobileDrawerOpen(false);
  state.shouldFocusLocationOnNextUpdate = false;
  state.selectedId = null;
  state.selectedObservationId = observationId;
  renderObservationDetail(feature);
  renderList();
  updateSelectedMarkerClass();

  const marker = state.observationMarkers.get(observationId);
  const latLng = observationLatLng(feature);
  const shouldCenter = options.pan !== false && latLng;
  if (shouldCenter) {
    centerMapOnPoint(latLng);
  }

  if (marker && options.openPopup !== false && !isMobileViewport()) {
    openSelectedPopup(marker, shouldCenter);
  }

  if (options.pan !== false) {
    window.requestAnimationFrame(scrollDetailIntoViewOnMobile);
  }
}

function scrollDetailIntoViewOnMobile() {
  if (!window.matchMedia("(max-width: 820px)").matches) return;
  state.map?.invalidateSize();
}

function centerMapOnPoint(latLng) {
  state.map.stop();
  const zoom = Math.max(state.map.getZoom(), SELECTED_POINT_ZOOM);
  state.map.flyTo(latLng, zoom, {
    animate: true,
    duration: 0.32,
    easeLinearity: 0.45,
  });
}

function openSelectedPopup(marker, delayed) {
  if (!delayed) {
    marker.openPopup();
    updateSelectedMarkerClass();
    return;
  }

  window.setTimeout(() => {
    marker.openPopup();
    updateSelectedMarkerClass();
  }, SELECTION_POPUP_DELAY_MS);
}

function selectedPopupOptions() {
  return {
    autoPan: false,
    keepInView: false,
  };
}

function updateSelectedMarkerClass() {
  state.markers.forEach((marker, markerId) => {
    marker.getElement()?.classList.toggle("selected-marker", markerId === state.selectedId);
  });
  state.observationMarkers.forEach((marker, markerId) => {
    marker.getElement()?.classList.toggle("selected-marker", markerId === state.selectedObservationId);
  });
}

function renderDetail(feature) {
  const props = feature.properties;
  const displayId = locationDisplayId(props);
  const title = props.knownFieldMatch
    ? `Known culvert · ${displayId}`
    : `Rank ${formatValue(props.rank)} · ${displayId}`;
  showDetailPanel();
  els.detail.innerHTML = `
    <div class="detail-panel-header">
      <h3>${escapeHtml(title)}</h3>
      <button type="button" class="detail-close" data-close-detail aria-label="Close details">Close</button>
    </div>
    <p>${escapeHtml(compactEvidenceSummary(props))}</p>
    <div class="quick-detail-grid">
      ${detailCell("Estimate", Math.round(props.score))}
      ${detailCell("Priority", labelBucket(props.bucket))}
      ${detailCell("Road", props.road_name || "Unnamed road")}
      ${detailCell("Drainage/source", drainageLabel(props))}
    </div>
    ${locationSummaryHtml(props)}
    <button id="open-detail-modal" type="button" class="secondary-action">More details</button>
    ${fieldFeedbackHtml("Field verification")}
  `;
  bindDetailCloseAction();
  els.detail.querySelector("#open-detail-modal")?.addEventListener("click", () => openDetailModal(feature));
  bindFeedbackActions((status, notes) => saveObservationForFeature(feature, status, notes));
}

function openDetailModal(feature) {
  const props = feature.properties;
  if (!els.detailModal || !els.detailModalBody || !els.detailModalTitle) return;

  els.detailModalTitle.textContent = `${locationDisplayId(props)} details`;
  els.detailModalBody.innerHTML = `
    <section class="modal-section">
      <h3>Location</h3>
      <div class="detail-grid">
        ${detailCell("Readable ID", locationDisplayId(props))}
        ${detailCell("Source ID", readableSourceId(props))}
        ${detailCell("Status", discoveryStatusLabel(props))}
        ${detailCell("Road", props.road_name || "Unnamed road")}
        ${detailCell("Drainage/source", drainageLabel(props))}
        ${detailCell("Latitude", formatNumber(props.latitude, ""))}
        ${detailCell("Longitude", formatNumber(props.longitude, ""))}
        ${detailCell("Crossing angle", formatNumber(props.crossing_angle_degrees, "deg"))}
      </div>
    </section>

    <section class="modal-section">
      <h3>Model estimates</h3>
      <div class="detail-grid">
        ${detailCell("Discovery estimate", Math.round(props.score))}
        ${detailCell("Model probability", formatPercent(props.culvert_probability))}
        ${detailCell("Model rank", formatScorePartFrom100(props.model_rank_score))}
        ${detailCell("Priority", labelBucket(props.bucket))}
        ${detailCell("GIS evidence", formatScorePartFrom100(props.evidence_score ?? props.culvert_likelihood_score))}
        ${detailCell("Road-drainage", formatScorePart(props.road_stream_proximity_score))}
        ${detailCell("Drainage strength", formatScorePart(props.drainage_strength_score))}
        ${detailCell("Crossing geometry", formatScorePart(props.crossing_geometry_score))}
        ${detailCell("Road context", formatScorePart(props.road_context_score))}
        ${detailCell("Valley position", formatScorePart(props.valley_position_score))}
        ${detailCell("Terrain break", formatScorePart(props.terrain_break_score))}
        ${detailCell("Field report", formatScorePart(props.field_report_support_score))}
      </div>
    </section>

    <section class="modal-section">
      <h3>Field data</h3>
      <div class="detail-grid">
        ${detailCell("Field match", props.knownFieldMatch ? "yes" : "no")}
        ${detailCell("Report date", props.nearest_field_report_date || props.field_report_date || "n/a")}
        ${detailCell("Culvert ID", formatReadableId(props.nearest_field_report_culvert_id || "n/a"))}
        ${detailCell("Field distance", formatNumber(props.dist_to_known_culvert_m, "m"))}
        ${detailCell("Valley depth", formatNumber(props.valley_depth_9x9_m, "m"))}
        ${detailCell("Slope", formatNumber(props.slope_degrees, "deg"))}
      </div>
    </section>

    <section class="modal-section">
      <h3>How the estimates are calculated</h3>
      <dl class="score-definitions">
        ${definitionItem("Discovery estimate", "Calculated in build_discovery_ranking from local GIS evidence and, when available, the supervised model rank. It is a prioritization estimate for field review, not a confirmed culvert label.")}
        ${definitionItem("Model probability", "Calculated in predict_culvert_probability from the trained scikit-learn model using numeric feature columns. It is the model's estimated probability that the candidate is a culvert.")}
        ${definitionItem("Priority", "Bucketed from the final estimate: low <= 35, medium > 35, high > 55, very high > 75. The list rank sorts undiscovered candidates by discovery estimate before known matches.")}
        ${definitionItem("Drainage/source", "This label is the stream or source name for the mapped drainage feature, not a numeric estimate.")}
        ${definitionItem("Drainage strength", "Calculated from stream order and stream-density percentile features. Higher means stronger nearby drainage evidence.")}
        ${definitionItem("Road-drainage", "Calculated from road-stream distance and exact intersection evidence. Distance uses 1 / (1 + distance_m / 20), plus an exact-intersection boost.")}
        ${definitionItem("Crossing geometry", "Calculated from crossing angle as 1 - abs(90 - angle) / 90. Crossings closer to 90 degrees score higher.")}
      </dl>
    </section>
  `;
  els.detailModal.showModal();
}

function locationSummaryHtml(props) {
  const latitude = formatNumber(props.latitude, "");
  const longitude = formatNumber(props.longitude, "");
  const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${props.latitude},${props.longitude}`;

  return `
    <section class="location-summary" aria-label="Selected location">
      <div>
        <span>Latitude</span>
        <strong>${escapeHtml(latitude)}</strong>
      </div>
      <div>
        <span>Longitude</span>
        <strong>${escapeHtml(longitude)}</strong>
      </div>
      <div class="actions location-actions">
        ${props.google_earth_url ? `<a href="${escapeAttr(props.google_earth_url)}" target="_blank" rel="noreferrer">Google Earth</a>` : ""}
        <a href="${escapeAttr(mapsUrl)}" target="_blank" rel="noreferrer">Google Maps</a>
      </div>
    </section>
  `;
}

function clearDetail() {
  state.selectedId = null;
  state.selectedObservationId = null;
  hideDetailPanel({ clearSelection: false });
}

function showDetailPanel() {
  if (!els.detail) return;
  els.detail.hidden = false;
}

function hideDetailPanel(options = {}) {
  const clearSelection = options.clearSelection !== false;
  if (state.placingPoint) {
    state.placingPoint = false;
    updatePlacePointButton();
  }
  if (clearSelection) {
    state.selectedId = null;
    state.selectedObservationId = null;
    renderList();
    updateSelectedMarkerClass();
    state.map?.closePopup();
  }
  if (!els.detail) return;
  els.detail.hidden = true;
  els.detail.innerHTML = `<div class="empty-state">Select a culvert from the map or list.</div>`;
  state.map?.invalidateSize();
}

function bindDetailCloseAction() {
  els.detail.querySelector("[data-close-detail]")?.addEventListener("click", () => hideDetailPanel());
}

function updateFeatureDistances() {
  state.features.forEach((feature) => {
    const latLng = featureLatLng(feature);
    feature.properties.distanceToUserMeters = state.userLocation && latLng
      ? distanceMeters(state.userLocation.lat, state.userLocation.lng, latLng[0], latLng[1])
      : null;
  });
}

function compareFeaturesForList(a, b) {
  const distanceA = Number(a.properties.distanceToUserMeters);
  const distanceB = Number(b.properties.distanceToUserMeters);
  if (Number.isFinite(distanceA) && Number.isFinite(distanceB)) {
    return distanceA - distanceB || a.properties.rank - b.properties.rank;
  }
  if (Number.isFinite(distanceA)) return -1;
  if (Number.isFinite(distanceB)) return 1;
  return a.properties.rank - b.properties.rank;
}

function listSubtitle(props) {
  const distance = formatDistance(props.distanceToUserMeters);
  const prefix = distance ? `${distance} away · ` : "";
  if (props.knownFieldMatch) {
    const date = props.nearest_field_report_date || props.field_report_date;
    return date ? `${prefix}field report match · ${date}` : `${prefix}field report match`;
  }
  const stream = props.stream_name && props.stream_name !== "route sample" ? props.stream_name : props.source;
  return stream ? `${prefix}undiscovered candidate · ${stream}` : `${prefix}undiscovered candidate`;
}

function compactEvidenceSummary(props) {
  if (props.knownFieldMatch) {
    return props.nearest_field_report_date
      ? `Known field match from ${props.nearest_field_report_date}.`
      : "Known field match.";
  }
  return props.evidence_summary || "No evidence summary available.";
}

function drainageLabel(props) {
  const label = props.stream_name && props.stream_name !== "route sample" ? props.stream_name : props.source;
  return label || "Unknown";
}

function locationDisplayId(props) {
  return formatReadableId(
    props.candidate_id ||
      props.nearest_field_report_culvert_id ||
      props.culvert_id ||
      props.observation_id ||
      props.road_name,
  );
}

function readableSourceId(props) {
  return formatReadableId(
    props.candidate_id ||
      props.nearest_field_report_culvert_id ||
      props.culvert_id ||
      props.nearest_field_report_route ||
      "n/a",
  );
}

function fitVisibleMarkers() {
  const latLngs = [
    ...state.filtered.filter((feature) => !feature.properties.knownFieldMatch).map(featureLatLng),
    ...knownFeatures().map(featureLatLng),
    ...state.observations.map(observationLatLng),
  ].filter(Boolean);

  if (!latLngs.length) return;
  state.map.stop();
  state.map.fitBounds(latLngs, { padding: [28, 28], maxZoom: 15 });
}

function toggleLocationTracking() {
  if (state.locationWatchId !== null) {
    stopLocationTracking();
    return;
  }
  startLocationTracking();
}

function startLocationTracking() {
  if (!navigator.geolocation) {
    setLocationStatus("Location is not available in this browser.");
    return;
  }

  if (els.locateMe) {
    els.locateMe.disabled = true;
    els.locateMe.textContent = "Locating";
  }
  state.shouldFocusLocationOnNextUpdate = true;
  setLocationStatus("Requesting location permission...");

  state.locationWatchId = navigator.geolocation.watchPosition(
    handleLocationSuccess,
    handleLocationError,
    {
      enableHighAccuracy: true,
      timeout: 12000,
      maximumAge: 5000,
    },
  );
}

function stopLocationTracking() {
  if (state.locationWatchId !== null) {
    navigator.geolocation.clearWatch(state.locationWatchId);
  }
  state.locationWatchId = null;
  state.userLocation = null;
  state.locationMarker = null;
  state.locationAccuracyCircle = null;
  state.lastLocationListRenderAt = 0;
  state.shouldFocusLocationOnNextUpdate = false;
  state.locationLayer?.clearLayers();
  updateLocationButton(false);
  setLocationStatus("");
  render();
}

function handleLocationSuccess(position) {
  const latitude = Number(position.coords.latitude);
  const longitude = Number(position.coords.longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    handleLocationError({ code: 0, message: "Invalid location returned by browser." });
    return;
  }

  const previousLocation = state.userLocation;
  const accuracy = Number(position.coords.accuracy);
  const movedMeters = previousLocation
    ? distanceMeters(previousLocation.lat, previousLocation.lng, latitude, longitude)
    : Infinity;
  if (previousLocation && movedMeters < LOCATION_MIN_MOVE_M && accuracy >= previousLocation.accuracy) {
    return;
  }

  state.userLocation = {
    lat: latitude,
    lng: longitude,
    accuracy,
  };
  renderLocationMarker();
  updateNearbyListFromLocation({ force: !previousLocation });
  if (state.shouldFocusLocationOnNextUpdate) {
    focusUserLocation();
    state.shouldFocusLocationOnNextUpdate = false;
  }
  updateLocationButton(true);
  setLocationStatus("Tracking is on. Nearby list updates as your position changes.");
}

function handleLocationError(error) {
  const permissionDenied = error?.code === 1;
  const message = permissionDenied
    ? "Location permission was denied. Enable location access for this site to use nearby culverts."
    : `Could not get your location${error?.message ? `: ${error.message}` : "."}`;
  if (state.locationWatchId !== null) {
    navigator.geolocation.clearWatch(state.locationWatchId);
  }
  state.locationWatchId = null;
  state.shouldFocusLocationOnNextUpdate = false;
  updateLocationButton(false);
  setLocationStatus(message);
}

function updateLocationButton(isTracking) {
  if (!els.locateMe) return;
  els.locateMe.disabled = false;
  els.locateMe.textContent = isTracking ? "Tracking" : "Locate";
  els.locateMe.setAttribute("aria-pressed", String(isTracking));
  els.locateMe.setAttribute("aria-label", isTracking ? "Stop location tracking" : "Start location tracking");
}

function setLocationStatus(message) {
  if (!els.locationStatus) return;
  els.locationStatus.textContent = message;
  els.locationStatus.hidden = !message;
}

function renderLocationMarker() {
  if (!state.locationLayer || !state.userLocation) return;
  const latLng = [state.userLocation.lat, state.userLocation.lng];
  const accuracy = Number.isFinite(state.userLocation.accuracy)
    ? Math.min(Math.max(state.userLocation.accuracy, 12), 250)
    : 25;

  if (!state.locationAccuracyCircle) {
    state.locationAccuracyCircle = L.circle(latLng, {
      radius: accuracy,
      color: "#1f6f57",
      fillColor: "#1f6f57",
      fillOpacity: 0.12,
      weight: 1,
      interactive: false,
    }).addTo(state.locationLayer);
  } else {
    state.locationAccuracyCircle.setLatLng(latLng);
    state.locationAccuracyCircle.setRadius(accuracy);
  }

  if (!state.locationMarker) {
    state.locationMarker = L.circleMarker(latLng, {
      radius: 8,
      color: "#ffffff",
      fillColor: "#1f6f57",
      fillOpacity: 1,
      weight: 3,
      className: "user-location-marker",
      interactive: false,
    }).addTo(state.locationLayer);
  } else {
    state.locationMarker.setLatLng(latLng);
  }
}

function focusUserLocation() {
  if (!state.userLocation) return;
  const latLng = [state.userLocation.lat, state.userLocation.lng];
  const zoom = Math.max(state.map.getZoom(), LOCATION_FOCUS_ZOOM);
  state.map.stop();
  state.map.flyTo(latLng, zoom, {
    animate: true,
    duration: 0.65,
    easeLinearity: 0.2,
  });
}

function updateNearbyListFromLocation(options = {}) {
  const now = performance.now();
  if (!options.force && now - state.lastLocationListRenderAt < LOCATION_LIST_THROTTLE_MS) return;
  state.lastLocationListRenderAt = now;
  updateFeatureDistances();
  state.filtered.sort(compareFeaturesForList);
  renderList();
  updateVisibleCount();
}

function nearestCandidateFeatures(limit) {
  return state.filtered
    .filter((feature) => !feature.properties.knownFieldMatch)
    .filter((feature) => Number.isFinite(Number(feature.properties.distanceToUserMeters)))
    .slice(0, limit);
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
  state.observationMarkers.clear();

  state.observations.forEach((feature) => {
    const latLng = observationLatLng(feature);
    if (!latLng) return;
    const observationId = observationIdOf(feature);

    const marker = L.marker(latLng, {
      icon: observationIcon(feature.properties),
      riseOnHover: true,
      zIndexOffset: 900,
    });
    marker.bindPopup(observationPopupHtml(feature.properties), selectedPopupOptions());
    marker.on("click", () => selectObservation(observationId, { pan: true }));
    marker.addTo(state.observationLayer);
    state.observationMarkers.set(observationId, marker);
  });
}

function observationIcon(props) {
  const normalized = observationStatus(props?.status);
  const text = normalized === "confirmed_culvert" ? "ABU" : normalized === "no_culvert" ? "NO" : "?";
  return L.divIcon({
    className: `observation-marker observation-${normalized}`,
    html: `<span class="observation-dot">${text}</span>`,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
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
    <div class="popup-meta">Estimate ${Math.round(props.score)} · ${labelBucket(props.bucket)}</div>
  `;
}

function knownCulvertLabel(props) {
  return formatReadableId(firstPresent([
    props.nearest_field_report_culvert_id,
    props.culvert_id,
    props.nearest_field_report_route,
    props.road_name,
    "Known culvert",
  ]));
}

function knownCulvertShortLabel(props) {
  const label = formatReadableId(knownCulvertLabel(props));
  if (/^sc[-_]?\d+/i.test(label)) return label.replace(/^sc/i, "SC").slice(0, 7);
  return "K";
}

function fieldFeedbackHtml(title) {
  return `
    <section class="field-feedback" aria-label="${escapeAttr(title)}">
      <h4>${escapeHtml(title)}</h4>
      <details class="notes-disclosure">
        <summary>Add optional notes</summary>
        <label for="field-notes">Field notes</label>
        <textarea id="field-notes" rows="2" placeholder="Optional notes from inspection"></textarea>
      </details>
      <div class="feedback-buttons">
        <button type="button" data-feedback-status="confirmed_culvert">Save culvert</button>
        <button type="button" data-feedback-status="no_culvert">No culvert</button>
        <button type="button" data-feedback-status="uncertain">Needs review</button>
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
        const storage = storageMessage(result.storage);
        if (statusOutput) {
          const training = status === "confirmed_culvert" ? " Saved as a positive training label." : "";
          statusOutput.textContent = `${statusLabel(status)}. ${storage}${training}`;
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

async function saveObservationAtPoint(latLng, status, notes, options = {}) {
  const context = options.context || {};
  const fieldId = options.fieldId || makeFieldCulvertId();
  const matched = context.matchedFeature?.properties || {};
  return saveObservation({
    status,
    notes,
    latitude: latLng.lat,
    longitude: latLng.lng,
    candidate_id: fieldId,
    field_culvert_id: fieldId,
    road_name: context.roadName || "",
    stream_name: context.streamName || "",
    source: "field_added_culvert",
    layout_source: context.layoutSource || "manual_map_point",
    layout_scan_summary: context.summary || "",
    nearest_candidate_id: matched.candidate_id || "",
    nearest_candidate_distance_m: context.distanceMeters,
    inferred_from_candidate: context.withinRadius ? 1 : 0,
    prediction_score: matched.score,
    priority_rank: matched.rank,
    priority_bucket: matched.bucket || "",
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
    if (saved.storage === "memory") {
      storeLocalObservation(savedFeature);
    }
    if (saved.findings && saved.summary) {
      applyDashboardData(saved.findings, saved.summary, saved.observations || { features: [savedFeature] }, {
        preserveSelection: true,
      });
    } else {
      addObservation(savedFeature);
    }
    return {
      feature: savedFeature,
      storage: saved.storage || "file",
      training: saved.training,
      warning: saved.warning,
    };
  } catch {
    storeLocalObservation(feature);
    addObservation(feature);
    return { feature, storage: "browser" };
  }
}

async function syncLocalObservationsToServer() {
  const local = loadLocalObservations();
  if (!local.length) return;

  let synced = 0;
  let latestRefresh = null;
  for (const feature of local) {
    const normalized = normalizeObservationFeature(feature);
    if (!normalized) continue;

    try {
      const response = await fetch(OBSERVATIONS_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(normalized.properties),
      });
      if (!response.ok) continue;

      const saved = await response.json();
      if (saved.storage === "vercel_blob" || saved.storage === "file") {
        synced += 1;
        latestRefresh = saved;
      }
    } catch {
      return;
    }
  }

  if (synced === local.length) {
    clearLocalObservations();
    if (latestRefresh?.findings && latestRefresh?.summary) {
      applyDashboardData(latestRefresh.findings, latestRefresh.summary, latestRefresh.observations, {
        preserveSelection: true,
      });
    }
  }
}

async function deleteObservationById(observationId, button) {
  const id = String(observationId || "").trim();
  if (!id) return;

  if (button) {
    button.disabled = true;
    button.textContent = "Deleting...";
  }

  try {
    const response = await fetch(`${OBSERVATIONS_URL}?id=${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error(`server returned ${response.status}`);
    }
    const deleted = await response.json();
    removeLocalObservation(id);
    if (deleted.findings && deleted.summary) {
      applyDashboardData(deleted.findings, deleted.summary, deleted.observations || { features: [] }, {
        preserveSelection: true,
      });
    } else {
      removeObservation(id);
    }
  } catch {
    removeLocalObservation(id);
    removeObservation(id);
  }
}

function storageMessage(storage) {
  if (storage === "vercel_blob") {
    return "Saved to Vercel and refreshed the deployed ranking.";
  }
  if (storage === "file") {
    return "Saved to data/processed/field_observations.geojson.";
  }
  if (storage === "memory") {
    return "Applied to this session. Configure Vercel Blob to persist it.";
  }
  return "Saved in this browser only.";
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
  renderList();
  updateVisibleCount();
}

function removeObservation(observationId) {
  state.observations = state.observations.filter(
    (feature) => feature.properties?.observation_id !== observationId,
  );
  if (state.selectedObservationId === observationId) {
    state.selectedObservationId = null;
    hideDetailPanel({ clearSelection: false });
  }
  state.map?.closePopup();
  renderObservationMarkers();
  renderList();
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
      field_culvert_id: payload.field_culvert_id || "",
      layout_source: payload.layout_source || "",
      layout_scan_summary: payload.layout_scan_summary || "",
      nearest_candidate_id: payload.nearest_candidate_id || "",
      nearest_candidate_distance_m: numberOrNull(payload.nearest_candidate_distance_m),
      inferred_from_candidate: numberOrNull(payload.inferred_from_candidate),
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

function observationIdOf(feature) {
  return String(feature?.properties?.observation_id || "");
}

function abuObservations() {
  return state.observations.filter((feature) => observationStatus(feature.properties?.status) === "confirmed_culvert");
}

function observationSubtitle(props) {
  const coordinates = `${formatNumber(props.latitude, "")}, ${formatNumber(props.longitude, "")}`;
  const date = props.observed_at ? shortDateTime(props.observed_at) : "unknown time";
  const label = props.field_culvert_id || props.road_name || props.candidate_id || coordinates;
  return `${label} · ${date}`;
}

function renderObservationDetail(feature) {
  const props = feature.properties || {};
  const title = observationTitle(props);
  const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${props.latitude},${props.longitude}`;
  showDetailPanel();
  els.detail.innerHTML = `
    <div class="detail-panel-header">
      <h3>${escapeHtml(title)}</h3>
      <button type="button" class="detail-close" data-close-detail aria-label="Close details">Close</button>
    </div>
    <p>${escapeHtml(props.notes || "User-added training point.")}</p>
    <div class="quick-detail-grid">
      ${detailCell("Status", statusLabel(props.status))}
      ${detailCell("Source", sourceLabel(props.source))}
      ${detailCell("Latitude", formatNumber(props.latitude, ""))}
      ${detailCell("Longitude", formatNumber(props.longitude, ""))}
    </div>
    <div class="actions location-actions">
      <a href="${escapeAttr(mapsUrl)}" target="_blank" rel="noreferrer">Google Maps</a>
      <button type="button" class="danger-action" data-observation-delete="${escapeAttr(props.observation_id || "")}">Delete ABU</button>
    </div>
  `;
  bindDetailCloseAction();
}

function observationPopupHtml(props) {
  const title = observationTitle(props);
  const road = props.field_culvert_id || props.road_name || props.candidate_id || "Manual field point";
  return `
    <div class="popup-title">${escapeHtml(title)}</div>
    <div class="popup-meta">${escapeHtml(road)}</div>
    <div class="popup-meta">${escapeHtml(props.observed_at || "")}</div>
    ${props.notes ? `<div class="popup-meta">${escapeHtml(props.notes)}</div>` : ""}
    <button type="button" class="popup-delete-observation" data-observation-delete="${escapeAttr(props.observation_id || "")}">Delete user point</button>
  `;
}

function observationTitle(props) {
  if (observationStatus(props?.status) === "confirmed_culvert" && props?.source === "field_added_culvert") {
    return "Added by user";
  }
  return statusLabel(props?.status);
}

function sourceLabel(source) {
  if (source === "field_added_culvert") return "Added on map";
  if (source === "prediction_review") return "Prediction review";
  if (source === "known_culvert_review") return "Known culvert review";
  return source || "Field review";
}

function fieldCulvertContextHtml(fieldId, context) {
  return `
    <section class="field-context" aria-label="Field culvert context">
      <div class="detail-grid">
        ${detailCell("Assigned ID", fieldId)}
        ${detailCell("Context source", context.withinRadius ? "nearest map candidate" : "manual point")}
        ${detailCell("Road", context.roadName || "unknown")}
        ${detailCell("Drainage/source", context.streamName || "unknown")}
        ${detailCell("Nearest candidate", context.nearestDisplayId || "none nearby")}
        ${detailCell("Distance", formatNumber(context.distanceMeters, "m"))}
      </div>
      <p class="context-note">${escapeHtml(context.summary)}</p>
    </section>
  `;
}

function mapContextForPoint(latLng) {
  const nearest = nearestFeatureToPoint(latLng);
  if (!nearest) {
    return {
      withinRadius: false,
      layoutSource: "manual_map_point",
      summary: "No current map candidate was close enough, so only the clicked coordinates and your field label will be used.",
    };
  }

  const props = nearest.feature.properties;
  const withinRadius = nearest.distanceMeters <= FIELD_CONTEXT_RADIUS_M;
  const roadName = props.road_name || "";
  const streamName = drainageLabel(props);
  const nearestDisplayId = locationDisplayId(props);
  return {
    matchedFeature: nearest.feature,
    withinRadius,
    distanceMeters: nearest.distanceMeters,
    roadName: withinRadius ? roadName : "",
    streamName: withinRadius ? streamName : "",
    nearestDisplayId,
    layoutSource: withinRadius ? "nearest_map_candidate" : "manual_map_point",
    summary: withinRadius
      ? `Copied road, drainage, rank, and estimate context from ${nearestDisplayId}, ${formatNumber(nearest.distanceMeters, "m")} from the clicked point.`
      : `Nearest map candidate is ${formatNumber(nearest.distanceMeters, "m")} away, outside the ${FIELD_CONTEXT_RADIUS_M} m context radius, so the point will be saved without inferred road/drainage context.`,
  };
}

function nearestFeatureToPoint(latLng) {
  const latitude = Number(latLng.lat);
  const longitude = Number(latLng.lng);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;

  let best = null;
  let bestDistance = Infinity;
  for (const feature of state.features) {
    const featurePoint = featureLatLng(feature);
    if (!featurePoint) continue;
    const distance = distanceMeters(latitude, longitude, featurePoint[0], featurePoint[1]);
    if (distance < bestDistance) {
      best = feature;
      bestDistance = distance;
    }
  }

  return best ? { feature: best, distanceMeters: bestDistance } : null;
}

function togglePlacePointMode() {
  state.placingPoint = !state.placingPoint;
  updatePlacePointButton();
  if (state.placingPoint) {
    setMobileDrawerOpen(false);
    state.selectedId = null;
    renderList();
    showDetailPanel();
    els.detail.innerHTML = `
      <div class="detail-panel-header">
        <h3>Add culvert point</h3>
        <button type="button" class="detail-close" data-close-detail aria-label="Close details">Close</button>
      </div>
      <p>Click the map at the culvert location. The app will assign an ID and use nearby map context for the training record.</p>
    `;
    bindDetailCloseAction();
  }
}

function updatePlacePointButton() {
  [els.placePoint, els.mobileAddPoint].filter(Boolean).forEach((button) => {
    button.setAttribute("aria-pressed", String(state.placingPoint));
    button.classList.toggle("active", state.placingPoint);
  });
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
  const fieldId = makeFieldCulvertId();
  const context = mapContextForPoint(latLng);
  showDetailPanel();
  els.detail.innerHTML = `
    <div class="detail-panel-header">
      <h3>New culvert ${escapeHtml(fieldId)}</h3>
      <button type="button" class="detail-close" data-close-detail aria-label="Close details">Close</button>
    </div>
    <p>Lat ${formatNumber(latLng.lat, "")}, Lon ${formatNumber(latLng.lng, "")}</p>
    ${fieldCulvertContextHtml(fieldId, context)}
    ${fieldFeedbackHtml("Save field training label")}
  `;
  bindDetailCloseAction();
  bindFeedbackActions((status, notes) => saveObservationAtPoint(latLng, status, notes, { fieldId, context }));
  window.requestAnimationFrame(scrollDetailIntoViewOnMobile);
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

function makeFieldCulvertId() {
  const now = new Date();
  const date = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
  ].join("");
  return `FC-${date}-${Math.random().toString(36).slice(2, 6).toUpperCase()}`;
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

function definitionItem(term, description) {
  return `<div><dt>${escapeHtml(term)}</dt><dd>${escapeHtml(description)}</dd></div>`;
}

function formatReadableId(value) {
  const text = String(value || "").trim();
  if (!text || text === "n/a") return "n/a";

  const candidateMatch = text.match(/^cand_0*(\d+)$/i);
  if (candidateMatch) return `C-${candidateMatch[1]}`;

  const observationMatch = text.match(/^obs_\d+_([a-z0-9]+)$/i);
  if (observationMatch) return `OBS-${observationMatch[1].toUpperCase()}`;

  const culvertMatch = text.match(/^sc[-_ ]?0*(\d+)$/i);
  if (culvertMatch) return `SC-${culvertMatch[1]}`;

  if (text.length > 18 && /^[a-z0-9_-]+$/i.test(text)) {
    return text.slice(-8).toUpperCase();
  }

  return text;
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

function formatDistance(value) {
  const meters = Number(value);
  if (!Number.isFinite(meters)) return "";
  const feet = meters * 3.28084;
  if (feet < 900) return `${Math.round(feet)} ft`;
  const miles = meters / 1609.344;
  return `${miles < 10 ? miles.toFixed(1) : Math.round(miles)} mi`;
}

function shortDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || "");
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function distanceMeters(latA, lonA, latB, lonB) {
  const radiusMeters = 6371008.8;
  const toRadians = (degrees) => (degrees * Math.PI) / 180;
  const phiA = toRadians(latA);
  const phiB = toRadians(latB);
  const deltaPhi = toRadians(latB - latA);
  const deltaLambda = toRadians(lonB - lonA);
  const haversine =
    Math.sin(deltaPhi / 2) ** 2 +
    Math.cos(phiA) * Math.cos(phiB) * Math.sin(deltaLambda / 2) ** 2;
  return 2 * radiusMeters * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));
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
  showDetailPanel();
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
