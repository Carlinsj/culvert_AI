import { readFile } from "node:fs/promises";
import path from "node:path";
import { gunzip } from "node:zlib";
import { promisify } from "node:util";

const gunzipAsync = promisify(gunzip);

const ROOT = process.cwd();
const ROUTE_COUNT_SOURCE_PATH = path.join(ROOT, "web", "data", "route_count_source.geojson.gz");

const DEFAULT_SCORE_COLUMNS = [
  "discovery_score",
  "culvert_likelihood_score",
  "culvert_probability",
];
const DEFAULT_SCORE_THRESHOLDS = [70, 68, 65, 60, 55];
const DEFAULT_PROBABILITY_THRESHOLDS = [0.85, 0.7, 0.5, 0.35, 0.2];
const UNKNOWN_EXCLUDED_STATUSES = new Set([
  "known_field_match",
  "field_denied",
  "confirmed_field_observation",
]);
const UNKNOWN_EXCLUDED_SOURCES = new Set(["field_report_observed_culvert"]);
const ROUTE_TOKEN_RE =
  /\b(?<prefix>NY|NYS|US|U\.S\.|I|CR|COUNTY|STATE)\s*-?\s*(?:(?:HWY|HIGHWAY|RTE|ROUTE|RT|ROAD|RD)\s*-?\s*)?(?<number>\d+[A-Z]?)\b/gi;
const GENERIC_ROUTE_TOKEN_RE =
  /\b(?:ROUTE|RTE|RT|HIGHWAY|HWY)\s*-?\s*(?<number>\d+[A-Z]?)\b/gi;

let routeCountSourceCache = null;

export async function loadRouteCountCollection(fallbackCollection) {
  if (routeCountSourceCache) return routeCountSourceCache;

  try {
    const compressed = await readFile(ROUTE_COUNT_SOURCE_PATH);
    const payload = await gunzipAsync(compressed);
    const collection = JSON.parse(payload.toString("utf8"));
    if (collection?.type === "FeatureCollection" && Array.isArray(collection.features)) {
      routeCountSourceCache = {
        collection,
        source: {
          name: "route_count_source",
          complete: true,
          path: "web/data/route_count_source.geojson.gz",
          rows: collection.features.length,
        },
      };
      return routeCountSourceCache;
    }
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
  }

  return {
    collection: fallbackCollection,
    source: {
      name: "dashboard_findings",
      complete: false,
      path: "web/data/findings.geojson",
      rows: Array.isArray(fallbackCollection?.features) ? fallbackCollection.features.length : 0,
    },
  };
}

export function buildRouteCountReport(collection, rawOptions = {}) {
  const options = normalizeOptions(rawOptions);
  const features = Array.isArray(collection?.features) ? collection.features : [];
  if (!features.length) {
    throw new Error("Route count source has no prediction rows.");
  }

  const rows = features.map(pointRow).filter(Boolean);
  if (!rows.length) {
    throw new Error("Route count source has no rows with valid point coordinates.");
  }

  const selectedScoreColumn = options.scoreColumn || scoreColumn(rows);
  if (!selectedScoreColumn) {
    throw new Error("Route count source has no usable score column.");
  }

  let pool = rows;
  if (!options.includeKnown) {
    pool = pool.filter(isUnknownPrediction);
  }
  if (options.rankLimit) {
    pool = pool.filter((row) => numericValue(row.props.discovery_rank) <= options.rankLimit);
  }
  if (options.route) {
    pool = pool.filter((row) => routeMatches(row, options.route));
  }
  if (options.bbox) {
    pool = pool.filter((row) => pointWithinBbox(row, options.bbox));
  }

  const clusters = clusterPredictions(pool, {
    clusterRadiusM: options.clusterRadiusM,
    scoreColumn: selectedScoreColumn,
    probabilityColumn: options.probabilityColumn,
  });

  return reportFromClusters({
    sourceRows: rows.length,
    filteredRows: pool.length,
    clusters,
    route: options.route,
    bbox: options.bbox,
    clusterRadiusM: options.clusterRadiusM,
    scoreColumn: selectedScoreColumn,
    probabilityColumn: options.probabilityColumn,
    scoreThreshold: options.scoreThreshold,
    thresholds: options.thresholds || defaultThresholds(selectedScoreColumn),
    topN: options.topN,
  });
}

function normalizeOptions(rawOptions) {
  const get = (key) => {
    if (rawOptions instanceof URLSearchParams) return rawOptions.get(key);
    return rawOptions?.[key];
  };
  const route = safeString(get("route"), 120);
  const scoreColumn = safeString(get("scoreColumn") || get("score_column"), 80);
  const probabilityColumn =
    safeString(get("probabilityColumn") || get("probability_column"), 80) || "culvert_probability";
  return {
    route,
    bbox: parseBbox(get("bbox")),
    includeKnown: parseBoolean(get("includeKnown") ?? get("include_known"), false),
    clusterRadiusM: clampNumber(get("clusterRadiusM") ?? get("cluster_radius_m"), 30, 0, 500),
    rankLimit: parsePositiveInteger(get("rankLimit") ?? get("rank_limit")),
    topN: clampInteger(get("topN") ?? get("top_n"), 12, 0, 100),
    scoreColumn: DEFAULT_SCORE_COLUMNS.includes(scoreColumn) ? scoreColumn : "",
    probabilityColumn,
    scoreThreshold: optionalNumber(get("scoreThreshold") ?? get("score_threshold")),
    thresholds: parseThresholds(get("thresholds")),
  };
}

function pointRow(feature, index) {
  const props = feature?.properties || {};
  const coordinates = feature?.geometry?.type === "Point" ? feature.geometry.coordinates : null;
  const longitude = Number(props.longitude ?? coordinates?.[0]);
  const latitude = Number(props.latitude ?? coordinates?.[1]);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return {
    feature,
    props,
    index,
    latitude,
    longitude,
  };
}

function isUnknownPrediction(row) {
  const status = String(row.props.discovery_status || "").toLowerCase();
  const source = String(row.props.source || "").toLowerCase();
  if (UNKNOWN_EXCLUDED_STATUSES.has(status)) return false;
  if (UNKNOWN_EXCLUDED_SOURCES.has(source)) return false;
  if (truthy(row.props.is_known_field_match) || truthy(row.props.is_culvert)) return false;
  return true;
}

function routeMatches(row, route) {
  const queryTokens = routeTokens(route);
  if (!queryTokens.size) {
    const normalizedQuery = normalizeRouteText(route);
    return normalizeRouteText(routeText(row.props)).includes(normalizedQuery);
  }

  const rowTokens = rowRouteTokens(row.props);
  for (const token of queryTokens) {
    if (rowTokens.has(token)) return true;
  }
  return false;
}

function rowRouteTokens(props) {
  const tokens = new Set();
  for (const column of ["matched_route", "road_name", "road_id", "nearest_field_report_route"]) {
    const value = props[column];
    if (value === null || value === undefined || value === "") continue;
    for (const piece of String(value).split(",")) {
      for (const token of routeTokens(piece)) {
        tokens.add(token);
      }
    }
  }
  return tokens;
}

function routeTokens(value) {
  const text = normalizeRouteText(value);
  const tokens = new Set();
  for (const match of text.matchAll(ROUTE_TOKEN_RE)) {
    const number = match.groups.number.toUpperCase();
    let prefix = match.groups.prefix.toUpperCase().replaceAll(".", "");
    if (prefix === "NYS" || prefix === "STATE") prefix = "NY";
    if (prefix === "COUNTY") prefix = "CR";
    tokens.add(`${prefix}${number}`);
    tokens.add(number);
  }
  for (const match of text.matchAll(GENERIC_ROUTE_TOKEN_RE)) {
    tokens.add(match.groups.number.toUpperCase());
  }
  const bare = text.match(/^\s*(\d+[A-Z]?)\s*$/i);
  if (bare) {
    tokens.add(bare[1].toUpperCase());
  }
  return tokens;
}

function routeText(props) {
  return ["matched_route", "road_name", "road_id", "nearest_field_report_route"]
    .map((column) => props[column])
    .filter((value) => value !== null && value !== undefined && value !== "")
    .join(" ");
}

function normalizeRouteText(value) {
  return String(value || "")
    .toUpperCase()
    .replaceAll(".", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function pointWithinBbox(row, bbox) {
  return (
    row.longitude >= bbox.west &&
    row.longitude <= bbox.east &&
    row.latitude >= bbox.south &&
    row.latitude <= bbox.north
  );
}

function clusterPredictions(rows, { clusterRadiusM, scoreColumn, probabilityColumn }) {
  if (!rows.length) return [];

  const meanLatitude = rows.reduce((sum, row) => sum + row.latitude, 0) / rows.length;
  const project = localProjection(meanLatitude);
  const ordered = rows
    .map((row) => {
      const point = project(row.latitude, row.longitude);
      return {
        ...row,
        x: point.x,
        y: point.y,
        routeCountScore: numericValue(row.props[scoreColumn]),
        routeCountProbability: clusterProbability(row.props, probabilityColumn),
        routeCountRank: numericValue(row.props.discovery_rank, Infinity),
      };
    })
    .sort((a, b) => {
      const scoreDiff = b.routeCountScore - a.routeCountScore;
      if (scoreDiff !== 0) return scoreDiff;
      const probabilityDiff = b.routeCountProbability - a.routeCountProbability;
      if (probabilityDiff !== 0) return probabilityDiff;
      return a.routeCountRank - b.routeCountRank;
    });

  const radius = Math.max(0, Number(clusterRadiusM) || 0);
  const cellSize = Math.max(radius, 1);
  const clusterPoints = [];
  const clusterMembers = [];
  const grid = new Map();

  for (const row of ordered) {
    const cell = [Math.floor(row.x / cellSize), Math.floor(row.y / cellSize)];
    let clusterId = nearbyClusterId(row, cell, grid, clusterPoints, radius);
    if (clusterId === null) {
      clusterId = clusterPoints.length;
      clusterPoints.push({ x: row.x, y: row.y, latitude: row.latitude, longitude: row.longitude });
      clusterMembers.push([]);
      const key = cellKey(cell);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(clusterId);
    }
    clusterMembers[clusterId].push(row);
  }

  return clusterMembers
    .map((members, index) => {
      const representative = members[0];
      const maxProbability = Math.max(...members.map((member) => member.routeCountProbability));
      const maxScore = Math.max(...members.map((member) => member.routeCountScore));
      const point = clusterPoints[index];
      return {
        representative,
        members,
        cluster_id: `rc_${String(index + 1).padStart(5, "0")}`,
        member_count: members.length,
        route_count_probability: maxProbability,
        route_count_score: maxScore,
        latitude: point.latitude,
        longitude: point.longitude,
      };
    })
    .sort((a, b) => {
      const scoreDiff = b.route_count_score - a.route_count_score;
      if (scoreDiff !== 0) return scoreDiff;
      return b.route_count_probability - a.route_count_probability;
    })
    .map((cluster, index) => ({
      ...cluster,
      rank: index + 1,
    }));
}

function nearbyClusterId(row, cell, grid, clusterPoints, radiusM) {
  if (radiusM <= 0) return null;
  let bestClusterId = null;
  let bestDistance = Infinity;
  for (const xOffset of [-1, 0, 1]) {
    for (const yOffset of [-1, 0, 1]) {
      const candidateCell = [cell[0] + xOffset, cell[1] + yOffset];
      for (const clusterId of grid.get(cellKey(candidateCell)) || []) {
        const point = clusterPoints[clusterId];
        const distance = Math.hypot(row.x - point.x, row.y - point.y);
        if (distance <= radiusM && distance < bestDistance) {
          bestClusterId = clusterId;
          bestDistance = distance;
        }
      }
    }
  }
  return bestClusterId;
}

function reportFromClusters({
  sourceRows,
  filteredRows,
  clusters,
  route,
  bbox,
  clusterRadiusM,
  scoreColumn,
  probabilityColumn,
  scoreThreshold,
  thresholds,
  topN,
}) {
  const probabilities = clusters.map((cluster) => clamp(cluster.route_count_probability, 0, 1));
  const expectedCount = probabilities.reduce((sum, probability) => sum + probability, 0);
  const interval = predictionInterval(probabilities, expectedCount);
  const recommended = recommendedCount(clusters, expectedCount, interval, scoreThreshold);

  return {
    route: route || "",
    bbox: bbox ? { ...bbox } : null,
    warnings: reportWarnings({ route, bbox, filteredRows }),
    source_rows: sourceRows,
    filtered_prediction_rows: filteredRows,
    cluster_radius_m: clusterRadiusM,
    score_column: scoreColumn,
    probability_column: probabilityColumn,
    candidate_clusters: clusters.length,
    recommended,
    threshold_counts: thresholdCounts(clusters, thresholds),
    top_clusters: topClusters(clusters, topN),
  };
}

function reportWarnings({ route, bbox, filteredRows }) {
  const warnings = [];
  if (route && !bbox) {
    warnings.push(
      "Route-only reports cover every matching candidate in the route-count source. Use current map view for a field-walk segment estimate.",
    );
  }
  if (filteredRows === 0) {
    warnings.push("No predictions matched the requested route and map-view filters.");
  }
  return warnings;
}

function recommendedCount(clusters, expectedCount, interval, scoreThreshold) {
  if (scoreThreshold !== null && scoreThreshold !== undefined) {
    const predictedCount = clusters.filter((cluster) => cluster.route_count_score >= scoreThreshold).length;
    return {
      method: "cluster_count_at_score_threshold",
      score_threshold: scoreThreshold,
      predicted_count: predictedCount,
      expected_count: expectedCount,
      prediction_interval_90: interval.prediction_interval_90,
    };
  }

  return {
    method: "sum_of_cluster_probabilities",
    score_threshold: null,
    predicted_count: Math.round(expectedCount),
    expected_count: expectedCount,
    prediction_interval_90: interval.prediction_interval_90,
  };
}

function predictionInterval(probabilities, expectedCount) {
  if (!probabilities.length) {
    return { prediction_interval_90: [0, 0], standard_deviation: 0 };
  }
  const variance = probabilities.reduce((sum, probability) => sum + probability * (1 - probability), 0);
  const standardDeviation = Math.sqrt(Math.max(0, variance));
  const lower = Math.max(0, Math.floor(expectedCount - 1.64 * standardDeviation));
  const upper = Math.max(lower, Math.ceil(expectedCount + 1.64 * standardDeviation));
  return {
    prediction_interval_90: [lower, upper],
    standard_deviation: standardDeviation,
  };
}

function thresholdCounts(clusters, thresholds) {
  return thresholds.map((threshold) => {
    const selected = clusters.filter((cluster) => cluster.route_count_score >= threshold);
    return {
      score_threshold: threshold,
      cluster_count: selected.length,
      expected_count: selected.reduce((sum, cluster) => sum + cluster.route_count_probability, 0),
    };
  });
}

function topClusters(clusters, topN) {
  if (topN <= 0) return [];
  return clusters.slice(0, topN).map((cluster) => {
    const props = cluster.representative.props;
    return {
      rank: cluster.rank,
      candidate_id: String(props.candidate_id || ""),
      cluster_id: cluster.cluster_id,
      predicted_site_probability: optionalFloat(cluster.route_count_probability),
      score: optionalFloat(cluster.route_count_score),
      discovery_rank: optionalFloat(props.discovery_rank),
      discovery_score: optionalFloat(props.discovery_score),
      culvert_probability: optionalFloat(props.culvert_probability),
      road_name: String(props.road_name || ""),
      stream_name: String(props.stream_name || ""),
      matched_route: String(props.matched_route || ""),
      source: String(props.source || ""),
      member_count: cluster.member_count,
      latitude: cluster.latitude,
      longitude: cluster.longitude,
    };
  });
}

function scoreColumn(rows) {
  for (const column of DEFAULT_SCORE_COLUMNS) {
    if (rows.some((row) => Number.isFinite(Number(row.props[column])))) {
      return column;
    }
  }
  return "";
}

function defaultThresholds(scoreColumnName) {
  return scoreColumnName === "culvert_probability"
    ? DEFAULT_PROBABILITY_THRESHOLDS
    : DEFAULT_SCORE_THRESHOLDS;
}

function clusterProbability(props, probabilityColumn) {
  const probability = Number(props[probabilityColumn]);
  if (Number.isFinite(probability)) return clamp(probability, 0, 1);

  const selectedScoreColumn = DEFAULT_SCORE_COLUMNS.find((column) => Number.isFinite(Number(props[column])));
  if (!selectedScoreColumn) return 0;
  const score = Number(props[selectedScoreColumn]);
  const scale = selectedScoreColumn === "culvert_probability" ? 1 : 100;
  return clamp(score / scale, 0, 1);
}

function localProjection(meanLatitude) {
  const latitudeRadians = (meanLatitude * Math.PI) / 180;
  const metersPerDegreeLatitude = 111_132.92 - 559.82 * Math.cos(2 * latitudeRadians);
  const metersPerDegreeLongitude = 111_412.84 * Math.cos(latitudeRadians);
  return (latitude, longitude) => ({
    x: longitude * metersPerDegreeLongitude,
    y: latitude * metersPerDegreeLatitude,
  });
}

function parseBbox(value) {
  const parts = String(value || "")
    .split(",")
    .map((part) => Number(part.trim()));
  if (parts.length !== 4 || parts.some((part) => !Number.isFinite(part))) return null;
  const [west, south, east, north] = parts;
  if (west < -180 || east > 180 || south < -90 || north > 90 || west >= east || south >= north) {
    return null;
  }
  return { west, south, east, north };
}

function parseThresholds(value) {
  const thresholds = String(value || "")
    .split(",")
    .map((part) => optionalNumber(part))
    .filter((part) => part !== null);
  return thresholds.length ? thresholds : null;
}

function optionalNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function clampNumber(value, fallback, min, max) {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return clamp(number, min, max);
}

function clampInteger(value, fallback, min, max) {
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number)) return fallback;
  return clamp(number, min, max);
}

function parsePositiveInteger(value) {
  const number = Number.parseInt(value, 10);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function parseBoolean(value, fallback) {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "boolean") return value;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function numericValue(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function optionalFloat(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function safeString(value, maxLength) {
  return String(value || "").trim().slice(0, maxLength);
}

function truthy(value) {
  if (value === true) return true;
  if (typeof value === "number") return value === 1;
  return ["1", "true", "yes"].includes(String(value || "").toLowerCase());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function cellKey(cell) {
  return `${cell[0]},${cell[1]}`;
}
