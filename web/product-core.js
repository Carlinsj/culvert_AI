export const VIEW_LIMITS = [25, 50, 100, 250, "all"];

export function priorityLabel(rank) {
  const value = Number(rank);
  if (value <= 25) return "High Priority";
  if (value <= 100) return "Medium Priority";
  return "Lower Priority";
}

export function filterSites(features, filters = {}, observations = new Map()) {
  const limit = filters.limit === "all" ? Infinity : Number(filters.limit || 25);
  const query = String(filters.road || "").trim().toLowerCase();
  const minRank = Number(filters.minRank || 1);
  const maxRank = Number(filters.maxRank || Infinity);
  return features
    .filter((feature) => {
      const props = feature.properties || {};
      const rank = Number(props.global_rank);
      const priority = priorityLabel(rank);
      const road = String(props.road_name || props.route_id || "Unnamed road").toLowerCase();
      const observation = observations.get(String(props.physical_site_id)) || null;
      if (rank > limit || rank < minRank || rank > maxRank) return false;
      if (query && !road.includes(query)) return false;
      if (filters.priority && filters.priority !== "all" && priority !== filters.priority) return false;
      if (filters.inspection === "inspected" && !observation) return false;
      if (filters.inspection === "uninspected" && observation) return false;
      if (filters.result && filters.result !== "all" && observation?.status !== filters.result) return false;
      return true;
    })
    .sort((a, b) => Number(a.properties.global_rank) - Number(b.properties.global_rank));
}

export function safeExportRows(features, observations = new Map()) {
  return features.map((feature) => {
    const props = feature.properties || {};
    const observation = observations.get(String(props.physical_site_id));
    return {
      global_rank: Number(props.global_rank),
      route: props.road_name || props.route_id || "",
      latitude: Number(props.latitude),
      longitude: Number(props.longitude),
      priority: priorityLabel(props.global_rank),
      recommendation_tier: "review",
      flow_c_score: numberOrBlank(props.flow_c_score),
      route_z_score: numberOrBlank(props.route_z_score),
      candidate_id: props.candidate_id || "",
      physical_site_id: props.physical_site_id || "",
      inspection_status: observation?.status || "uninspected",
    };
  });
}

export function rowsToCsv(rows) {
  if (!rows.length) return "";
  const columns = Object.keys(rows[0]);
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [columns.map(quote).join(","), ...rows.map((row) => columns.map((key) => quote(row[key])).join(","))].join("\n");
}

export function exportGeoJson(features, observations = new Map()) {
  const rows = safeExportRows(features, observations);
  return {
    type: "FeatureCollection",
    features: features.map((feature, index) => ({
      type: "Feature",
      geometry: feature.geometry,
      properties: rows[index],
    })),
  };
}

export function explanationLines(properties = {}) {
  const lines = [];
  if (Number.isFinite(Number(properties.distance_to_nearest_stream_m)) && Number(properties.distance_to_nearest_stream_m) <= 60) lines.push("Mapped stream information is close to the road location.");
  if (Number(properties.route_low_point_score) > 0.5 || Number(properties.dem_culvert_terrain_score) > 0.5) lines.push("Terrain measurements indicate a local low-point pattern.");
  if (Number.isFinite(Number(properties.local_road_stream_separation_m)) && Number(properties.local_road_stream_separation_m) <= 40) lines.push("Road and drainage geometry are locally close together.");
  if (Number(properties.route_local_rank) <= 10) lines.push("This location ranks strongly among sites on the same road.");
  return lines.length ? lines : ["The combined road, terrain, drainage, and hydrologic measurements produced a strong inspection rank."];
}

function numberOrBlank(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : "";
}
