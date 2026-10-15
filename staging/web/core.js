export function displayedSites(data, budget) {
  const sites = [...(data?.sites?.features || [])].sort((a,b) => a.properties.global_rank-b.properties.global_rank || a.properties.candidate_id.localeCompare(b.properties.candidate_id));
  return budget === 'all' ? sites : sites.slice(0, Number(budget));
}
export function measurement(value, suffix='') {
  return value === null || value === undefined || !Number.isFinite(Number(value)) ? 'Unavailable' : Number(value).toLocaleString(undefined,{maximumFractionDigits:3})+suffix;
}
export function exportSites(data, budget) {
  return {type:'FeatureCollection', experimental:true, field_verification_notice:'Recommendations require field verification.', score_interpretation:'Uncalibrated relative ranking score, not a probability.', model:data.model, features:displayedSites(data,budget)};
}
