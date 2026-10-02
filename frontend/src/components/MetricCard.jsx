export function MetricCard({ label, value, unit = '', icon: Icon, accent = '' }) {
  const shown = value === null || value === undefined ? 'N/A' : value
  return (
    <article className={`metric-card ${accent}`}>
      <div className="metric-head">
        <span>{label}</span>
        {Icon && <Icon size={16} strokeWidth={1.8} aria-hidden="true" />}
      </div>
      <div className="metric-value-row">
        <strong>{shown}</strong>
        {value !== null && value !== undefined && unit && <small>{unit}</small>}
      </div>
    </article>
  )
}