import { Activity } from 'lucide-react'

export function TrafficStatus({ status, index }) {
  const value = status || 'N/A'
  const tone = ['LOW', 'MODERATE', 'HIGH', 'SEVERE'].includes(value) ? value.toLowerCase() : 'unknown'
  return (
    <section className={`traffic-status status-${tone}`} aria-live="polite">
      <div className="traffic-status-top">
        <span className="eyebrow">TRAFFIC STATE</span>
        <Activity size={16} />
      </div>
      <div className="traffic-status-value">{value}</div>
      <div className="traffic-status-index">
        CONGESTION INDEX <strong>{index === null || index === undefined ? 'N/A' : `${index}%`}</strong>
      </div>
      <div className="status-scale" aria-hidden="true">
        <span className={tone === 'low' ? 'current' : ''} />
        <span className={tone === 'moderate' ? 'current' : ''} />
        <span className={tone === 'high' ? 'current' : ''} />
        <span className={tone === 'severe' ? 'current' : ''} />
      </div>
      <div className="status-scale-labels"><span>LOW</span><span>MOD</span><span>HIGH</span><span>SEVERE</span></div>
    </section>
  )
}