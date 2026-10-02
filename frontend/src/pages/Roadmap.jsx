import { ArrowRight, Check, Circle, CircleDot } from 'lucide-react'

const MILESTONES = [
  { id: 'M1', title: 'Detection', state: 'complete' },
  { id: 'M2', title: 'Tracking', state: 'complete' },
  { id: 'M3', title: 'Counting', state: 'complete' },
  { id: 'M4', title: 'Traffic Analysis', state: 'complete' },
  { id: 'M5', title: 'Benchmark', state: 'complete' },
  { id: 'M6', title: 'ViT Context', state: 'complete' },
  { id: 'M7', title: 'Feature Fusion', state: 'complete' },
  { id: 'M8', title: 'Explainability', state: 'complete' },
  { id: 'M9', title: 'CPU Optimization', state: 'next' },
  { id: 'M10', title: 'Demand Prediction', state: 'planned' },
  { id: 'M11', title: 'Adaptive Signal Timing', state: 'planned' },
  { id: 'M12', title: 'Emergency Detection', state: 'planned' },
  { id: 'M13', title: 'Emergency Priority', state: 'planned' },
  { id: 'M14', title: 'Decision Engine', state: 'planned' },
]

const STATUS = {
  complete: { label: 'COMPLETE', Icon: Check },
  next: { label: 'NEXT', Icon: CircleDot },
  planned: { label: 'PLANNED', Icon: Circle },
}

export function Roadmap() {
  return (
    <div className="content-page roadmap-page">
      <div className="dashboard-title-row"><div><span className="eyebrow">MILESTONE TRACKER</span><h2>Development roadmap</h2></div><span className="roadmap-progress">08 <i>/</i> 14 MILESTONES</span></div>
      <section className="roadmap-list" aria-label="TrafficIQ development milestones">
        {MILESTONES.map(({ id, title, state }) => {
          const { label, Icon } = STATUS[state]
          return <div className={`roadmap-row roadmap-${state}`} key={id}><span className="roadmap-id">{id}</span><span className="roadmap-marker"><Icon size={15} /></span><strong>{title}</strong><span className="roadmap-status">{label}</span>{state === 'next' && <ArrowRight className="roadmap-arrow" size={16} />}</div>
        })}
      </section>
    </div>
  )
}