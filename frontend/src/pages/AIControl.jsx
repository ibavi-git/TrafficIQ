import { Check, Circle, Cpu } from 'lucide-react'

const ACTIVE = ['Detection', 'Tracking', 'Counting', 'Traffic Analysis', 'ViT Context', 'Feature Fusion', 'Explainability']
const NEXT = ['Traffic Demand Prediction', 'Adaptive Signal Timing', 'Emergency Vehicle Detection', 'Emergency Vehicle Prioritization', 'AI Traffic Decision Engine']

export function AIControl() {
  return (
    <div className="content-page">
      <div className="dashboard-title-row"><div><span className="eyebrow">SYSTEM CAPABILITIES</span><h2>AI control</h2></div><span className="control-badge"><Cpu size={15} /> M1-M8 READY</span></div>
      <section className="capability-section"><div className="section-heading"><div><span className="eyebrow">AVAILABLE NOW</span><h3>Current AI capabilities</h3></div><span className="capability-count">07 ACTIVE</span></div><div className="capability-list">{ACTIVE.map((item, index) => <div className="capability-row" key={item}><span className="capability-index">0{index + 1}</span><span className="capability-check"><Check size={15} /></span><strong>{item}</strong><span className="capability-live">AVAILABLE</span></div>)}</div></section>
      <section className="capability-section coming-section"><div className="section-heading"><div><span className="eyebrow">NOT IMPLEMENTED</span><h3>Coming next</h3></div><span className="capability-count muted-count">05 PLANNED</span></div><div className="capability-list">{NEXT.map((item, index) => <div className="capability-row coming-row" key={item}><span className="capability-index">{String(index + 9).padStart(2, '0')}</span><span className="capability-check"><Circle size={14} /></span><strong>{item}</strong><span className="capability-live">PLANNED</span></div>)}</div></section>
    </div>
  )
}