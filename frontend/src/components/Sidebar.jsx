import { Activity, BrainCircuit, CircuitBoard, Gauge, Route } from 'lucide-react'

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Traffic monitor', icon: Activity },
  { id: 'explainability', label: 'Explainable AI', icon: BrainCircuit },
  { id: 'control', label: 'AI control', icon: Gauge },
  { id: 'roadmap', label: 'Roadmap', icon: Route },
]

export function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <button className="brand-lockup" onClick={() => onNavigate('dashboard')} aria-label="TrafficIQ home">
        <span className="brand-mark"><CircuitBoard size={20} strokeWidth={1.8} /></span>
        <span className="brand-wordmark">TRAFFIC<span>IQ</span></span>
      </button>

      <div className="sidebar-label">WORKSPACE</div>
      <nav className="primary-nav" aria-label="Main navigation">
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`nav-item ${activePage === id ? 'active' : ''}`}
            onClick={() => onNavigate(id)}
            aria-label={label}
            aria-current={activePage === id ? 'page' : undefined}
          >
            <Icon size={17} strokeWidth={1.8} />
            <span>{label}</span>
            {activePage === id && <span className="nav-active-mark" />}
          </button>
        ))}
      </nav>

      <div className="sidebar-foot">
        <span className="sidebar-foot-dot" />
        <div><strong>CORE ONLINE</strong><small>M1-M8 available</small></div>
      </div>
    </aside>
  )
}