import { Radio } from 'lucide-react'

export function TopBar({ title, eyebrow, connection, jobStatus }) {
  const connected = connection === 'connected'
  const running = jobStatus === 'processing' || jobStatus === 'uploading' || jobStatus === 'queued'
  return (
    <header className="topbar">
      <div className="topbar-heading">
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
      </div>
      <div className="topbar-meta">
        <div className={`connection-state ${connected ? 'is-connected' : ''}`}>
          <Radio size={15} />
          <span>{connected ? 'STREAM CONNECTED' : running ? 'CONNECTING' : 'READY'}</span>
        </div>
        <div className="topbar-date">LOCAL PROCESSING <span>·</span> CPU</div>
      </div>
    </header>
  )
}