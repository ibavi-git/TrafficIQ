import { Check, Circle, LoaderCircle } from 'lucide-react'

const PIPELINE = ['VIDEO LOADED', 'YOLO DETECTION + BYTETRACK', 'VEHICLE COUNTING', 'TRAFFIC ANALYSIS']

export function ProcessingStatus({ status, progress, metrics, filename, warning }) {
  const hasFrames = Boolean(metrics)
  const done = status === 'completed'
  const failed = status === 'error'
  const activeStep = hasFrames ? (done ? PIPELINE.length : 3) : 1
  const currentFrame = metrics?.frame ?? 0
  const totalFrames = metrics?.total_frames

  return (
    <section className="processing-panel">
      <div className="section-heading compact-heading">
        <div>
          <span className="eyebrow">ANALYSIS PIPELINE</span>
          <h2>{failed ? 'Processing stopped' : done ? 'Analysis complete' : 'Processing traffic video'}</h2>
        </div>
        <span className={`job-state state-${status}`}>{status.toUpperCase()}</span>
      </div>
      <div className="processing-file"><span>INPUT</span><strong title={filename}>{filename}</strong></div>
      <ol className="pipeline-list">
        {PIPELINE.map((step, index) => {
          const complete = index === 0 || index < activeStep
          const active = !complete && index === activeStep && !done && !failed
          const Icon = complete ? Check : active ? LoaderCircle : Circle
          return (
            <li className={`${complete ? 'complete' : ''} ${active ? 'active' : ''}`} key={step}>
              <span className="pipeline-icon"><Icon size={14} /></span>
              <span>{step}</span>
            </li>
          )
        })}
      </ol>
      <div className="progress-labels"><span>PROGRESS</span><strong>{progress}%</strong></div>
      <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
      <div className="frame-readout">
        <div><span>CURRENT FRAME</span><strong>{metrics ? currentFrame.toLocaleString() : 'N/A'}</strong></div>
        <div><span>PROCESSED</span><strong>{metrics?.processed_frames?.toLocaleString() ?? 'N/A'}</strong></div>
        <div><span>TOTAL FRAMES</span><strong>{totalFrames ? totalFrames.toLocaleString() : 'N/A'}</strong></div>
      </div>
      {warning && <p className="inline-warning" role="status">{warning}</p>}
    </section>
  )
}