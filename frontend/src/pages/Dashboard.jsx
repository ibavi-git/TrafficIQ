import { useEffect, useState } from 'react'
import { AlertTriangle, Clock3, RotateCcw, Users, CarFront, Layers3 } from 'lucide-react'
import { MetricCard } from '../components/MetricCard.jsx'
import { ProcessingStatus } from '../components/ProcessingStatus.jsx'
import { TrafficStatus } from '../components/TrafficStatus.jsx'
import { UploadPanel } from '../components/UploadPanel.jsx'
import { VideoPanel } from '../components/VideoPanel.jsx'

export function Dashboard({ jobId, jobStatus, progress, metrics, filename, error, connectionWarning, xaiStatus, xaiProgress, xaiError, onGenerateXai, onStart, onReset }) {
  const [busy, setBusy] = useState(false)
  const [localPreview, setLocalPreview] = useState('')
  const preview = metrics?.frame_jpeg_base64 ? `data:image/jpeg;base64,${metrics.frame_jpeg_base64}` : ''

  useEffect(() => {
    if (!jobId) return undefined
    setLocalPreview('')
    return undefined
  }, [jobId])

  async function handleStart(file) {
    setBusy(true)
    try {
      await onStart(file)
    } finally {
      setBusy(false)
    }
  }

  if (!jobId && jobStatus === 'idle') {
    return <UploadPanel onStart={handleStart} error={error} busy={busy} />
  }

  const value = (key) => metrics?.[key] ?? null
  return (
    <div className="dashboard-view">
      <div className="dashboard-title-row">
        <div><span className="eyebrow">TRAFFIC MONITOR</span><h2>{jobStatus === 'completed' ? 'Run summary' : 'Live analysis'}</h2></div>
        <button className="quiet-button" onClick={onReset}><RotateCcw size={15} /> NEW ANALYSIS</button>
      </div>
      <div className="dashboard-grid">
        <div className="dashboard-main-column">
          {jobStatus === 'uploading' ? (
            <section className="processing-panel upload-progress-panel"><span className="eyebrow">VIDEO UPLOAD</span><h2>Sending video to TrafficIQ</h2><div className="progress-track"><span className="progress-indeterminate" /></div></section>
          ) : (
            <VideoPanel preview={preview} localPreview={localPreview} status={jobStatus} filename={filename} />
          )}
          <section className="metrics-section">
            <div className="section-heading metrics-heading"><div><span className="eyebrow">FRAME-LEVEL MEASUREMENTS</span><h2>Traffic metrics</h2></div><span className="metric-source"><i /> FROM UPLOADED VIDEO</span></div>
            <div className="metrics-grid">
              <MetricCard label="UNIQUE VEHICLES" value={value('unique_vehicles')} icon={Users} />
              <MetricCard label="PEAK VEHICLES" value={value('peak_vehicles')} icon={Layers3} accent="metric-amber" />
              <MetricCard label="CURRENT VEHICLES" value={value('vehicles')} icon={CarFront} />
              <MetricCard label="CONGESTION INDEX" value={value('congestion_index')} unit="/ 100" icon={AlertTriangle} accent="metric-cyan" />
              <MetricCard label="OCCUPANCY" value={value('occupancy')} unit="%" icon={Layers3} />
              <MetricCard label="ANOMALIES" value={value('anomalies')} icon={AlertTriangle} accent="metric-red" />
              <MetricCard label="TIMESTAMP" value={value('timestamp') === null ? null : `${Number(value('timestamp')).toFixed(2)} s`} icon={Clock3} />
            </div>
          </section>
        </div>
        <aside className="dashboard-side-column">
          <TrafficStatus status={metrics?.congestion_status} index={metrics?.congestion_index} />
          <ProcessingStatus status={jobStatus} progress={progress} metrics={metrics} filename={filename} warning={connectionWarning} />
          {jobStatus === 'completed' && (
            <section className="xai-launch-panel">
              <span className="eyebrow">ON-DEMAND M8</span>
              <button
                className="primary-button xai-launch-button"
                disabled={xaiStatus === 'queued' || xaiStatus === 'processing' || xaiStatus === 'completed'}
                onClick={onGenerateXai}
              >
                <span>{xaiStatus === 'processing' || xaiStatus === 'queued' ? 'GENERATING XAI...' : xaiStatus === 'completed' ? 'EXPLAINABLE AI READY' : xaiStatus === 'error' ? 'RETRY EXPLAINABLE AI' : 'GENERATE EXPLAINABLE AI'}</span>
                <span className="button-arrow">{xaiStatus === 'completed' ? '✓' : '→'}</span>
              </button>
              {(xaiStatus === 'queued' || xaiStatus === 'processing') && (
                <div className="xai-progress" role="status">
                  <div className="progress-labels"><span>GRAD-CAM SAMPLE JOB</span><strong>{xaiProgress}%</strong></div>
                  <div className="progress-track"><span style={{ width: `${xaiProgress}%` }} /></div>
                </div>
              )}
              {xaiStatus === 'completed' && <p className="xai-launch-note">Sampled Grad-CAM output is ready on the Explainable AI page.</p>}
              {xaiError && <p className="form-error" role="alert">{xaiError}</p>}
            </section>
          )}
          {jobStatus === 'error' && error && <div className="error-panel" role="alert"><AlertTriangle size={17} /><span>{error}</span></div>}
        </aside>
      </div>
    </div>
  )
}