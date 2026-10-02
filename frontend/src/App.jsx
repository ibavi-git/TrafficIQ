import { useEffect, useState } from 'react'
import { Sidebar } from './components/Sidebar.jsx'
import { TopBar } from './components/TopBar.jsx'
import { Dashboard } from './pages/Dashboard.jsx'
import { ExplainableAI } from './pages/ExplainableAI.jsx'
import { AIControl } from './pages/AIControl.jsx'
import { Roadmap } from './pages/Roadmap.jsx'

const PAGE_META = {
  dashboard: { title: 'Traffic monitor', eyebrow: 'LIVE OPERATIONS' },
  explainability: { title: 'Explainable AI', eyebrow: 'MODEL INSPECTION' },
  control: { title: 'AI control', eyebrow: 'CAPABILITY REGISTER' },
  roadmap: { title: 'Roadmap', eyebrow: 'DEVELOPMENT TRACK' },
}

const EMPTY_XAI = {
  status: 'idle',
  progress: 0,
  videoUrl: null,
  metadataUrl: null,
  sourceFilename: null,
  error: '',
}

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [metrics, setMetrics] = useState(null)
  const [filename, setFilename] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [connection, setConnection] = useState('offline')
  const [connectionWarning, setConnectionWarning] = useState('')
  const [xai, setXai] = useState(EMPTY_XAI)
  const xaiRunning = xai.status === 'queued' || xai.status === 'processing'

  useEffect(() => {
    if (!jobId) return undefined

    let disposed = false
    let terminal = false
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/${jobId}`)

    socket.onopen = () => {
      if (!disposed) {
        setConnection('connected')
        setConnectionWarning('')
      }
    }
    socket.onmessage = (event) => {
      let update
      try {
        update = JSON.parse(event.data)
      } catch {
        setConnectionWarning('Received an unreadable update from the processing service.')
        return
      }

      if (update.type === 'metrics') {
        setMetrics(update)
        setProgress(update.progress ?? 0)
        setJobStatus('processing')
      } else if (update.type === 'status') {
        setJobStatus(update.status)
        if (typeof update.progress === 'number') setProgress(update.progress)
        if (update.status === 'error') setUploadError(update.message || 'Processing failed.')
        if (update.status === 'completed' || update.status === 'error') terminal = true
      }
    }
    socket.onerror = () => {
      if (!disposed) setConnection('disconnected')
    }
    socket.onclose = () => {
      if (!disposed) {
        setConnection('disconnected')
        if (!terminal) setConnectionWarning('Live updates disconnected before the job finished.')
      }
    }

    return () => {
      disposed = true
      socket.close()
    }
  }, [jobId])

  useEffect(() => {
    if (!jobId || !xaiRunning) return undefined

    let disposed = false
    let terminal = false
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/${jobId}/xai`)

    socket.onmessage = (event) => {
      let update
      try {
        update = JSON.parse(event.data)
      } catch {
        return
      }
      if (update.type !== 'xai_status') return
      if (update.status === 'completed' || update.status === 'error') terminal = true
      setXai((current) => ({
        ...current,
        status: update.status,
        progress: update.progress ?? current.progress,
        videoUrl: update.video_url ?? current.videoUrl,
        metadataUrl: update.metadata_url ?? current.metadataUrl,
        sourceFilename: update.source_filename ?? current.sourceFilename,
        error: update.message || '',
      }))
    }
    socket.onclose = () => {
      if (!disposed && !terminal) {
        setXai((current) => current.status === 'completed' || current.status === 'error'
          ? current
          : { ...current, status: 'error', error: 'XAI status connection was interrupted. Retry generation.' })
      }
    }

    return () => {
      disposed = true
      socket.close()
    }
  }, [jobId, xaiRunning])

  async function startAnalysis(file) {
    setUploadError('')
    setConnectionWarning('')
    setMetrics(null)
    setProgress(0)
    setXai(EMPTY_XAI)
    setFilename(file.name)
    setJobId(null)
    setConnection('connecting')
    setJobStatus('uploading')

    const body = new FormData()
    body.append('file', file)
    try {
      const response = await fetch('/api/upload', { method: 'POST', body })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(result.detail || 'The video could not be uploaded.')
      }
      setJobId(result.job_id)
      setJobStatus(result.status || 'queued')
    } catch (error) {
      setJobStatus('idle')
      setConnection('offline')
      setUploadError(
        error instanceof TypeError
          ? 'TrafficIQ could not reach the backend. Start the FastAPI server and try again.'
          : error.message,
      )
    }
  }

  async function generateXai() {
    if (!jobId || jobStatus !== 'completed') return
    setXai({ ...EMPTY_XAI, status: 'queued' })
    try {
      const response = await fetch(`/api/jobs/${jobId}/xai`, { method: 'POST' })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(result.detail || 'Could not start Explainable AI.')
      if (result.status === 'completed') {
        setXai((current) => ({
          ...current,
          status: 'completed',
          progress: 100,
          videoUrl: result.video_url,
          metadataUrl: result.metadata_url,
          sourceFilename: filename,
          error: '',
        }))
      }
    } catch (error) {
      setXai((current) => ({
        ...current,
        status: 'error',
        progress: 0,
        error: error instanceof TypeError
          ? 'TrafficIQ could not reach the backend. Try again when the service is available.'
          : error.message,
      }))
    }
  }

  function resetAnalysis() {
    setJobId(null)
    setJobStatus('idle')
    setProgress(0)
    setMetrics(null)
    setXai(EMPTY_XAI)
    setFilename('')
    setUploadError('')
    setConnectionWarning('')
    setConnection('offline')
  }

  const activePage = PAGE_META[page] ? page : 'dashboard'
  const content = {
    dashboard: (
      <Dashboard
        jobId={jobId}
        jobStatus={jobStatus}
        progress={progress}
        metrics={metrics}
        filename={filename}
        error={uploadError}
        connectionWarning={connectionWarning}
        xaiStatus={xai.status}
        xaiProgress={xai.progress}
        xaiError={xai.error}
        onGenerateXai={generateXai}
        onStart={startAnalysis}
        onReset={resetAnalysis}
      />
    ),
    explainability: (
      <ExplainableAI
        jobStatus={jobStatus}
        jobFilename={filename}
        xaiStatus={xai.status}
        xaiProgress={xai.progress}
        xaiVideoUrl={xai.videoUrl}
        xaiMetadataUrl={xai.metadataUrl}
        xaiSourceFilename={xai.sourceFilename}
        xaiError={xai.error}
      />
    ),
    control: <AIControl />,
    roadmap: <Roadmap />,
  }[activePage]

  return (
    <div className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setPage} />
      <div className="app-main">
        <TopBar
          title={PAGE_META[activePage].title}
          eyebrow={PAGE_META[activePage].eyebrow}
          connection={connection}
          jobStatus={jobStatus}
        />
        <main className="page-content">{content}</main>
      </div>
    </div>
  )
}