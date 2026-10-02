import { useEffect, useRef, useState } from 'react'
import { FileVideo2, FolderOpen, UploadCloud } from 'lucide-react'

const ACCEPTED = new Set(['mp4', 'avi', 'mov'])
const MAX_BYTES = 500 * 1024 * 1024

function formatSize(bytes) {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`
}

export function UploadPanel({ onStart, error, busy }) {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [localError, setLocalError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState('')

  useEffect(() => {
    if (!file) {
      setPreviewUrl('')
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function choose(nextFile) {
    setLocalError('')
    if (!nextFile) return
    const extension = nextFile.name.split('.').pop()?.toLowerCase()
    if (!ACCEPTED.has(extension)) {
      setFile(null)
      setLocalError('Unsupported video format. Choose an MP4, AVI, or MOV file.')
      return
    }
    if (nextFile.size > MAX_BYTES) {
      setFile(null)
      setLocalError('This file exceeds the 500 MB upload limit.')
      return
    }
    if (nextFile.size === 0) {
      setFile(null)
      setLocalError('This file is empty. Choose another video.')
      return
    }
    setFile(nextFile)
  }

  return (
    <section className="upload-screen">
      <div className="upload-intro">
        <span className="eyebrow"><i className="eyebrow-line" /> AI TRAFFIC INTELLIGENCE</span>
        <h2>See the flow.<br /><span>Understand the road.</span></h2>
        <p>Analyze an uploaded traffic video with detection, tracking, and frame-level traffic metrics.</p>
      </div>

      <div className="upload-workspace">
        <div className="upload-controls">
          <div className="section-heading">
            <div><span className="eyebrow">NEW ANALYSIS</span><h3>Upload traffic video</h3></div>
            <span className="upload-index">01 <i>/ 01</i></span>
          </div>
          <button
            className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files[0]) }}
          >
            {file ? <FileVideo2 size={26} /> : <UploadCloud size={26} />}
            <strong>{file ? file.name : 'Drop a video to begin'}</strong>
            <span>{file ? `${formatSize(file.size)} · Ready to analyze` : 'or choose a file from your device'}</span>
            <span className="browse-chip"><FolderOpen size={14} /> BROWSE FILES</span>
          </button>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".mp4,.avi,.mov,video/mp4,video/x-msvideo,video/quicktime"
            onChange={(event) => choose(event.target.files?.[0])}
          />
          <div className="upload-specs"><span>SUPPORTED</span><strong>MP4 <i>/</i> AVI <i>/</i> MOV</strong><span>MAX FILE SIZE</span><strong>500 MB</strong></div>
          {(localError || error) && <p className="form-error" role="alert">{localError || error}</p>}
          <button className="primary-button" disabled={!file || busy} onClick={() => onStart(file)}>
            <span>{busy ? 'UPLOADING VIDEO' : 'START ANALYSIS'}</span>
            <span className="button-arrow">{busy ? '…' : '→'}</span>
          </button>
          <p className="upload-disclaimer">Processing begins only after you start the analysis. Video processing runs locally on the TrafficIQ backend.</p>
        </div>
        <div className="upload-preview">
          {previewUrl ? (
            <video src={previewUrl} controls muted playsInline />
          ) : (
            <div className="preview-placeholder">
              <div className="road-grid" aria-hidden="true"><span /><span /><span /></div>
              <div className="preview-crosshair" aria-hidden="true" />
              <div className="preview-caption"><span>INPUT PREVIEW</span><strong>AWAITING VIDEO SOURCE</strong></div>
            </div>
          )}
          <div className="preview-meta"><span>LOCAL FILE PREVIEW</span><span>ANALYSIS IS MANUAL</span></div>
        </div>
      </div>
    </section>
  )
}