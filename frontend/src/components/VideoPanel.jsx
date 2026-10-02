import { ScanLine, Video } from 'lucide-react'

export function VideoPanel({ preview, localPreview, status, filename }) {
  return (
    <section className="video-panel">
      <div className="video-panel-head">
        <div className="video-heading">
          <ScanLine size={17} />
          <div><span className="eyebrow">FRAME INSPECTION</span><h2>{preview ? 'Processed traffic feed' : 'Video source'}</h2></div>
        </div>
        <span className={`feed-indicator ${preview ? 'feed-live' : ''}`}><i />{preview ? 'LIVE FRAME' : status.toUpperCase()}</span>
      </div>
      <div className="video-stage">
        {preview ? (
          <img src={preview} alt="Live annotated frame from the uploaded traffic video" />
        ) : localPreview ? (
          <video src={localPreview} controls muted playsInline />
        ) : (
          <div className="video-empty">
            <Video size={28} strokeWidth={1.4} />
            <span>{filename || 'Waiting for video frames'}</span>
          </div>
        )}
        <span className="stage-corner top-left" /><span className="stage-corner top-right" />
        <span className="stage-corner bottom-left" /><span className="stage-corner bottom-right" />
      </div>
      <div className="video-panel-foot">
        <span>{filename || 'NO SOURCE SELECTED'}</span>
        <span>{preview ? 'YOLO · BYTETRACK · M4 ANALYSIS' : 'ANNOTATED PREVIEW WILL APPEAR HERE'}</span>
      </div>
    </section>
  )
}