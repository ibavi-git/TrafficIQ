import { useEffect, useState } from 'react'
import { BrainCircuit, ExternalLink } from 'lucide-react'

const NOTE = 'This visualization explains the pretrained ViT representation and is not a validated traffic-congestion classifier explanation.'

export function ExplainableAI({ jobStatus, jobFilename, xaiStatus, xaiProgress, xaiVideoUrl, xaiMetadataUrl, xaiSourceFilename, xaiError }) {
  const [available, setAvailable] = useState(true)
  const showSample = !xaiVideoUrl && xaiStatus === 'idle'
  const videoUrl = xaiVideoUrl || (showSample ? '/api/xai/preview' : null)

  useEffect(() => {
    setAvailable(true)
  }, [videoUrl])

  return (
    <div className="content-page explain-page">
      <div className="dashboard-title-row">
        <div><span className="eyebrow">M8 · REPRESENTATION SENSITIVITY</span><h2>Explainable AI</h2></div>
        {xaiMetadataUrl && <a className="quiet-button link-button" href={xaiMetadataUrl} target="_blank" rel="noreferrer"><ExternalLink size={15} /> OPEN METADATA</a>}
      </div>
      <section className="xai-player-panel">
        <div className="video-panel-head"><div className="video-heading"><BrainCircuit size={18} /><div><span className="eyebrow">PRETRAINED VISION TRANSFORMER</span><h2>{xaiVideoUrl ? 'Generated job explanation' : showSample ? 'Sample Grad-CAM preview' : 'Explainable AI generation'}</h2></div></div><span className="xai-stage-tag">{xaiVideoUrl ? 'JOB-SPECIFIC M8' : showSample ? 'SAMPLE / DEMO' : xaiStatus.toUpperCase()}</span></div>
        {videoUrl && available ? (
          <video className="xai-video" src={videoUrl} controls playsInline onError={() => setAvailable(false)} />
        ) : xaiStatus === 'queued' || xaiStatus === 'processing' ? (
          <div className="xai-unavailable"><BrainCircuit size={25} /><strong>Generating XAI...</strong><span>Grad-CAM is processing a bounded sample of {jobFilename || 'the uploaded video'} on CPU.</span><div className="progress-track"><span style={{ width: `${xaiProgress}%` }} /></div><small>{xaiProgress}%</small></div>
        ) : xaiError ? (
          <div className="xai-unavailable" role="alert"><BrainCircuit size={25} /><strong>Explainable AI generation failed</strong><span>{xaiError}</span></div>
        ) : available && showSample ? (
          <video className="xai-video" src="/api/xai/preview" controls playsInline onError={() => setAvailable(false)} />
        ) : (
          <div className="xai-unavailable"><BrainCircuit size={25} /><strong>{jobStatus === 'completed' ? 'Generate an explanation for this upload' : 'No uploaded-video result yet'}</strong><span>Start Explainable AI from the completed Traffic Analysis dashboard. M8 runs only when requested.</span></div>
        )}
        {xaiVideoUrl && <div className="xai-result-source"><span>GENERATED FOR</span><strong>{xaiSourceFilename || jobFilename}</strong></div>}
        {showSample && <p className="sample-warning"><strong>SAMPLE / DEMO ONLY</strong> This is the existing preview, not an explanation for your uploaded video.</p>}
      </section>
      <section className="xai-details">
        <div className="xai-model"><span className="eyebrow">MODEL</span><strong>google/vit-base-patch16-224</strong></div>
        <div><span className="eyebrow">EMBEDDING</span><strong>768 dimensions</strong></div>
        <div><span className="eyebrow">PATCH GRID</span><strong>14 × 14</strong></div>
        <div><span className="eyebrow">TARGET LAYER</span><strong>layers.11.layernorm_before</strong></div>
        <div className="xai-target"><span className="eyebrow">TARGET</span><strong>Strongest-magnitude CLS representation component</strong></div>
      </section>
      <p className="method-note"><span>METHOD NOTE</span>{NOTE}</p>
    </div>
  )
}