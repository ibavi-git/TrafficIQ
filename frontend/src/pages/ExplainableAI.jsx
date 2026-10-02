import { useState } from 'react'
import { BrainCircuit, ExternalLink } from 'lucide-react'

const NOTE = 'This visualization explains the pretrained ViT representation and is not a validated traffic-congestion classifier explanation.'

export function ExplainableAI() {
  const [available, setAvailable] = useState(true)
  return (
    <div className="content-page explain-page">
      <div className="dashboard-title-row">
        <div><span className="eyebrow">M8 · REPRESENTATION SENSITIVITY</span><h2>Explainable AI</h2></div>
        <a className="quiet-button link-button" href="/api/xai/preview" target="_blank" rel="noreferrer"><ExternalLink size={15} /> OPEN PREVIEW</a>
      </div>
      <section className="xai-player-panel">
        <div className="video-panel-head"><div className="video-heading"><BrainCircuit size={18} /><div><span className="eyebrow">PRETRAINED VISION TRANSFORMER</span><h2>ViT Grad-CAM preview</h2></div></div><span className="xai-stage-tag">M8 OUTPUT</span></div>
        {available ? (
          <video className="xai-video" src="/api/xai/preview" controls playsInline onError={() => setAvailable(false)} />
        ) : (
          <div className="xai-unavailable"><BrainCircuit size={25} /><strong>M8 preview unavailable</strong><span>Generate outputs/xai/m8_gradcam_preview.mp4 to view the existing explanation.</span></div>
        )}
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