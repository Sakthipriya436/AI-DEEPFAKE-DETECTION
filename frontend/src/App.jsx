import { useState, useRef, useCallback } from 'react'
import './App.css'

// Signal icon map
const SIGNAL_ICONS = { warning: '⚠', ok: '✓', caution: '◈' }

function SignalRow({ label, type }) {
  return (
    <div className={`signal-row signal-${type}`}>
      <span className="signal-icon">{SIGNAL_ICONS[type]}</span>
      <span className="signal-label">{label}</span>
    </div>
  )
}

function PipelineDot({ label, sublabel, state }) {
  // state: 'idle' | 'active' | 'done'
  return (
    <div className={`pipe-step pipe-${state}`}>
      <div className="pipe-dot" />
      <div className="pipe-text">
        <span className="pipe-name">{label}</span>
        <span className="pipe-sub">{sublabel}</span>
      </div>
    </div>
  )
}

const PIPELINE_STEPS = [
  { label: 'Upload',            sublabel: 'Transfer file' },
  { label: 'BlazeFace',         sublabel: 'Detect faces' },
  { label: 'Xception + ResNeXt',sublabel: 'Classify frames' },
  { label: 'Ensemble',          sublabel: 'Fuse scores' },
]

export default function App() {
  const [file, setFile]       = useState(null)
  const [preview, setPreview] = useState(null)
  const [isVideo, setIsVideo] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage]     = useState(0)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef(null)

  const applyFile = useCallback((f) => {
    if (!f) return
    setFile(f); setResult(null); setError(null); setStage(0); setProgress(0)
    setPreview(URL.createObjectURL(f))
    setIsVideo(f.type.startsWith('video/'))
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    applyFile(e.dataTransfer.files[0])
  }, [applyFile])

  const reset = () => {
    setFile(null); setPreview(null); setResult(null); setError(null)
    setStage(0); setProgress(0)
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null); setStage(1)
    const fd = new FormData()
    fd.append('file', file)

    const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', `${API_BASE}/api/detect`)
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) setProgress(Math.round(e.loaded / e.total * 100))
        }
        const t1 = setTimeout(() => setStage(2), 600)
        const t2 = setTimeout(() => setStage(3), 1800)
        xhr.onload = () => {
          clearTimeout(t1); clearTimeout(t2); setStage(4)
          if (xhr.status === 200) resolve(JSON.parse(xhr.responseText))
          else reject(new Error(`Server error ${xhr.status}`))
        }
        xhr.onerror = () => reject(new Error('Cannot connect to backend (port 8000)'))
        xhr.send(fd)
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
      setStage(0)
    } finally {
      setLoading(false)
    }
  }

  const r = result || {}
  const isAI = r.is_deepfake
  const signals = r.signals || []
  const pipeline = r.pipeline || {}
  const videoMeta = r.video_metadata || {}

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="hdr">
        <div className="hdr-inner">
          <div className="brand">
            <div className="brand-icon">
              <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                <polygon points="11,2 20,6.5 20,15.5 11,20 2,15.5 2,6.5" stroke="url(#hg)" strokeWidth="1.4" fill="none"/>
                <circle cx="11" cy="11" r="4" fill="url(#hg)"/>
                <defs><linearGradient id="hg" x1="0" y1="0" x2="22" y2="22">
                  <stop offset="0%" stopColor="#a78bfa"/><stop offset="100%" stopColor="#38bdf8"/>
                </linearGradient></defs>
              </svg>
            </div>
            <span className="brand-name">DeepGuard</span>
          </div>
          <div className="hdr-chips">
            <span className="chip chip-violet">BlazeFace · Xception · ResNeXt</span>
            <span className="chip chip-green">● Live</span>
          </div>
        </div>
      </header>

      <div className="page">
        {/* ── Left Panel ── */}
        <aside className="panel-left">

          {/* Drop zone */}
          <div
            id="dropzone"
            className={`dropzone ${dragging ? 'dz-drag' : ''} ${file ? 'dz-filled' : ''}`}
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onClick={() => !file && fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept="image/*,video/*"
              onChange={e => applyFile(e.target.files[0])} style={{ display: 'none' }} />

            {!file ? (
              <div className="dz-empty">
                <div className="dz-upload-icon">
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                    <path d="M16 22V10M16 10L11 15M16 10L21 15" stroke="url(#ug)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M6 24H26" stroke="url(#ug)" strokeWidth="1.4" strokeLinecap="round"/>
                    <defs><linearGradient id="ug" x1="6" y1="10" x2="26" y2="26">
                      <stop offset="0%" stopColor="#a78bfa"/><stop offset="100%" stopColor="#38bdf8"/>
                    </linearGradient></defs>
                  </svg>
                </div>
                <p className="dz-title">Drop video or image here</p>
                <p className="dz-hint">or <span className="dz-link">browse</span> — up to 500MB</p>
                <p className="dz-formats">MP4 · AVI · MOV · JPG · PNG · WEBM</p>
              </div>
            ) : (
              <div className="dz-preview">
                {isVideo
                  ? <video src={preview} className="preview-media" controls muted playsInline />
                  : <img src={preview} className="preview-media" alt="preview" />
                }
                <div className="dz-file-info">
                  <span className="dz-fname">{file.name}</span>
                  <span className="dz-fsize">{(file.size/1024/1024).toFixed(1)} MB</span>
                </div>
                <button className="dz-remove" onClick={e => { e.stopPropagation(); reset() }}>✕ Remove</button>
              </div>
            )}
          </div>

          {/* Progress bar */}
          {loading && (
            <div className="upload-prog">
              <div className="up-bar"><div className="up-fill" style={{ width: `${progress}%` }} /></div>
              <span className="up-label">{progress < 100 ? `Uploading ${progress}%` : 'Processing…'}</span>
            </div>
          )}

          {/* Analyze button */}
          <button id="analyze-btn" className={`analyze-btn ${loading ? 'ana-busy' : ''}`}
            onClick={analyze} disabled={!file || loading}>
            {loading
              ? <><span className="spin" /><span>Analyzing…</span></>
              : <><span className="btn-icon">⬡</span><span>Detect AI / Real</span></>}
          </button>

          {/* Pipeline */}
          <div className="pipeline-box">
            <p className="box-label">PIPELINE</p>
            {PIPELINE_STEPS.map((s, i) => (
              <PipelineDot key={i} label={s.label} sublabel={s.sublabel}
                state={stage === 0 ? 'idle' : stage === i + 1 ? 'active' : stage > i + 1 ? 'done' : 'idle'} />
            ))}
          </div>

          {/* Architecture chips */}
          <div className="arch-box">
            <p className="box-label">ARCHITECTURE</p>
            <div className="arch-chips">
              {[['⚡','BlazeFace','Face detection'],['🧠','Xception','Texture analysis'],
                ['🔬','ResNeXt','Pattern analysis'],['⚖️','Ensemble','55% + 45% fusion']
              ].map(([icon,name,desc]) => (
                <div className="arch-chip" key={name}>
                  <span>{icon}</span>
                  <div><strong>{name}</strong><br/><small>{desc}</small></div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* ── Right Panel (Results) ── */}
        <main className="panel-right">

          {/* Error */}
          {error && (
            <div className="result-box rb-error">
              <div className="rb-icon-wrap rb-icon-error">⚠</div>
              <h2 className="rb-title">Analysis Failed</h2>
              <p className="rb-sub">{error}</p>
            </div>
          )}

          {/* Idle */}
          {!result && !error && !loading && (
            <div className="result-box rb-idle">
              <div className="idle-art">
                <svg width="90" height="90" viewBox="0 0 90 90" fill="none">
                  <circle cx="45" cy="45" r="40" stroke="rgba(167,139,250,0.12)" strokeWidth="1.5" strokeDasharray="8 5"/>
                  <circle cx="45" cy="45" r="26" stroke="rgba(56,189,248,0.1)" strokeWidth="1.5"/>
                  <circle cx="45" cy="45" r="12" fill="rgba(167,139,250,0.08)"/>
                  <path d="M31 31L59 59M59 31L31 59" stroke="rgba(167,139,250,0.1)" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
              </div>
              <h2 className="rb-title idle-title">Awaiting Analysis</h2>
              <p className="rb-sub">Upload a video or image to detect AI generation</p>
              <div className="idle-features">
                <div className="idle-feat">
                  <span>🎯</span>
                  <span>Detects GAN-generated faces</span>
                </div>
                <div className="idle-feat">
                  <span>⚡</span>
                  <span>Sub-second face detection</span>
                </div>
                <div className="idle-feat">
                  <span>🔬</span>
                  <span>Multi-model ensemble fusion</span>
                </div>
              </div>
            </div>
          )}

          {/* Analyzing */}
          {loading && (
            <div className="result-box rb-analyzing">
              <div className="scan-anim">
                <div className="scan-ring sr1" />
                <div className="scan-ring sr2" />
                <div className="scan-ring sr3" />
                <span className="scan-core">⬡</span>
              </div>
              <h2 className="rb-title">
                {stage === 1 && 'Uploading…'}
                {stage === 2 && 'Detecting Faces…'}
                {stage === 3 && 'Running Neural Networks…'}
                {stage === 4 && 'Computing Final Verdict…'}
              </h2>
              <p className="rb-sub">
                {stage === 2 && 'BlazeFace extracting face regions'}
                {stage === 3 && 'Xception + ResNeXt analyzing patterns'}
                {stage === 4 && 'Fusing ensemble predictions'}
              </p>
              <div className="analyzing-steps">
                {PIPELINE_STEPS.map((s, i) => (
                  <div key={i} className={`ana-step ${stage > i ? 'ana-done' : stage === i + 1 ? 'ana-active' : ''}`}>
                    <span>{stage > i + 1 ? '✓' : stage === i + 1 ? '◌' : '○'}</span>
                    <span>{s.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Verdict */}
          {result && (
            <div className={`result-box rb-verdict ${isAI ? 'rb-ai' : 'rb-real'}`}>

              {/* Big verdict */}
              <div className="verdict-hero">
                <div className={`verdict-icon-ring ${isAI ? 'vring-ai' : 'vring-real'}`}>
                  <span className="verdict-emoji">{isAI ? '🤖' : '✅'}</span>
                </div>
                <div className="verdict-text">
                  <div className={`verdict-main ${isAI ? 'vt-ai' : 'vt-real'}`}>
                    {isAI ? 'AI GENERATED' : 'REAL'}
                  </div>
                  <div className="verdict-sub">{r.analysis}</div>
                </div>
              </div>

              {/* Evidence signals */}
              {signals.length > 0 && (
                <div className="signals-section">
                  <p className="box-label">EVIDENCE SIGNALS</p>
                  <div className="signals-list">
                    {signals.map((s, i) => (
                      <SignalRow key={i}
                        label={Array.isArray(s) ? s[0] : s.label || s}
                        type={Array.isArray(s) ? s[1] : s.type || 'ok'} />
                    ))}
                  </div>
                </div>
              )}

              {/* Quick stats */}
              <div className="quick-stats">
                <div className="qs-item">
                  <span className="qs-val">{r.frames_analyzed ?? 1}</span>
                  <span className="qs-label">Frames</span>
                </div>
                <div className="qs-divider" />
                <div className="qs-item">
                  <span className="qs-val">{r.total_faces_analyzed ?? r.faces_detected ?? 0}</span>
                  <span className="qs-label">Faces</span>
                </div>
                <div className="qs-divider" />
                <div className="qs-item">
                  <span className="qs-val">
                    {pipeline.total_ms ? (pipeline.total_ms < 1000
                      ? `${Math.round(pipeline.total_ms)}ms`
                      : `${(pipeline.total_ms/1000).toFixed(1)}s`) : '—'}
                  </span>
                  <span className="qs-label">Time</span>
                </div>
                <div className="qs-divider" />
                <div className="qs-item">
                  <span className="qs-val">{r.model_agreement != null ? `${(r.model_agreement*100).toFixed(0)}%` : '—'}</span>
                  <span className="qs-label">Agreement</span>
                </div>
              </div>

              {/* Video meta if available */}
              {r.file_type === 'video' && videoMeta.fps && (
                <div className="vmeta">
                  <span>{videoMeta.resolution}</span>
                  <span className="vmeta-dot">·</span>
                  <span>{videoMeta.duration_sec}s</span>
                  <span className="vmeta-dot">·</span>
                  <span>{videoMeta.fps} fps</span>
                  <span className="vmeta-dot">·</span>
                  <span>{videoMeta.file_size_mb} MB</span>
                </div>
              )}

              {/* Detector tag */}
              <div className="detector-tag">
                Analyzed by &nbsp;<strong>Xception + ResNeXt Ensemble</strong>&nbsp;·&nbsp;
                <span style={{color: 'var(--text3)'}}>Face detector: {pipeline.face_detector || 'Haar Cascade'}</span>
              </div>
            </div>
          )}
        </main>
      </div>

      <footer className="foot">
        <span>DeepGuard — BlazeFace · Xception · ResNeXt · Ensemble Fusion</span>
        <span className="foot-sep">|</span>
        <span>AI :5000 · Backend :8000 · Frontend :5174</span>
      </footer>
    </div>
  )
}
