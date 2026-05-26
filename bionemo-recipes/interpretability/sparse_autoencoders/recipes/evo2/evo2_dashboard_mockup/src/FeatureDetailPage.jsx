import React, { useState, useEffect, useRef } from 'react'
import SequenceView, { computeAlignInfo } from './SequenceView'
import { getRegionLabel } from './utils'

const styles = {
  overlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.5)',
    zIndex: 2000,
    overflowY: 'auto',
  },
  page: {
    maxWidth: '960px',
    margin: '20px auto',
    background: 'var(--bg-card)',
    borderRadius: '8px',
    boxShadow: '0 4px 24px rgba(0,0,0,0.2)',
    color: 'var(--text)',
  },
  header: {
    padding: '12px 20px',
    borderBottom: '1px solid var(--border-light)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: '14px',
    fontWeight: '700',
    color: 'var(--text-heading)',
  },
  closeBtn: {
    background: 'none',
    border: '1px solid var(--border-input)',
    borderRadius: '4px',
    padding: '3px 10px',
    cursor: 'pointer',
    fontSize: '11px',
    color: 'var(--text-secondary)',
  },
  section: {
    padding: '10px 20px',
    borderBottom: '1px solid var(--border-light)',
  },
  sectionTitle: {
    fontSize: '11px',
    fontWeight: '600',
    marginBottom: '6px',
    color: 'var(--text-heading)',
    textTransform: 'uppercase',
  },
  example: {
    marginBottom: '6px',
    padding: '6px 8px',
    background: 'var(--bg-example)',
    borderRadius: '4px',
    border: '1px solid var(--border-light)',
  },
  exampleMeta: {
    fontSize: '10px',
    color: 'var(--text-secondary)',
    marginBottom: '4px',
    fontFamily: 'monospace',
    display: 'flex',
    justifyContent: 'space-between',
  },
  placeholder: {
    border: '1px dashed var(--border)',
    borderRadius: '6px',
    padding: '24px',
    textAlign: 'center',
    color: 'var(--text-muted)',
    fontSize: '12px',
    fontStyle: 'italic',
  },
  placeholderLabel: {
    fontSize: '13px',
    fontWeight: '500',
    color: 'var(--text-muted)',
    marginBottom: '8px',
  },
}

export default function FeatureDetailPage({ feature, examples, onClose }) {
  const [alignMode, setAlignMode] = useState('max_activation')
  const scrollGroupRef = useRef(null)

  const freq = feature.activation_freq || 0
  const maxAct = feature.max_activation || 0
  const description = feature.description || feature.label || `Feature ${feature.feature_id}`

  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  const visibleExamples = (examples || []).slice(0, 30)
  const { anchor: alignAnchor, totalLength } = computeAlignInfo(visibleExamples.slice(0, 6), alignMode)

  return (
    <div style={styles.overlay} onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div style={styles.page}>

        <div style={styles.header}>
          <div>
            <div style={styles.title}>
              Feature #{feature.feature_id}
              <span style={{ fontWeight: 400, fontSize: '11px', color: 'var(--text-secondary)', marginLeft: '8px' }}>
                {description}
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ display: 'flex', gap: '10px', fontSize: '10px', color: 'var(--text-secondary)' }}>
              <span>freq: <strong>{(freq * 100).toFixed(1)}%</strong></span>
              <span>max: <strong>{maxAct.toFixed(1)}</strong></span>
            </div>
            <button style={styles.closeBtn} onClick={onClose}>✕</button>
          </div>
        </div>

        <div style={styles.section}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <div style={styles.sectionTitle}>Top Activating Sequences</div>
            <div style={{ display: 'flex', gap: '4px', fontSize: '10px' }}>
              {['start', 'first_activation', 'max_activation'].map(mode => (
                <button
                  key={mode}
                  onClick={() => setAlignMode(mode)}
                  style={{
                    padding: '2px 8px', borderRadius: '3px', cursor: 'pointer', fontSize: '10px',
                    border: alignMode === mode ? '1px solid var(--accent)' : '1px solid var(--border-input)',
                    background: alignMode === mode ? 'var(--bg-card-expanded)' : 'var(--bg-input)',
                    color: alignMode === mode ? 'var(--accent)' : 'var(--text-secondary)',
                    fontWeight: alignMode === mode ? '600' : '400',
                  }}
                >
                  {mode === 'start' ? 'seq start' : mode === 'first_activation' ? 'first act.' : 'max act.'}
                </button>
              ))}
            </div>
          </div>

          {visibleExamples.length > 0 ? (
            visibleExamples.map((ex, i) => (
              <div key={i} style={styles.example}>
                <div style={styles.exampleMeta}>
                  <span><strong style={{ color: 'var(--text-heading)' }}>{getRegionLabel(ex)}</strong></span>
                  <span style={{ fontFamily: 'monospace' }}>max: {ex.max_activation?.toFixed(3)}</span>
                </div>
                <SequenceView
                  sequence={ex.sequence}
                  activations={ex.activations}
                  maxActivation={ex.max_activation}
                  alignMode={alignMode}
                  alignAnchor={alignAnchor}
                  totalLength={totalLength}
                  scrollGroupRef={scrollGroupRef}
                />
              </div>
            ))
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px', fontStyle: 'italic' }}>No examples loaded</div>
          )}
        </div>

        {/* v2 roadmap placeholders — populated when annotation + conservation pipelines land. */}
        <div style={styles.section}>
          <div style={styles.placeholderLabel}>Annotations</div>
          <div style={styles.placeholder} title="Annotation overlay (RefSeq, Rfam, JASPAR) — coming in v2">
            Annotation overlay (RefSeq, Rfam, JASPAR) — coming in v2
          </div>
        </div>

        <div style={styles.section}>
          <div style={styles.placeholderLabel}>Conservation</div>
          <div style={styles.placeholder} title="Conservation track (phyloP) — coming in v2">
            Conservation track (phyloP) — coming in v2
          </div>
        </div>

      </div>
    </div>
  )
}
