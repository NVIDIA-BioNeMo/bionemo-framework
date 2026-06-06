import React, { useEffect } from 'react'
import TextSequence from './TextSequence'
import TokenLogits from './TokenLogits'

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
  sectionSubtitle: {
    fontSize: '9px',
    color: 'var(--text-muted)',
    marginBottom: '8px',
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
}

export default function FeatureDetailPage({ feature, examples, vocabLogits, onClose }) {
  const fid = String(feature.feature_id)
  const logits = vocabLogits ? vocabLogits[fid] : null

  const freq = feature.activation_freq || 0
  const maxAct = feature.max_activation || 0
  const description = feature.label || feature.description || `Feature ${feature.feature_id}`

  // Close on Escape
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  const visibleExamples = (examples || []).slice(0, 8)

  return (
    <div style={styles.overlay} onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div style={styles.page}>

        {/* Header + stats in one row */}
        <div style={styles.header}>
          <div>
            <div style={styles.title}>
              Feature #{feature.feature_id}
              <span style={{ fontWeight: 400, fontSize: '11px', color: 'var(--text-secondary)', marginLeft: '8px' }}>{description}</span>
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

        {/* Decoder Logits */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Decoder Logits — Promoted / Suppressed Tokens</div>
          <div style={styles.sectionSubtitle}>
            Projection of this feature's decoder weight vector through the language model's prediction head, with the
            mean logit vector subtracted across all features. Values reflect what this feature <em>specifically</em>{' '}
            promotes (green) or suppresses (red) relative to the average feature — i.e. what it pushes the model to
            output, which is conceptually distinct from what activates it.
          </div>
          {logits ? <TokenLogits logits={logits} limit={20} /> : (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>No decoder logits available</div>
          )}
        </div>

        {/* Top Activating Sequences */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Top Activating Sequences</div>
          <div style={styles.sectionSubtitle}>
            Text spans where this feature fires most strongly. Each token is highlighted by its activation value —
            brighter means the feature responds more strongly at that position.
          </div>
          {visibleExamples.length > 0 ? (
            visibleExamples.map((ex, i) => (
              <div key={i} style={styles.example}>
                <div style={styles.exampleMeta}>
                  <strong style={{ color: 'var(--text-heading)' }}>
                    {ex.text_id != null ? `#${ex.text_id}` : `Example ${i + 1}`}
                  </strong>
                  <span style={{ fontFamily: 'monospace' }}>max: {ex.max_activation?.toFixed(3)}</span>
                </div>
                <TextSequence
                  tokens={ex.tokens}
                  activations={ex.activations}
                  maxActivation={ex.max_activation}
                />
              </div>
            ))
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px', fontStyle: 'italic' }}>No examples loaded</div>
          )}
        </div>

      </div>
    </div>
  )
}
