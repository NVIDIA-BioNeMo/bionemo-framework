import React, { useState, useEffect, useRef, forwardRef } from 'react'
import TextSequence from './TextSequence'
import TokenLogits from './TokenLogits'
import FeatureDetailPage from './FeatureDetailPage'
import { joinTokens } from './utils'

const styles = {
  card: {
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    flexShrink: 0,
  },
  cardHighlighted: {
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '2px solid var(--highlight-border)',
    flexShrink: 0,
    boxShadow: '0 2px 8px var(--highlight-shadow)',
  },
  header: {
    padding: '12px 14px',
    borderBottom: '1px solid var(--border-light)',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '10px',
  },
  headerLeft: {
    flex: 1,
    minWidth: 0,
  },
  featureId: {
    fontSize: '11px',
    color: 'var(--text-tertiary)',
    fontFamily: 'monospace',
    marginBottom: '2px',
  },
  description: {
    fontSize: '13px',
    fontWeight: '500',
    wordBreak: 'break-word',
    lineHeight: '1.4',
    color: 'var(--text)',
  },
  userTitle: {
    fontSize: '13px',
    fontWeight: '500',
    wordBreak: 'break-word',
    lineHeight: '1.4',
    color: 'var(--accent)',
    fontStyle: 'italic',
  },
  stats: {
    display: 'flex',
    gap: '12px',
    fontSize: '11px',
    color: 'var(--text-secondary)',
    flexShrink: 0,
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  statLabel: {
    color: 'var(--text-muted)',
    fontSize: '9px',
    textTransform: 'uppercase',
  },
  statValue: {
    fontFamily: 'monospace',
    fontWeight: '500',
  },
  expandIcon: {
    color: 'var(--text-muted)',
    fontSize: '10px',
    marginLeft: '6px',
  },
  expandedContent: {
    padding: '10px 14px',
    background: 'var(--bg-card-expanded)',
    maxHeight: '900px',
    overflowY: 'auto',
  },
  sectionHeader: {
    fontSize: '10px',
    color: 'var(--text-tertiary)',
    textTransform: 'uppercase',
    marginBottom: '8px',
    fontWeight: '500',
  },
  example: {
    marginBottom: '8px',
    padding: '8px 10px',
    background: 'var(--bg-example)',
    borderRadius: '4px',
    border: '1px solid var(--border-light)',
  },
  exampleMeta: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    marginBottom: '4px',
    fontFamily: 'monospace',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  textId: {
    color: 'var(--text-heading)',
    fontWeight: '700',
  },
  annotation: {
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
    marginLeft: '8px',
  },
  noExamples: {
    color: 'var(--text-muted)',
    fontSize: '12px',
    fontStyle: 'italic',
  },
  densityBar: {
    width: '50px',
    height: '3px',
    background: 'var(--density-bar-bg)',
    borderRadius: '2px',
    overflow: 'hidden',
    marginTop: '3px',
  },
  densityFill: {
    height: '100%',
    background: '#76b900',
    borderRadius: '2px',
  },
}

const FeatureCard = forwardRef(function FeatureCard({ feature, isHighlighted, forceExpanded, onClick, loadExamples, vocabLogits }, ref) {
  const [expanded, setExpanded] = useState(false)
  const [showDetailPage, setShowDetailPage] = useState(false)
  const [examples, setExamples] = useState([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const examplesCacheRef = useRef(null)
  const [editingTitle, setEditingTitle] = useState(false)
  const [userTitle, setUserTitle] = useState('')
  const inputRef = useRef(null)

  // Load user-provided title from localStorage
  useEffect(() => {
    const stored = localStorage.getItem(`featureTitle_${feature.feature_id}`)
    if (stored) {
      setUserTitle(stored)
    }
  }, [feature.feature_id])

  // Focus input when editing starts
  useEffect(() => {
    if (editingTitle && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [editingTitle])

  // If forceExpanded changes to true, expand the card
  useEffect(() => {
    if (forceExpanded) {
      setExpanded(true)
    }
  }, [forceExpanded])

  // Lazy-load examples from DuckDB when card is expanded
  useEffect(() => {
    if (!expanded || !loadExamples || examplesCacheRef.current) return
    let cancelled = false
    setLoadingExamples(true)
    loadExamples(feature.feature_id).then(result => {
      if (cancelled) return
      examplesCacheRef.current = result
      setExamples(result)
      setLoadingExamples(false)
    }).catch(err => {
      if (cancelled) return
      console.error('Error loading examples for feature', feature.feature_id, err)
      setLoadingExamples(false)
    })
    return () => { cancelled = true }
  }, [expanded, loadExamples, feature.feature_id])

  const freq = feature.activation_freq || 0
  const maxAct = feature.max_activation || 0
  const rawDesc = feature.label || feature.description || `Feature ${feature.feature_id}`
  const description = rawDesc

  const handleClick = () => {
    const willExpand = !expanded
    // Update UMAP highlight immediately, defer card expansion so it doesn't block
    if (onClick) {
      onClick(feature.feature_id, willExpand)
    }
    requestAnimationFrame(() => {
      setExpanded(willExpand)
    })
  }

  const handleSaveTitle = () => {
    if (userTitle.trim()) {
      localStorage.setItem(`featureTitle_${feature.feature_id}`, userTitle.trim())
    } else {
      localStorage.removeItem(`featureTitle_${feature.feature_id}`)
      setUserTitle('')
    }
    setEditingTitle(false)
  }

  const handleCancelEdit = () => {
    const stored = localStorage.getItem(`featureTitle_${feature.feature_id}`)
    setUserTitle(stored || '')
    setEditingTitle(false)
  }

  const displayTitle = userTitle || description

  const handleTitleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSaveTitle()
    } else if (e.key === 'Escape') {
      handleCancelEdit()
    }
  }

  const exportToCSV = () => {
    const lines = []

    lines.push('=== FEATURE METADATA ===')
    lines.push(`Feature ID,${feature.feature_id}`)
    lines.push(`Label,${displayTitle}`)
    if (userTitle) {
      lines.push(`User Title,${userTitle}`)
    }
    lines.push(`Activation Frequency,${(freq * 100).toFixed(2)}%`)
    lines.push(`Max Activation,${maxAct.toFixed(4)}`)
    lines.push('')

    const logits = vocabLogits?.[String(feature.feature_id)]
    if (logits) {
      lines.push('=== TOP PROMOTED TOKENS ===')
      lines.push('Token,Logit Value')
      ;(logits.top_positive || []).forEach(([token, val]) => {
        lines.push(`${JSON.stringify(token)},${val.toFixed(4)}`)
      })
      lines.push('')
      lines.push('=== TOP SUPPRESSED TOKENS ===')
      lines.push('Token,Logit Value')
      ;(logits.top_negative || []).forEach(([token, val]) => {
        lines.push(`${JSON.stringify(token)},${val.toFixed(4)}`)
      })
      lines.push('')
    }

    if (examples && examples.length > 0) {
      lines.push('=== ACTIVATION EXAMPLES ===')
      lines.push('Rank,Text ID,Max Activation,Text')
      examples.forEach((ex, i) => {
        const text = joinTokens(ex.tokens).replace(/[\r\n]+/g, ' ')
        lines.push(`${i + 1},${ex.text_id || ''},${ex.max_activation?.toFixed(4) || ''},${JSON.stringify(text)}`)
      })
    }

    const csv = lines.join('\n')
    const filename = `feature_${feature.feature_id}_${displayTitle.replace(/[^a-z0-9]/gi, '_').substring(0, 20)}.csv`
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.setAttribute('href', URL.createObjectURL(blob))
    link.setAttribute('download', filename)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div ref={ref} style={isHighlighted ? styles.cardHighlighted : styles.card}>
      <div style={styles.header} onClick={handleClick}>
        <div style={styles.headerLeft}>
          <div style={styles.featureId}>Feature #{feature.feature_id}</div>
          {editingTitle ? (
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <input
                ref={inputRef}
                type="text"
                value={userTitle}
                onChange={(e) => setUserTitle(e.target.value)}
                onKeyDown={handleTitleKeyDown}
                onClick={(e) => e.stopPropagation()}
                style={{
                  fontSize: '13px',
                  fontWeight: '500',
                  padding: '4px 8px',
                  border: '1px solid #76b900',
                  borderRadius: '4px',
                  flex: 1,
                }}
              />
              <button
                onClick={(e) => { e.stopPropagation(); handleSaveTitle() }}
                style={{
                  padding: '2px 6px', fontSize: '10px', background: '#76b900',
                  color: '#fff', border: 'none', borderRadius: '3px', cursor: 'pointer',
                }}
              >
                ✓
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); handleCancelEdit() }}
                style={{
                  padding: '2px 6px', fontSize: '10px', background: '#ddd',
                  color: '#333', border: 'none', borderRadius: '3px', cursor: 'pointer',
                }}
              >
                ✕
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <div style={userTitle ? styles.userTitle : styles.description}>{displayTitle}</div>
              <span
                onClick={(e) => { e.stopPropagation(); setEditingTitle(true) }}
                style={{
                  fontSize: '11px', color: '#999', cursor: 'pointer',
                  padding: '2px 4px', borderRadius: '3px', userSelect: 'none',
                }}
                title="Click to edit title"
              >
                ✎
              </span>
            </div>
          )}
        </div>
        <div style={styles.stats}>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Freq</span>
            <span style={styles.statValue}>{(freq * 100).toFixed(1)}%</span>
            <div style={styles.densityBar}>
              <div style={{ ...styles.densityFill, width: `${Math.min(freq * 100 * 10, 100)}%` }} />
            </div>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Max</span>
            <span style={styles.statValue}>{maxAct.toFixed(1)}</span>
          </div>
          <span style={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
        </div>
      </div>

      {/* Details and export buttons - shown when expanded */}
      {expanded && (
        <div style={{ padding: '0 14px 8px', borderBottom: '1px solid var(--border-light)', display: 'flex', gap: '8px' }}>
          <button
            onClick={(e) => { e.stopPropagation(); setShowDetailPage(true) }}
            style={{
              background: 'var(--bg-card-expanded)', border: '1px solid var(--accent)', borderRadius: '4px',
              padding: '4px 12px', fontSize: '11px', color: 'var(--accent)', cursor: 'pointer', fontWeight: '500',
            }}
          >
            Full analysis
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); exportToCSV() }}
            style={{
              background: 'none', border: '1px solid var(--border-input)', borderRadius: '4px',
              padding: '4px 12px', fontSize: '11px', color: 'var(--text-secondary)', cursor: 'pointer',
            }}
          >
            Export
          </button>
        </div>
      )}

      {expanded && (
        <div style={styles.expandedContent}>
          {/* Decoder logits - top promoted/suppressed tokens */}
          {vocabLogits && vocabLogits[String(feature.feature_id)] && (
            <div style={{ marginBottom: '12px' }}>
              <div style={styles.sectionHeader}>Decoder Logits</div>
              <TokenLogits logits={vocabLogits[String(feature.feature_id)]} compact />
            </div>
          )}

          {/* Text examples */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={styles.sectionHeader}>Top Activating Sequences</div>
          </div>
          {loadingExamples ? (
            <div style={{ textAlign: 'center', padding: '20px', color: '#888', fontSize: '13px' }}>
              Loading examples...
            </div>
          ) : examples.length > 0 ? (
            examples.slice(0, 6).map((ex, i) => (
              <div key={i} style={styles.example}>
                <div style={styles.exampleMeta}>
                  <span>
                    <span style={styles.textId}>{ex.text_id != null ? `#${ex.text_id}` : `Example ${i + 1}`}</span>
                    {ex.best_annotation && (
                      <span style={styles.annotation}>{ex.best_annotation}</span>
                    )}
                  </span>
                  <span>max: {ex.max_activation?.toFixed(3) || 'N/A'}</span>
                </div>
                <TextSequence
                  tokens={ex.tokens}
                  activations={ex.activations}
                  maxActivation={ex.max_activation}
                />
              </div>
            ))
          ) : (
            <div style={styles.noExamples}>No examples available</div>
          )}
        </div>
      )}

      {showDetailPage && (
        <FeatureDetailPage
          feature={feature}
          examples={examples}
          vocabLogits={vocabLogits}
          onClose={() => setShowDetailPage(false)}
        />
      )}
    </div>
  )
})

export default FeatureCard
