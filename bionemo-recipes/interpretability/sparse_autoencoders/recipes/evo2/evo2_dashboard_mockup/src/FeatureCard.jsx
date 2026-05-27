import React, { useState, useEffect, useRef, forwardRef } from 'react'
import SequenceView, { computeAlignInfo } from './SequenceView'
import FeatureDetailPage from './FeatureDetailPage'
import { getRegionLabel } from './utils'

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
  proteinId: {
    color: 'var(--text-heading)',
    fontWeight: '700',
  },
  annotation: {
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
    marginLeft: '8px',
  },
  uniprotLink: {
    color: 'var(--link)',
    textDecoration: 'none',
    fontSize: '11px',
    marginLeft: '4px',
    opacity: 0.6,
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
  alignBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    marginBottom: '10px',
    fontSize: '10px',
    color: '#888',
  },
  alignLabel: {
    textTransform: 'uppercase',
    fontWeight: '500',
  },
  alignBtn: {
    padding: '2px 8px',
    border: '1px solid #ddd',
    borderRadius: '3px',
    background: '#fff',
    cursor: 'pointer',
    fontSize: '10px',
    color: '#555',
  },
  alignBtnActive: {
    padding: '2px 8px',
    border: '1px solid #76b900',
    borderRadius: '3px',
    background: '#f0f9e0',
    cursor: 'pointer',
    fontSize: '10px',
    color: '#333',
    fontWeight: '600',
  },
}

const FeatureCard = forwardRef(function FeatureCard({ feature, isHighlighted, forceExpanded, onClick, loadExamples }, ref) {
  const [expanded, setExpanded] = useState(false)
  const [showDetailPage, setShowDetailPage] = useState(false)
  const [examples, setExamples] = useState([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const examplesCacheRef = useRef(null)
  const [alignMode, setAlignMode] = useState('start')
  const scrollGroupRef = useRef([])
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

  // Reset scroll group when alignment changes
  useEffect(() => { scrollGroupRef.current = [] }, [alignMode])

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
  const description = rawDesc.toLowerCase().includes('common codons') ? 'Unidentified Feature' : rawDesc


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

    // Feature metadata section
    lines.push('=== FEATURE METADATA ===')
    lines.push(`Feature ID,${feature.feature_id}`)
    lines.push(`Label,${displayTitle}`)
    if (userTitle) {
      lines.push(`User Title,${userTitle}`)
    }
    lines.push(`Activation Frequency,${(freq * 100).toFixed(2)}%`)
    lines.push(`Max Activation,${maxAct.toFixed(4)}`)
    lines.push('')

    // Examples section
    if (examples && examples.length > 0) {
      lines.push('=== ACTIVATION EXAMPLES ===')
      lines.push('Rank,Region,Max Activation,Sequence')
      examples.forEach((ex, i) => {
        lines.push(`${i + 1},${getRegionLabel(ex) || ''},${ex.max_activation?.toFixed(4) || ''},${ex.sequence || ''}`)
      })
    }

    // Generate CSV
    const csv = lines.join('\n')

    // Create download link
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
                  padding: '2px 6px',
                  fontSize: '10px',
                  background: '#76b900',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                }}
              >
                ✓
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); handleCancelEdit() }}
                style={{
                  padding: '2px 6px',
                  fontSize: '10px',
                  background: '#ddd',
                  color: '#333',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
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
                  fontSize: '11px',
                  color: '#999',
                  cursor: 'pointer',
                  padding: '2px 4px',
                  borderRadius: '3px',
                  userSelect: 'none',
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
          {/* v2 roadmap placeholders — populated when real eval pipeline lands. */}
          <div style={styles.stat} title="Top annotation database match (RefSeq / Rfam / JASPAR). Coming in v2.">
            <span style={{ ...styles.statLabel, color: 'var(--text-muted)' }}>Annotation</span>
            <span style={{ ...styles.statValue, color: 'var(--text-muted)' }}>—</span>
          </div>
          <div style={styles.stat} title="Recall against annotation database. Coming in v2.">
            <span style={{ ...styles.statLabel, color: 'var(--text-muted)' }}>Sensitivity</span>
            <span style={{ ...styles.statValue, color: 'var(--text-muted)' }}>—</span>
          </div>
          <div style={styles.stat} title="Reconstruction loss change from ablating this feature. Coming in v2.">
            <span style={{ ...styles.statLabel, color: 'var(--text-muted)' }}>Recon Δ</span>
            <span style={{ ...styles.statValue, color: 'var(--text-muted)' }}>—</span>
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
              padding: '4px 12px', fontSize: '11px', color: 'var(--accent)', cursor: 'pointer',
              fontWeight: '500',
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
          {feature.logo_path && (
            <div style={{ marginBottom: '14px' }}>
              <div style={styles.sectionHeader}>Sequence Logo</div>
              <img
                src={feature.logo_path}
                alt={`Sequence logo for ${displayTitle}`}
                style={{ maxWidth: '100%', height: 'auto', display: 'block' }}
              />
            </div>
          )}
          {/* Sequence examples */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={styles.sectionHeader}>Top Activating Sequences</div>
            <div style={styles.alignBar}>
              <span style={styles.alignLabel}>Align by:</span>
              {['start', 'first_activation', 'max_activation'].map(mode => (
                <button
                  key={mode}
                  style={alignMode === mode ? styles.alignBtnActive : styles.alignBtn}
                  onClick={(e) => { e.stopPropagation(); setAlignMode(mode) }}
                >
                  {mode === 'start' ? 'sequence start' : mode === 'first_activation' ? 'first activation' : 'max activation'}
                </button>
              ))}
            </div>
          </div>
          {loadingExamples ? (
            <div style={{ textAlign: 'center', padding: '20px', color: '#888', fontSize: '13px' }}>
              Loading examples...
            </div>
          ) : examples.length > 0 ? (
            <>
              {(() => {
                const visibleExamples = examples.slice(0, 6)
                const { anchor: alignAnchor, totalLength } = computeAlignInfo(visibleExamples, alignMode)
                return visibleExamples.map((ex, i) => (
                  <div key={i} style={styles.example}>
                    <div style={styles.exampleMeta}>
                      <span>
                        <span style={styles.proteinId}>{getRegionLabel(ex)}</span>
                        {ex.best_annotation && (
                          <span style={styles.annotation}>{ex.best_annotation}</span>
                        )}
                      </span>
                      <span>max: {ex.max_activation?.toFixed(3) || 'N/A'}</span>
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
              })()}

            </>
          ) : (
            <div style={styles.noExamples}>No examples available</div>
          )}
        </div>
      )}

      {showDetailPage && (
        <FeatureDetailPage
          feature={feature}
          examples={examples}
          onClose={() => setShowDetailPage(false)}
        />
      )}
    </div>
  )
})

export default FeatureCard
