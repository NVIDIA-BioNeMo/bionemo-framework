import React, { useState, useEffect, useRef, forwardRef } from 'react'
import TokenHighlight from './TokenHighlight'

const styles = {
  card: {
    background: '#fff',
    borderRadius: '8px',
    border: '1px solid #e0e0e0',
    flexShrink: 0,
  },
  cardHighlighted: {
    background: '#fff',
    borderRadius: '8px',
    border: '2px solid #222',
    flexShrink: 0,
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
  },
  header: {
    padding: '12px 14px',
    borderBottom: '1px solid #eee',
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
    color: '#888',
    fontFamily: 'monospace',
    marginBottom: '2px',
  },
  description: {
    fontSize: '13px',
    fontWeight: '500',
    wordBreak: 'break-word',
    lineHeight: '1.4',
  },
  stats: {
    display: 'flex',
    gap: '12px',
    fontSize: '11px',
    color: '#666',
    flexShrink: 0,
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  statLabel: {
    color: '#999',
    fontSize: '9px',
    textTransform: 'uppercase',
  },
  statValue: {
    fontFamily: 'monospace',
    fontWeight: '500',
  },
  expandIcon: {
    color: '#999',
    fontSize: '10px',
    marginLeft: '6px',
  },
  examples: {
    padding: '10px 14px',
    background: '#fafafa',
    maxHeight: '400px',
    overflowY: 'auto',
  },
  exampleHeader: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    marginBottom: '8px',
    fontWeight: '500',
  },
  example: {
    marginBottom: '8px',
    padding: '8px 10px',
    background: '#fff',
    borderRadius: '4px',
    border: '1px solid #eee',
  },
  exampleMeta: {
    fontSize: '10px',
    color: '#999',
    marginBottom: '4px',
    fontFamily: 'monospace',
  },
  noExamples: {
    color: '#999',
    fontSize: '12px',
    fontStyle: 'italic',
  },
  logitsSection: {
    padding: '10px 14px',
    borderBottom: '1px solid #eee',
  },
  logitsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '6px',
  },
  logitsLabel: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    fontWeight: '500',
    width: '70px',
    flexShrink: 0,
  },
  logitsPills: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '4px',
  },
  pillPositive: {
    padding: '2px 6px',
    fontSize: '11px',
    fontFamily: 'monospace',
    background: '#e8f5e9',
    color: '#2e7d32',
    borderRadius: '3px',
    border: '1px solid #c8e6c9',
  },
  pillNegative: {
    padding: '2px 6px',
    fontSize: '11px',
    fontFamily: 'monospace',
    background: '#ffebee',
    color: '#c62828',
    borderRadius: '3px',
    border: '1px solid #ffcdd2',
  },
  densityBar: {
    width: '50px',
    height: '3px',
    background: '#eee',
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

const FeatureCard = forwardRef(function FeatureCard({ feature, isHighlighted, forceExpanded, onClick, loadExamples }, ref) {
  const [expanded, setExpanded] = useState(false)
  const [examples, setExamples] = useState([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const examplesCacheRef = useRef(null)

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
  const description = feature.description || `Feature ${feature.feature_id}`

  const handleClick = () => {
    const willExpand = !expanded
    setExpanded(willExpand)
    if (onClick) {
      onClick(feature.feature_id, willExpand)
    }
  }

  return (
    <div ref={ref} style={isHighlighted ? styles.cardHighlighted : styles.card}>
      <div style={styles.header} onClick={handleClick}>
        <div style={styles.headerLeft}>
          <div style={styles.featureId}>Feature #{feature.feature_id}</div>
          <div style={styles.description}>{description}</div>
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

      {expanded && (
        <>
          {/* Top logits section */}
          {(feature.top_positive_logits?.length > 0 || feature.top_negative_logits?.length > 0) && (
            <div style={styles.logitsSection}>
              {feature.top_positive_logits?.length > 0 && (
                <div style={styles.logitsRow}>
                  <span style={styles.logitsLabel}>Promotes</span>
                  <div style={styles.logitsPills}>
                    {feature.top_positive_logits.slice(0, 8).map(([token, value], i) => (
                      <span key={i} style={styles.pillPositive} title={`logit: ${value}`}>
                        {token}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {feature.top_negative_logits?.length > 0 && (
                <div style={{...styles.logitsRow, marginBottom: 0}}>
                  <span style={styles.logitsLabel}>Suppresses</span>
                  <div style={styles.logitsPills}>
                    {feature.top_negative_logits.slice(0, 8).map(([token, value], i) => (
                      <span key={i} style={styles.pillNegative} title={`logit: ${value}`}>
                        {token}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <div style={styles.examples}>
            <div style={styles.exampleHeader}>Top Activating Examples</div>
            {loadingExamples ? (
              <div style={{ textAlign: 'center', padding: '20px', color: '#888', fontSize: '13px' }}>
                Loading examples...
              </div>
            ) : examples.length > 0 ? (
              examples.slice(0, 5).map((ex, i) => (
                <div key={i} style={styles.example}>
                  <div style={styles.exampleMeta}>
                    activation: {ex.max_activation?.toFixed(3) || 'N/A'}
                  </div>
                  <TokenHighlight tokens={ex.tokens} />
                </div>
              ))
            ) : (
              <div style={styles.noExamples}>No examples available</div>
            )}
          </div>
        </>
      )}
    </div>
  )
})

export default FeatureCard
