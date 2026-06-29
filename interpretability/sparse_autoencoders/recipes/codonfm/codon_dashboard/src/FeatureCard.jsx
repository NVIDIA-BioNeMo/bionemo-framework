import React, { useState, useEffect, useRef, forwardRef } from 'react'
import ProteinSequence, { computeAlignInfo } from './ProteinSequence'
import FeatureDetailPage from './FeatureDetailPage'
import { getAccession, uniprotUrl } from './utils'

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

const FeatureCard = forwardRef(function FeatureCard({ feature, isHighlighted, forceExpanded, onClick, loadExamples, vocabLogits, featureAnalysis }, ref) {
  const [expanded, setExpanded] = useState(false)
  const [showDetailPage, setShowDetailPage] = useState(false)
  const [examples, setExamples] = useState([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const examplesCacheRef = useRef(null)
  const [alignMode, setAlignMode] = useState('start')
  const scrollGroupRef = useRef([])
  const [hoveredCodon, setHoveredCodon] = useState(null)
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
  const highScoreFrac = feature.high_score_fraction
  const variantDelta = feature.mean_variant_delta
  const siteDelta = feature.mean_site_delta
  const localDelta = feature.mean_local_delta
  const clinvarFrac = feature.clinvar_fraction
  const phylop = feature.mean_phylop
  const gcMean = feature.gc_mean
  const trinucEntropy = feature.trinuc_entropy
  const geneEntropy = feature.gene_entropy
  const geneNUnique = feature.gene_n_unique
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

    // Vocab logits section
    const logits = vocabLogits?.[String(feature.feature_id)]
    if (logits) {
      lines.push('=== TOP PROMOTED CODONS ===')
      lines.push('Codon,Amino Acid,Logit Value')
      const CODON_AA = {
        'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
        'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
        'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
        'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
        'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
        'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
        'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
        'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
      }
      ;(logits.top_positive || []).forEach(([codon, val]) => {
        lines.push(`${codon},${CODON_AA[codon] || '?'},${val.toFixed(4)}`)
      })
      lines.push('')

      lines.push('=== TOP SUPPRESSED CODONS ===')
      lines.push('Codon,Amino Acid,Logit Value')
      ;(logits.top_negative || []).forEach(([codon, val]) => {
        lines.push(`${codon},${CODON_AA[codon] || '?'},${val.toFixed(4)}`)
      })
      lines.push('')
    }

    // Codon annotations section
    const analysis = featureAnalysis?.[String(feature.feature_id)]
    if (analysis?.codon_annotations) {
      lines.push('=== CODON ANNOTATIONS ===')
      const ann = analysis.codon_annotations
      if (ann.amino_acid) {
        lines.push(`Amino Acid,${ann.amino_acid.aa}`)
        lines.push(`AA Frequency,${(ann.amino_acid.fraction * 100).toFixed(1)}%`)
      }
      if (ann.codon_usage) {
        lines.push(`Codon Usage,${ann.codon_usage.bias}`)
      }
      if (ann.wobble) {
        lines.push(`Wobble Position,${ann.wobble.preference}`)
      }
      if (ann.cpg) {
        lines.push(`CpG Enriched,Yes`)
      }
      if (ann.position) {
        lines.push(`Position,${ann.position.label}`)
      }
      lines.push('')
    }

    // GSEA enrichment section
    const gseaCsvFields = [
      { key: 'gsea_overall_best', label: 'GSEA Overall Best' },
      { key: 'gsea_GO_Biological_Process', label: 'GSEA GO Biological Process' },
      { key: 'gsea_GO_Molecular_Function', label: 'GSEA GO Molecular Function' },
      { key: 'gsea_GO_Cellular_Component', label: 'GSEA GO Cellular Component' },
      { key: 'gsea_InterPro_Domains', label: 'GSEA InterPro Domains' },
      { key: 'gsea_GO_Slim', label: 'GSEA GO Slim' },
    ]
    const gseaLines = gseaCsvFields
      .filter(({ key }) => feature[key] && feature[key] !== 'unlabeled')
      .map(({ key, label }) => `${label},${feature[key]}`)
    if (gseaLines.length > 0) {
      lines.push('=== GSEA ENRICHMENT ===')
      gseaLines.forEach(l => lines.push(l))
      lines.push('')
    }

    // Examples section
    if (examples && examples.length > 0) {
      lines.push('=== ACTIVATION EXAMPLES ===')
      lines.push('Rank,Protein ID,Max Activation,Sequence')
      examples.forEach((ex, i) => {
        lines.push(`${i + 1},${ex.protein_id || ''},${ex.max_activation?.toFixed(4) || ''},${ex.sequence || ''}`)
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
          {highScoreFrac != null && !isNaN(highScoreFrac) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>Hi-Score</span>
              <span style={{ ...styles.statValue, color: highScoreFrac > 0.6 ? '#d32f2f' : highScoreFrac < 0.4 ? '#388e3c' : '#666' }}>
                {(highScoreFrac * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {variantDelta != null && !isNaN(variantDelta) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>Δ Var</span>
              <span style={{ ...styles.statValue, color: Math.abs(variantDelta) > 0.5 ? '#1565c0' : '#666' }}>
                {variantDelta > 0 ? '+' : ''}{variantDelta.toFixed(2)}
              </span>
            </div>
          )}
          {siteDelta != null && !isNaN(siteDelta) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>Δ Site</span>
              <span style={{ ...styles.statValue, color: Math.abs(siteDelta) > 0.5 ? '#7b1fa2' : '#666' }}>
                {siteDelta > 0 ? '+' : ''}{siteDelta.toFixed(2)}
              </span>
            </div>
          )}
          {localDelta != null && !isNaN(localDelta) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>Δ Local</span>
              <span style={{ ...styles.statValue, color: Math.abs(localDelta) > 0.5 ? '#00695c' : '#666' }}>
                {localDelta > 0 ? '+' : ''}{localDelta.toFixed(2)}
              </span>
            </div>
          )}
          {clinvarFrac != null && !isNaN(clinvarFrac) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>ClinVar</span>
              <span style={styles.statValue}>{(clinvarFrac * 100).toFixed(0)}%</span>
            </div>
          )}
          {phylop != null && !isNaN(phylop) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>PhyloP</span>
              <span style={styles.statValue}>{phylop.toFixed(1)}</span>
            </div>
          )}
          {gcMean != null && !isNaN(gcMean) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>GC</span>
              <span style={{ ...styles.statValue, color: Math.abs(gcMean - 0.5) > 0.1 ? '#e65100' : '#666' }}>
                {(gcMean * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {trinucEntropy != null && !isNaN(trinucEntropy) && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>Trinuc H</span>
              <span style={{ ...styles.statValue, color: trinucEntropy < 3 ? '#ad1457' : '#666' }}>
                {trinucEntropy.toFixed(1)}
              </span>
            </div>
          )}
          {geneNUnique != null && geneNUnique > 0 && (
            <div style={styles.stat}>
              <span style={styles.statLabel}>Genes</span>
              <span style={{ ...styles.statValue, color: geneNUnique < 5 ? '#4527a0' : '#666' }}>
                {geneNUnique}
              </span>
            </div>
          )}
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
          {/* Vocabulary logits - all codons grouped by amino acid */}
          {vocabLogits && vocabLogits[String(feature.feature_id)] && (() => {
            const logits = vocabLogits[String(feature.feature_id)]
            const CODON_AA = {
              'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
              'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
              'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
              'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
              'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
              'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
              'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
              'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
            }
            // Build codon logit map from all entries, excluding stop codons
            const codonLogitMap = {}
            for (const [codon, val] of (logits.top_positive || [])) {
              if (CODON_AA[codon] !== '*') codonLogitMap[codon] = val
            }
            for (const [codon, val] of (logits.top_negative || [])) {
              if (CODON_AA[codon] !== '*') codonLogitMap[codon] = val
            }
            const maxAbs = Math.max(...Object.values(codonLogitMap).map(Math.abs), 0.001)
            // Group by AA, excluding stop codons
            const aaGroups = {}
            for (const [codon, aa] of Object.entries(CODON_AA)) {
              if (aa === '*') continue
              if (!aaGroups[aa]) aaGroups[aa] = []
              aaGroups[aa].push(codon)
            }
            const aaOrder = Object.keys(aaGroups).sort()
            return (
              <div style={{ marginBottom: '8px' }}>
                <div style={styles.sectionHeader}>Decoder Logits</div>
                <div style={{ position: 'relative' }}>
                  {hoveredCodon && (() => {
                    const val = codonLogitMap[hoveredCodon] || 0
                    const aa = CODON_AA[hoveredCodon]
                    return (
                      <div style={{
                        position: 'absolute', top: '-18px', left: '50%', transform: 'translateX(-50%)',
                        fontSize: '9px', fontFamily: 'monospace', fontWeight: '600', color: '#333',
                        background: '#fff', border: '1px solid #ddd', borderRadius: '3px',
                        padding: '1px 5px', whiteSpace: 'nowrap', zIndex: 1,
                        pointerEvents: 'none',
                      }}>
                        {hoveredCodon} ({aa}): {val.toFixed(3)}
                      </div>
                    )
                  })()}
                  <div style={{ display: 'flex', width: '100%', gap: '1px' }}>
                    {aaOrder.map(aa => {
                      const codons = aaGroups[aa] || []
                      return (
                        <div key={aa} style={{ flex: codons.length, minWidth: 0 }}>
                          <div style={{ fontSize: '7px', fontWeight: '700', color: '#555', textAlign: 'center' }}>{aa}</div>
                          <div style={{ display: 'flex', alignItems: 'flex-end', height: '28px' }}>
                            {codons.sort().map(codon => {
                              const val = codonLogitMap[codon] || 0
                              const h = Math.max(1, (Math.abs(val) / maxAbs) * 24)
                              const isHovered = hoveredCodon === codon
                              const barColor = val === 0 ? '#ccc' : val > 0 ? '#76b900' : '#e57373'
                              return (
                                <div key={codon} style={{
                                  flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                                  justifyContent: 'flex-end', height: '28px',
                                }}
                                  onMouseEnter={() => setHoveredCodon(codon)}
                                  onMouseLeave={() => setHoveredCodon(null)}
                                >
                                  <div style={{
                                    width: '100%', maxWidth: '12px', height: `${h}px`, borderRadius: '1px',
                                    background: barColor,
                                    opacity: val === 0 ? 0.5 : Math.abs(val) / maxAbs * 0.7 + 0.3,
                                    outline: isHovered ? '1.5px solid #333' : 'none',
                                    outlineOffset: '-0.5px',
                                  }} />
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            )
          })()}

          {/* Analysis summary tags */}
          {featureAnalysis && featureAnalysis[String(feature.feature_id)] && (() => {
            const analysis = featureAnalysis[String(feature.feature_id)]
            const tags = []
            const ann = analysis.codon_annotations || {}

            if (ann.amino_acid) tags.push({ label: `AA: ${ann.amino_acid.aa} (${(ann.amino_acid.fraction * 100).toFixed(0)}%)`, color: '#e3f2fd' })
            if (ann.codon_usage) tags.push({ label: `${ann.codon_usage.bias} codons`, color: '#fff3e0' })
            if (ann.wobble) tags.push({ label: `wobble ${ann.wobble.preference}`, color: '#f3e5f5' })
            if (ann.cpg) tags.push({ label: `CpG enriched`, color: '#fce4ec' })
            if (ann.position) tags.push({ label: `N-terminal`, color: '#e8f5e9' })

            // GSEA enrichment tags
            const gseaFields = [
              { key: 'gsea_GO_Biological_Process', prefix: 'GO:BP', color: '#e8eaf6' },
              { key: 'gsea_GO_Molecular_Function', prefix: 'GO:MF', color: '#ede7f6' },
              { key: 'gsea_GO_Cellular_Component', prefix: 'GO:CC', color: '#e0f2f1' },
              { key: 'gsea_InterPro_Domains', prefix: 'InterPro', color: '#fff8e1' },
              { key: 'gsea_GO_Slim', prefix: 'GO Slim', color: '#f1f8e9' },
            ]
            for (const { key, prefix, color } of gseaFields) {
              const val = feature[key]
              if (val && val !== 'unlabeled' && val !== 'other') {
                tags.push({ label: `${prefix}: ${val}`, color })
              }
            }

            // Codon optimality metrics from annotations
            if (ann.cai != null) tags.push({ label: `CAI: ${ann.cai.toFixed(3)}`, color: '#e0f7fa' })
            if (ann.tai != null) tags.push({ label: `tAI: ${ann.tai.toFixed(3)}`, color: '#e0f7fa' })
            if (ann.rscu != null) tags.push({ label: `RSCU: ${ann.rscu.toFixed(2)}`, color: '#e0f7fa' })

            if (tags.length === 0) return null
            return (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '10px' }}>
                {tags.map((t, i) => (
                  <span key={i} style={{
                    padding: '2px 6px', borderRadius: '3px', fontSize: '9px',
                    fontWeight: '500', background: t.color, color: '#333',
                  }}>{t.label}</span>
                ))}
              </div>
            )
          })()}

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
                        <span style={styles.proteinId}>{ex.protein_id}</span>
                        <a
                          href={uniprotUrl(getAccession(ex.protein_id))}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={styles.uniprotLink}
                          onClick={e => e.stopPropagation()}
                          title="View on UniProt"
                        >
                          ↗
                        </a>
                        {ex.best_annotation && (
                          <span style={styles.annotation}>{ex.best_annotation}</span>
                        )}
                      </span>
                      <span>max: {ex.max_activation?.toFixed(3) || 'N/A'}</span>
                    </div>
                    <ProteinSequence
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
          vocabLogits={vocabLogits}
          featureAnalysis={featureAnalysis}
          onClose={() => setShowDetailPage(false)}
        />
      )}
    </div>
  )
})

export default FeatureCard
