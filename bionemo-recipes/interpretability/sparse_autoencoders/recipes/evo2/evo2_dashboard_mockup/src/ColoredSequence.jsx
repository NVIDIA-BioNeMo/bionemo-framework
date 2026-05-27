import React, { useMemo, useState } from 'react'

// Tableau 10 colorblind-friendlier palette. Feature IDs map to colors by
// (feature_id % palette.length). Same feature -> same color, deterministically.
const PALETTE = [
  '#4E79A7', // blue
  '#F28E2B', // orange
  '#E15759', // red
  '#76B7B2', // teal
  '#59A14F', // green
  '#EDC948', // yellow
  '#B07AA1', // purple
  '#FF9DA7', // pink
  '#9C755F', // brown
  '#BAB0AC', // gray
  '#499894', // dark teal
  '#D37295', // dark pink
]

// Show plain text (no background) when the top feature's activation is below
// this fraction of the sequence-wide max. Tweak if "interesting" regions
// look too sparse or too noisy.
const ACTIVATION_FLOOR_FRAC = 0.10

// Opacity floor + ceiling. Keep ceiling well below 1 so the base letter
// stays readable against the colored background.
const OPACITY_FLOOR = 0.15
const OPACITY_CEILING = 0.85

// Line-wrap at 60 bases — standard biology convention (FASTA, GenBank).
const BASES_PER_LINE = 60


function colorForFeature(featureId) {
  if (featureId == null || featureId < 0) return null
  return PALETTE[featureId % PALETTE.length]
}


function labelForFeature(featureId, catalog) {
  if (!catalog || featureId == null || featureId < 0) return `feature_${featureId}`
  const entry = catalog[featureId] || catalog.get?.(featureId)
  if (!entry) return `feature_${featureId}`
  return entry.label || entry.description || `feature_${featureId}`
}


// Build mock /analyze-shaped data for development without the backend.
function buildMockAnalysis(sequence) {
  const length = sequence.length
  // 5 hand-rolled features with locality patterns so colored regions are
  // visible. Real backend output will have ~10 top features per position.
  const FEATURES = [101, 207, 314, 422, 588]
  const topk_features = []
  const topk_activations = []
  for (let i = 0; i < length; i++) {
    const acts = [0, 0, 0, 0, 0]
    // Feature 101: peaks around bases 50-100
    acts[0] = Math.max(0, 0.8 - Math.abs(i - 75) / 30)
    // Feature 207: peaks at start codon-ish positions (every 90 bases)
    acts[1] = Math.exp(-Math.pow((i % 90) - 5, 2) / 10) * 0.6
    // Feature 314: peaks in middle third
    acts[2] = i > length / 3 && i < (2 * length) / 3 ? 0.4 + 0.3 * Math.sin(i / 7) : 0
    // Feature 422: low baseline + occasional spikes
    acts[3] = i % 17 === 3 ? 0.7 : 0.05
    // Feature 588: tapering toward end
    acts[4] = Math.max(0, (i - length / 2) / (length / 2)) * 0.5
    // Sort descending, keep top 10 (or all 5 here)
    const pairs = FEATURES.map((f, k) => [f, acts[k]]).sort((a, b) => b[1] - a[1])
    topk_features.push(pairs.map((p) => p[0]))
    topk_activations.push(pairs.map((p) => p[1]))
  }
  return { length, topk_features, topk_activations }
}


export default function ColoredSequence({
  sequence,
  analysis,
  featureCatalog,
  mode = 'top', // 'top' | 'single'
  singleFeatureId = null,
  onBaseClick,
  baseFontSize = 13,
}) {
  // Fall back to mock data when no analysis prop is supplied. Pluggable: pass
  // a real /analyze response and the component renders the real signal.
  const data = useMemo(() => analysis ?? buildMockAnalysis(sequence), [sequence, analysis])

  // Sequence-wide max activation across top-1 per position. Used to normalize
  // each position's opacity into [OPACITY_FLOOR, OPACITY_CEILING].
  const maxAct = useMemo(() => {
    if (mode === 'single' && singleFeatureId != null) {
      let m = 0
      for (let i = 0; i < sequence.length; i++) {
        const feats = data.topk_features[i] || []
        const acts = data.topk_activations[i] || []
        const idx = feats.indexOf(singleFeatureId)
        if (idx >= 0 && acts[idx] > m) m = acts[idx]
      }
      return m
    }
    let m = 0
    for (let i = 0; i < sequence.length; i++) {
      const a = data.topk_activations[i]?.[0] ?? 0
      if (a > m) m = a
    }
    return m
  }, [sequence, data, mode, singleFeatureId])

  // Count how many positions each color shows up at — drives the legend.
  const colorUsage = useMemo(() => {
    const counts = new Map() // featureId -> count
    for (let i = 0; i < sequence.length; i++) {
      const feats = data.topk_features[i] || []
      const acts = data.topk_activations[i] || []
      if (mode === 'single') {
        if (singleFeatureId == null) continue
        const idx = feats.indexOf(singleFeatureId)
        if (idx < 0 || acts[idx] < ACTIVATION_FLOOR_FRAC * maxAct) continue
        counts.set(singleFeatureId, (counts.get(singleFeatureId) || 0) + 1)
      } else {
        const f = feats[0]
        const a = acts[0] ?? 0
        if (f == null || a < ACTIVATION_FLOOR_FRAC * maxAct) continue
        counts.set(f, (counts.get(f) || 0) + 1)
      }
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1])
  }, [sequence, data, mode, singleFeatureId, maxAct])

  // Pre-compute per-base styling + tooltip data. Keeps render fast even on
  // long sequences (e.g. 5 kb genes).
  const baseStyles = useMemo(() => {
    const styles = []
    for (let i = 0; i < sequence.length; i++) {
      const feats = data.topk_features[i] || []
      const acts = data.topk_activations[i] || []
      let featureId = null
      let activation = 0
      if (mode === 'single' && singleFeatureId != null) {
        const idx = feats.indexOf(singleFeatureId)
        if (idx >= 0) {
          featureId = singleFeatureId
          activation = acts[idx]
        }
      } else {
        featureId = feats[0]
        activation = acts[0] ?? 0
      }
      const color = colorForFeature(featureId)
      const threshold = ACTIVATION_FLOOR_FRAC * maxAct
      let bg = 'transparent'
      if (color && activation >= threshold && maxAct > 0) {
        const t = Math.min(1, activation / maxAct)
        const alpha = OPACITY_FLOOR + t * (OPACITY_CEILING - OPACITY_FLOOR)
        bg = hexWithAlpha(color, alpha)
      }
      styles.push({ bg, featureId, activation })
    }
    return styles
  }, [sequence, data, mode, singleFeatureId, maxAct])

  // Split the sequence into lines of BASES_PER_LINE for the standard
  // sequence-view layout (position numbers at line breaks).
  const lines = []
  for (let start = 0; start < sequence.length; start += BASES_PER_LINE) {
    const end = Math.min(start + BASES_PER_LINE, sequence.length)
    lines.push({ start, end, bases: sequence.slice(start, end) })
  }

  return (
    <div style={styles.container}>
      <div style={styles.sequenceBlock}>
        {lines.map((line) => (
          <div key={line.start} style={styles.line}>
            <span style={{ ...styles.position, fontSize: baseFontSize - 2 }}>
              {String(line.start + 1).padStart(6, ' ')}
            </span>
            <span style={{ ...styles.bases, fontSize: baseFontSize }}>
              {[...line.bases].map((base, j) => {
                const i = line.start + j
                const { bg, featureId, activation } = baseStyles[i]
                return (
                  <Base
                    key={i}
                    base={base}
                    position={i}
                    background={bg}
                    featureId={featureId}
                    activation={activation}
                    topFeatures={data.topk_features[i] || []}
                    topActivations={data.topk_activations[i] || []}
                    featureCatalog={featureCatalog}
                    onClick={onBaseClick}
                  />
                )
              })}
            </span>
          </div>
        ))}
      </div>

      <Legend usage={colorUsage} catalog={featureCatalog} maxAct={maxAct} />
    </div>
  )
}


function Base({
  base,
  position,
  background,
  featureId,
  activation,
  topFeatures,
  topActivations,
  featureCatalog,
  onClick,
}) {
  const [hover, setHover] = useState(false)
  const handleClick = (e) => {
    e.stopPropagation()
    onClick?.({ position, base, topFeatures, topActivations })
  }
  return (
    <span
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={handleClick}
      style={{
        background,
        padding: '0 1px',
        borderRadius: '2px',
        cursor: 'pointer',
        position: 'relative',
      }}
    >
      {base}
      {hover && (
        <Tooltip
          position={position}
          base={base}
          topFeatures={topFeatures}
          topActivations={topActivations}
          featureCatalog={featureCatalog}
        />
      )}
    </span>
  )
}


function Tooltip({ position, base, topFeatures, topActivations, featureCatalog }) {
  const rows = topFeatures.slice(0, 5).map((f, k) => ({
    featureId: f,
    label: labelForFeature(f, featureCatalog),
    activation: topActivations[k] ?? 0,
    color: colorForFeature(f),
  }))
  return (
    <div style={styles.tooltip}>
      <div style={styles.tooltipHeader}>
        pos {position + 1} <span style={{ color: 'var(--text-secondary)' }}>(base {base})</span>
      </div>
      <table style={styles.tooltipTable}>
        <tbody>
          {rows.map((r) => (
            <tr key={r.featureId}>
              <td>
                <span style={{ ...styles.colorDot, background: r.color }} /> {r.label}
              </td>
              <td style={styles.tooltipActCell}>{r.activation.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}


function Legend({ usage, catalog, maxAct }) {
  if (usage.length === 0) {
    return <div style={styles.legendEmpty}>No features above activation threshold</div>
  }
  return (
    <div style={styles.legend}>
      <div style={styles.legendTitle}>Colors in this sequence (by # of positions):</div>
      <div style={styles.legendItems}>
        {usage.map(([fid, n]) => (
          <span key={fid} style={styles.legendItem}>
            <span style={{ ...styles.colorDot, background: colorForFeature(fid) }} />
            {labelForFeature(fid, catalog)}
            <span style={styles.legendCount}>×{n}</span>
          </span>
        ))}
      </div>
      <div style={styles.legendMeta}>max activation in sequence: {maxAct.toFixed(3)}</div>
    </div>
  )
}


// Add an alpha channel to a 6-digit hex color. Keeps the palette source as
// plain hex so palette swaps don't have to redo opacity math.
function hexWithAlpha(hex, alpha) {
  const a = Math.round(alpha * 255).toString(16).padStart(2, '0')
  return `${hex}${a}`
}


const styles = {
  container: {
    fontFamily: 'monospace',
    background: 'var(--bg-card-expanded)',
    border: '1px solid var(--border-light)',
    borderRadius: '6px',
    padding: '10px 12px',
  },
  sequenceBlock: {
    whiteSpace: 'pre-wrap',
    lineHeight: '1.5',
    color: 'var(--text)',
  },
  line: {
    display: 'flex',
    gap: '8px',
    alignItems: 'baseline',
  },
  position: {
    color: 'var(--text-muted)',
    userSelect: 'none',
    fontFamily: 'monospace',
    minWidth: '50px',
    textAlign: 'right',
  },
  bases: {
    fontFamily: 'monospace',
    letterSpacing: '0.5px',
  },
  tooltip: {
    position: 'absolute',
    top: '110%',
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 100,
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    padding: '6px 8px',
    fontSize: '11px',
    minWidth: '220px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
    pointerEvents: 'none',
    fontFamily: 'system-ui, sans-serif',
  },
  tooltipHeader: {
    fontWeight: 600,
    marginBottom: '4px',
    paddingBottom: '4px',
    borderBottom: '1px solid var(--border-light)',
    color: 'var(--text-heading)',
  },
  tooltipTable: {
    width: '100%',
    borderCollapse: 'collapse',
  },
  tooltipActCell: {
    textAlign: 'right',
    fontFamily: 'monospace',
    color: 'var(--text-secondary)',
  },
  colorDot: {
    display: 'inline-block',
    width: '8px',
    height: '8px',
    borderRadius: '2px',
    marginRight: '6px',
    verticalAlign: 'middle',
  },
  legend: {
    marginTop: '10px',
    paddingTop: '8px',
    borderTop: '1px solid var(--border-light)',
    fontSize: '11px',
    fontFamily: 'system-ui, sans-serif',
    color: 'var(--text-secondary)',
  },
  legendTitle: {
    fontSize: '10px',
    textTransform: 'uppercase',
    color: 'var(--text-tertiary)',
    marginBottom: '6px',
  },
  legendItems: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '12px',
  },
  legendItem: {
    display: 'inline-flex',
    alignItems: 'center',
    fontSize: '11px',
  },
  legendCount: {
    marginLeft: '4px',
    fontFamily: 'monospace',
    color: 'var(--text-muted)',
  },
  legendMeta: {
    marginTop: '6px',
    fontSize: '10px',
    color: 'var(--text-muted)',
  },
  legendEmpty: {
    fontSize: '11px',
    fontStyle: 'italic',
    color: 'var(--text-muted)',
    marginTop: '8px',
  },
}
