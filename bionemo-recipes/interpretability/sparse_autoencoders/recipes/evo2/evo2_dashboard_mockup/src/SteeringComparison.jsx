import React, { useEffect, useMemo, useState } from 'react'

// Side-by-side suppress / baseline / amplify comparison for a chosen
// (seed_sequence, feature) pair. All synthetic — loads
// /steering_examples.json. No backend, no inference. Selectors are
// instant-apply (no cosmetic "Run" button) so when real inference
// lands we only swap the data source.

const BASES = ['A', 'C', 'G', 'T']
const BASE_COLORS = {
  A: '#59A14F', // green
  C: '#4E79A7', // blue
  G: '#F28E2B', // orange
  T: '#E15759', // red
}

const INTERVENTIONS = [
  { id: 'suppress', label: 'Suppress', badge: '0×', color: '#9C755F' },
  { id: 'baseline', label: 'Baseline', badge: '1×', color: '#76B7B2' },
  { id: 'amplify', label: 'Amplify', badge: '2×', color: '#E15759' },
]

// Null-result threshold: comparisons with effect_size below this get
// a "no signal" badge in the headline.
const NULL_EFFECT_THRESHOLD = 0.05


export default function SteeringComparison() {
  const [data, setData] = useState(null)
  const [seedId, setSeedId] = useState('ecoli_16s')
  const [featureId, setFeatureId] = useState(null) // resolves on first data load
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/steering_examples.json')
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  // When data lands, pick a sensible default feature for the chosen seed.
  useEffect(() => {
    if (!data || featureId != null) return
    const firstKey = Object.keys(data.comparisons).find((k) => k.startsWith(`${seedId}__`))
    if (firstKey) setFeatureId(data.comparisons[firstKey].feature_label)
  }, [data, seedId, featureId])

  if (error) return <div style={styles.error}>Failed to load steering_examples.json: {error}</div>
  if (!data) return <div style={styles.loading}>Loading steering examples…</div>

  // Features that have at least one comparison defined (across all seeds).
  // Filtered subset — we don't expose features that have no synthetic data.
  const availableFeatures = useMemo(() => {
    const seen = new Set()
    for (const key of Object.keys(data.comparisons)) {
      seen.add(data.comparisons[key].feature_label)
    }
    return [...seen]
  }, [data])

  // Features that have a comparison FOR THE CURRENT SEED (shrinks dropdown
  // to genuinely available pairs, with the others shown as disabled).
  const availableForSeed = useMemo(() => {
    const seen = new Set()
    for (const key of Object.keys(data.comparisons)) {
      if (key.startsWith(`${seedId}__`)) {
        seen.add(data.comparisons[key].feature_label)
      }
    }
    return seen
  }, [data, seedId])

  const seed = data.seeds[seedId]
  const comparisonKey = `${seedId}__${featureId}`
  const comparison = data.comparisons[comparisonKey]

  return (
    <div style={styles.container}>
      <div style={styles.banner}>
        MOCKUP — synthetic data, not from real inference. Probability values are hand-rolled
        per (seed, feature) pair to demonstrate the comparison-view UX.
      </div>

      <Controls
        seeds={data.seeds}
        availableFeatures={availableFeatures}
        availableForSeed={availableForSeed}
        seedId={seedId}
        setSeedId={(id) => {
          setSeedId(id)
          // Reset feature; will re-resolve to a default that exists for the new seed.
          setFeatureId(null)
        }}
        featureId={featureId}
        setFeatureId={setFeatureId}
      />

      {comparison ? (
        <>
          <DiffSummary comparison={comparison} seed={seed} />
          <Columns seed={seed} comparison={comparison} />
        </>
      ) : (
        <div style={styles.noPair}>
          No synthetic steering data for <code>{seedId}</code> ×{' '}
          <code>{featureId || '(none)'}</code>. Pick a different combination —{' '}
          {availableForSeed.size} features have data for this seed.
        </div>
      )}
    </div>
  )
}


function Controls({ seeds, availableFeatures, availableForSeed, seedId, setSeedId, featureId, setFeatureId }) {
  return (
    <div style={styles.controls}>
      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Seed sequence:</label>
        <div style={styles.seedButtons}>
          {Object.entries(seeds).map(([id, s]) => (
            <button
              key={id}
              onClick={() => setSeedId(id)}
              style={seedId === id ? styles.seedBtnActive : styles.seedBtn}
              title={s.description}
            >
              {s.name}
            </button>
          ))}
        </div>
      </div>

      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Steer feature:</label>
        <select
          value={featureId || ''}
          onChange={(e) => setFeatureId(e.target.value)}
          style={styles.featureSelect}
        >
          {availableFeatures.map((f) => (
            <option key={f} value={f} disabled={!availableForSeed.has(f)}>
              {f}
              {availableForSeed.has(f) ? '' : '  (no data for this seed)'}
            </option>
          ))}
        </select>
      </div>

      <div style={styles.explanation}>
        Steering clamps a chosen feature high (amplify) or low (suppress) inside the SAE during a
        single forward pass at a masked position. The model's predicted base distribution shifts
        based on what the feature controls.
      </div>
    </div>
  )
}


function DiffSummary({ comparison, seed }) {
  const isNull = comparison.effect_size < NULL_EFFECT_THRESHOLD
  const baseline = comparison.baseline
  const amplify = comparison.amplify
  const top = amplify.top_base
  const baselineProb = baseline.p_base[top]
  const amplifyProb = amplify.p_base[top]

  return (
    <div style={styles.diffSummary}>
      <div style={styles.diffHeader}>
        <span style={styles.diffTitle}>What changed</span>
        {isNull ? (
          <span style={styles.nullBadge}>NULL RESULT — minimal effect</span>
        ) : (
          <span style={styles.signalBadge}>effect size: {(comparison.effect_size * 100).toFixed(0)}pp</span>
        )}
      </div>
      <div style={styles.diffRow}>
        Amplifying <b>{comparison.feature_label}</b> shifted{' '}
        <code>P({top})</code> from <b>{(baselineProb * 100).toFixed(0)}%</b> to{' '}
        <b>{(amplifyProb * 100).toFixed(0)}%</b>
        <span style={{ color: amplifyProb > baselineProb ? '#5a9' : '#c66', marginLeft: '6px' }}>
          ({amplifyProb > baselineProb ? '+' : ''}{((amplifyProb - baselineProb) * 100).toFixed(0)}pp)
        </span>
      </div>
      <div style={styles.diffRow}>
        Feature activation at masked position:{' '}
        <code>{comparison.suppress.feature_activation.toFixed(1)}</code> →{' '}
        <code>{comparison.baseline.feature_activation.toFixed(1)}</code> →{' '}
        <code>{comparison.amplify.feature_activation.toFixed(1)}</code>
      </div>
      {comparison.narrative && (
        <div style={styles.callout}>
          <b>Note:</b> {comparison.narrative}
        </div>
      )}
    </div>
  )
}


function Columns({ seed, comparison }) {
  return (
    <div style={styles.columnRow}>
      {INTERVENTIONS.map((iv) => (
        <Column key={iv.id} intervention={iv} seed={seed} result={comparison[iv.id]} />
      ))}
    </div>
  )
}


function Column({ intervention, seed, result }) {
  return (
    <div style={styles.column}>
      <div style={styles.columnHeader}>
        <span style={{ ...styles.badge, background: intervention.color }}>{intervention.badge}</span>
        <span style={styles.columnLabel}>{intervention.label}</span>
      </div>
      <SequenceWithMask sequence={seed.sequence} maskPosition={seed.mask_position} />
      <BarChart pBase={result.p_base} topBase={result.top_base} />
      <ActivationPanel value={result.feature_activation} />
    </div>
  )
}


function SequenceWithMask({ sequence, maskPosition }) {
  // Standard FASTA-style 60bp line wrap, mask position highlighted.
  const lines = []
  const linewidth = 60
  for (let start = 0; start < sequence.length; start += linewidth) {
    lines.push({ start, end: Math.min(start + linewidth, sequence.length) })
  }
  return (
    <div style={styles.seqWrap}>
      {lines.map((line) => (
        <div key={line.start} style={styles.seqLine}>
          <span style={styles.seqPos}>{String(line.start + 1).padStart(3, ' ')}</span>
          <span style={styles.seqBases}>
            {[...sequence.slice(line.start, line.end)].map((base, j) => {
              const pos = line.start + j
              const isMask = pos === maskPosition
              return (
                <span key={pos} style={isMask ? styles.maskBase : styles.normalBase}>
                  {isMask ? '?' : base}
                </span>
              )
            })}
          </span>
        </div>
      ))}
    </div>
  )
}


function BarChart({ pBase, topBase }) {
  const max = Math.max(...BASES.map((b) => pBase[b] || 0))
  return (
    <div style={styles.barChart}>
      <div style={styles.barChartTitle}>P(base) at masked position</div>
      {BASES.map((base) => {
        const p = pBase[base] || 0
        const isTop = base === topBase
        const pctWidth = (p / max) * 100
        return (
          <div key={base} style={styles.barRow}>
            <span style={{ ...styles.barLabel, color: BASE_COLORS[base] }}>{base}</span>
            <div style={styles.barTrack}>
              <div
                style={{
                  ...styles.barFill,
                  width: `${pctWidth}%`,
                  background: BASE_COLORS[base],
                  border: isTop ? `2px solid ${BASE_COLORS[base]}` : 'none',
                  filter: isTop ? 'none' : 'opacity(0.65)',
                }}
              />
            </div>
            <span style={{ ...styles.barValue, fontWeight: isTop ? 700 : 400 }}>
              {(p * 100).toFixed(0)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}


function ActivationPanel({ value }) {
  return (
    <div style={styles.actPanel}>
      <span style={styles.actLabel}>Feature activation:</span>
      <span style={styles.actValue}>{value.toFixed(1)}</span>
    </div>
  )
}


const styles = {
  container: {
    fontFamily: 'system-ui, sans-serif',
    color: 'var(--text, #222)',
    padding: '0',
  },
  banner: {
    background: '#fff3cd',
    border: '1px solid #ffeeba',
    color: '#856404',
    padding: '8px 14px',
    borderRadius: '4px',
    fontSize: '11px',
    marginBottom: '14px',
  },
  controls: {
    background: 'var(--bg-card, #fff)',
    border: '1px solid var(--border, #ddd)',
    borderRadius: '6px',
    padding: '12px 16px',
    marginBottom: '14px',
  },
  controlRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '8px',
    flexWrap: 'wrap',
  },
  controlLabel: {
    fontSize: '12px',
    fontWeight: 600,
    color: 'var(--text-secondary, #555)',
    minWidth: '110px',
  },
  seedButtons: { display: 'flex', gap: '6px', flexWrap: 'wrap' },
  seedBtn: {
    padding: '4px 10px',
    border: '1px solid var(--border, #ddd)',
    background: '#fff',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    color: 'var(--text-secondary, #555)',
  },
  seedBtnActive: {
    padding: '4px 10px',
    border: '1px solid var(--accent, #76b900)',
    background: 'var(--bg-card-expanded, #f0f8e8)',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    color: 'var(--accent, #76b900)',
    fontWeight: 600,
  },
  featureSelect: {
    padding: '4px 8px',
    fontSize: '12px',
    border: '1px solid var(--border, #ddd)',
    borderRadius: '4px',
    background: '#fff',
    minWidth: '260px',
  },
  explanation: {
    marginTop: '8px',
    paddingTop: '8px',
    borderTop: '1px solid var(--border-light, #eee)',
    fontSize: '11px',
    color: 'var(--text-secondary, #666)',
    lineHeight: '1.4',
  },
  diffSummary: {
    background: 'var(--bg-card, #fff)',
    border: '1px solid var(--border, #ddd)',
    borderLeft: '3px solid var(--accent, #76b900)',
    borderRadius: '6px',
    padding: '12px 16px',
    marginBottom: '14px',
    fontSize: '12px',
    lineHeight: '1.6',
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  diffHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '6px',
  },
  diffTitle: {
    fontSize: '11px',
    fontWeight: 700,
    textTransform: 'uppercase',
    color: 'var(--text-tertiary, #888)',
  },
  signalBadge: {
    fontSize: '10px',
    fontWeight: 600,
    color: 'var(--accent, #76b900)',
    background: 'var(--bg-card-expanded, #f0f8e8)',
    border: '1px solid var(--accent, #76b900)',
    borderRadius: '3px',
    padding: '2px 6px',
  },
  nullBadge: {
    fontSize: '10px',
    fontWeight: 600,
    color: '#856404',
    background: '#fff3cd',
    border: '1px solid #ffeeba',
    borderRadius: '3px',
    padding: '2px 6px',
  },
  diffRow: { fontSize: '12px' },
  callout: {
    marginTop: '8px',
    padding: '6px 10px',
    background: '#eef6ff',
    border: '1px solid #bcd9ff',
    borderRadius: '4px',
    fontSize: '11px',
    color: '#1a3a6a',
  },
  columnRow: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '14px',
  },
  column: {
    background: 'var(--bg-card, #fff)',
    border: '1px solid var(--border, #ddd)',
    borderRadius: '6px',
    padding: '12px',
  },
  columnHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '10px',
    paddingBottom: '8px',
    borderBottom: '1px solid var(--border-light, #eee)',
  },
  badge: {
    color: 'white',
    fontWeight: 700,
    fontSize: '11px',
    padding: '3px 8px',
    borderRadius: '3px',
    fontFamily: 'monospace',
  },
  columnLabel: {
    fontWeight: 600,
    fontSize: '13px',
    color: 'var(--text-heading, #222)',
  },
  seqWrap: {
    fontFamily: 'monospace',
    fontSize: '11px',
    background: 'var(--bg-card-expanded, #f8f8f8)',
    border: '1px solid var(--border-light, #eee)',
    borderRadius: '4px',
    padding: '6px 8px',
    marginBottom: '10px',
    lineHeight: '1.5',
  },
  seqLine: { display: 'flex', gap: '6px' },
  seqPos: { color: 'var(--text-muted, #aaa)', minWidth: '24px', textAlign: 'right' },
  seqBases: { letterSpacing: '0.5px' },
  normalBase: {},
  maskBase: {
    background: '#ffd966',
    color: '#7a5b00',
    fontWeight: 700,
    padding: '0 2px',
    borderRadius: '2px',
    border: '1px solid #b89732',
  },
  barChart: {
    marginBottom: '10px',
  },
  barChartTitle: {
    fontSize: '10px',
    fontWeight: 600,
    textTransform: 'uppercase',
    color: 'var(--text-tertiary, #888)',
    marginBottom: '6px',
  },
  barRow: {
    display: 'grid',
    gridTemplateColumns: '20px 1fr 40px',
    gap: '6px',
    alignItems: 'center',
    marginBottom: '4px',
  },
  barLabel: {
    fontFamily: 'monospace',
    fontWeight: 700,
    fontSize: '12px',
  },
  barTrack: {
    height: '14px',
    background: '#f0f0f0',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: '3px',
    transition: 'width 0.2s ease',
  },
  barValue: {
    fontFamily: 'monospace',
    fontSize: '11px',
    textAlign: 'right',
    color: 'var(--text, #333)',
  },
  actPanel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '6px 10px',
    background: 'var(--bg-card-expanded, #f8f8f8)',
    border: '1px solid var(--border-light, #eee)',
    borderRadius: '4px',
  },
  actLabel: {
    fontSize: '10px',
    fontWeight: 600,
    textTransform: 'uppercase',
    color: 'var(--text-tertiary, #888)',
  },
  actValue: {
    fontFamily: 'monospace',
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--text-heading, #222)',
  },
  noPair: {
    padding: '24px',
    textAlign: 'center',
    background: 'var(--bg-card-expanded, #f8f8f8)',
    border: '1px dashed var(--border, #ddd)',
    borderRadius: '6px',
    fontSize: '12px',
    color: 'var(--text-muted, #888)',
  },
  loading: {
    padding: '40px',
    textAlign: 'center',
    color: 'var(--text-muted, #aaa)',
    fontStyle: 'italic',
  },
  error: {
    padding: '20px',
    background: '#fee',
    color: '#c34',
    borderRadius: '4px',
    fontFamily: 'monospace',
    fontSize: '12px',
  },
}
