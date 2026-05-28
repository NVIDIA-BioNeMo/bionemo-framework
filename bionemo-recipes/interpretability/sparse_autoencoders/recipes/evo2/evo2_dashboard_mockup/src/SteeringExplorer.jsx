import React, { useEffect, useMemo, useState } from 'react'

// Whole-sequence steering visualization. Loads /steering_data.json containing
// per-position P(A/C/G/T) baseline + steered distributions at clamp ∈ {-2,0,+2,+5}
// for each (seed, feature) pair. Slider linearly interpolates between the
// discrete clamp points.

const BASES = ['A', 'C', 'G', 'T']
const BASE_COLORS = {
  A: '#59A14F',
  C: '#4E79A7',
  G: '#F28E2B',
  T: '#E15759',
}
const POS_PER_LINE = 60
const BAR_HEIGHT = 24      // vertical pixels for the stacked prob bar above each base
const CELL_WIDTH = 11       // each base + its bar slot
const ENTROPY_DELTA_THRESHOLD = 0.15  // highlight positions with meaningful shifts


// Interpolate distributions between two discrete clamp values.
function interpolate(distLow, distHigh, t) {
  // t in [0,1]: 0 returns distLow, 1 returns distHigh
  const out = new Array(distLow.length)
  for (let i = 0; i < distLow.length; i++) {
    const a = distLow[i]
    const b = distHigh[i]
    out[i] = {
      A: a.A + (b.A - a.A) * t,
      C: a.C + (b.C - a.C) * t,
      G: a.G + (b.G - a.G) * t,
      T: a.T + (b.T - a.T) * t,
    }
  }
  return out
}


// Find the two flanking clamp points and t in [0,1] between them.
function pickClampSegment(clampPoints, value) {
  // clampPoints sorted ascending, e.g. [-2, 0, 2, 5]
  if (value <= clampPoints[0]) return { lo: clampPoints[0], hi: clampPoints[0], t: 0 }
  if (value >= clampPoints[clampPoints.length - 1]) {
    const last = clampPoints[clampPoints.length - 1]
    return { lo: last, hi: last, t: 0 }
  }
  for (let i = 0; i < clampPoints.length - 1; i++) {
    if (value >= clampPoints[i] && value <= clampPoints[i + 1]) {
      const t = (value - clampPoints[i]) / (clampPoints[i + 1] - clampPoints[i])
      return { lo: clampPoints[i], hi: clampPoints[i + 1], t }
    }
  }
  return { lo: clampPoints[0], hi: clampPoints[0], t: 0 }
}


function topBase(dist) {
  let mb = 'A'
  let mp = 0
  for (const b of BASES) {
    if (dist[b] > mp) { mp = dist[b]; mb = b }
  }
  return mb
}


function entropy(dist) {
  let h = 0
  for (const b of BASES) {
    const p = dist[b]
    if (p > 0) h -= p * Math.log2(p)
  }
  return h
}


export default function SteeringExplorer() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [seedId, setSeedId] = useState('ecoli_16s')
  const [featureLabel, setFeatureLabel] = useState(null)
  const [clamp, setClamp] = useState(0)
  const [hoverPos, setHoverPos] = useState(null)

  // --- data loading
  useEffect(() => {
    fetch('/steering_data.json').then((r) => r.json()).then(setData).catch((e) => setError(e.message))
  }, [])

  // pick a sensible default feature when data lands or seed changes
  useEffect(() => {
    if (!data) return
    const prefix = `${seedId}__`
    const firstKey = Object.keys(data.comparisons).find((k) => k.startsWith(prefix))
    if (firstKey) {
      const lbl = data.comparisons[firstKey].feature_label
      setFeatureLabel(lbl)
    } else {
      setFeatureLabel(null)
    }
  }, [data, seedId])

  // --- derived (all hooks must run regardless of data) ---
  const availableForSeed = useMemo(() => {
    if (!data) return new Set()
    const seen = new Set()
    for (const k of Object.keys(data.comparisons)) {
      if (k.startsWith(`${seedId}__`)) seen.add(data.comparisons[k].feature_label)
    }
    return seen
  }, [data, seedId])

  const allFeatures = useMemo(() => {
    if (!data) return []
    return data.features_available
  }, [data])

  const comparisonKey = `${seedId}__${featureLabel}`
  const comparison = data?.comparisons[comparisonKey] || null

  // Interpolated steered distribution for the current clamp value
  const steered = useMemo(() => {
    if (!comparison || !data) return null
    const { lo, hi, t } = pickClampSegment(data.clamp_points, clamp)
    return interpolate(comparison.steered_distributions[String(lo)], comparison.steered_distributions[String(hi)], t)
  }, [comparison, data, clamp])

  // Position indices where the steered top base differs from baseline AND the
  // change is substantive (entropy delta beyond a threshold).
  const flippedPositions = useMemo(() => {
    if (!comparison || !steered) return new Set()
    const out = new Set()
    for (let i = 0; i < steered.length; i++) {
      const bTop = topBase(comparison.baseline_distributions[i])
      const sTop = topBase(steered[i])
      if (bTop !== sTop) {
        const dH = Math.abs(entropy(comparison.baseline_distributions[i]) - entropy(steered[i]))
        if (dH > ENTROPY_DELTA_THRESHOLD || Math.abs(steered[i][sTop] - comparison.baseline_distributions[i][sTop]) > 0.2) {
          out.add(i)
        }
      }
    }
    return out
  }, [comparison, steered])

  const summary = useMemo(() => {
    if (!comparison || !steered) return null
    let entropyDelta = 0
    for (let i = 0; i < steered.length; i++) {
      entropyDelta += entropy(steered[i]) - entropy(comparison.baseline_distributions[i])
    }
    return {
      nFlipped: flippedPositions.size,
      meanEntropyDelta: entropyDelta / steered.length,
    }
  }, [comparison, steered, flippedPositions])

  // --- early returns AFTER hooks ---
  if (error) return <div style={styles.error}>Failed to load steering_data.json: {error}</div>
  if (!data) return <div style={styles.loading}>Loading steering data…</div>

  const seed = data.seeds[seedId]

  return (
    <div style={styles.container}>
      <div style={styles.banner}>
        MOCKUP — synthetic data. Per-position probabilities are generated algorithmically from
        per-pair effect rules, not real model inference.
      </div>

      <Controls
        seeds={data.seeds}
        allFeatures={allFeatures}
        availableForSeed={availableForSeed}
        seedId={seedId}
        setSeedId={(id) => { setSeedId(id); setHoverPos(null) }}
        featureLabel={featureLabel}
        setFeatureLabel={setFeatureLabel}
        clamp={clamp}
        setClamp={setClamp}
      />

      {comparison && steered ? (
        <>
          <div style={styles.heatmapPanel}>
            <Strip
              label="Baseline"
              sequence={seed.sequence}
              distributions={comparison.baseline_distributions}
              flipped={new Set()}
              hoverPos={hoverPos}
              setHoverPos={setHoverPos}
            />
            <Strip
              label="Steered"
              sequence={seed.sequence}
              distributions={steered}
              flipped={flippedPositions}
              hoverPos={hoverPos}
              setHoverPos={setHoverPos}
              showFlippedMarkers
            />
          </div>

          {hoverPos != null && (
            <PositionDetail
              pos={hoverPos}
              sequence={seed.sequence}
              baseline={comparison.baseline_distributions[hoverPos]}
              steered={steered[hoverPos]}
              baselineAct={comparison.feature_activation['0'][hoverPos]}
              steeredAct={interpolateAct(comparison.feature_activation, data.clamp_points, clamp, hoverPos)}
            />
          )}

          <SummaryStats summary={summary} narrative={comparison.narrative} clamp={clamp} />
        </>
      ) : (
        <div style={styles.noPair}>
          No synthetic steering data for <code>{seedId}</code> ×{' '}
          <code>{featureLabel || '(none)'}</code>. Pick a different combination —{' '}
          {availableForSeed.size} features have data for this seed.
        </div>
      )}
    </div>
  )
}


function interpolateAct(featureAct, clampPoints, clamp, pos) {
  const { lo, hi, t } = pickClampSegment(clampPoints, clamp)
  const a = featureAct[String(lo)][pos]
  const b = featureAct[String(hi)][pos]
  return a + (b - a) * t
}


function Controls({ seeds, allFeatures, availableForSeed, seedId, setSeedId, featureLabel, setFeatureLabel, clamp, setClamp }) {
  return (
    <div style={styles.controls}>
      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Sequence:</label>
        <div style={styles.seedButtons}>
          {Object.entries(seeds).map(([id, s]) => (
            <button
              key={id}
              onClick={() => setSeedId(id)}
              style={seedId === id ? styles.seedBtnActive : styles.seedBtn}
            >
              {s.name}
            </button>
          ))}
        </div>
        <span style={styles.futureHint}>(v2: paste your own sequence)</span>
      </div>

      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Feature:</label>
        <select
          value={featureLabel || ''}
          onChange={(e) => setFeatureLabel(e.target.value)}
          style={styles.featureSelect}
        >
          {allFeatures.map((f) => (
            <option key={f.label} value={f.label} disabled={!availableForSeed.has(f.label)}>
              {f.label}{availableForSeed.has(f.label) ? '' : '  (no data)'}
            </option>
          ))}
        </select>
      </div>

      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Clamp:</label>
        <div style={styles.sliderWrap}>
          <input
            type="range"
            min={-2}
            max={5}
            step={0.1}
            value={clamp}
            onChange={(e) => setClamp(parseFloat(e.target.value))}
            style={styles.slider}
          />
          <div style={styles.sliderTicks}>
            <span style={styles.tick} onClick={() => setClamp(-2)}>−2 Suppress</span>
            <span style={styles.tick} onClick={() => setClamp(0)}>0 Baseline</span>
            <span style={styles.tick} onClick={() => setClamp(2)}>+2 Amplify</span>
            <span style={styles.tick} onClick={() => setClamp(5)}>+5 Strong</span>
          </div>
          <span style={styles.clampValue}>= {clamp.toFixed(1)}</span>
        </div>
      </div>

      <div style={styles.explanation}>
        Clamping forces the SAE feature to activate at the chosen strength at every position. The
        model's predicted base at each position shifts based on what the feature controls.
      </div>
    </div>
  )
}


function Strip({ label, sequence, distributions, flipped, hoverPos, setHoverPos, showFlippedMarkers }) {
  // Layout: position groups of POS_PER_LINE bases per row, with stacked prob bars
  // above each base.
  const rows = []
  for (let start = 0; start < sequence.length; start += POS_PER_LINE) {
    rows.push({ start, end: Math.min(start + POS_PER_LINE, sequence.length) })
  }
  return (
    <div style={styles.strip}>
      <div style={styles.stripLabel}>{label}</div>
      {rows.map(({ start, end }) => (
        <div key={start} style={styles.row}>
          <div style={styles.rowIndex}>{String(start + 1).padStart(4, ' ')}</div>
          <div style={styles.bars}>
            {Array.from({ length: end - start }).map((_, j) => {
              const pos = start + j
              return (
                <ProbCell
                  key={pos}
                  pos={pos}
                  base={sequence[pos]}
                  dist={distributions[pos]}
                  isHover={hoverPos === pos}
                  isFlipped={flipped.has(pos)}
                  showFlippedMarkers={showFlippedMarkers}
                  onEnter={() => setHoverPos(pos)}
                  onLeave={() => setHoverPos(null)}
                />
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}


function ProbCell({ pos, base, dist, isHover, isFlipped, showFlippedMarkers, onEnter, onLeave }) {
  // Stacked vertical bar above the base letter. Total height = BAR_HEIGHT;
  // each base contributes proportional to dist[b].
  let yCursor = 0
  const segments = BASES.map((b) => {
    const h = dist[b] * BAR_HEIGHT
    const seg = (
      <div
        key={b}
        style={{
          height: `${h}px`,
          background: BASE_COLORS[b],
          width: '100%',
          opacity: isHover ? 1.0 : 0.85,
        }}
      />
    )
    yCursor += h
    return seg
  })
  return (
    <div
      style={{
        ...styles.cell,
        outline: isHover ? '2px solid #76b900' : 'none',
        outlineOffset: '-1px',
      }}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      <div style={styles.barStack}>{segments}</div>
      <div style={{ ...styles.baseLetter, color: BASE_COLORS[base] }}>{base}</div>
      {showFlippedMarkers && isFlipped && <div style={styles.flippedMarker}>▲</div>}
    </div>
  )
}


function PositionDetail({ pos, sequence, baseline, steered, baselineAct, steeredAct }) {
  const topB = topBase(baseline)
  const topS = topBase(steered)
  return (
    <div style={styles.detailPanel}>
      <div style={styles.detailHeader}>
        Position {pos + 1} — base <code>{sequence[pos]}</code>
      </div>
      <div style={styles.detailColumns}>
        <DetailBars title="Baseline" dist={baseline} top={topB} />
        <DetailBars title="Steered" dist={steered} top={topS} />
      </div>
      <div style={styles.detailReadout}>
        Top base: <b>{topB}</b> ({(baseline[topB] * 100).toFixed(0)}%) →{' '}
        <b style={{ color: topB === topS ? 'var(--text)' : '#c34' }}>{topS}</b>{' '}
        ({(steered[topS] * 100).toFixed(0)}%)
        {topB !== topS && <span style={styles.flipBadge}> FLIPPED</span>}
      </div>
      <div style={styles.detailReadout}>
        Feature activation: <code>{baselineAct.toFixed(2)}</code> → <code>{steeredAct.toFixed(2)}</code>
      </div>
    </div>
  )
}


function DetailBars({ title, dist, top }) {
  const max = Math.max(...BASES.map((b) => dist[b]))
  return (
    <div style={styles.detailCol}>
      <div style={styles.detailColTitle}>{title}</div>
      {BASES.map((b) => (
        <div key={b} style={styles.detailBarRow}>
          <span style={{ ...styles.detailBarLabel, color: BASE_COLORS[b] }}>{b}</span>
          <div style={styles.detailBarTrack}>
            <div
              style={{
                width: `${(dist[b] / max) * 100}%`,
                height: '100%',
                background: BASE_COLORS[b],
                opacity: b === top ? 1 : 0.55,
                borderRadius: '2px',
              }}
            />
          </div>
          <span style={{ ...styles.detailBarVal, fontWeight: b === top ? 700 : 400 }}>
            {(dist[b] * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  )
}


function SummaryStats({ summary, narrative, clamp }) {
  if (!summary) return null
  return (
    <div style={styles.summary}>
      <div style={styles.summaryRow}>
        <span style={styles.summaryLabel}>Top-base flipped positions:</span>
        <span style={styles.summaryValue}>{summary.nFlipped}</span>
      </div>
      <div style={styles.summaryRow}>
        <span style={styles.summaryLabel}>Mean Δ-entropy across sequence:</span>
        <span style={styles.summaryValue}>{summary.meanEntropyDelta.toFixed(3)} bits</span>
      </div>
      <div style={styles.summaryRow}>
        <span style={styles.summaryLabel}>Slider:</span>
        <span style={styles.summaryValue}>clamp = {clamp.toFixed(1)}</span>
      </div>
      {narrative && (
        <div style={styles.narrative}>
          <b>Note:</b> {narrative}
        </div>
      )}
    </div>
  )
}


const styles = {
  container: { fontFamily: 'system-ui, sans-serif', color: 'var(--text, #222)' },
  banner: {
    background: '#fff3cd', border: '1px solid #ffeeba', color: '#856404',
    padding: '8px 14px', borderRadius: '4px', fontSize: '11px', marginBottom: '12px',
  },
  controls: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '12px 16px', marginBottom: '14px',
    position: 'sticky', top: 0, zIndex: 10,
  },
  controlRow: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px', flexWrap: 'wrap' },
  controlLabel: { fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary, #555)', minWidth: '90px' },
  seedButtons: { display: 'flex', gap: '6px', flexWrap: 'wrap' },
  futureHint: {
    fontSize: '10px',
    fontStyle: 'italic',
    color: 'var(--text-muted, #aaa)',
    marginLeft: '8px',
  },
  seedBtn: {
    padding: '4px 10px', border: '1px solid var(--border, #ddd)', background: '#fff',
    borderRadius: '4px', cursor: 'pointer', fontSize: '11px', color: 'var(--text-secondary, #555)',
  },
  seedBtnActive: {
    padding: '4px 10px', border: '1px solid var(--accent, #76b900)',
    background: 'var(--bg-card-expanded, #f0f8e8)', borderRadius: '4px', cursor: 'pointer',
    fontSize: '11px', color: 'var(--accent, #76b900)', fontWeight: 600,
  },
  featureSelect: {
    padding: '4px 8px', fontSize: '12px', border: '1px solid var(--border, #ddd)',
    borderRadius: '4px', background: '#fff', minWidth: '260px',
  },
  sliderWrap: { display: 'flex', alignItems: 'center', gap: '10px', flex: 1 },
  slider: { flex: 1, maxWidth: '500px' },
  sliderTicks: { display: 'flex', gap: '12px', fontSize: '10px', color: 'var(--text-muted, #888)' },
  tick: { cursor: 'pointer', userSelect: 'none' },
  clampValue: { fontFamily: 'monospace', fontSize: '12px', fontWeight: 600, minWidth: '60px' },
  explanation: {
    marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--border-light, #eee)',
    fontSize: '11px', color: 'var(--text-secondary, #666)', lineHeight: '1.4',
  },
  heatmapPanel: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '14px', marginBottom: '14px', overflowX: 'auto',
  },
  strip: { marginBottom: '14px' },
  stripLabel: {
    fontSize: '11px', fontWeight: 700, textTransform: 'uppercase',
    color: 'var(--text-tertiary, #888)', marginBottom: '6px',
  },
  row: { display: 'flex', alignItems: 'flex-end', marginBottom: '10px', gap: '6px' },
  rowIndex: {
    fontFamily: 'monospace', color: 'var(--text-muted, #888)',
    fontSize: '10px', minWidth: '32px', textAlign: 'right',
  },
  bars: { display: 'flex', alignItems: 'flex-end' },
  cell: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    width: `${CELL_WIDTH}px`, position: 'relative', cursor: 'pointer',
  },
  barStack: {
    display: 'flex', flexDirection: 'column-reverse', height: `${BAR_HEIGHT}px`,
    width: `${CELL_WIDTH - 2}px`, marginBottom: '2px',
    background: '#f5f5f5', borderRadius: '1px',
  },
  baseLetter: { fontFamily: 'monospace', fontSize: '11px', fontWeight: 700 },
  flippedMarker: {
    position: 'absolute', bottom: '-9px', fontSize: '8px', color: '#c34', fontWeight: 700,
  },
  detailPanel: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--accent, #76b900)',
    borderLeft: '3px solid var(--accent, #76b900)', borderRadius: '6px',
    padding: '12px 14px', marginBottom: '14px',
  },
  detailHeader: { fontSize: '13px', fontWeight: 600, marginBottom: '8px' },
  detailColumns: { display: 'flex', gap: '16px', marginBottom: '8px' },
  detailCol: { flex: 1, fontSize: '11px' },
  detailColTitle: {
    fontSize: '10px', textTransform: 'uppercase', fontWeight: 600,
    color: 'var(--text-tertiary, #888)', marginBottom: '4px',
  },
  detailBarRow: {
    display: 'grid', gridTemplateColumns: '14px 1fr 36px',
    gap: '6px', alignItems: 'center', marginBottom: '3px',
  },
  detailBarLabel: { fontFamily: 'monospace', fontWeight: 700, fontSize: '11px' },
  detailBarTrack: { height: '10px', background: '#f0f0f0', borderRadius: '2px', overflow: 'hidden' },
  detailBarVal: { fontFamily: 'monospace', fontSize: '10px', textAlign: 'right' },
  detailReadout: { fontSize: '11px', color: 'var(--text-secondary, #444)', marginTop: '4px' },
  flipBadge: {
    marginLeft: '6px', background: '#fcebea', color: '#c34', padding: '1px 6px',
    borderRadius: '3px', fontSize: '9px', fontWeight: 700,
  },
  summary: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '10px 14px', fontSize: '12px',
  },
  summaryRow: { display: 'flex', justifyContent: 'space-between', marginBottom: '2px' },
  summaryLabel: { color: 'var(--text-secondary, #555)' },
  summaryValue: { fontFamily: 'monospace', fontWeight: 600 },
  narrative: {
    marginTop: '8px', padding: '6px 10px', background: '#eef6ff', border: '1px solid #bcd9ff',
    borderRadius: '4px', fontSize: '11px', color: '#1a3a6a',
  },
  noPair: {
    padding: '24px', textAlign: 'center', background: 'var(--bg-card-expanded, #f8f8f8)',
    border: '1px dashed var(--border, #ddd)', borderRadius: '6px', fontSize: '12px',
    color: 'var(--text-muted, #888)',
  },
  loading: { padding: '40px', textAlign: 'center', color: 'var(--text-muted, #aaa)', fontStyle: 'italic' },
  error: { padding: '20px', background: '#fee', color: '#c34', borderRadius: '4px', fontSize: '12px', fontFamily: 'monospace' },
}
