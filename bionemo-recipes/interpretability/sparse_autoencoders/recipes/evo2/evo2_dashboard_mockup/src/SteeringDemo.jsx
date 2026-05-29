import React, { useEffect, useMemo, useState } from 'react'

// Position-targeted steering demo. Click a position in the sequence, pick a
// feature, drag the clamp. Two side-by-side P(base) bar charts compare
// baseline vs steered.

// Evo2 is DNA-tokenized; the model emits P(A/C/G/T) on every input, including
// rRNA contexts. We display T everywhere — never U — to match what the model
// actually predicts.
const BASES = ['A', 'C', 'G', 'T']
const COLORS = {
  A: '#59A14F', C: '#4E79A7', G: '#F28E2B', T: '#E15759',
  accent: '#76b900',
  pass: '#5a9c3f',
  fail: '#c34a4a',
  baseline: '#9C755F',
  steered: '#76B7B2',
}

// UI clamp axis is a multiplier on the feature's natural peak activation:
// 0× = baseline (no intervention), 1× = the highest value we'd see naturally
// in the training data (in-distribution upper bound), 2× = pushed past what
// the model has been trained on (out-of-distribution).
// Underlying JSON stores discrete shifts at {-2, 0, 2, 5}; the UI -0.5×/0×/1×/2×
// map to those four points so we can interpolate without re-mocking the data.
const UI_CLAMP_POINTS = [-0.5, 0, 1, 2]
const JSON_CLAMP_KEYS = ['-2', '0', '2', '5']
const UI_CLAMP_MIN = -0.5
const UI_CLAMP_MAX = 2
const OOD_THRESHOLD = 1.0   // anything past 1× is out-of-distribution

const BASES_PER_LINE = 60


function interpProbs(low, high, t) {
  const out = {}
  let s = 0
  for (const b of Object.keys(low)) {
    out[b] = low[b] + (high[b] - low[b]) * t
    s += out[b]
  }
  // renormalize (interp can drift from 1.0)
  for (const b of Object.keys(out)) out[b] /= s
  return out
}


// Map a UI clamp value (e.g. 0.65 = 0.65× natural peak) to two adjacent JSON
// clamp keys and an interpolation factor t in [0,1] between them.
function pickSegment(uiValue) {
  if (uiValue <= UI_CLAMP_POINTS[0]) return { lo: JSON_CLAMP_KEYS[0], hi: JSON_CLAMP_KEYS[0], t: 0 }
  if (uiValue >= UI_CLAMP_POINTS[UI_CLAMP_POINTS.length - 1]) {
    const last = JSON_CLAMP_KEYS[JSON_CLAMP_KEYS.length - 1]
    return { lo: last, hi: last, t: 0 }
  }
  for (let i = 0; i < UI_CLAMP_POINTS.length - 1; i++) {
    if (uiValue >= UI_CLAMP_POINTS[i] && uiValue <= UI_CLAMP_POINTS[i + 1]) {
      const t = (uiValue - UI_CLAMP_POINTS[i]) / (UI_CLAMP_POINTS[i + 1] - UI_CLAMP_POINTS[i])
      return { lo: JSON_CLAMP_KEYS[i], hi: JSON_CLAMP_KEYS[i + 1], t }
    }
  }
  return { lo: JSON_CLAMP_KEYS[0], hi: JSON_CLAMP_KEYS[0], t: 0 }
}


const NARRATIVES = {
  headline_amr:
    "Matches the known A1408G aminoglycoside-resistance mutation in E. coli 16S rRNA. The model learned this association without supervision; steering the kanamycin-resistance SAE feature reproduces the resistance mutation.",
  tata_demo:
    "Steering the TATA-box feature concentrates probability at A — the canonical first base of the TATAAA consensus.",
  structural_demo:
    "Amplifying the α-helix feature in a coding region biases the predicted base toward G — consistent with codons encoding helix-favoring amino acids.",
  null_result:
    "No meaningful shift. This is a random control with no biological context that would make any feature appropriate. A well-behaved feature shouldn't shift predictions where it has no reason to fire.",
}


// Pick the default pair to surface for a given seed.
function defaultPairForSeed(seedId, data) {
  const prefix = `${seedId}__`
  return Object.keys(data.comparisons).find((k) => k.startsWith(prefix))
}


export default function SteeringDemo() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [seedId, setSeedId] = useState('ecoli_16s')
  // Multi-feature clamping: clamp 1..N features simultaneously, all at the
  // same slider value. With one selected the page behaves as before; with
  // multiple, the per-feature steered-minus-baseline deltas sum and we
  // renormalize. Cheap mock; the real backend would compute a joint forward.
  const [featureIds, setFeatureIds] = useState([12])
  const [clamp, setClamp] = useState(1) // start at the in-distribution upper bound (1× natural peak)
  const [mode, setMode] = useState('position') // 'position' = targeted; 'global' = clamp every position
  const [neighbors, setNeighbors] = useState(1)
  const [targetPos, setTargetPos] = useState(null)

  useEffect(() => {
    fetch('/steering_data.json').then((r) => r.json()).then(setData).catch((e) => setError(e.message))
  }, [])

  // Whenever data lands or the seed changes, find the matching pair (if any)
  // and align the feature + target position to it.
  useEffect(() => {
    if (!data) return
    const pairKey = defaultPairForSeed(seedId, data)
    if (pairKey) {
      const cmp = data.comparisons[pairKey]
      setFeatureIds([cmp.feature_id])
      setTargetPos(cmp.target_position)
    } else {
      setTargetPos(data.seeds[seedId].default_target_position)
    }
  }, [data, seedId])

  // Find the seed's default comparison for narrative + baseline. The
  // "primary" comparison is the one matching the first selected feature
  // (if it exists), else the first comparison for this seed.
  const primaryComparison = useMemo(() => {
    if (!data) return null
    const primaryFid = featureIds[0]
    const exactKey = Object.keys(data.comparisons).find((k) => {
      const c = data.comparisons[k]
      return c.seed === seedId && c.feature_id === primaryFid && c.target_position === targetPos
    })
    if (exactKey) return data.comparisons[exactKey]
    const seedKey = defaultPairForSeed(seedId, data)
    return seedKey ? data.comparisons[seedKey] : null
  }, [data, seedId, featureIds, targetPos])

  // Per-feature steered distributions at the current clamp value, additively
  // combined into a single steered distribution. With one feature it's just
  // that feature's steered probs; with more, sum (steered_f - baseline) over
  // f and add to baseline, then renormalize.
  //
  // Global mode short-circuits: clamping every position smears the output, so
  // we return a fixed low-confidence distribution with no clean winner — that
  // pattern is the visible point of contrast with position-targeted steering.
  const interpolated = useMemo(() => {
    if (!data || !primaryComparison) return null
    const { lo, hi, t } = pickSegment(clamp)
    const baseline = primaryComparison.results_by_clamp['0']?.baseline
                  || primaryComparison.results_by_clamp[lo].baseline
    if (!baseline) return null

    if (mode === 'global') {
      // Clamping every position smears toward a low-confidence distribution.
      // We interpolate from baseline (at clamp 0) to a target smear at the
      // extreme |clamp| = 2, so the slider still has visible effect — just
      // never a clean winner.
      const SMEAR = { A: 0.31, G: 0.34, C: 0.19, T: 0.16 }
      const t = Math.min(1, Math.abs(clamp) / UI_CLAMP_MAX)
      const out = {}
      let z = 0
      for (const b of BASES) {
        out[b] = (1 - t) * (baseline[b] ?? 0) + t * SMEAR[b]
        z += out[b]
      }
      for (const b of BASES) out[b] /= z
      return { baseline, steered: out }
    }

    // Find one comparison per selected feature (matching seed, falling back
    // to any pair using that feature on this seed).
    const perFeatureSteered = featureIds.map((fid) => {
      const k = Object.keys(data.comparisons).find((key) => {
        const c = data.comparisons[key]
        return c.seed === seedId && c.feature_id === fid
      })
      if (!k) return null
      const c = data.comparisons[k]
      const loSet = c.results_by_clamp[lo]
      const hiSet = c.results_by_clamp[hi]
      if (!loSet || !hiSet) return null
      return interpProbs(loSet.steered, hiSet.steered, t)
    }).filter(Boolean)

    if (perFeatureSteered.length === 0) return null
    if (perFeatureSteered.length === 1) return { baseline, steered: perFeatureSteered[0] }

    // Multi-feature combine: sum the deltas from baseline, add to baseline, renormalize.
    const combined = { ...baseline }
    for (const b of BASES) {
      let d = 0
      for (const s of perFeatureSteered) {
        d += (s[b] ?? 0) - (baseline[b] ?? 0)
      }
      combined[b] = Math.max(1e-6, (baseline[b] ?? 0) + d)
    }
    const z = BASES.reduce((acc, b) => acc + combined[b], 0)
    for (const b of BASES) combined[b] /= z
    return { baseline, steered: combined }
  }, [data, seedId, featureIds, primaryComparison, clamp])

  const comparison = primaryComparison // used downstream for the no-data fallback message

  if (error) return <div style={styles.error}>Failed to load steering_data.json: {error}</div>
  if (!data) return <div style={styles.loading}>Loading steering demo…</div>

  const seed = data.seeds[seedId]

  return (
    <div style={styles.container}>
      <div style={styles.banner}>
        MOCKUP — hand-rolled probability distributions per (seed, feature, clamp). Position-targeted
        steering protocol.
      </div>

      <Controls
        data={data}
        seedId={seedId}
        setSeedId={setSeedId}
        featureIds={featureIds}
        setFeatureIds={setFeatureIds}
        clamp={clamp}
        setClamp={setClamp}
        neighbors={neighbors}
        setNeighbors={setNeighbors}
        mode={mode}
        setMode={setMode}
      />

      <SequenceTarget
        seed={seed}
        targetPos={targetPos}
        setTargetPos={setTargetPos}
        neighbors={neighbors}
      />

      {mode === 'global' && comparison && (
        <SequenceStrip
          seed={seed}
          seedId={seedId}
          featureIds={featureIds}
          clamp={clamp}
        />
      )}

      {comparison && interpolated ? (
        <BarComparison
          targetPos={targetPos}
          baseline={interpolated.baseline}
          steered={interpolated.steered}
          mode={mode}
        />
      ) : (
        <div style={styles.empty}>
          No demo data for this combination. Pick a seed; the feature + target position will snap
          to the demo pair available for that seed.
        </div>
      )}
    </div>
  )
}


function Controls({ data, seedId, setSeedId, featureIds, setFeatureIds, clamp, setClamp, neighbors, setNeighbors, mode, setMode }) {
  const primaryFid = featureIds[0]
  const additionalFids = featureIds.slice(1)

  const setPrimary = (fid) => {
    // Move new primary to front; drop it from additional if present.
    const remaining = featureIds.filter((x) => x !== fid)
    setFeatureIds([fid, ...remaining])
  }
  const toggleAdditional = (fid) => {
    if (additionalFids.includes(fid)) {
      setFeatureIds([primaryFid, ...additionalFids.filter((x) => x !== fid)])
    } else {
      setFeatureIds([primaryFid, ...additionalFids, fid])
    }
  }

  return (
    <div style={styles.controls}>
      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Sequence:</label>
        <select value={seedId} onChange={(e) => setSeedId(e.target.value)} style={styles.select}>
          {Object.entries(data.seeds).map(([id, s]) => (
            <option key={id} value={id}>{s.name}</option>
          ))}
        </select>
      </div>

      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Feature to steer:</label>
        <select
          value={primaryFid}
          onChange={(e) => setPrimary(parseInt(e.target.value, 10))}
          style={styles.select}
        >
          {data.features_available.map((f) => (
            <option key={f.id} value={f.id}>
              {f.label}{f.is_amr ? ' (AMR)' : ''}
            </option>
          ))}
        </select>
      </div>

      {data.features_available.length > 1 && (
        <div style={styles.controlRow}>
          <label style={styles.controlLabel}>Also clamp:</label>
          <div style={styles.featureChips}>
            {data.features_available
              .filter((f) => f.id !== primaryFid)
              .map((f) => {
                const active = additionalFids.includes(f.id)
                return (
                  <label key={f.id} style={active ? styles.checkLabelActive : styles.checkLabel}>
                    <input
                      type="checkbox"
                      checked={active}
                      onChange={() => toggleAdditional(f.id)}
                      style={styles.checkbox}
                    />
                    {f.label}{f.is_amr ? ' (AMR)' : ''}
                  </label>
                )
              })}
          </div>
        </div>
      )}
      <div style={styles.multiHint}>
        {featureIds.length === 1
          ? 'Clamping 1 feature.'
          : `Clamping ${featureIds.length} features at the same clamp value. Per-feature shifts add and renormalize (mock).`}
      </div>

      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Clamp:</label>
        <div style={styles.sliderColumn}>
          <input
            type="range"
            min={UI_CLAMP_MIN}
            max={UI_CLAMP_MAX}
            step={0.05}
            value={clamp}
            onChange={(e) => setClamp(parseFloat(e.target.value))}
            style={styles.slider}
          />
          <div style={styles.sliderTicks}>
            <span onClick={() => setClamp(-0.5)} style={styles.tick}>−0.5× suppress</span>
            <span onClick={() => setClamp(0)}    style={styles.tick}>0× baseline</span>
            <span onClick={() => setClamp(1)}    style={{ ...styles.tick, fontWeight: 600 }}>1× natural peak</span>
            <span onClick={() => setClamp(2)}    style={styles.tick}>2× OOD</span>
          </div>
        </div>
        <span style={styles.clampValue}>
          = {clamp.toFixed(2)}×{clamp > OOD_THRESHOLD ? ' (OOD)' : ''}
        </span>
      </div>

      <div style={styles.controlRow}>
        <label style={styles.controlLabel}>Steering mode:</label>
        {[
          { id: 'position', label: 'Position-restricted' },
          { id: 'global',   label: 'Global (all positions)' },
        ].map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            style={mode === m.id ? styles.modeBtnActive : styles.modeBtn}
            title={
              m.id === 'position'
                ? 'Clamp the target position (and its upstream neighbors) only — surgical intervention.'
                : 'Clamp every position in the sequence. Smears the output; no clean winner — useful as a contrast.'
            }
          >
            {m.label}
          </button>
        ))}
      </div>

      <div style={styles.controlRow}>
        <label style={{ ...styles.controlLabel, opacity: mode === 'global' ? 0.4 : 1 }}>
          Neighbors clamped:
          <span
            title="Number of bp upstream of the target that are also clamped. Clamping only the target alone often doesn't take; 1-4 prior positions usually help."
            style={styles.infoIcon}
          >ⓘ</span>
        </label>
        {[0, 1, 2, 3, 4].map((n) => (
          <button
            key={n}
            onClick={() => setNeighbors(n)}
            disabled={mode === 'global'}
            style={{
              ...(neighbors === n ? styles.neighborBtnActive : styles.neighborBtn),
              opacity: mode === 'global' ? 0.4 : 1,
              cursor: mode === 'global' ? 'not-allowed' : 'pointer',
            }}
          >
            {n}
          </button>
        ))}
      </div>
    </div>
  )
}


// Three-segment colored bar above the slider that visually marks the zones:
// suppress (left of 0×), in-distribution (0× to 1×), OOD (past 1×). Widths are
// proportional to the actual UI range so the bar lines up under the slider thumb.
function ClampZoneBar() {
  const total = UI_CLAMP_MAX - UI_CLAMP_MIN
  const w = (a, b) => `${((b - a) / total) * 100}%`
  return (
    <div style={styles.zoneBar} title="Suppress / in-distribution / out-of-distribution zones">
      <div style={{ ...styles.zone, width: w(UI_CLAMP_MIN, 0),                background: '#fff3cd', color: '#856404' }}>
        suppress
      </div>
      <div style={{ ...styles.zone, width: w(0, OOD_THRESHOLD),              background: '#e8f5e9', color: '#2e7d32' }}>
        in-distribution
      </div>
      <div style={{ ...styles.zone, width: w(OOD_THRESHOLD, UI_CLAMP_MAX),   background: '#fcebea', color: '#a13', borderRight: 'none' }}>
        out-of-distribution
      </div>
    </div>
  )
}


function SequenceTarget({ seed, targetPos, setTargetPos, neighbors }) {
  const seq = seed.sequence
  const lines = []
  for (let start = 0; start < seq.length; start += BASES_PER_LINE) {
    lines.push({ start, end: Math.min(start + BASES_PER_LINE, seq.length) })
  }
  // Neighbor positions: `neighbors` bases immediately before the target.
  const neighborSet = new Set()
  if (targetPos != null) {
    for (let k = 1; k <= neighbors; k++) {
      const p = targetPos - k
      if (p >= 0) neighborSet.add(p)
    }
  }
  return (
    <div style={styles.seqPanel}>
      <div style={styles.seqHeader}>
        Click a position to target it. Currently targeting <b>position {targetPos != null ? targetPos + 1 : '—'}</b>.
      </div>
      <div style={styles.seqBody}>
        {lines.map(({ start, end }) => (
          <div key={start} style={styles.seqLine}>
            <span style={styles.seqIndex}>{String(start + 1).padStart(4, ' ')}</span>
            <span style={styles.seqBases}>
              {[...seq.slice(start, end)].map((base, j) => {
                const pos = start + j
                const isTarget = pos === targetPos
                const isNeighbor = neighborSet.has(pos)
                let style = styles.baseChar
                if (isTarget) style = { ...style, ...styles.baseTarget }
                else if (isNeighbor) style = { ...style, ...styles.baseNeighbor }
                return (
                  <span
                    key={pos}
                    onClick={() => setTargetPos(pos)}
                    style={style}
                    title={isNeighbor ? `Position ${pos + 1} (clamped neighbor)` : `Position ${pos + 1}`}
                  >
                    {base}
                  </span>
                )
              })}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}


// Global-mode visualization: clamping every position degrades the argmax
// sequence almost everywhere. We render the baseline argmax (~= input seq, since
// Evo2 reproduces a real bio sequence on its own context) vs a deterministically
// scrambled steered argmax. Fraction of flipped positions scales with |clamp|.
function SequenceStrip({ seed, seedId, featureIds, clamp }) {
  const SMEAR_WEIGHTS = [['A', 0.31], ['G', 0.34], ['C', 0.19], ['T', 0.16]]
  const baseChars = seed.sequence.toUpperCase().split('').filter((c) => 'ACGT'.includes(c))

  // mulberry32 PRNG so the scramble is stable for a given (seed, feature, clamp)
  function hashStr(s) {
    let h = 2166136261
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = (h * 16777619) >>> 0 }
    return h >>> 0
  }
  const mulberry32 = (a) => () => {
    a = (a + 0x6D2B79F5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const rngKey = `${seedId}|${featureIds.join(',')}|${clamp.toFixed(2)}`
  const rand = mulberry32(hashStr(rngKey))

  const flipFrac = Math.min(1, Math.abs(clamp) / UI_CLAMP_MAX)

  const steeredChars = baseChars.map((c) => {
    if (rand() >= flipFrac) return c
    // pick a new base weighted by the smear distribution
    let r = rand(), acc = 0
    for (const [b, w] of SMEAR_WEIGHTS) { acc += w; if (r < acc) return b }
    return SMEAR_WEIGHTS[SMEAR_WEIGHTS.length - 1][0]
  })

  const changed = baseChars.reduce((n, c, i) => n + (steeredChars[i] !== c ? 1 : 0), 0)
  const preserved = baseChars.length - changed

  return (
    <div style={styles.stripPanel}>
      <div style={styles.stripHeader}>
        <span style={styles.stripTitle}>Effect across all positions (global clamp)</span>
        <span style={styles.stripStat}>
          argmax preserved:{' '}
          <b>{preserved} / {baseChars.length}</b>{' '}
          ({((preserved / baseChars.length) * 100).toFixed(0)}%)
        </span>
      </div>

      <div style={styles.stripRow}>
        <span style={styles.stripLabel}>Baseline</span>
        <span style={styles.stripSeq}>
          {baseChars.map((c, i) => (
            <span key={i} style={{ ...styles.stripCell, color: COLORS[c] }}>{c}</span>
          ))}
        </span>
      </div>
      <div style={styles.stripRow}>
        <span style={styles.stripLabel}>Steered</span>
        <span style={styles.stripSeq}>
          {steeredChars.map((c, i) => {
            const flipped = c !== baseChars[i]
            return (
              <span
                key={i}
                style={{
                  ...styles.stripCell,
                  color: flipped ? '#fff' : COLORS[c],
                  background: flipped ? COLORS.fail : 'transparent',
                }}
              >
                {c}
              </span>
            )
          })}
        </span>
      </div>

      <div style={styles.stripFooter}>
        Position-restricted steering would change exactly 1 position (or 1 + neighbors).
        Global clamping degrades the prediction nearly everywhere.
      </div>
    </div>
  )
}


function BarComparison({ targetPos, baseline, steered, mode }) {
  // top base in each
  let topB = BASES[0], topS = BASES[0]
  for (const b of BASES) {
    if (baseline[b] > baseline[topB]) topB = b
    if ((steered[b] ?? 0) > (steered[topS] ?? 0)) topS = b
  }
  const flipped = topB !== topS
  return (
    <div style={styles.barPanel}>
      <div style={styles.barTitle}>
        Predicted base distribution at position {targetPos + 1}
      </div>
      <div style={styles.barCharts}>
        <BarChart title="Baseline" dist={baseline} top={topB} />
        <BarChart title="Steered" dist={steered} top={topS} />
      </div>
      <div style={styles.barSummary}>
        {mode === 'global' ? (
          <>
            Top base:{' '}
            <b style={{ color: COLORS[topB] }}>{topB}</b> ({baseline[topB].toFixed(2)}) →{' '}
            <b style={{ color: COLORS[topS] }}>{topS}</b> ({steered[topS].toFixed(2)})
            <span style={styles.degradedBadge}>no clean flip — degraded</span>
          </>
        ) : flipped ? (
          <>
            Top base changed:{' '}
            <b style={{ color: COLORS[topB] }}>{topB}</b> ({baseline[topB].toFixed(2)}) →{' '}
            <b style={{ color: COLORS[topS] }}>{topS}</b> ({steered[topS].toFixed(2)})
            <span style={styles.flipBadge}>FLIPPED</span>
          </>
        ) : (
          <>
            Top base unchanged:{' '}
            <b style={{ color: COLORS[topB] }}>{topB}</b> ({baseline[topB].toFixed(2)}) → {topS}{' '}
            ({steered[topS].toFixed(2)})
          </>
        )}
      </div>
    </div>
  )
}


function BarChart({ title, dist, top }) {
  return (
    <div style={styles.barCard}>
      <div style={styles.barCardTitle}>{title}</div>
      {BASES.map((b) => {
        const p = dist[b] ?? 0
        const isTop = b === top
        return (
          <div key={b} style={styles.barRow}>
            <span style={{ ...styles.barBaseLabel, color: COLORS[b] }}>{b}</span>
            <div style={styles.barTrack}>
              <div
                style={{
                  ...styles.barFill,
                  width: `${p * 100}%`,
                  background: COLORS[b],
                  border: isTop ? `2px solid ${COLORS[b]}` : 'none',
                  filter: isTop ? 'none' : 'opacity(0.65)',
                }}
              />
            </div>
            <span style={{ ...styles.barProb, fontWeight: isTop ? 700 : 400 }}>{p.toFixed(2)}</span>
          </div>
        )
      })}
    </div>
  )
}


function Selectivity({ rows, clamp }) {
  return (
    <div style={styles.selPanel}>
      <div style={styles.selTitle}>Selectivity check — does only the right feature shift the prediction?</div>
      <table style={styles.selTable}>
        <thead>
          <tr style={styles.selTableHeader}>
            <th style={{ ...styles.selCell, textAlign: 'left' }}>Feature</th>
            <th style={styles.selCell}>Steered top</th>
            <th style={styles.selCell}>P(top)</th>
            <th style={styles.selCell}>Related?</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.feature_id} style={styles.selRow}>
              <td style={{ ...styles.selCell, textAlign: 'left' }}>{r.feature_label}</td>
              <td style={{ ...styles.selCell, color: COLORS[r.steered_top_base] }}>
                <b>{r.steered_top_base}</b>
              </td>
              <td style={{ ...styles.selCell, fontFamily: 'monospace' }}>{r.p_top.toFixed(2)}</td>
              <td style={styles.selCell}>
                {r.is_amr ? (
                  <span style={{ color: COLORS.pass, fontWeight: 700 }}>✓</span>
                ) : (
                  <span style={{ color: COLORS.fail }}>✗</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={styles.selFootnote}>
        Selectivity values shown at clamp = +5 (canonical headline strength); current slider at {clamp.toFixed(1)}.
      </div>
    </div>
  )
}


function Narrative({ type }) {
  const text = NARRATIVES[type]
  if (!text) return null
  return (
    <div style={styles.narrative}>
      <span style={styles.narrativeIcon}>💡</span>
      <span>{text}</span>
    </div>
  )
}


const styles = {
  container: { fontFamily: 'system-ui, sans-serif', color: 'var(--text, #222)' },
  banner: {
    background: '#fff3cd', border: '1px solid #ffeeba', color: '#856404',
    padding: '6px 12px', borderRadius: '4px', fontSize: '11px', marginBottom: '12px',
  },
  controls: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '10px 14px', marginBottom: '12px',
    position: 'sticky', top: 0, zIndex: 10,
  },
  controlRow: { display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '6px', flexWrap: 'wrap' },
  controlLabel: { fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary, #555)', minWidth: '120px' },
  select: { padding: '4px 8px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--border, #ddd)', background: '#fff', minWidth: '260px' },
  featureChips: { display: 'flex', flexWrap: 'wrap', gap: '10px' },
  checkLabel: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    padding: '3px 8px',
    border: '1px solid var(--border, #ddd)',
    background: '#fff',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    color: 'var(--text-secondary, #555)',
    userSelect: 'none',
  },
  checkLabelActive: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    padding: '3px 8px',
    border: '1px solid var(--accent, #76b900)',
    background: 'var(--bg-card-expanded, #f0f8e8)',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    color: 'var(--accent, #76b900)',
    fontWeight: 600,
    userSelect: 'none',
  },
  checkbox: { margin: 0, cursor: 'pointer' },
  multiHint: {
    marginLeft: '120px',
    fontSize: '10px',
    fontStyle: 'italic',
    color: 'var(--text-muted, #888)',
    marginBottom: '8px',
  },
  sliderColumn: { display: 'flex', flexDirection: 'column', flex: 1, maxWidth: '520px', gap: '4px' },
  slider: { width: '100%' },
  sliderTicks: { display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted, #888)' },
  tick: { cursor: 'pointer', userSelect: 'none' },
  clampValue: { fontFamily: 'monospace', fontSize: '12px', fontWeight: 600, minWidth: '80px', whiteSpace: 'nowrap' },
  zoneBar: {
    display: 'flex',
    height: '14px',
    border: '1px solid var(--border, #ddd)',
    borderRadius: '3px',
    overflow: 'hidden',
    fontSize: '9px',
    fontWeight: 600,
    textTransform: 'uppercase',
  },
  zone: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRight: '1px solid #fff',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
  },
  infoIcon: { marginLeft: '4px', color: 'var(--text-muted, #aaa)', cursor: 'help', fontSize: '11px' },
  neighborBtn: {
    padding: '3px 12px', border: '1px solid var(--border, #ddd)', background: '#fff',
    borderRadius: '4px', cursor: 'pointer', fontSize: '11px', fontFamily: 'monospace', color: 'var(--text-secondary, #555)',
  },
  neighborBtnActive: {
    padding: '3px 12px', border: '1px solid var(--accent, #76b900)',
    background: 'var(--bg-card-expanded, #f0f8e8)', borderRadius: '4px', cursor: 'pointer',
    fontSize: '11px', fontFamily: 'monospace', color: 'var(--accent, #76b900)', fontWeight: 700,
  },
  seqPanel: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '10px 14px', marginBottom: '12px',
  },
  seqHeader: { fontSize: '11px', color: 'var(--text-secondary, #555)', marginBottom: '8px' },
  seqBody: { fontFamily: 'monospace', fontSize: '13px', lineHeight: '1.7' },
  seqLine: { display: 'flex', gap: '8px', alignItems: 'baseline' },
  seqIndex: { color: 'var(--text-muted, #aaa)', fontSize: '11px', minWidth: '32px', textAlign: 'right' },
  seqBases: { letterSpacing: '1px' },
  baseChar: { padding: '0 1px', cursor: 'pointer', borderRadius: '2px' },
  baseTarget: {
    outline: `2px solid ${COLORS.accent}`,
    background: 'rgba(118, 185, 0, 0.18)',
    fontWeight: 700,
  },
  baseNeighbor: {
    background: 'rgba(118, 185, 0, 0.10)',
    outline: `1px dashed ${COLORS.accent}`,
  },
  barPanel: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '10px 14px', marginBottom: '12px',
  },
  barTitle: { fontSize: '12px', fontWeight: 600, color: 'var(--text-heading, #222)', marginBottom: '10px' },
  barCharts: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' },
  barCard: {
    background: 'var(--bg-card-expanded, #fafafa)',
    border: '1px solid var(--border-light, #eee)', borderRadius: '4px', padding: '10px',
  },
  barCardTitle: {
    fontSize: '10px', textTransform: 'uppercase', fontWeight: 600,
    color: 'var(--text-tertiary, #888)', marginBottom: '6px',
  },
  barRow: { display: 'grid', gridTemplateColumns: '18px 1fr 40px', gap: '6px', alignItems: 'center', marginBottom: '4px' },
  barBaseLabel: { fontFamily: 'monospace', fontWeight: 700 },
  barTrack: { height: '14px', background: '#f0f0f0', borderRadius: '3px', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: '3px' },
  barProb: { fontFamily: 'monospace', fontSize: '11px', textAlign: 'right' },
  barSummary: { marginTop: '8px', fontSize: '12px', color: 'var(--text-secondary, #444)' },
  flipBadge: {
    marginLeft: '8px', background: '#fcebea', color: '#c34', padding: '1px 6px',
    borderRadius: '3px', fontSize: '9px', fontWeight: 700,
  },
  degradedBadge: {
    marginLeft: '8px', background: '#f0f0f0', color: '#666', padding: '1px 6px',
    borderRadius: '3px', fontSize: '10px', fontStyle: 'italic',
  },
  modeBtn: {
    padding: '4px 10px', border: '1px solid var(--border, #ddd)', background: '#fff',
    borderRadius: '4px', cursor: 'pointer', fontSize: '11px', color: 'var(--text-secondary, #555)',
  },
  modeBtnActive: {
    padding: '4px 10px', border: '1px solid var(--accent, #76b900)',
    background: 'var(--bg-card-expanded, #f0f8e8)', borderRadius: '4px', cursor: 'pointer',
    fontSize: '11px', color: 'var(--accent, #76b900)', fontWeight: 600,
  },
  stripPanel: {
    border: '1px solid var(--border, #ddd)', background: 'var(--bg-card, #fff)',
    borderRadius: '6px', padding: '12px 16px', marginBottom: '12px',
  },
  stripHeader: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
    marginBottom: '8px', fontSize: '12px',
  },
  stripTitle: { fontWeight: 600, color: 'var(--text, #222)' },
  stripStat: { color: 'var(--text-secondary, #555)', fontSize: '11px' },
  stripRow: { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '2px' },
  stripLabel: {
    width: '60px', fontSize: '10px', color: 'var(--text-secondary, #666)',
    textTransform: 'uppercase', letterSpacing: '0.5px',
  },
  stripSeq: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace', fontSize: '11px',
    letterSpacing: '0', whiteSpace: 'pre-wrap', wordBreak: 'break-all', lineHeight: 1.5,
  },
  stripCell: { display: 'inline-block', width: '11px', textAlign: 'center', fontWeight: 600 },
  stripFooter: {
    marginTop: '8px', fontSize: '10px', color: 'var(--text-secondary, #888)',
    fontStyle: 'italic',
  },
  selPanel: {
    background: 'var(--bg-card, #fff)', border: '1px solid var(--border, #ddd)',
    borderRadius: '6px', padding: '10px 14px', marginBottom: '12px',
  },
  selTitle: { fontSize: '12px', fontWeight: 600, color: 'var(--text-heading, #222)', marginBottom: '8px' },
  selTable: { width: '100%', borderCollapse: 'collapse', fontSize: '12px' },
  selTableHeader: {
    fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-tertiary, #888)', fontWeight: 600,
    borderBottom: '1px solid var(--border-light, #eee)',
  },
  selRow: { borderBottom: '1px solid var(--border-light, #f5f5f5)' },
  selCell: { padding: '5px 8px', textAlign: 'center' },
  selFootnote: { marginTop: '6px', fontSize: '10px', color: 'var(--text-muted, #888)', fontStyle: 'italic' },
  narrative: {
    display: 'flex', alignItems: 'flex-start', gap: '8px',
    background: '#eef6ff', border: '1px solid #bcd9ff',
    borderRadius: '4px', padding: '8px 12px',
    fontSize: '12px', color: '#1a3a6a', lineHeight: '1.5',
  },
  narrativeIcon: { fontSize: '16px' },
  loading: { padding: '40px', textAlign: 'center', color: 'var(--text-muted, #aaa)', fontStyle: 'italic' },
  empty: {
    padding: '24px', textAlign: 'center', background: 'var(--bg-card-expanded, #f8f8f8)',
    border: '1px dashed var(--border, #ddd)', borderRadius: '6px', fontSize: '12px',
    color: 'var(--text-muted, #888)',
  },
  error: { padding: '20px', background: '#fee', color: '#c34', borderRadius: '4px', fontSize: '12px', fontFamily: 'monospace' },
}
