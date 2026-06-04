import React, { useEffect, useMemo, useState } from 'react'
import { useHealth, postJSON, getJSON, activationColor, legendGradient, cleanDNA } from './backend'

// Sequence inspector: paste DNA -> per-base SAE activations from the live backend
// (/annotate). DNA-only display; absolute 0->max coloring (clear -> green).
// Feature selection mirrors the steering tab: by-name, multi-feature.

const DEFAULT_SEQ = 'ATGGCTGAAAAGCTGGAAGCGGCAATTGAGCAGGCTGCAGTGGCAAATCAAGCG'
const BASES_PER_LINE = 80

export default function SequenceInspector() {
  const health = useHealth()
  const organismTags = health.info?.organism_tags
  const [catalog, setCatalog] = useState([])

  const [sequence, setSequence] = useState(DEFAULT_SEQ)
  const [organism, setOrganism] = useState('Human')
  const [tag, setTag] = useState(null) // editable phylo tag; null until prefilled from health
  const [mode, setMode] = useState('topk') // 'topk' | 'pick'
  const [k, setK] = useState(8)
  const [pickRows, setPickRows] = useState([{ q: '' }])

  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (health.status !== 'ready') return
    if (tag === null && organismTags) setTag(organismTags[organism] ?? '')
    if (!catalog.length) getJSON('/features').then(setCatalog).catch(() => {})
  }, [health.status, organismTags])

  const cleaned = useMemo(() => cleanDNA(sequence), [sequence])

  const annotate = async () => {
    setBusy(true)
    setError(null)
    try {
      const body = { sequence: cleaned, organism, tag: tag ?? (organismTags?.[organism] ?? ''), mode, k: Number(k) }
      if (mode === 'pick') {
        body.feature_ids = pickRows.map((r) => resolveFeatureId(catalog, r.q)).filter((x) => x != null)
        if (!body.feature_ids.length) throw new Error('pick at least one feature by name or #id')
      }
      setResult(await postJSON('/annotate', body))
    } catch (e) {
      setError(String(e.message || e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={S.wrap}>
      <BackendBanner health={health} />

      <div style={S.card}>
        <Row label="DNA sequence:">
          <div style={{ flex: 1 }}>
            <textarea value={sequence} onChange={(e) => setSequence(e.target.value)} rows={3}
              style={S.textarea} placeholder="Paste DNA (FASTA okay — headers/whitespace stripped)…" />
            <div style={S.hint}>{cleaned.length} bp after cleanup</div>
          </div>
        </Row>

        <OrganismField {...{ organismTags, organism, setOrganism, tag, setTag }} />

        <Row label="Features to show:">
          <Toggle active={mode === 'topk'} onClick={() => setMode('topk')}>Top-K by max activation</Toggle>
          <Toggle active={mode === 'pick'} onClick={() => setMode('pick')}>Pick features</Toggle>
        </Row>

        {mode === 'topk' ? (
          <Row label="Top-K:">
            <span style={S.inlineField}>K&nbsp;=&nbsp;
              <input type="number" min={1} max={64} value={k} onChange={(e) => setK(e.target.value)} style={S.num} />
            </span>
            <span style={S.help}>the K features that fire hardest on this sequence</span>
          </Row>
        ) : (
          <Row label="Features:">
            <FeaturePicker catalog={catalog} rows={pickRows} setRows={setPickRows} withStrength={false} />
          </Row>
        )}

        <div style={S.actions}>
          <button onClick={annotate}
            disabled={busy || !cleaned.length || health.status !== 'ready'}
            style={{ ...S.primary, opacity: busy || !cleaned.length || health.status !== 'ready' ? 0.5 : 1 }}>
            {busy ? 'Annotating…' : 'Annotate sequence'}
          </button>
          {health.status !== 'ready' && <span style={S.down}>× backend {health.status === 'offline' ? 'down' : 'loading'}</span>}
          {error && <span style={S.down}>× {error}</span>}
        </div>
      </div>

      {!result ? (
        <div style={S.empty}>Paste a DNA sequence above and click <b>Annotate sequence</b> to see per-base SAE activations.</div>
      ) : (
        <Result result={result} />
      )}
    </div>
  )
}

function Result({ result }) {
  const tagLen = result.tag_len || 0 // DNA-only: always strip the phylo prefix
  const bases = result.bases.slice(tagLen)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={S.resultMeta}>
        {result.features.length} feature{result.features.length === 1 ? '' : 's'} · {bases.length} bases ·
        layer {result.layer} · organism {result.organism}
        {result.tag_len > 0 ? ` · phylo tag (${result.tag_len} bp) stripped` : ''}
      </div>
      <Legend label="SAE activation (Viridis)" note="each feature scaled to its own max" />
      {result.features.map((f) => <FeatureHeatmap key={f.feature_id} feature={f} tagLen={tagLen} bases={bases} />)}
    </div>
  )
}

function FeatureHeatmap({ feature, tagLen, bases }) {
  const acts = feature.activations.slice(tagLen)
  const max = acts.length ? Math.max(...acts) : 0
  const lines = []
  for (let i = 0; i < bases.length; i += BASES_PER_LINE) lines.push(i)
  return (
    <div style={S.featCard}>
      <div style={S.featHead}>
        <span style={S.featLabel}>{feature.label || `feature ${feature.feature_id}`}</span>
        <span style={S.featId}>#{feature.feature_id}</span>
        <span style={S.featMax}>max {feature.max_activation?.toFixed(2)}</span>
      </div>
      <Heat bases={bases} acts={acts} max={max} lines={lines} />
    </div>
  )
}

export function Heat({ bases, acts, max, lines }) {
  // Per-cell letter color so text stays legible across the Viridis ramp: dark
  // text on the light (high-activation) end, light text on the dark/empty end.
  const dark = typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  const empty = dark ? '#dcdcdc' : '#333'
  return (
    <div style={S.heatBody}>
      {lines.map((start) => (
        <div key={start} style={S.heatLine}>
          <span style={S.heatIdx}>{String(start + 1).padStart(5, ' ')}</span>
          <span style={S.heatSeq}>
            {bases.slice(start, start + BASES_PER_LINE).map((b, j) => {
              const idx = start + j
              const a = acts[idx] ?? 0
              const t = max > 0 ? Math.min(1, a / max) : 0
              const letter = a <= 0 || t < 0.02 ? empty : t > 0.45 ? '#0a0a0a' : '#f4f4f4'
              return (
                <span key={idx} title={`pos ${idx + 1}: ${a.toFixed(3)}`}
                  style={{ background: activationColor(a, max), color: letter }}>{b}</span>
              )
            })}
          </span>
        </div>
      ))}
    </div>
  )
}

// Viridis colorbar legend.
export function Legend({ label = 'SAE activation', note }) {
  return (
    <div style={S.legend}>
      <span style={S.legendLabel}>{label}</span>
      <span>low</span>
      <span style={{ ...S.legendBar, background: legendGradient() }} />
      <span>high</span>
      {note && <span style={S.legendNote}>{note}</span>}
    </div>
  )
}

// Resolve a picker row's text ("#123 …" or an exact label) to a feature id.
export function resolveFeatureId(catalog, q) {
  const m = String(q).match(/#?(\d+)/)
  if (m) {
    const id = Number(m[1])
    if (catalog.some((f) => f.id === id)) return id
  }
  const lab = String(q).trim()
  const hit = catalog.find((f) => f.label === lab)
  return hit ? hit.id : null
}

// Shared by-name feature picker (used by both tabs). withStrength adds a clamp value.
export function FeaturePicker({ catalog, rows, setRows, withStrength }) {
  const byId = useMemo(() => Object.fromEntries(catalog.map((f) => [f.id, f])), [catalog])
  const setRow = (i, patch) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const add = () => setRows((rs) => [...rs, withStrength ? { q: '', strength: 4 } : { q: '' }])
  const del = (i) => setRows((rs) => (rs.length > 1 ? rs.filter((_, j) => j !== i) : rs))
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {rows.map((r, i) => {
        const fid = resolveFeatureId(catalog, r.q)
        const f = fid != null ? byId[fid] : null
        return (
          <div key={i} style={S.pickRow}>
            <input list="evo2-feature-catalog" value={r.q} onChange={(e) => setRow(i, { q: e.target.value })}
              placeholder="feature name or #id…" style={S.featInput} />
            <span style={S.resolved}>{f ? `→ #${f.id} ${f.label}` : '— not resolved'}</span>
            {withStrength && (
              <span style={S.strengthWrap}>clamp to&nbsp;
                <input type="range" min={-2} max={10} step={0.5} value={r.strength}
                  onChange={(e) => setRow(i, { strength: parseFloat(e.target.value) })} style={{ width: '140px' }} />
                <input type="number" step={0.5} value={r.strength} onChange={(e) => setRow(i, { strength: e.target.value })} style={S.num} />
              </span>
            )}
            <button onClick={() => del(i)} style={S.del} title="remove">✕</button>
          </div>
        )
      })}
      <div><button onClick={add} style={S.addBtn}>+ Add feature</button></div>
      <datalist id="evo2-feature-catalog">
        {catalog.slice(0, 2000).map((f) => <option key={f.id} value={`#${f.id} ${f.label}`} />)}
      </datalist>
    </div>
  )
}

// Organism preset dropdown + an always-editable phylo tag (prefilled from the preset).
export function OrganismField({ organismTags, organism, setOrganism, tag, setTag }) {
  const names = Object.keys(organismTags || { 'None (raw DNA)': '' })
  return (
    <Row label="Organism:">
      <select value={organism} onChange={(e) => { const v = e.target.value; setOrganism(v); setTag(organismTags?.[v] ?? '') }} style={S.select}>
        {names.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
      <input value={tag ?? ''} onChange={(e) => setTag(e.target.value)} style={S.customTag}
        title="Phylogenetic tag prepended to the sequence — edit freely to use a custom lineage"
        placeholder="|d__…;s__…|  phylo tag (editable)" />
    </Row>
  )
}

export function BackendBanner({ health }) {
  if (health.status === 'ready') {
    const i = health.info || {}
    return <div style={{ ...S.banner, ...S.bannerOk }}>● Backend live — Evo2 1B layer {i.layer}, {i.n_features} SAE features ({i.n_labels} labeled) on {i.device}.</div>
  }
  if (health.status === 'loading') return <div style={{ ...S.banner, ...S.bannerWarn }}>◐ Backend loading model + SAE… (~1 min at startup)</div>
  return <div style={{ ...S.banner, ...S.bannerWarn }}>Backend offline. Start <code>steering_server.py</code> on port 8001 (EVO2_CKPT_DIR + SAE_CKPT_PATH, C13 layer 19).</div>
}

export function Row({ label, children }) {
  return <div style={S.row}><label style={S.rowLabel}>{label}</label><div style={S.rowBody}>{children}</div></div>
}
export function Toggle({ active, onClick, children }) {
  return <button onClick={onClick} style={active ? S.toggleOn : S.toggleOff}>{children}</button>
}

export const S = {
  wrap: { padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '14px', maxWidth: '1200px', margin: '0 auto' },
  banner: { padding: '8px 14px', borderRadius: '6px', fontSize: '12px' },
  bannerOk: { background: 'rgba(118,185,0,0.12)', border: '1px solid var(--accent)', color: 'var(--accent)' },
  bannerWarn: { background: 'rgba(255,193,7,0.10)', border: '1px solid #b8860b', color: '#d9a400' },
  card: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: '8px', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '10px' },
  row: { display: 'flex', alignItems: 'flex-start', gap: '14px' },
  rowLabel: { width: '120px', flexShrink: 0, fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', paddingTop: '6px' },
  rowBody: { flex: 1, display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
  textarea: { width: '100%', fontFamily: 'monospace', fontSize: '12px', padding: '8px', border: '1px solid var(--border-input)', borderRadius: '6px', background: 'var(--bg-input)', color: 'var(--text)', boxSizing: 'border-box', resize: 'vertical' },
  hint: { fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' },
  select: { padding: '5px 8px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--border-input)', background: 'var(--bg-input)', color: 'var(--text)', minWidth: '170px' },
  customTag: { flex: 1, minWidth: '320px', fontFamily: 'monospace', fontSize: '11px', padding: '5px 8px', borderRadius: '4px', border: '1px solid var(--border-input)', background: 'var(--bg-input)', color: 'var(--text)' },
  inlineField: { fontSize: '12px', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center' },
  help: { fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' },
  num: { width: '64px', padding: '4px 6px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--border-input)', background: 'var(--bg-input)', color: 'var(--text)' },
  actions: { display: 'flex', alignItems: 'center', gap: '12px', marginTop: '4px' },
  primary: { padding: '7px 16px', border: '1px solid var(--accent)', background: 'var(--accent)', color: '#000', borderRadius: '5px', cursor: 'pointer', fontSize: '12px', fontWeight: 700 },
  down: { color: '#d9534f', fontSize: '12px' },
  empty: { padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic', border: '1px dashed var(--border)', borderRadius: '8px' },
  resultMeta: { fontSize: '11px', color: 'var(--text-muted)' },
  toggleOn: { padding: '5px 12px', border: '1px solid var(--accent)', background: 'var(--bg-card-expanded)', color: 'var(--accent)', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', fontWeight: 600 },
  toggleOff: { padding: '5px 12px', border: '1px solid var(--border-input)', background: 'var(--bg-input)', color: 'var(--text-secondary)', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' },
  pickRow: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
  featInput: { width: '230px', fontSize: '12px', padding: '5px 8px', borderRadius: '4px', border: '1px solid var(--border-input)', background: 'var(--bg-input)', color: 'var(--text)' },
  resolved: { fontSize: '11px', color: 'var(--text-muted)', minWidth: '210px', fontFamily: 'monospace' },
  strengthWrap: { display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-secondary)' },
  del: { border: '1px solid var(--border-input)', background: 'transparent', color: 'var(--text-muted)', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', padding: '3px 7px' },
  addBtn: { border: '1px dashed var(--border-input)', background: 'transparent', color: 'var(--text-secondary)', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', padding: '4px 10px' },
  featCard: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: '8px', padding: '10px 14px' },
  featHead: { display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '8px' },
  featLabel: { fontSize: '13px', fontWeight: 600, color: 'var(--text-heading)' },
  featId: { fontFamily: 'monospace', fontSize: '11px', color: 'var(--text-tertiary)' },
  featMax: { marginLeft: 'auto', fontFamily: 'monospace', fontSize: '11px', color: 'var(--text-secondary)' },
  heatBody: { fontFamily: 'ui-monospace, Menlo, monospace', fontSize: '13px', lineHeight: 1.7 },
  heatLine: { display: 'flex', gap: '8px', alignItems: 'baseline' },
  heatIdx: { color: 'var(--text-muted)', fontSize: '11px', minWidth: '40px', textAlign: 'right', whiteSpace: 'pre' },
  heatSeq: { letterSpacing: '1px', wordBreak: 'break-all' },
  legend: { display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', color: 'var(--text-muted)' },
  legendLabel: { fontWeight: 600, color: 'var(--text-secondary)' },
  legendBar: { width: '160px', height: '10px', borderRadius: '3px', border: '1px solid var(--border)' },
  legendNote: { fontStyle: 'italic' },
}
