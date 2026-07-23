import React, { useEffect, useMemo, useState } from 'react'
import { useHealth, postJSON, getJSON, cleanDNA, workTimeout } from './backend'
import { S, BASES_PER_LINE, Heat, Legend, resolveFeatureId, FeaturePicker, OrganismField, BackendBanner, RestartEngineButton, Row, Toggle, useUserLabels } from './components'

// Sequence inspector: paste DNA -> per-base SAE activations from the live backend
// (/annotate). DNA-only display; absolute 0->max coloring (clear -> green).
// Feature selection mirrors the steering tab: by-name, multi-feature.
// Shared widgets/styles (Heat, FeaturePicker, S, …) live in components.jsx.

const DEFAULT_SEQ = 'ATGGCTGAAAAGCTGGAAGCGGCAATTGAGCAGGCTGCAGTGGCAAATCAAGCG'

// Re-export so existing importers of these from './SequenceInspector' keep working.
export { resolveFeatureId, userLabel } from './components'

export default function SequenceInspector() {
  const labelTick = useUserLabels() // re-render + refetch the catalog when a feature is renamed in any tab
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
    // Refetch on every rename (labelTick) too, so a feature named anywhere becomes searchable here.
    getJSON('/features').then(setCatalog).catch(() => {})
  }, [health.status, organismTags, labelTick])

  // Clear the "Engine restarting…" note once the worker is back (BackendBanner then shows it live again).
  useEffect(() => {
    if (health.status === 'ready') setError((e) => (typeof e === 'string' && e.startsWith('Engine restarting') ? null : e))
  }, [health.status])

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
      // One forward pass; scale the wait with sequence length (a full 8192 bp seq gets ~3.5 min,
      // a short paste still fails fast if the backend is hung). See workTimeout in backend.js.
      setResult(await postJSON('/annotate', body, { timeoutMs: workTimeout(cleaned.length, { perUnit: 25, ceilMs: 900000 }) }))
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
          <RestartEngineButton enabled={!!health.info?.restart_enabled} busy={busy}
            onRestart={() => { setBusy(false); setResult(null); setError('Engine restarting — reloading the model (~1 min)…') }} />
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
