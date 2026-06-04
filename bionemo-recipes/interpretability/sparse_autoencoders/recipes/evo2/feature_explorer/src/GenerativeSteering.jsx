import React, { useEffect, useMemo, useState } from 'react'
import { useHealth, postJSON, getJSON, cleanDNA } from './backend'
import { BackendBanner, OrganismField, FeaturePicker, resolveFeatureId, Heat, Row } from './SequenceInspector'

// Generative steering: autoregressively generate DNA from Evo2 while ADDITIVELY
// clamping one or more SAE features (picked by name) on the generated
// continuation only. Real model + real SAE via backend /generate.

const BASES_PER_LINE = 80

export default function GenerativeSteering() {
  const health = useHealth()
  const organismTags = health.info?.organism_tags

  const [catalog, setCatalog] = useState([])
  const [organism, setOrganism] = useState('Human')
  const [tag, setTag] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [rows, setRows] = useState([{ q: '', strength: 4 }])
  const [nTokens, setNTokens] = useState(120)
  const [temperature, setTemperature] = useState(1.0)

  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (health.status !== 'ready') return
    if (tag === null && organismTags) setTag(organismTags[organism] ?? '')
    if (!catalog.length) getJSON('/features').then(setCatalog).catch(() => {})
  }, [health.status, organismTags])

  const clamps = rows
    .map((r) => ({ id: resolveFeatureId(catalog, r.q), strength: Number(r.strength) }))
    .filter((c) => c.id != null)

  const generate = async () => {
    setBusy(true)
    setError(null)
    try {
      const body = {
        prompt: cleanDNA(prompt),
        organism,
        tag: tag ?? (organismTags?.[organism] ?? ''),
        features: clamps.map((c) => ({ feature_id: c.id, strength: c.strength })),
        n_tokens: Number(nTokens),
        temperature: Number(temperature),
      }
      setResult(await postJSON('/generate', body))
    } catch (e) {
      setError(String(e.message || e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  const canRun = clamps.length > 0 && health.status === 'ready' && !busy

  return (
    <div style={S.wrap}>
      <BackendBanner health={health} />
      <Formula />

      <div style={S.card}>
        <OrganismField {...{ organismTags, organism, setOrganism, tag, setTag }} />

        <Row label="Prompt (seed):">
          <div style={{ flex: 1 }}>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={2} style={S.textarea}
              placeholder="DNA to seed generation — leave blank to generate from the organism tag alone. Clamping is applied to what's generated AFTER this prompt." />
            <div style={S.hint}>{cleanDNA(prompt).length} bp seed · clamp applies to the generated continuation only</div>
          </div>
        </Row>

        <Row label="Clamp features:">
          <FeaturePicker catalog={catalog} rows={rows} setRows={setRows} withStrength={true} />
        </Row>

        <Row label="Temperature:">
          <input type="range" min={0} max={2} step={0.05} value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))} style={{ width: '220px' }} />
          <span style={S.mono}>{Number(temperature).toFixed(2)}</span>
          <span style={S.help}>{temperature == 0 ? 'greedy (argmax)' : temperature < 0.8 ? 'conservative' : temperature > 1.2 ? 'diverse' : 'balanced'}</span>
        </Row>

        <Row label="Length:">
          <span style={S.inlineField}>tokens&nbsp;
            <input type="number" min={1} max={400} value={nTokens} onChange={(e) => setNTokens(e.target.value)} style={S.num} />
          </span>
        </Row>

        <div style={S.actions}>
          <button onClick={generate} disabled={!canRun} style={{ ...S.primary, opacity: canRun ? 1 : 0.5 }}>
            {busy ? 'Generating…' : `Generate (clamp ${clamps.length} feature${clamps.length === 1 ? '' : 's'})`}
          </button>
          {health.status !== 'ready' && <span style={S.down}>× backend {health.status === 'offline' ? 'down' : 'loading'}</span>}
          {error && <span style={S.down}>× {error}</span>}
        </div>
      </div>

      {!result ? (
        <div style={S.empty}>Pick one or more features, set their clamp values, and click <b>Generate</b> to compare an unsteered vs feature-steered Evo2 sample.</div>
      ) : (
        <SteerResult result={result} />
      )}
    </div>
  )
}

function Formula() {
  return (
    <div style={S.formula}>
      <div style={S.formulaTitle}>Additive steering (feature clamp) — applied to the residual stream at layer 19, generated positions only:</div>
      <div style={S.formulaEq}>h ← h + Σ<sub>f</sub> ( t<sub>f</sub> − a<sub>f</sub>(h) ) · d<sub>f</sub></div>
      <div style={S.formulaLegend}>
        <span><b>h</b> = base-model hidden state</span>
        <span><b>a<sub>f</sub></b> = relu((h − b<sub>pre</sub>)·W<sub>enc</sub>[f] + b<sub>f</sub>) current activation</span>
        <span><b>d<sub>f</sub></b> = SAE decoder column for feature f</span>
        <span><b>t<sub>f</sub></b> = the activation you clamp feature f to</span>
      </div>
    </div>
  )
}

function SteerResult({ result }) {
  const feats = result.features
  const maxByFid = {}
  for (const f of feats) {
    const b = result.baseline.activations[f.id] || []
    const s = result.steered.activations[f.id] || []
    maxByFid[f.id] = Math.max(0, ...b, ...s)
  }
  const mean = (a) => (a && a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div style={S.resultMeta}>
        Clamped {feats.length} feature{feats.length === 1 ? '' : 's'} on the generated continuation ({result.organism}).
        Mean activation baseline → steered:&nbsp;
        {feats.map((f) => (
          <span key={f.id} style={{ marginRight: '12px' }}>
            <b>#{f.id} {f.label}</b> {mean(result.baseline.activations[f.id]).toFixed(3)} →{' '}
            <b style={{ color: 'var(--accent)' }}>{mean(result.steered.activations[f.id]).toFixed(3)}</b> (clamp {f.strength})
          </span>
        ))}
      </div>
      <SteerBlock title="Baseline (no steering)" seq={result.baseline.sequence} activations={result.baseline.activations} feats={feats} maxByFid={maxByFid} />
      <SteerBlock title="Steered" seq={result.steered.sequence} activations={result.steered.activations} feats={feats} maxByFid={maxByFid} />
    </div>
  )
}

function SteerBlock({ title, seq, activations, feats, maxByFid }) {
  const bases = [...seq]
  const lines = []
  for (let i = 0; i < bases.length; i += BASES_PER_LINE) lines.push(i)
  const gc = bases.length ? (bases.filter((b) => b === 'G' || b === 'C').length / bases.length) * 100 : 0
  return (
    <div style={S.block}>
      <div style={S.blockHead}><span style={S.blockTitle}>{title}</span><span style={S.blockMeta}>{bases.length} bp · GC {gc.toFixed(0)}%</span></div>
      {feats.map((f) => (
        <div key={f.id} style={{ marginBottom: '8px' }}>
          <div style={S.trackLabel}>#{f.id} {f.label}</div>
          <Heat bases={bases} acts={activations[f.id] || []} max={maxByFid[f.id]} lines={lines} />
        </div>
      ))}
    </div>
  )
}

const S = {
  wrap: { padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '14px', maxWidth: '1200px', margin: '0 auto' },
  formula: { background: 'var(--bg-card-expanded)', border: '1px solid var(--border)', borderRadius: '8px', padding: '10px 14px' },
  formulaTitle: { fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px' },
  formulaEq: { fontFamily: 'ui-monospace, Menlo, monospace', fontSize: '15px', color: 'var(--text-heading)', marginBottom: '6px' },
  formulaLegend: { display: 'flex', flexWrap: 'wrap', gap: '14px', fontSize: '11px', color: 'var(--text-muted)' },
  card: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: '8px', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: '10px' },
  textarea: { width: '100%', fontFamily: 'monospace', fontSize: '12px', padding: '8px', border: '1px solid var(--border-input)', borderRadius: '6px', background: 'var(--bg-input)', color: 'var(--text)', boxSizing: 'border-box', resize: 'vertical' },
  hint: { fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' },
  mono: { fontFamily: 'monospace', fontSize: '12px', fontWeight: 600, minWidth: '42px' },
  help: { fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' },
  inlineField: { fontSize: '12px', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center' },
  num: { width: '64px', padding: '4px 6px', fontSize: '12px', borderRadius: '4px', border: '1px solid var(--border-input)', background: 'var(--bg-input)', color: 'var(--text)' },
  actions: { display: 'flex', alignItems: 'center', gap: '12px', marginTop: '4px' },
  primary: { padding: '7px 16px', border: '1px solid var(--accent)', background: 'var(--accent)', color: '#000', borderRadius: '5px', cursor: 'pointer', fontSize: '12px', fontWeight: 700 },
  down: { color: '#d9534f', fontSize: '12px' },
  empty: { padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic', border: '1px dashed var(--border)', borderRadius: '8px' },
  resultMeta: { fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.6 },
  block: { background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: '8px', padding: '10px 14px' },
  blockHead: { display: 'flex', alignItems: 'baseline', gap: '10px', marginBottom: '8px' },
  blockTitle: { fontSize: '13px', fontWeight: 600, color: 'var(--text-heading)' },
  blockMeta: { marginLeft: 'auto', fontFamily: 'monospace', fontSize: '11px', color: 'var(--text-secondary)' },
  trackLabel: { fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace', marginBottom: '2px' },
}
