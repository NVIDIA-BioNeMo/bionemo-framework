import React, { useEffect, useMemo, useRef, useState } from 'react'
import { useHealth, postJSON, getJSON, cleanDNA, workTimeout } from './backend'
import { BackendBanner, OrganismField, FeaturePicker, RestartEngineButton, resolveFeatureId, Row, userLabel, useUserLabels } from './components'

// Render a LaTeX string with KaTeX (loaded from the CDN in index.html — no npm dep).
function Tex({ expr, block = false }) {
  const ref = useRef(null)
  useEffect(() => {
    if (ref.current && window.katex) {
      window.katex.render(expr, ref.current, { throwOnError: false, displayMode: block })
    }
  }, [expr, block])
  return <span ref={ref}>{expr}</span> // falls back to the raw string until KaTeX loads
}

// Generative steering: autoregressively generate DNA from Evo2 while ADDITIVELY
// clamping one or more SAE features (picked by name) on the generated
// continuation only. Real model + real SAE via backend /generate.

const BASES_PER_LINE = 80

export default function GenerativeSteering() {
  useUserLabels() // re-render when a feature is renamed in any tab
  const health = useHealth()
  const organismTags = health.info?.organism_tags

  const [catalog, setCatalog] = useState([])
  const [organism, setOrganism] = useState('None (raw text)')
  const [tag, setTag] = useState(null)
  // Open as a working demo: a coding-DNA seed + a known-steerable feature (#643) clamped to 0
  // (suppress), greedy decoding, baseline comparison on — so the first "Generate" visibly steers.
  const [prompt, setPrompt] = useState('The weather today is')
  const [rows, setRows] = useState([{ q: '643', strength: 0 }])
  const [nTokens, setNTokens] = useState(120)
  const [temperature, setTemperature] = useState(0)
  const [compareBaseline, setCompareBaseline] = useState(true)

  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (health.status !== 'ready') return
    if (tag === null && organismTags) setTag(organismTags[organism] ?? '')
    if (!catalog.length) getJSON('/features').then(setCatalog).catch(() => {})
  }, [health.status, organismTags])

  // Clear the "Engine restarting…" note once the worker is back (BackendBanner then shows it live again).
  useEffect(() => {
    if (health.status === 'ready') setError((e) => (typeof e === 'string' && e.startsWith('Engine restarting') ? null : e))
  }, [health.status])

  const nFeatures = health.info?.n_features
  const clamps = rows
    .map((r) => ({ id: resolveFeatureId(catalog, r.q), strength: Number(r.strength) }))
    .filter((c) => c.id != null && (nFeatures == null || (c.id >= 0 && c.id < nFeatures)))

  // Context budget: tag + prompt + generated tokens must fit the model's bp window. genMax = tokens
  // still generatable; promptTooLong = the prompt alone already overflows (Steering rejects rather
  // than clamps — so we block it client-side instead of round-tripping to a 413).
  const maxSeqLen = health.info?.max_seq_len || 8192
  const tagLen = (tag ?? organismTags?.[organism] ?? '').length
  const promptBp = cleanDNA(prompt).length
  const genMax = Math.max(1, maxSeqLen - tagLen - promptBp)
  const promptTooLong = tagLen + promptBp > maxSeqLen
  // Rough runtime estimate: decode measured at ~37 tok/s (unsteered); steering + a baseline pass make
  // it slower, so use a conservative ~30 tok/s and double for the baseline. Just a heads-up, not exact.
  const TOK_PER_S = 30
  const estSec = Math.round((Number(nTokens) || 0) * (compareBaseline ? 2 : 1) / TOK_PER_S)
  const estStr = estSec < 90 ? `~${estSec}s` : `~${(estSec / 60).toFixed(1)} min`

  const generate = async () => {
    setBusy(true)
    setError(null)
    try {
      const nTok = Math.max(1, Math.min(genMax, Number(nTokens) || 1)) // never request past the budget
      const body = {
        prompt: cleanDNA(prompt),
        organism,
        tag: tag ?? (organismTags?.[organism] ?? ''),
        features: clamps.map((c) => ({ feature_id: c.id, strength: c.strength })),
        n_tokens: nTok,
        temperature: Number(temperature),
        compare_baseline: compareBaseline,
      }
      // Autoregressive: wall-clock scales with tokens generated (x2 when also generating an unsteered
      // baseline). A 120-token sample fails fast; a full 8192-token continuation gets the minutes it needs.
      setResult(await postJSON('/generate', body, {
        // Ceiling raised to 30 min: a full 8192-token steered run with a baseline pass can take many
        // minutes, and the old 15-min default was cutting it off. perUnit stays generous headroom.
        timeoutMs: workTimeout(nTok * (compareBaseline ? 2 : 1), { perUnit: 120, baseMs: 45000, ceilMs: 3_600_000 }),
      }))
    } catch (e) {
      setError(String(e.message || e))
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  const canRun = health.status === 'ready' && !busy // clamps optional — [] = plain generation


  return (
    <div style={S.wrap}>
      <BackendBanner health={health} />
      <Formula />

      <div style={S.card}>

        <Row label="Prompt (seed):">
          <div style={{ flex: 1 }}>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={2} style={S.textarea}
              placeholder="Clamping is applied to what's generated AFTER this prompt." />
            <div style={S.hint}>{cleanDNA(prompt).length} characters · clamp applies to the generated continuation only{promptTooLong && <span style={{ color: '#ef4444' }}> · over the {maxSeqLen}-token context — shorten to generate</span>}</div>
          </div>
        </Row>

        <Row label="Clamp features:">
          <FeaturePicker catalog={catalog} rows={rows} setRows={setRows} withStrength={true} nFeatures={nFeatures} />
        </Row>

        <Row label="Temperature:">
          <input type="range" min={0} max={2} step={0.05} value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))} style={{ width: '220px' }} />
          <span style={S.mono}>{Number(temperature).toFixed(2)}</span>
          <span style={S.help}>{temperature == 0 ? 'greedy (argmax)' : temperature < 0.8 ? 'conservative' : temperature > 1.2 ? 'diverse' : 'balanced'}</span>
        </Row>

        <Row label="Length:">
          <span style={S.inlineField}>tokens&nbsp;
            <input type="number" min={1} max={genMax} value={nTokens}
              onChange={(e) => { const v = e.target.value; setNTokens(v === '' ? '' : Math.min(genMax, Math.max(1, Number(v) || 1))) }} style={S.num} />
            <span style={S.help}>up to {genMax} more tokens (prompt + generation must fit the {maxSeqLen}-token context) · est. {estStr}{compareBaseline ? ' (incl. baseline)' : ''}</span>
          </span>
        </Row>

        <Row label="Baseline:">
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={compareBaseline} onChange={(e) => setCompareBaseline(e.target.checked)} disabled={clamps.length === 0} />
            also generate an unsteered baseline to compare
          </label>
          <span style={S.help}>{clamps.length === 0 ? '(no clamp — single plain generation)' : 'off by default; doubles generation time'}</span>
        </Row>

        <div style={S.actions}>
          <button onClick={generate} disabled={!canRun || promptTooLong} style={{ ...S.primary, opacity: canRun && !promptTooLong ? 1 : 0.5 }}>
            {busy ? 'Generating…' : clamps.length ? `Generate (clamp ${clamps.length} feature${clamps.length === 1 ? '' : 's'})` : 'Generate (no clamp)'}
          </button>
          <RestartEngineButton enabled={!!health.info?.restart_enabled} busy={busy}
            onRestart={() => { setBusy(false); setResult(null); setError('Engine restarting — reloading the model (~1 min)…') }} />
          {health.status !== 'ready' && <span style={S.down}>× backend {health.status === 'offline' ? 'down' : 'loading'}</span>}
          {error && <span style={S.down}>× {error}</span>}
        </div>
      </div>

      {!result ? (
        <div style={S.empty}>Pick one or more features, set their clamp values, and click <b>Generate</b> to compare an unsteered vs feature-steered sample.</div>
      ) : (
        <SteerResult result={result} />
      )}
    </div>
  )
}

function Formula() {
  return (
    <div style={S.formula}>
      <div style={S.formulaTitle}>Additive steering (feature clamp) — applied to the residual stream at the SAE layer, generated positions only:</div>
      <div style={S.formulaEq}><Tex block expr="h \leftarrow h + \sum_f \left( t_f - a_f(h) \right)\, d_f" /></div>
      <div style={S.formulaLegend}>
        <span><Tex expr="h" /> = base-model hidden state</span>
        <span><Tex expr="a_f = \mathrm{relu}\!\left((h - b_{pre}) \cdot W_{enc}[f] + b_f\right)" /> current activation</span>
        <span><Tex expr="d_f" /> = SAE decoder column for feature f</span>
        <span><Tex expr="t_f" /> = the activation you clamp feature f to</span>
      </div>
    </div>
  )
}

function SteerResult({ result }) {
  const feats = result.features || []
  const gen = result.generation
  const base = result.baseline
  const mean = (a) => (a && a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {feats.length ? (
        <div style={S.resultMeta}>
          Clamped {feats.length} feature{feats.length === 1 ? '' : 's'} on the generated continuation.&nbsp;
          {feats.map((f) => (
            <span key={f.id} style={{ marginRight: '12px' }}>
              <b>#{f.id} {userLabel(f.id) || f.label}</b> mean <b style={{ color: 'var(--accent)' }}>{mean(gen.activations[f.id]).toFixed(3)}</b>
              {base ? ` (baseline ${mean(base.activations[f.id]).toFixed(3)})` : ''} @ clamp {f.strength}
            </span>
          ))}
        </div>
      ) : (
        <div style={S.resultMeta}>Unsteered generation · {gen.sequence.length} characters.</div>
      )}
      <SteerBlock title={feats.length ? 'Steered generation' : 'Generation'} seq={gen.sequence} />
      {base && <SteerBlock title="Baseline (unsteered)" seq={base.sequence} />}
    </div>
  )
}

// Save a generated sequence as a FASTA file (full length — no truncation).
function downloadFasta(title, seq) {
  const name = title.replace(/\s+/g, '_')
  const fa = `${name} (${seq.length} chars)\n${seq}\n`
  const url = URL.createObjectURL(new Blob([fa], { type: 'text/plain' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `${name}.fasta`
  a.click()
  URL.revokeObjectURL(url)
}

// Plain DNA readout — black monospace text on a light panel (no activation colormap).
function SteerBlock({ title, seq }) {
  const bases = [...seq]
  const lines = []
  for (let i = 0; i < bases.length; i += BASES_PER_LINE) lines.push(i)
  const gc = bases.length ? (bases.filter((b) => b === 'G' || b === 'C').length / bases.length) * 100 : 0
  return (
    <div style={S.block}>
      <div style={S.blockHead}>
        <span style={S.blockTitle}>{title}</span>
        <span style={S.blockMeta}>
          {bases.length} bp · GC {gc.toFixed(0)}%
          <button onClick={() => downloadFasta(title, seq)} style={{ marginLeft: 10, cursor: 'pointer' }}>Download FASTA</button>
        </span>
      </div>
      <div style={S.seqReadout}>
        {lines.map((start) => (
          <div key={start} style={S.seqLine}>
            <span style={S.seqIdx}>{String(start + 1).padStart(5, ' ')}</span>
            <span style={S.seqText}>{bases.slice(start, start + BASES_PER_LINE).join('')}</span>
          </div>
        ))}
      </div>
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
  seqReadout: { background: '#ffffff', border: '1px solid #e0e0e0', borderRadius: '6px', padding: '8px 10px', fontFamily: 'ui-monospace, Menlo, monospace', fontSize: '13px', lineHeight: 1.7 },
  seqLine: { display: 'flex', gap: '8px', alignItems: 'baseline' },
  seqIdx: { color: '#999', fontSize: '11px', minWidth: '40px', textAlign: 'right', whiteSpace: 'pre' },
  seqText: { color: '#111', letterSpacing: '1px', wordBreak: 'break-all' },
}
