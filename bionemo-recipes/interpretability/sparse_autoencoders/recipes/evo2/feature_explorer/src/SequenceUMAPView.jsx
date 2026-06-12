// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Apache2
//
// Sequence UMAP: embed a set of sequences live (Evo2 -> layer-L -> SAE, mean-pooled
// per sequence) via /api/gene_embed, UMAP them client-side, then recolor or
// *reorganize* the layout by a chosen SAE feature. Adapted from the dashboard
// mockup's GeneUMAPView (umap-js recolor + reorganize core), with two input modes:
//   - Preset: pick from a bundled labeled library (/sequence_library.json)
//   - Custom: paste FASTA (>name|label) or TSV (name<TAB>label<TAB>seq)
// Feature ids map to the live SAE's labels (same SAE as the feature atlas).
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { UMAP } from 'umap-js'

const BACKEND = '/api'
const LAMBDA = 4.0 // feature amplification for "reorganize"
const ANIM_MS = 700
const CAT_COLORS = ['#76b900', '#3b82f6', '#ef4444', '#f59e0b', '#a855f7', '#14b8a6', '#ec4899', '#84cc16', '#06b6d4', '#f97316']
const NOISE = '#555'
// Perceptual "turbo-lite" ramp: blue -> cyan -> green -> amber -> red. Bright at
// both ends so it reads on dark AND light themes (unlike viridis' near-black low end).
const RAMP = [[59, 76, 192], [34, 211, 238], [118, 185, 0], [251, 191, 36], [239, 68, 68]]

function hashHue(s) {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0
  return ((h % 360) + 360) % 360
}
function colorForLabel(label) {
  if (label == null) return NOISE
  // Deterministic per-label color (hash of the label string) so a given label keeps the
  // SAME color across embeds / subsets / sessions — independent of data order.
  return `hsl(${hashHue(String(label))}, 62%, 55%)`
}
function ramp(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1)
  const i = Math.floor(x), f = x - i, a = RAMP[i], b = RAMP[Math.min(RAMP.length - 1, i + 1)]
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`
}

function parseCustom(text) {
  const items = []
  if (text.trim().startsWith('>')) {
    // FASTA: >name|label  \n SEQ...
    let name = null, label = null, seq = []
    const flush = () => { if (name) items.push({ symbol: name, label, sequence: seq.join('') }) }
    for (const line of text.split('\n')) {
      if (line.startsWith('>')) {
        flush(); seq = []
        const h = line.slice(1).trim().split('|')
        name = h[0]?.trim() || `seq${items.length}`; label = h[1]?.trim() || null
      } else seq.push(line.trim())
    }
    flush()
  } else {
    // TSV: name <tab> label <tab> sequence
    for (const line of text.split('\n')) {
      const p = line.split('\t')
      if (p.length >= 3 && p[2].trim()) items.push({ symbol: p[0].trim(), label: p[1].trim() || null, sequence: p[2].trim() })
    }
  }
  return items.filter((s) => (s.sequence || '').replace(/[^ACGTNacgtn]/g, '').length >= 3)
}

export default function SequenceUMAPView({ height = 600 }) {
  const [mode, setMode] = useState('preset')
  const [library, setLibrary] = useState([])
  const [picked, setPicked] = useState(new Set())
  const [customText, setCustomText] = useState('')
  const [organism, setOrganism] = useState('None (raw DNA)')
  const [organisms, setOrganisms] = useState(['None (raw DNA)'])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [bundle, setBundle] = useState(null) // {G(active), Gmean, Gmax, nf, ng, meta, items:[{name,label,species,x,y}], stats}
  const [pooling, setPooling] = useState('mean') // 'mean' | 'max' — toggled client-side, no re-forward
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [editingFeature, setEditingFeature] = useState(null) // feature_id whose label is being edited
  const [editText, setEditText] = useState('')
  const [reorgCoords, setReorgCoords] = useState(null)
  const [anim, setAnim] = useState(1)
  const [hover, setHover] = useState(null)
  const [backendReady, setBackendReady] = useState(null) // null=unknown, false=loading, true=ready
  const canvasRef = useRef(null)
  const plotRef = useRef(null)
  const [size, setSize] = useState({ w: 720, h: 480 })

  useEffect(() => {
    fetch('/sequence_library.json').then((r) => (r.ok ? r.json() : [])).then(setLibrary).catch(() => setLibrary([]))
    let stop = false, timer
    const poll = () => {
      fetch(`${BACKEND}/health`).then((r) => r.json()).then((h) => {
        if (stop) return
        setOrganisms(h.organisms || ['None (raw DNA)'])
        setBackendReady(!!h.ready)
        if (!h.ready) timer = setTimeout(poll, 3000) // keep polling until model+SAE finish loading
      }).catch(() => { if (!stop) timer = setTimeout(poll, 3000) })
    }
    poll()
    return () => { stop = true; clearTimeout(timer) }
  }, [])

  // Keep the canvas sized to its container (responsive to window/panel resize).
  useEffect(() => {
    const el = plotRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setSize({ w: Math.max(80, Math.floor(r.width)), h: Math.max(80, Math.floor(r.height)) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [bundle])

  async function embed() {
    const genes = mode === 'preset' ? library.filter((_, i) => picked.has(i)) : parseCustom(customText)
    if (!genes.length) { setError('Pick or paste at least one sequence (>=3 nt).'); return }
    setBusy(true); setError(null); setBundle(null); setSelectedFeature(null); setReorgCoords(null); setPooling('mean')
    try {
      const resp = await fetch(`${BACKEND}/gene_embed`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ genes, organism }),
      })
      if (!resp.ok) throw new Error(`${resp.status}: ${(await resp.text()).slice(0, 200)}`)
      const r = await resp.json()
      const dec = (b64) => new Float32Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer)
      const Gmean = dec(r.G_b64)
      const Gmax = r.Gmax_b64 ? dec(r.Gmax_b64) : Gmean // back-compat if server only sends mean
      const nf = r.n_features, ng = r.n_genes
      const items = buildItems(Gmean, nf, ng, r.genes) // default pooling = mean
      setBundle({ G: Gmean, Gmean, Gmax, nf, ng, meta: r.genes, items, stats: r.feature_stats, saeId: r.sae_id })
    } catch (e) { setError(String(e.message || e)) } finally { setBusy(false) }
  }

  // Switch mean<->max pooling instantly (both came from the same forward); just
  // re-lay-out client-side from the stored matrix — no re-running the model.
  async function setPool(p) {
    if (!bundle || p === pooling) return
    setBusy(true); setError(null); setReorgCoords(null)
    try {
      const G = p === 'max' ? bundle.Gmax : bundle.Gmean
      await new Promise((r) => setTimeout(r, 16))
      const items = buildItems(G, bundle.nf, bundle.ng, bundle.meta)
      setBundle({ ...bundle, G, items })
      setPooling(p)
    } catch (e) { setError('re-pool failed: ' + (e.message || e)) } finally { setBusy(false) }
  }

  const colorInfo = useMemo(() => {
    if (!bundle) return null
    const { G, nf, items } = bundle
    if (selectedFeature == null) {
      const cats = [...new Set(items.map((it) => it.label))]
      return { mode: 'label', colors: items.map((it) => colorForLabel(it.label)), firing: null, cats }
    }
    // feature mode: split firing vs silent; scale by the 95th pct of FIRING values
    // (SAE activations are heavy-tailed, so a plain /max washes everything to one end)
    // and sqrt-spread so low-but-nonzero points stay distinguishable.
    const vals = items.map((_, i) => G[i * nf + selectedFeature])
    const firing = vals.map((v) => v > 0)
    const pos = vals.filter((v) => v > 0).sort((a, b) => a - b)
    const vmax = Math.max(...vals, 1e-9)
    const p95 = pos.length ? pos[Math.min(pos.length - 1, Math.floor(0.95 * pos.length))] : vmax
    const colors = vals.map((v) => (v > 0 ? ramp(Math.sqrt(Math.min(1, v / (p95 || vmax)))) : null))
    return { mode: 'feature', colors, firing, vals, vmin: pos[0] ?? 0, vmax, nFiring: pos.length }
  }, [bundle, selectedFeature])

  const coords = useMemo(() => {
    if (!bundle) return null
    const base = bundle.items.map((it) => [it.x, it.y])
    if (!reorgCoords) return base
    return base.map((b, i) => [b[0] + (reorgCoords[i][0] - b[0]) * anim, b[1] + (reorgCoords[i][1] - b[1]) * anim])
  }, [bundle, reorgCoords, anim])

  useEffect(() => {
    if (!bundle || !coords || !colorInfo) return
    const cv = canvasRef.current; if (!cv) return
    const dpr = window.devicePixelRatio || 1
    const w = size.w, h = size.h
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr)
    const ctx = cv.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0) // draw in CSS pixels, render at device resolution
    ctx.clearRect(0, 0, w, h)
    let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity
    for (const [x, y] of coords) { mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mny = Math.min(mny, y); mxy = Math.max(mxy, y) }
    const pad = 30, s = Math.min((w - 2 * pad) / Math.max(1e-9, mxx - mnx), (h - 2 * pad) / Math.max(1e-9, mxy - mny))
    const X = (i) => pad + (coords[i][0] - mnx) * s, Y = (i) => pad + (coords[i][1] - mny) * s
    // draw silent (non-firing) points first, hottest last, so peaks sit on top
    const order = [...coords.keys()]
    if (colorInfo.mode === 'feature') order.sort((a, b) => colorInfo.vals[a] - colorInfo.vals[b])
    for (const i of order) {
      const silent = colorInfo.mode === 'feature' && !colorInfo.firing[i]
      ctx.globalAlpha = hover != null && i !== hover ? 0.3 : 1
      ctx.beginPath(); ctx.arc(X(i), Y(i), hover === i ? 7 : 4.5, 0, 6.2832)
      if (silent) { ctx.strokeStyle = NOISE; ctx.lineWidth = 1.2; ctx.stroke() }
      else { ctx.fillStyle = colorInfo.colors[i]; ctx.fill() }
    }
    ctx.globalAlpha = 1
  }, [bundle, coords, colorInfo, hover, size])

  useEffect(() => {
    if (!reorgCoords) { setAnim(1); return }
    let raf; const t0 = performance.now()
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / ANIM_MS); setAnim(t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    setAnim(0); raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf)
  }, [reorgCoords])

  async function reorganize() {
    if (!bundle || selectedFeature == null) return
    setBusy(true)
    try {
      const { G, nf, ng } = bundle
      // Amplifying one column among 65k by a small factor is invisible. Instead
      // z-score the selected feature across sequences and scale it to ~the typical
      // row norm so it DOMINATES the layout -> sequences pull together by that feature.
      let normSum = 0
      const col = new Float64Array(ng)
      for (let i = 0; i < ng; i++) {
        let nrm = 0; const b = i * nf
        for (let f = 0; f < nf; f++) nrm += G[b + f] * G[b + f]
        normSum += Math.sqrt(nrm); col[i] = G[b + selectedFeature]
      }
      const meanNorm = normSum / ng || 1
      const cmean = col.reduce((a, b) => a + b, 0) / ng
      let cv = 0; for (let i = 0; i < ng; i++) cv += (col[i] - cmean) ** 2
      const cstd = Math.sqrt(cv / ng) || 1
      const W = LAMBDA * meanNorm // feature dim becomes ~LAMBDA x the whole-vector scale
      const vecs = Array.from({ length: ng }, (_, i) => {
        const row = Array.from(G.subarray(i * nf, (i + 1) * nf))
        row[selectedFeature] = ((col[i] - cmean) / cstd) * W
        return row
      })
      await new Promise((r) => setTimeout(r, 16))
      const coords2 = new UMAP({ nComponents: 2, nNeighbors: Math.min(15, Math.max(2, ng - 1)), minDist: 0.1 }).fit(vecs)
      setReorgCoords(coords2)
    } catch (e) {
      console.error('reorganize failed:', e)
      setError('reorganize failed: ' + (e.message || e))
    } finally {
      setBusy(false)
    }
  }

  // Biologist-contributed label: persist via the backend (scoped to this SAE), reflect locally.
  async function saveLabel(fid, text) {
    const label = (text || '').trim()
    setEditingFeature(null)
    try {
      const resp = await fetch(`${BACKEND}/label`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feature_id: fid, label, sae_id: bundle?.saeId }),
      })
      if (!resp.ok) throw new Error(`${resp.status}: ${(await resp.text()).slice(0, 160)}`)
      const r = await resp.json()
      setBundle((b) => (b ? { ...b, stats: b.stats.map((s) => (s.feature_id === fid ? { ...s, label: r.label } : s)) } : b))
    } catch (e) { setError('label save failed: ' + (e.message || e)) }
  }

  const onMove = (e) => {
    if (!bundle || !coords) return
    const cv = canvasRef.current, rect = cv.getBoundingClientRect()
    const w = rect.width, h = rect.height // CSS pixels — matches the DPR-scaled draw
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity
    for (const [x, y] of coords) { mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mny = Math.min(mny, y); mxy = Math.max(mxy, y) }
    const pad = 30, s = Math.min((w - 2 * pad) / Math.max(1e-9, mxx - mnx), (h - 2 * pad) / Math.max(1e-9, mxy - mny))
    let best = null, bd = 144
    for (let i = 0; i < coords.length; i++) {
      const px = pad + (coords[i][0] - mnx) * s, py = pad + (coords[i][1] - mny) * s
      const d = (px - mx) ** 2 + (py - my) ** 2; if (d < bd) { bd = d; best = i }
    }
    setHover(best)
  }

  return (
    <div style={{ padding: 16, color: 'var(--text)' }}>
      <h3 style={{ margin: '0 0 8px' }}>Sequence UMAP <span style={{ fontWeight: 400, opacity: 0.7, fontSize: 13 }}>— embed sequences live, color or reorganize by an SAE feature</span></h3>
      {!bundle && (
        <div style={{ border: '1px solid var(--border, #333)', borderRadius: 8, padding: 12, maxWidth: 720 }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <button onClick={() => setMode('preset')} style={tabStyle(mode === 'preset')}>Preset library ({library.length})</button>
            <button onClick={() => setMode('custom')} style={tabStyle(mode === 'custom')}>Paste your own</button>
            <select value={organism} onChange={(e) => setOrganism(e.target.value)} style={{ marginLeft: 'auto' }}>
              {organisms.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
          {mode === 'preset' ? (
            <div style={{ maxHeight: 220, overflow: 'auto', fontSize: 13 }}>
              <label style={{ display: 'block', marginBottom: 4 }}>
                <input type="checkbox" checked={picked.size === library.length && library.length > 0}
                  onChange={(e) => setPicked(e.target.checked ? new Set(library.map((_, i) => i)) : new Set())} /> select all
              </label>
              {library.map((g, i) => (
                <label key={i} style={{ display: 'block' }}>
                  <input type="checkbox" checked={picked.has(i)} onChange={() => {
                    const s = new Set(picked); s.has(i) ? s.delete(i) : s.add(i); setPicked(s)
                  }} /> {g.symbol} <span style={{ opacity: 0.6 }}>[{g.label}]</span>
                </label>
              ))}
            </div>
          ) : (
            <textarea value={customText} onChange={(e) => setCustomText(e.target.value)} rows={8}
              placeholder={'>MYSEQ|my_label\nACGT...\n\nor TSV:\nname<TAB>label<TAB>ACGT...'}
              style={{ width: '100%', fontFamily: 'monospace', fontSize: 12 }} />
          )}
          {backendReady === false && (
            <div style={{ marginTop: 8, fontSize: 12, color: '#f59e0b' }}>◐ Evo2 model + SAE loading… (~1 min at startup) — Embed will enable when ready.</div>
          )}
          <div style={{ marginTop: 8 }}>
            <button onClick={embed} disabled={busy || backendReady === false} style={{ ...tabStyle(true), opacity: backendReady === false ? 0.5 : 1, cursor: backendReady === false ? 'not-allowed' : 'pointer' }}>
              {busy ? 'Embedding… (one Evo2 pass per sequence)' : `Embed ${mode === 'preset' ? picked.size : parseCustom(customText).length} sequences`}
            </button>
          </div>
          <div style={{ marginTop: 6, fontSize: 11, opacity: 0.6 }}>
            One Evo2 forward per sequence (no batching) — embedding many sequences or long sequences can take a while.
          </div>
          {error && <div style={{ color: '#ef4444', marginTop: 8, fontSize: 12 }}>{error}</div>}
        </div>
      )}

      {bundle && (
        <div style={{ display: 'flex', gap: 12, height }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{ display: 'flex', gap: 8, marginBottom: 6, alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: 13, opacity: 0.8 }}>{bundle.ng} sequences × {bundle.nf} features · color: {selectedFeature == null ? 'label' : `feature #${selectedFeature}`}</span>
              <span style={{ display: 'inline-flex', border: '1px solid var(--border,#444)', borderRadius: 6, overflow: 'hidden' }} title="mean = how densely a feature fires across the sequence; max = its peak (sharper for sparse motifs)">
                {['mean', 'max'].map((p) => (
                  <button key={p} onClick={() => setPool(p)} disabled={busy}
                    style={{ padding: '5px 10px', border: 'none', cursor: 'pointer', fontSize: 12, background: pooling === p ? '#76b900' : 'transparent', color: pooling === p ? '#000' : 'var(--text)', fontWeight: pooling === p ? 600 : 400 }}>
                    {p}-pool
                  </button>
                ))}
              </span>
              <button onClick={reorganize} disabled={selectedFeature == null || busy} style={tabStyle(selectedFeature != null)}>
                {busy ? 'Working…' : 'Reorganize by feature'}
              </button>
              {reorgCoords && <button onClick={() => setReorgCoords(null)} style={tabStyle(false)}>Reset layout</button>}
              <button onClick={() => { setBundle(null); setReorgCoords(null) }} style={tabStyle(false)}>New set</button>
            </div>
            <div ref={plotRef} style={{ flex: 1, position: 'relative', minHeight: 0 }}>
              <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
                style={{ border: '1px solid var(--border,#333)', borderRadius: 8, width: '100%', height: '100%', display: 'block' }} />
              {hover != null && (
                <div style={{ position: 'absolute', top: 8, left: 8, background: 'var(--bg,#111)', border: '1px solid var(--border,#333)', borderRadius: 6, padding: 8, fontSize: 12, pointerEvents: 'none' }}>
                  <b>{bundle.items[hover].name}</b><br />label: {bundle.items[hover].label ?? '—'}<br />
                  {selectedFeature != null && <>feat #{selectedFeature}: {bundle.G[hover * bundle.nf + selectedFeature].toFixed(3)}</>}
                </div>
              )}
            </div>
            {colorInfo && (
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center', fontSize: 11, marginTop: 6, opacity: 0.9 }}>
                {colorInfo.mode === 'label'
                  ? colorInfo.cats.map((c) => (
                    <span key={c ?? '_'} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 5, background: colorForLabel(c) }} />{c ?? '—'}
                    </span>
                  ))
                  : (
                    <>
                      <span style={{ opacity: 0.7 }}>{colorInfo.nFiring}/{bundle.ng} firing</span>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                        <span style={{ width: 9, height: 9, borderRadius: 5, border: `1.5px solid ${NOISE}` }} />silent
                      </span>
                      <span style={{ opacity: 0.7 }}>low {(colorInfo.vmin || 0).toFixed(2)}</span>
                      <span style={{ width: 130, height: 10, borderRadius: 3, background: 'linear-gradient(to right,rgb(59,76,192),rgb(34,211,238),rgb(118,185,0),rgb(251,191,36),rgb(239,68,68))' }} />
                      <span style={{ opacity: 0.7 }}>{colorInfo.vmax.toFixed(2)} act.</span>
                    </>
                  )}
              </div>
            )}
          </div>
          <div style={{ width: 240, overflow: 'auto', fontSize: 12 }}>
            <div style={{ fontWeight: 600 }}>{bundle.stats.length} active SAE features</div>
            <div style={{ opacity: 0.55, fontSize: 11 }}>click to color the map · then “Reorganize” · n = sequences it fires in</div>
            {bundle.saeId && (
              <div title="Feature ids/labels belong to this SAE only — they do NOT correspond to a different SAE's atlas unless the id matches."
                style={{ opacity: 0.5, marginBottom: 6, fontSize: 10, fontFamily: 'monospace', wordBreak: 'break-all' }}>
                SAE: {bundle.saeId}
              </div>
            )}
            {bundle.stats.slice(0, 200).map((s) => (
              <div key={s.feature_id}
                onClick={() => editingFeature !== s.feature_id && setSelectedFeature(s.feature_id === selectedFeature ? null : s.feature_id)}
                style={{ cursor: 'pointer', padding: '2px 4px', borderRadius: 4, display: 'flex', gap: 6, alignItems: 'center', background: s.feature_id === selectedFeature ? 'rgba(118,185,0,0.25)' : 'transparent' }}>
                <span style={{ fontFamily: 'monospace', opacity: 0.75 }}>#{s.feature_id}</span>
                {editingFeature === s.feature_id ? (
                  <input autoFocus value={editText} onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') saveLabel(s.feature_id, editText); else if (e.key === 'Escape') setEditingFeature(null) }}
                    onBlur={() => saveLabel(s.feature_id, editText)}
                    placeholder="label…" style={{ flex: 1, minWidth: 0, fontSize: 11 }} />
                ) : (
                  <>
                    <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', opacity: s.label ? 1 : 0.4, fontStyle: s.label ? 'normal' : 'italic' }}>
                      {s.label || 'add label…'}
                    </span>
                    <button title="name this feature" onClick={(e) => { e.stopPropagation(); setEditingFeature(s.feature_id); setEditText(s.label || '') }}
                      style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--text)', opacity: 0.5, fontSize: 11, padding: 0 }}>✎</button>
                  </>
                )}
                <span style={{ opacity: 0.45 }}>{s.n_firing}/{bundle.ng}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Client-side base UMAP from a pooled matrix G (Float32Array, [ng*nf]) -> items with x/y.
function buildItems(G, nf, ng, meta) {
  const vecs = Array.from({ length: ng }, (_, i) => Array.from(G.subarray(i * nf, (i + 1) * nf)))
  const coords = new UMAP({ nComponents: 2, nNeighbors: Math.min(15, Math.max(2, ng - 1)), minDist: 0.1 }).fit(vecs)
  return meta.map((g, i) => ({ name: g.gene_symbol, label: g.label, species: g.species, x: coords[i][0], y: coords[i][1] }))
}

function tabStyle(on) {
  return {
    padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border,#444)', cursor: 'pointer',
    background: on ? '#76b900' : 'transparent', color: on ? '#000' : 'var(--text)', fontWeight: on ? 600 : 400,
  }
}
