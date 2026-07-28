// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Apache2
//
// Text UMAP: embed a set of sequences live (Evo2 -> layer-L -> SAE, mean-pooled
// per sequence) via /api/gene_embed, UMAP them client-side, then recolor or
// *reorganize* the layout by a chosen SAE feature. Adapted from the dashboard
// mockup's GeneUMAPView (umap-js recolor + reorganize core), with two input modes:
//   - Preset: pick from a bundled labeled library (/sequence_library.json)
//   - Custom: paste FASTA (>name|label) or TSV (name<TAB>label<TAB>seq)
// Feature ids map to the live SAE's labels (same SAE as the feature atlas).
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { UMAP } from 'umap-js'
import { BACKEND, fetchWithTimeout, workTimeout } from './backend'
import { RestartEngineButton, userLabel, setUserLabel, useUserLabels } from './components'

// Static, precomputed bundle (dashboard.py embeddings) — lets this tab run with NO live backend.
// Same JSON shape as the /gene_embed response. Overridable via ?embeddings=<url>.
export const EMBEDDINGS_URL =
  (typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('embeddings')) ||
  '/sequmap_embeddings.json'
const LAMBDA = 4.0 // feature amplification for "reorganize"
const ANIM_MS = 700
const FEAT_LIST_CAP = 300 // max feature rows rendered at once (after search filtering) — keeps the DOM light
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

// Custom-input formats the user explicitly picks (no more auto-detect guessing). Labels are optional
// everywhere and drive the map's label-coloring; a missing name is auto-filled seq1, seq2, …
const CUSTOM_PLACEHOLDER = {
  fasta: '>BRCA1|tumor_suppressor\nATGGATCCA...\n>TP53|tumor_suppressor\nATGGAGCCG...',
  tsv: 'BRCA1\ttumor_suppressor\tATGGATCCA...\nTP53\ttumor_suppressor\tATGGAGCCG...',
  lines: 'ATGGATCCA...\nATGGAGCCG...\n(one sequence per line — auto-named seq1, seq2…)',
}

function parseCustom(text, format = 'fasta') {
  const items = []
  if (format === 'lines') {
    // One sequence per line, auto-named, no labels.
    for (const line of text.split('\n')) {
      const seq = line.trim()
      if (seq) items.push({ symbol: `seq${items.length + 1}`, label: null, sequence: seq })
    }
  } else if (format === 'tsv') {
    // name <TAB> label <TAB> sequence (label may be blank).
    for (const line of text.split('\n')) {
      if (!line.trim()) continue
      const p = line.split('\t')
      if (p.length >= 3 && p[2].trim()) {
        items.push({ symbol: p[0].trim() || `seq${items.length + 1}`, label: p[1].trim() || null, sequence: p[2].trim() })
      }
    }
  } else {
    // FASTA: >name|label  \n SEQ...
    let name = null, label = null, seq = []
    const flush = () => { if (name) items.push({ symbol: name, label, sequence: seq.join('') }) }
    for (const line of text.split('\n')) {
      if (line.startsWith('>')) {
        flush(); seq = []
        const h = line.slice(1).trim().split('|')
        name = h[0]?.trim() || `seq${items.length + 1}`; label = h[1]?.trim() || null
      } else seq.push(line.trim())
    }
    flush()
  }
  return items.filter((s) => (s.sequence || '').trim().length >= 3)
}

// Live-preview counts for the input box: how many sequences parse, and how many input "records"
// (FASTA headers, or non-blank lines) were ignored (unparseable / < 3 nt) — so nothing drops silently.
function previewCustom(text, format = 'fasta') {
  const items = parseCustom(text, format)
  const lines = text.split('\n')
  const records = format === 'fasta' ? lines.filter((l) => l.startsWith('>')).length : lines.filter((l) => l.trim()).length
  return { items, ignored: Math.max(0, records - items.length) }
}

export default function SequenceUMAPView({ height = 600 }) {
  useUserLabels() // re-render when a feature is renamed in any tab
  const [mode, setMode] = useState('preset')
  const [library, setLibrary] = useState([])
  const [picked, setPicked] = useState(new Set())
  const [customText, setCustomText] = useState('')
  const [customFormat, setCustomFormat] = useState('fasta') // 'fasta' | 'tsv' | 'lines' — chosen explicitly, not guessed
  const [organism, setOrganism] = useState('None (raw text)')
  const [organisms, setOrganisms] = useState(['None (raw text)'])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null) // non-fatal warning: e.g. some sequences dropped before embedding
  const [bundle, setBundle] = useState(null) // {G(active), Gmean, Gmax, nf, ng, meta, items:[{name,label,species,x,y}], stats}
  const [initialLoading, setInitialLoading] = useState(true) // loading the precomputed snapshot on mount — suppress the input form until it lands (or fails)
  const [pooling, setPooling] = useState('mean') // 'mean' | 'max' — toggled client-side, no re-forward
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [editingFeature, setEditingFeature] = useState(null) // feature_id whose label is being edited
  const [featQuery, setFeatQuery] = useState('') // search box over the active-features list (label or #id)
  const [editText, setEditText] = useState('')
  const [reorgCoords, setReorgCoords] = useState(null)
  const [anim, setAnim] = useState(1)
  const [hover, setHover] = useState(null)
  const [selectedSeq, setSelectedSeq] = useState(null) // library row / map point selected for cross-highlight
  const [backendReady, setBackendReady] = useState(null) // null=unknown, false=loading, true=ready
  const [restartEnabled, setRestartEnabled] = useState(false) // server allows POST /api/restart (shows the button)
  const canvasRef = useRef(null)
  const plotRef = useRef(null)
  const selRowRef = useRef(null) // the selected sequence's list row — scroll it into view when picked on the map
  const [size, setSize] = useState({ w: 720, h: 480 })

  useEffect(() => {
    fetch('/sequence_library.json').then((r) => (r.ok ? r.json() : [])).then(setLibrary).catch(() => setLibrary([]))
    let stop = false, timer
    const poll = () => {
      fetchWithTimeout(`${BACKEND}/health`, {}, 8000).then((r) => r.json()).then((h) => {
        if (stop) return
        setOrganisms(h.organisms || ['None (raw text)'])
        setBackendReady(!!h.ready)
        setRestartEnabled(!!h.restart_enabled)
        if (!h.ready) timer = setTimeout(poll, 3000) // keep polling until model+SAE finish loading
      }).catch(() => { if (!stop) timer = setTimeout(poll, 3000) })
    }
    poll()
    return () => { stop = true; clearTimeout(timer) }
  }, [])

  // Clear the "Engine restarting…" note once the worker is back ready.
  useEffect(() => {
    if (backendReady) setError((e) => (typeof e === 'string' && e.startsWith('Engine restarting') ? null : e))
  }, [backendReady])

  // Land on the precomputed snapshot: load the static bundle on mount and show IT directly. While it
  // loads we show a small "loading" placeholder — NOT the input form — so the input UI never flashes
  // before the snapshot. If the bundle is absent/unreachable, we fall through to the input form.
  useEffect(() => {
    let cancelled = false
    fetchStaticBundle(EMBEDDINGS_URL).then((r) => {
      if (cancelled) return
      if (r) { try { ingestBundle(r, { precomputed: true }) } catch { /* malformed -> input form */ } }
      setInitialLoading(false)
    })
    return () => { cancelled = true }
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

  // Decode a /gene_embed-shaped bundle (live response OR the precomputed static JSON) and install it
  // as the active layout — shared by the live embed, the offline fallback, and the initial on-load
  // load. `precomputed` tags the static library bundle so the UI can label it and skip the
  // "dropped N sequences" notice (which only makes sense for a live embed of a chosen set).
  function ingestBundle(r, { precomputed = false, genes = null } = {}) {
    const dec = (b64) => {
      const bytes = Uint8Array.from(atob(b64 || ''), (c) => c.charCodeAt(0))
      if (bytes.length % 4 !== 0) throw new Error('corrupt embedding bundle (byte length not a multiple of 4)')
      return new Float32Array(bytes.buffer)
    }
    const Gmean = dec(r.G_b64)
    const Gmax = r.Gmax_b64 ? dec(r.Gmax_b64) : Gmean // back-compat if server only sends mean
    const nf = r.n_features, ng = r.n_genes
    if (Gmean.length !== nf * ng) throw new Error(`embedding bundle size mismatch: ${Gmean.length} != ${nf}*${ng}`)
    // The server ships only the firing columns; feature_ids[col] -> real SAE feature id.
    // colOf maps a feature id back to its matrix column (null bundle => identity, for old bundles).
    const colOf = r.feature_ids ? new Map(r.feature_ids.map((fid, c) => [fid, c])) : null
    const items = buildItems(Gmean, nf, ng, r.genes) // default pooling = mean
    // Stash the raw sequences we embedded (if any) so the "view sequence" card can show them — the
    // /gene_embed response itself doesn't echo the sequence text back.
    const seqMap = genes
      ? new Map(genes.map((g) => [g.symbol || g.gene_symbol || g.name, g.sequence]).filter(([k, v]) => k && v))
      : undefined
    setBundle({ G: Gmean, Gmean, Gmax, nf, ng, meta: r.genes, items, stats: r.feature_stats, saeId: r.sae_id, colOf, precomputed, seqByName: seqMap })
    if (precomputed) return
    // The backend drops sequences it can't embed (over the per-request cap, too short, or longer
    // than the context window) — surface that instead of silently UMAPing fewer than submitted.
    const dropped = (r.n_skipped_short || 0) + (r.n_dropped_over_cap || 0)
    const notes = []
    if (dropped > 0) {
      const parts = []
      if (r.n_dropped_over_cap) parts.push(`${r.n_dropped_over_cap} past the ${r.max_genes}-sequence cap`)
      if (r.n_skipped_short) parts.push(`${r.n_skipped_short} shorter than 3 nt`)
      notes.push(`dropped ${parts.join(', ')}`)
    }
    if (r.n_clamped) notes.push(`${r.n_clamped} clamped to the ${r.max_seq_len} bp context`)
    if (notes.length) setNotice(`Embedded ${ng} of ${r.n_received} sequences — ${notes.join('; ')}.`)
  }

  async function embed() {
    const genes = mode === 'preset' ? library.filter((_, i) => picked.has(i)) : parseCustom(customText, customFormat)
    if (!genes.length) { setError('Pick or paste at least one sequence (>=3 nt).'); return }
    setBusy(true); setError(null); setNotice(null); setBundle(null); setSelectedFeature(null); setSelectedSeq(null); setReorgCoords(null); setPooling('mean')
    try {
      // Offline fast-path: if /health says the model isn't up, skip the live call entirely — a
      // length-scaled timeout would otherwise make the user wait minutes for a request that can't
      // succeed. Preset mode then serves the precomputed bundle; custom paste needs the model.
      const online = backendReady !== false
      const live = online
        ? await fetchWithTimeout(`${BACKEND}/gene_embed`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ genes, organism }),
            // Slowest call — one encode per sequence on the single GPU — so scale the wait with the
            // count (1000 seqs -> ~12 min) instead of a fixed cap that's too short for a big batch.
          }, workTimeout(genes.length, { perUnit: 700, baseMs: 45000, ceilMs: 3_600_000 })).catch(() => null)
        : null
      let r
      if (live && live.ok) {
        r = await live.json()
      } else if (mode === 'preset') {
        r = await fetchStaticBundle(EMBEDDINGS_URL) // precomputed full-library bundle (same shape as /gene_embed)
        if (!r) throw new Error(`${online ? 'No live backend' : 'Model offline'} and no precomputed embeddings bundle found.`)
      } else {
        throw new Error('Embedding custom sequences needs a live backend; the precomputed bundle covers the preset library only.')
      }
      ingestBundle(r, { precomputed: !(live && live.ok), genes })
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

  // Live preview of the custom-paste box (parsed count + ignored lines), recomputed as you type.
  const customPreview = useMemo(
    () => (mode === 'custom' ? previewCustom(customText, customFormat) : { items: [], ignored: 0 }),
    [mode, customText, customFormat],
  )

  const colorInfo = useMemo(() => {
    if (!bundle) return null
    const { G, nf, items, colOf } = bundle
    if (selectedFeature == null) {
      const cats = [...new Set(items.map((it) => it.label))]
      return { mode: 'label', colors: items.map((it) => colorForLabel(it.label)), firing: null, cats }
    }
    // feature mode: split firing vs silent, then color firing points on a LINEAR, absolute scale
    // 0 -> vmax (the feature's peak activation across the shown sequences). Color is directly
    // proportional to activation, so the ramp reads as real values and is comparable point-to-point.
    // (Was 95th-pct-normalized + sqrt-spread, which boosted contrast on the heavy tail but made the
    // scale relative and hard to interpret.)
    const fcol = colOf ? colOf.get(selectedFeature) : selectedFeature // feature id -> matrix column
    const vals = items.map((_, i) => (fcol == null ? 0 : G[i * nf + fcol]))
    const firing = vals.map((v) => v > 0)
    const pos = vals.filter((v) => v > 0)
    const vmax = Math.max(...vals, 1e-9)
    const colors = vals.map((v) => (v > 0 ? ramp(Math.min(1, v / vmax)) : null))
    return { mode: 'feature', colors, firing, vals, vmin: pos.length ? Math.min(...pos) : 0, vmax, nFiring: pos.length }
  }, [bundle, selectedFeature])

  const coords = useMemo(() => {
    if (!bundle) return null
    const base = bundle.items.map((it) => [it.x, it.y])
    if (!reorgCoords) return base
    return base.map((b, i) => [b[0] + (reorgCoords[i][0] - b[0]) * anim, b[1] + (reorgCoords[i][1] - b[1]) * anim])
  }, [bundle, reorgCoords, anim])

  // Sequences grouped by label (category) for the browsable side list — largest category first.
  // Each entry is [label, [item indices]]; indices key back into bundle.items / coords / colorInfo.
  const seqGroups = useMemo(() => {
    if (!bundle) return []
    const m = new Map()
    bundle.items.forEach((it, i) => {
      const k = it.label ?? '—'
      if (!m.has(k)) m.set(k, [])
      m.get(k).push(i)
    })
    return [...m.entries()].sort((a, b) => b[1].length - a[1].length)
  }, [bundle])

  // name -> raw DNA, for the "view sequence" card. The bundle carries only names/labels, but the
  // preset library JSON carries sequences (join by name) and a live/custom embed stashes what it
  // sent (bundle.seqByName) — so both the precomputed and the just-embedded sets resolve.
  const seqByName = useMemo(() => {
    const m = new Map()
    for (const g of library) if (g && g.symbol && g.sequence) m.set(g.symbol, g.sequence)
    if (bundle?.seqByName) for (const [k, v] of bundle.seqByName) m.set(k, v)
    return m
  }, [library, bundle])

  // Active-features list filtered by the search box — matches a label substring or the feature id
  // (with or without a leading '#'). Searches the FULL set, not just the rendered slice.
  const matchedStats = useMemo(() => {
    if (!bundle) return []
    const q = featQuery.trim().toLowerCase().replace(/^#/, '')
    if (!q) return bundle.stats
    return bundle.stats.filter((s) => {
      const lbl = (userLabel(s.feature_id) || s.label || '').toLowerCase() // search the shared rename too
      return String(s.feature_id).includes(q) || lbl.includes(q)
    })
  }, [bundle, featQuery])

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
    // Selected-sequence highlight (from the side list or a map click): a ring + name chip on top.
    if (selectedSeq != null && coords[selectedSeq]) {
      const px = X(selectedSeq), py = Y(selectedSeq)
      ctx.beginPath(); ctx.arc(px, py, 8.5, 0, 6.2832)
      ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.stroke()
      ctx.lineWidth = 2; ctx.strokeStyle = '#fff'; ctx.stroke()
      const nm = bundle.items[selectedSeq]?.name || ''
      if (nm) {
        ctx.font = '12px sans-serif'
        const tw = ctx.measureText(nm).width
        const lx = Math.min(w - tw - 6, px + 11), ly = Math.max(15, py - 11)
        ctx.fillStyle = 'rgba(17,17,17,0.82)'; ctx.fillRect(lx - 4, ly - 12, tw + 8, 16)
        ctx.fillStyle = '#fff'; ctx.fillText(nm, lx, ly)
      }
    }
  }, [bundle, coords, colorInfo, hover, size, selectedSeq])

  // When a point is picked on the map, scroll its row into view in the side list.
  useEffect(() => { selRowRef.current?.scrollIntoView({ block: 'nearest' }) }, [selectedSeq])

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
    const fcol = bundle.colOf ? bundle.colOf.get(selectedFeature) : selectedFeature // feature id -> column
    if (fcol == null) return // selected feature isn't in the shipped (firing) set
    setBusy(true)
    try {
      const { G, nf, ng } = bundle
      // Amplifying one column by a small factor is invisible. Instead z-score the selected
      // feature across sequences and scale it to ~the typical row norm so it DOMINATES the
      // layout -> sequences pull together by that feature.
      let normSum = 0
      const col = new Float64Array(ng)
      for (let i = 0; i < ng; i++) {
        let nrm = 0; const b = i * nf
        for (let f = 0; f < nf; f++) nrm += G[b + f] * G[b + f]
        normSum += Math.sqrt(nrm); col[i] = G[b + fcol]
      }
      const meanNorm = normSum / ng || 1
      const cmean = col.reduce((a, b) => a + b, 0) / ng
      let cv = 0; for (let i = 0; i < ng; i++) cv += (col[i] - cmean) ** 2
      const cstd = Math.sqrt(cv / ng) || 1
      const W = LAMBDA * meanNorm // feature dim becomes ~LAMBDA x the whole-vector scale
      const vecs = Array.from({ length: ng }, (_, i) => {
        const row = Array.from(G.subarray(i * nf, (i + 1) * nf))
        row[fcol] = ((col[i] - cmean) / cstd) * W
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
  function saveLabel(fid, text) {
    const label = (text || '').trim()
    setEditingFeature(null)
    // Persist to the SHARED cross-tab store (localStorage, keyed by feature id) that the atlas +
    // steering + inspector already use — so a name set here shows everywhere (and vice versa). The
    // local stats update just re-renders this row immediately; display reads userLabel() first.
    setUserLabel(fid, label)
    setBundle((b) => (b ? { ...b, stats: b.stats.map((s) => (s.feature_id === fid ? { ...s, label } : s)) } : b))
  }

  // Nearest point under the cursor (or null if none within the hit radius) — shared by hover + click.
  const pointAt = (e) => {
    if (!bundle || !coords) return null
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
    return best
  }
  const onMove = (e) => setHover(pointAt(e))
  // Click a point to select its sequence (and highlight its row in the side list); click empty space
  // or the same point again to deselect.
  const onClickCanvas = (e) => { const i = pointAt(e); setSelectedSeq((cur) => (cur === i ? null : i)) }

  return (
    <div style={{ padding: 16, color: 'var(--text)' }}>
      <h3 style={{ margin: '0 0 8px' }}>Text UMAP <span style={{ fontWeight: 400, opacity: 0.7, fontSize: 13 }}>— embed texts live, color or reorganize by an SAE feature</span></h3>
      {!bundle && initialLoading && (
        <div style={{ padding: 16, opacity: 0.7, fontSize: 13 }}>Loading text map…</div>
      )}
      {!bundle && !initialLoading && (
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
            <div>
              <div style={{ display: 'flex', gap: 10, marginBottom: 6, fontSize: 12, alignItems: 'center' }}>
                <span style={{ opacity: 0.7 }}>Format:</span>
                {[['fasta', 'FASTA'], ['tsv', 'TSV'], ['lines', 'one per line']].map(([v, lbl]) => (
                  <label key={v} style={{ display: 'inline-flex', alignItems: 'center', gap: 3, cursor: 'pointer' }}>
                    <input type="radio" name="customfmt" checked={customFormat === v} onChange={() => setCustomFormat(v)} /> {lbl}
                  </label>
                ))}
              </div>
              <textarea value={customText} onChange={(e) => setCustomText(e.target.value)} rows={8}
                placeholder={CUSTOM_PLACEHOLDER[customFormat]}
                style={{ width: '100%', fontFamily: 'monospace', fontSize: 12 }} />
              <div style={{ marginTop: 4, fontSize: 11, opacity: 0.8 }}>
                {customText.trim() ? (
                  <>
                    <span style={{ color: customPreview.items.length ? '#76b900' : '#ef4444' }}>✓ {customPreview.items.length} parsed</span>
                    {customPreview.ignored > 0 && <span style={{ color: '#f59e0b' }}> · {customPreview.ignored} {customFormat === 'fasta' ? 'record' : 'line'}{customPreview.ignored === 1 ? '' : 's'} ignored (unparseable or &lt; 3 nt)</span>}
                    {customPreview.items.length > 0 && (
                      <span style={{ opacity: 0.65 }}> — {customPreview.items.slice(0, 4).map((s) => s.symbol + (s.label ? ` [${s.label}]` : '')).join(', ')}{customPreview.items.length > 4 ? ', …' : ''}</span>
                    )}
                  </>
                ) : (
                  <span style={{ opacity: 0.6 }}>labels are optional — they drive the map’s label-coloring</span>
                )}
              </div>
            </div>
          )}
          {backendReady === false && (
            <div style={{ marginTop: 8, fontSize: 12, color: '#f59e0b' }}>
              {mode === 'preset'
                ? '◐ GPT-2 model + SAE loading… — Embed will serve the precomputed library until it’s ready.'
                : '◐ GPT-2 model + SAE loading… (~1 min at startup) — pasting your own text needs the live model.'}
            </div>
          )}
          {(() => {
            // Custom paste needs the live model; preset can fall back to the precomputed bundle, so
            // stay enabled when offline (embed() short-circuits to the static bundle — no long wait).
            const blocked = busy || (backendReady === false && mode !== 'preset')
            return (
              <div style={{ marginTop: 8 }}>
                <button onClick={embed} disabled={blocked} style={{ ...tabStyle(true), opacity: blocked ? 0.5 : 1, cursor: blocked ? 'not-allowed' : 'pointer' }}>
                  {busy ? 'Embedding… (batched on the GPU)' : `Embed ${mode === 'preset' ? picked.size : customPreview.items.length} sequences`}
                </button>{' '}
                <RestartEngineButton enabled={restartEnabled} busy={busy}
                  onRestart={() => { setBusy(false); setError('Engine restarting — reloading the model (~1 min)…') }} />
              </div>
            )
          })()}
          <div style={{ marginTop: 6, fontSize: 11, opacity: 0.6 }}>
            Batched on the GPU (8 sequences per forward, length-bucketed to limit padding) — many or long sequences still take a while.
          </div>
          {error && <div style={{ color: '#ef4444', marginTop: 8, fontSize: 12 }}>{error}</div>}
          {notice && <div style={{ color: '#f59e0b', marginTop: 8, fontSize: 12 }}>⚠ {notice}</div>}
        </div>
      )}

      {bundle && (
        <div style={{ display: 'flex', gap: 12, height }}>
          <div style={{ width: 210, overflow: 'auto', fontSize: 12, flexShrink: 0 }}>
            <div style={{ fontWeight: 600 }}>{bundle.precomputed ? 'Preset library' : 'Your sequence set'}</div>
            <div style={{ opacity: 0.6, fontSize: 11, marginBottom: 6 }}>
              {bundle.ng} sequences · {seqGroups.length} categor{seqGroups.length === 1 ? 'y' : 'ies'}. Click a name to find it on the map; click a point to find it here.
            </div>
            {seqGroups.map(([label, idxs]) => (
              <div key={label ?? '_'} style={{ marginBottom: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 600, opacity: 0.85, margin: '5px 0 2px' }}>
                  <span style={{ width: 9, height: 9, borderRadius: 5, background: colorForLabel(label === '—' ? null : label), flexShrink: 0 }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
                  <span style={{ opacity: 0.55, fontWeight: 400 }}>({idxs.length})</span>
                </div>
                {idxs.map((i) => (
                  <div key={i}
                    ref={i === selectedSeq ? selRowRef : null}
                    onClick={() => setSelectedSeq((cur) => (cur === i ? null : i))}
                    onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                    title={bundle.items[i].species || bundle.items[i].name}
                    style={{ cursor: 'pointer', padding: '2px 6px', borderRadius: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                      background: i === selectedSeq ? 'rgba(118,185,0,0.28)' : (i === hover ? 'rgba(255,255,255,0.07)' : 'transparent') }}>
                    {bundle.items[i].name}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{ display: 'flex', gap: 8, marginBottom: 6, alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: 13, opacity: 0.8 }}>{bundle.ng} sequences × {bundle.nf} features · color: {selectedFeature == null ? 'label' : `feature #${selectedFeature}`}</span>
              {bundle.precomputed && (
                <span title="A precomputed library bundle (no model call). Click “New set” to embed your own sequences live." style={{ fontSize: 12, color: '#f59e0b' }}>
                  ◐ precomputed library{backendReady === false ? ' (model offline)' : ' — “New set” to embed live'}
                </span>
              )}
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
              <button onClick={() => { setBundle(null); setReorgCoords(null); setSelectedFeature(null); setSelectedSeq(null) }} style={tabStyle(false)}>New set</button>
              <RestartEngineButton enabled={restartEnabled} busy={busy}
                onRestart={() => { setBusy(false); setError('Engine restarting — reloading the model (~1 min)…') }} />
            </div>
            <div ref={plotRef} style={{ flex: 1, position: 'relative', minHeight: 0 }}>
              <canvas ref={canvasRef} onMouseMove={onMove} onMouseLeave={() => setHover(null)} onClick={onClickCanvas}
                style={{ border: '1px solid var(--border,#333)', borderRadius: 8, width: '100%', height: '100%', display: 'block', cursor: 'pointer' }} />
              {hover != null && (
                <div style={{ position: 'absolute', top: 8, left: 8, background: 'var(--bg,#111)', border: '1px solid var(--border,#333)', borderRadius: 6, padding: 8, fontSize: 12, pointerEvents: 'none' }}>
                  <b>{bundle.items[hover].name}</b><br />label: {bundle.items[hover].label ?? '—'}<br />
                  {selectedFeature != null && <>feat #{selectedFeature}: {bundle.G[hover * bundle.nf + selectedFeature].toFixed(3)}</>}
                </div>
              )}
              {/* Selected-sequence card: floats over the bottom of the plot only while something is
                  selected, and is dismissible — so you can read the DNA without it permanently taking space. */}
              {selectedSeq != null && (() => {
                const sel = bundle.items[selectedSeq]
                const seq = seqByName.get(sel.name)
                const btn = { border: '1px solid var(--border,#444)', background: 'transparent', color: 'var(--text)', borderRadius: 4, cursor: 'pointer', fontSize: 11, padding: '1px 7px' }
                return (
                  <div style={{ position: 'absolute', left: 8, right: 8, bottom: 8, background: 'var(--bg,#111)', border: '1px solid var(--border,#333)', borderRadius: 8, padding: '7px 9px', fontSize: 12, boxShadow: '0 2px 10px rgba(0,0,0,0.45)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 5, background: colorForLabel(sel.label), flexShrink: 0 }} />
                      <b style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sel.name}</b>
                      {sel.label && <span style={{ opacity: 0.7 }}>[{sel.label}]</span>}
                      {sel.species && <span style={{ opacity: 0.55, fontStyle: 'italic', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sel.species}</span>}
                      {seq && <span style={{ opacity: 0.6, whiteSpace: 'nowrap' }}>· {seq.length} chars</span>}
                      <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, flexShrink: 0 }}>
                        {seq && <button style={btn} onClick={() => navigator.clipboard?.writeText(seq)}>copy</button>}
                        <button style={btn} title="close" onClick={() => setSelectedSeq(null)}>✕</button>
                      </span>
                    </div>
                    {seq
                      ? <div style={{ fontFamily: 'monospace', fontSize: 11, lineHeight: 1.4, maxHeight: 76, overflow: 'auto', wordBreak: 'break-all', background: 'rgba(127,127,127,0.10)', border: '1px solid var(--border,#222)', borderRadius: 6, padding: '5px 7px' }}>{seq}</div>
                      : <div style={{ opacity: 0.6, fontSize: 11 }}>Sequence text isn’t available for this set (embed it live, or it’s a precomputed set without sequences).</div>}
                  </div>
                )
              })()}
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
                      <span style={{ opacity: 0.7 }}>0</span>
                      <span style={{ width: 130, height: 10, borderRadius: 3, background: 'linear-gradient(to right,rgb(59,76,192),rgb(34,211,238),rgb(118,185,0),rgb(251,191,36),rgb(239,68,68))' }} />
                      <span style={{ opacity: 0.7 }}>{colorInfo.vmax.toFixed(2)} act.</span>
                    </>
                  )}
              </div>
            )}
          </div>
          <div style={{ width: 240, overflow: 'auto', fontSize: 12 }}>
            <div style={{ fontWeight: 600 }}>{featQuery.trim() ? `${matchedStats.length} of ${bundle.stats.length}` : bundle.stats.length} active SAE features</div>
            <div style={{ opacity: 0.55, fontSize: 11 }}>click to color the map · then “Reorganize” · n = texts it fires in</div>
            <input value={featQuery} onChange={(e) => setFeatQuery(e.target.value)} placeholder="search by label or #id…"
              style={{ width: '100%', boxSizing: 'border-box', margin: '5px 0 6px', fontSize: 12, padding: '3px 6px' }} />
            {bundle.saeId && (
              <div title="Feature ids/labels belong to this SAE only — they do NOT correspond to a different SAE's atlas unless the id matches."
                style={{ opacity: 0.5, marginBottom: 6, fontSize: 10, fontFamily: 'monospace', wordBreak: 'break-all' }}>
                SAE: {bundle.saeId}
              </div>
            )}
            {matchedStats.slice(0, FEAT_LIST_CAP).map((s) => (
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
                    {(() => {
                      const lbl = userLabel(s.feature_id) || s.label // shared rename overlays the base label
                      return (
                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', opacity: lbl ? 1 : 0.4, fontStyle: lbl ? 'normal' : 'italic' }}>
                          {lbl || 'add label…'}
                        </span>
                      )
                    })()}
                    <button title="name this feature" onClick={(e) => { e.stopPropagation(); setEditingFeature(s.feature_id); setEditText(userLabel(s.feature_id) || s.label || '') }}
                      style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: 'var(--text)', opacity: 0.5, fontSize: 11, padding: 0 }}>✎</button>
                  </>
                )}
                <span style={{ opacity: 0.45 }}>{s.n_firing}/{bundle.ng}</span>
              </div>
            ))}
            {matchedStats.length > FEAT_LIST_CAP && (
              <div style={{ opacity: 0.55, fontSize: 11, padding: '4px 2px' }}>+{matchedStats.length - FEAT_LIST_CAP} more — refine your search to narrow the list</div>
            )}
            {matchedStats.length === 0 && (
              <div style={{ opacity: 0.55, fontSize: 11, padding: '4px 2px' }}>no features match “{featQuery.trim()}”</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// Fetch a precomputed bundle, tolerating a dev server's SPA HTML fallback: a MISSING static file
// still 200s with text/html (not the bundle), which would otherwise surface as an opaque JSON parse
// error. Returns the parsed bundle only if it really is one (JSON with G_b64), else null.
async function fetchStaticBundle(url) {
  try {
    const r = await fetch(url)
    if (!r.ok || !(r.headers.get('content-type') || '').includes('json')) return null
    const b = await r.json()
    return b && b.G_b64 ? b : null
  } catch {
    return null
  }
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
