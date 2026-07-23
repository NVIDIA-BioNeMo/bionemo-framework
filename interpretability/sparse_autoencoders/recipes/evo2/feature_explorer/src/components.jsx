// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Apache2
//
// Shared widgets + styles for the live-backend panes (Sequence inspector, Generative
// steering, …). These used to live inside SequenceInspector.jsx, which made one pane a
// dependency of another; they belong in a neutral module both panes import from.

import React, { useMemo, useState, useEffect } from 'react'
import { BACKEND, activationColor, legendGradient, getJSON, postJSON } from './backend'

export const BASES_PER_LINE = 80

// Steering clamp targets are absolute SAE-code values; mirrors the backend MAX_CLAMP_STRENGTH
// guard so the UI can't request a target the engine would reject/cap.
const CLAMP_MAX = 300

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
// Resolve a rename (a typed display name) back to a feature id via the overlay stores. Covers ANY
// feature — including ones absent from the labeled /features catalog (e.g. an arbitrary atlas feature
// you just renamed). Checks the server overlay, then the localStorage overlay.
function renameToId(name) {
  for (const [id, lab] of _serverRenames) if (lab === name) return id
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)
      if (k && k.startsWith('featureTitle_') && localStorage.getItem(k) === name) {
        return Number(k.slice('featureTitle_'.length))
      }
    }
  } catch {
    /* private mode — no localStorage overlay to scan */
  }
  return null
}

export function resolveFeatureId(catalog, q) {
  // A bare numeric id resolves to ANY feature (every feature 0..n_features-1 is clampable, even if
  // unlabeled). Otherwise match a label — the effective (renamed) label first, then the base catalog
  // label, then the rename overlay, so a renamed-but-uncataloged feature still resolves by its new name.
  const m = String(q).match(/#?(\d+)/)
  if (m) {
    const id = Number(m[1])
    if (Number.isInteger(id) && id >= 0) return id
  }
  const lab = String(q).trim()
  if (!lab) return null
  const hit = catalog.find((f) => (userLabel(f.id) || f.label) === lab)
  if (hit) return hit.id
  return renameToId(lab)
}

// Feature renames are stored SERVER-SIDE (POST /rename -> a JSON sidecar next to the annotations),
// so they survive reloads/redeploys and are visible to every viewer. `_serverRenames` mirrors that
// store in-process; it's primed once on boot from GET /renames (primeServerRenames). localStorage is
// kept only as an instant, offline-friendly local overlay that wins for immediate feedback.
const _serverRenames = new Map()

// Prime the server-rename overlay on boot. Call once at app startup (see App.jsx). Idempotent.
export async function primeServerRenames() {
  try {
    const data = await getJSON('/renames')
    _serverRenames.clear()
    for (const [k, v] of Object.entries(data || {})) _serverRenames.set(Number(k), v)
    _labelListeners.forEach((fn) => { try { fn() } catch { /* ignore */ } })
  } catch {
    /* backend unreachable (atlas-only mode) — fall back to localStorage overlay */
  }
}

// The effective label overlay: a local (this-browser) rename wins for instant feedback; otherwise the
// server-persisted rename (shared across browsers). Returns null when neither has one.
export function userLabel(id) {
  try {
    const local = localStorage.getItem(`featureTitle_${id}`)
    if (local) return local
  } catch {
    /* private mode — fall through to the server overlay */
  }
  return _serverRenames.get(Number(id)) || null
}

// Subscribers that re-render when any feature is renamed. Needed because tabs are now kept mounted
// (keep-alive), so they no longer remount-and-reread localStorage on a tab switch — without this, a
// rename in one tab wouldn't show in the others until they happened to re-render.
const _labelListeners = new Set()

// Call in any component that displays feature names, so it re-renders when a rename happens anywhere.
// Returns a monotonically-increasing tick, usable as a useEffect dependency to re-run imperative
// label updates (e.g. the WebGPU atlas) on a rename.
export function useUserLabels() {
  const [tick, bump] = useState(0)
  useEffect(() => {
    const fn = () => bump((n) => n + 1)
    _labelListeners.add(fn)
    // Also react to renames from OTHER browser windows/tabs — localStorage fires a native `storage`
    // event cross-window (the in-process _labelListeners set only covers this document).
    const onStorage = (e) => { if (!e.key || e.key.startsWith('featureTitle_')) fn() }
    window.addEventListener('storage', onStorage)
    return () => { _labelListeners.delete(fn); window.removeEventListener('storage', onStorage) }
  }, [])
  return tick
}

// Persist (or clear when blank) a user-provided feature title everywhere: localStorage (instant,
// local), the in-process server overlay (optimistic), and the backend (durable, shared). Then notify
// subscribers so every mounted tab updates. Safe in private-mode/quota and when the backend is down.
export function setUserLabel(id, text) {
  const t = (text || '').trim()
  try {
    if (t) localStorage.setItem(`featureTitle_${id}`, t)
    else localStorage.removeItem(`featureTitle_${id}`)
  } catch {
    /* private mode / quota exceeded — the server copy below is the durable one anyway */
  }
  if (t) _serverRenames.set(Number(id), t)
  else _serverRenames.delete(Number(id))
  // The UI already reflects it optimistically; on server confirm, notify again so the picker tabs
  // refetch /features and the rename becomes searchable there (a failure just means it isn't durable).
  postJSON('/rename', { feature_id: Number(id), label: t })
    .then(() => { _labelListeners.forEach((fn) => { try { fn() } catch { /* ignore */ } }) })
    .catch(() => { /* backend down / atlas-only */ })
  _labelListeners.forEach((fn) => { try { fn() } catch { /* ignore */ } })
}

// Shared by-name feature picker (used by both tabs). withStrength adds a clamp value.
export function FeaturePicker({ catalog, rows, setRows, withStrength, nFeatures }) {
  const byId = useMemo(() => Object.fromEntries(catalog.map((f) => [f.id, f])), [catalog])
  const setRow = (i, patch) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const add = () => setRows((rs) => [...rs, withStrength ? { q: '', strength: 0 } : { q: '' }])
  const del = (i) => setRows((rs) => (rs.length > 1 ? rs.filter((_, j) => j !== i) : rs))
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
      {rows.map((r, i) => {
        const fid = resolveFeatureId(catalog, r.q)
        const f = fid != null ? byId[fid] : null
        const ul = fid != null ? userLabel(fid) : null
        const outOfRange = fid != null && nFeatures != null && (fid < 0 || fid >= nFeatures)
        return (
          <div key={i} style={S.pickRow}>
            <input list="evo2-feature-catalog" value={r.q} onChange={(e) => setRow(i, { q: e.target.value })}
              placeholder="feature name or #id…" style={S.featInput} />
            <span title={f?.description || undefined} style={{ ...S.resolved, ...(outOfRange ? { color: '#ef4444' } : {}) }}>
              {outOfRange
                ? `✗ #${fid} out of range (0–${nFeatures - 1})`
                : ul
                  ? `→ #${fid} ${ul}`
                  : f
                    ? `→ #${f.id} ${f.label}`
                    : fid != null
                      ? `→ #${fid} (unlabeled)`
                      : '— not resolved'}
            </span>
            {withStrength && (
              <span style={S.strengthWrap}>clamp&nbsp;to&nbsp;
                <input type="range" min={0} max={CLAMP_MAX} step={5} value={r.strength}
                  onChange={(e) => setRow(i, { strength: parseFloat(e.target.value) })} style={{ width: '140px' }} />
                <input type="number" min={0} max={CLAMP_MAX} step={5} value={r.strength}
                  onChange={(e) => setRow(i, { strength: Math.max(0, Math.min(CLAMP_MAX, Number(e.target.value))) })} style={S.num} />
                <span style={S.help}>{f?.natural_peak != null ? `peak ≈ ${Math.round(f.natural_peak)} · 0 = suppress` : '0 = suppress'}</span>
              </span>
            )}
            <button onClick={() => del(i)} style={S.del} title="remove">✕</button>
          </div>
        )
      })}
      <div><button onClick={add} style={S.addBtn}>+ Add feature</button></div>
      <datalist id="evo2-feature-catalog">
        {catalog.slice(0, 2000).map((f) => <option key={f.id} value={`#${f.id} ${userLabel(f.id) || f.label}`} />)}
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
    return <div style={{ ...S.banner, ...S.bannerOk }}>● Backend live — Evo2 layer {i.layer}, {i.n_features} SAE features ({i.n_labels} labeled) on {i.device}.</div>
  }
  if (health.status === 'loading') return <div style={{ ...S.banner, ...S.bannerWarn }}>◐ Backend loading model + SAE… (~1 min at startup)</div>
  return <div style={{ ...S.banner, ...S.bannerWarn }}>Backend offline. Start the backend: <code>launch_inference.sh serve</code> on port 8001 (7B, layer 26).</div>
}

// Shared "restart the engine" control for the backend-backed tabs. Shown only when the server enables
// it (`ALLOW_ENGINE_RESTART`, set by launch_inference.sh — it runs under the supervisor that respawns
// the worker). Kills the in-flight request and reloads the model (~1 min) — the only way to free the
// single GPU mid-run. `onRestart` lets the tab clear its own busy/result state and show a note;
// useHealth then polls back to ready on its own.
export function RestartEngineButton({ enabled, busy, onRestart }) {
  if (!enabled) return null
  const click = async () => {
    if (!window.confirm('Restart the engine? This kills the running request and reloads the model (~1 min), affecting all tabs.')) return
    onRestart?.()
    try {
      await fetch(`${BACKEND}/restart`, { method: 'POST' })
    } catch {
      /* the connection drops as the worker exits — expected */
    }
  }
  return (
    <button onClick={click} title="Kill the running request and reload the model (~1 min)"
      style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border,#555)', background: 'transparent', color: 'var(--text)', cursor: 'pointer', fontSize: 12 }}>
      {busy ? 'Cancel — restart engine' : 'Restart engine'}
    </button>
  )
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
