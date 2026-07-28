// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Apache2
//
// Shared widgets + styles for the live-backend panes (Sequence inspector, Generative
// steering, …). These used to live inside SequenceInspector.jsx, which made one pane a
// dependency of another; they belong in a neutral module both panes import from.

import React, { useMemo, useState, useEffect } from 'react'
import { BACKEND, activationColor, legendGradient } from './backend'

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
export function resolveFeatureId(catalog, q) {
  // A bare numeric id resolves to ANY feature (the catalog only lists the labeled subset, but
  // every feature 0..n_features-1 is clampable). Otherwise match an exact label from the catalog.
  const m = String(q).match(/#?(\d+)/)
  if (m) {
    const id = Number(m[1])
    if (Number.isInteger(id) && id >= 0) return id
  }
  const lab = String(q).trim()
  if (!lab) return null
  const exact = catalog.find((f) => f.label === lab)
  if (exact) return exact.id
  // Fall back to the best (highest-peak, since the catalog is peak-sorted) label containing the query,
  // so typing a concept like "Paris" resolves even without picking an autocomplete suggestion.
  const low = lab.toLowerCase()
  const sub = catalog.find((f) => (f.label || '').toLowerCase().includes(low))
  return sub ? sub.id : null
}

// In-UI feature renames live in localStorage (featureTitle_<id>). Overlay them so a name set
// in the atlas also shows in the steering/inspector pickers (cross-tab carry-over, per browser).
export function userLabel(id) {
  try {
    return localStorage.getItem(`featureTitle_${id}`) || null
  } catch {
    return null
  }
}

// Subscribers that re-render when any feature is renamed. Needed because tabs are now kept mounted
// (keep-alive), so they no longer remount-and-reread localStorage on a tab switch — without this, a
// rename in one tab wouldn't show in the others until they happened to re-render.
const _labelListeners = new Set()

// Call in any component that displays feature names, so it re-renders when a rename happens anywhere.
export function useUserLabels() {
  const [, bump] = useState(0)
  useEffect(() => {
    const fn = () => bump((n) => n + 1)
    _labelListeners.add(fn)
    // Cross-window: _labelListeners is per-document, so setUserLabel only notifies THIS tab/window.
    // localStorage is shared across same-origin tabs, and the `storage` event fires in the *other*
    // documents (never the one that wrote it) — so listen for it to carry a rename made in another
    // browser tab/window into this one. Together with _labelListeners (same-document) this covers
    // both in-app tabs and separate browser windows. e.key === null is a localStorage.clear().
    const onStorage = (e) => { if (e.key == null || e.key.startsWith('featureTitle_')) fn() }
    window.addEventListener('storage', onStorage)
    return () => { _labelListeners.delete(fn); window.removeEventListener('storage', onStorage) }
  }, [])
}

// Persist (or clear when blank) a user-provided feature title, then notify subscribers so every
// mounted tab updates. Safe in private-mode/quota: a throwing localStorage is swallowed.
export function setUserLabel(id, text) {
  try {
    const t = (text || '').trim()
    if (t) localStorage.setItem(`featureTitle_${id}`, t)
    else localStorage.removeItem(`featureTitle_${id}`)
  } catch {
    /* private mode / quota exceeded — ignore */
  }
  _labelListeners.forEach((fn) => { try { fn() } catch { /* ignore */ } })
  // Write-through to the backend so the rename PERSISTS and is shared across browsers/users. Best-effort:
  // the localStorage cache above already updated the UI, so a down/absent backend (static mode) is fine.
  try {
    fetch(`${BACKEND}/label`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature_id: Number(id), label: (text || '').trim() }),
    }).catch(() => {})
  } catch {
    /* ignore */
  }
}

// Pull the backend's persisted renames into the local cache on app load, so this browser (e.g. a
// teammate's) sees names saved elsewhere. Overlays onto localStorage and re-renders subscribers.
export async function hydrateUserLabels() {
  try {
    const r = await fetch(`${BACKEND}/labels`, { cache: 'no-store' })
    if (!r.ok) return
    const map = await r.json()
    let changed = false
    for (const [id, text] of Object.entries(map || {})) {
      if (text) {
        try { localStorage.setItem(`featureTitle_${id}`, text); changed = true } catch { /* ignore */ }
      }
    }
    if (changed) _labelListeners.forEach((fn) => { try { fn() } catch { /* ignore */ } })
  } catch {
    /* offline / no backend — the localStorage cache stands */
  }
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
        // Live typeahead over the full labeled catalog (client-side, top 50) so any of the ~24k
        // Neuronpedia concepts is discoverable without shipping 24k <option>s per row.
        const ql = r.q.trim().toLowerCase()
        const sugg = (ql ? catalog.filter((c) => (c.label || '').toLowerCase().includes(ql)) : catalog).slice(0, 50)
        return (
          <div key={i} style={S.pickRow}>
            <input list={`feat-cat-${i}`} value={r.q} onChange={(e) => setRow(i, { q: e.target.value })}
              placeholder="search a concept or #id…" style={S.featInput} />
            <datalist id={`feat-cat-${i}`}>
              {sugg.map((c) => <option key={c.id} value={`#${c.id} ${userLabel(c.id) || c.label}`} />)}
            </datalist>
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
    </div>
  )
}

// Organism / phylogenetic tagging is a genomics-only concept; for natural-language text there is only
// "raw text", so the field is hidden. Kept as a no-op export so existing imports/props still resolve.
export function OrganismField() {
  return null
}

export function BackendBanner({ health }) {
  if (health.status === 'ready') {
    const i = health.info || {}
    return <div style={{ ...S.banner, ...S.bannerOk }}>● Backend live — GPT-2 layer {i.layer}, {i.n_features} SAE features ({i.n_labels} labeled) on {i.device}.</div>
  }
  if (health.status === 'loading') return <div style={{ ...S.banner, ...S.bannerWarn }}>◐ Backend loading model + SAE… (~1 min at startup)</div>
  return <div style={{ ...S.banner, ...S.bannerWarn }}>Backend offline. Start it with <code>python gpt2_server.py 8749</code>.</div>
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
