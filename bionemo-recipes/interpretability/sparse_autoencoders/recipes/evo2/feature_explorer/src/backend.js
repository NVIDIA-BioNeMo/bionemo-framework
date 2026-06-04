// Shared helpers for the live backend (steering_server.py).
//
// All calls go through the Vite dev-server proxy (/api -> http://localhost:8001),
// so only the Vite port needs to be tunneled. Override with VITE_BACKEND.
import { useEffect, useRef, useState } from 'react'

export const BACKEND = (import.meta.env && import.meta.env.VITE_BACKEND) || '/api'

// Per-nucleotide letter colors (shared with the steering strips).
export const BASE_COLORS = { A: '#59A14F', C: '#4E79A7', G: '#F28E2B', T: '#E15759', N: '#888', U: '#E15759' }

// Poll /health so each tab can show a live banner and react when the model/SAE
// finish loading. status: 'loading' | 'ready' | 'offline'.
export function useHealth(pollMs = 4000) {
  const [health, setHealth] = useState({ status: 'loading' })
  const timer = useRef(null)
  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const r = await fetch(`${BACKEND}/health`, { cache: 'no-store' })
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const info = await r.json()
        if (alive) setHealth({ status: info.ready ? 'ready' : 'loading', info })
      } catch (e) {
        if (alive) setHealth({ status: 'offline', error: String(e) })
      }
    }
    tick()
    timer.current = setInterval(tick, pollMs)
    return () => {
      alive = false
      clearInterval(timer.current)
    }
  }, [pollMs])
  return health
}

// Activation -> background color, absolute 0->max. No (or negligible) activation
// is fully clear; real activation ramps clear -> green and saturates toward `max`.
// Theme-aware: dark-green on the light theme, brighter green on the dark theme so
// the high end stays legible against the background.
export function activationColor(value, max) {
  if (!(max > 0) || value <= 0) return 'transparent'
  const t = Math.max(0, Math.min(1, value / max))
  if (t < 0.03) return 'transparent' // keep the floor clean — only real activation shows
  const dark = typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  const lo = dark ? [46, 92, 46] : [221, 242, 210] // near-clear green
  const hi = dark ? [126, 217, 87] : [9, 74, 22] // saturated -> dark green
  const r = Math.round(lo[0] + (hi[0] - lo[0]) * t)
  const g = Math.round(lo[1] + (hi[1] - lo[1]) * t)
  const b = Math.round(lo[2] + (hi[2] - lo[2]) * t)
  return `rgba(${r}, ${g}, ${b}, ${(0.18 + 0.82 * t).toFixed(3)})`
}

export function cleanDNA(raw) {
  return (raw || '').toUpperCase().replace(/[^ACGTN]/g, '')
}

export async function postJSON(path, body) {
  const r = await fetch(`${BACKEND}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = `HTTP ${r.status}`
    try {
      const j = await r.json()
      detail = j.detail || detail
    } catch (_) {}
    throw new Error(detail)
  }
  return r.json()
}

export async function getJSON(path) {
  const r = await fetch(`${BACKEND}${path}`, { cache: 'no-store' })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}
