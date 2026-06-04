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

// Viridis — the de-facto perceptually-uniform scientific colormap (matplotlib default).
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]
const _l = (a, b, t) => Math.round(a + (b - a) * t)

export function viridis(t) {
  t = Math.max(0, Math.min(1, t))
  const n = VIRIDIS.length - 1
  const x = t * n
  const i = Math.min(n - 1, Math.floor(x))
  const f = x - i
  const a = VIRIDIS[i]
  const b = VIRIDIS[i + 1]
  return [_l(a[0], b[0], f), _l(a[1], b[1], f), _l(a[2], b[2], f)]
}

// CSS gradient for the legend bar.
export function legendGradient() {
  return (
    'linear-gradient(90deg,' +
    VIRIDIS.map((c, i) => `rgb(${c[0]},${c[1]},${c[2]}) ${Math.round((100 * i) / (VIRIDIS.length - 1))}%`).join(',') +
    ')'
  )
}

// Activation -> Viridis color, absolute 0->max. Alpha ramps in so zero activation
// is fully clear (no fill) and intensity rises toward `max`.
export function activationColor(value, max) {
  if (!(max > 0) || value <= 0) return 'transparent'
  const t = Math.max(0, Math.min(1, value / max))
  if (t < 0.02) return 'transparent'
  const [r, g, b] = viridis(t)
  return `rgba(${r}, ${g}, ${b}, ${(0.22 + 0.78 * t).toFixed(3)})`
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
