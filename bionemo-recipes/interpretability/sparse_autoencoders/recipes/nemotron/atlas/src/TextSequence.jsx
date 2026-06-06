import React, { useState } from 'react'
import { displayToken, parseTokens } from './utils'

/**
 * Activation -> background color. Mirrors the codon dashboard's ramp
 * (white -> NVIDIA-green/amber) so highlighted spans read the same way,
 * but applied to flowing tokenized text instead of fixed-width codons.
 */
function activationColorHex(value, maxValue) {
  if (maxValue <= 0 || value <= 0) return 'transparent'
  const n = Math.min(value / maxValue, 1)
  const r = Math.round(255 - n * 137)
  const g = Math.round(255 - n * 70)
  const b = Math.round(255 * (1 - n))
  const toHex = (c) => c.toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

const styles = {
  container: {
    fontFamily: 'Monaco, Menlo, "Courier New", monospace',
    fontSize: '12px',
    lineHeight: '1.7',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    position: 'relative',
  },
  token: {
    borderRadius: '2px',
    padding: '1px 0',
    cursor: 'default',
  },
  spaceToken: {
    color: 'var(--text-muted)',
  },
  tooltip: {
    position: 'fixed',
    background: 'var(--bg-card)',
    color: 'var(--text)',
    border: '1px solid var(--border)',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '10px',
    fontFamily: 'monospace',
    zIndex: 1000,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
  },
}

export default function TextSequence({ tokens, activations, maxActivation }) {
  const [tooltip, setTooltip] = useState(null)

  const toks = parseTokens(tokens)
  const acts = activations ? activations.slice(0, toks.length) : []
  const maxAct = maxActivation || Math.max(...acts, 0.001)

  if (!toks.length) {
    return <span style={{ color: 'var(--text-muted)' }}>No text</span>
  }

  const handleMouseEnter = (e, token, idx, act) => {
    setTooltip({
      x: e.clientX + 10,
      y: e.clientY - 25,
      text: `"${token}" · pos ${idx + 1} · activation: ${act.toFixed(4)}`,
    })
  }

  const handleMouseMove = (e) => {
    setTooltip((prev) => (prev ? { ...prev, x: e.clientX + 10, y: e.clientY - 25 } : null))
  }

  const handleMouseLeave = () => setTooltip(null)

  return (
    <div style={styles.container}>
      {toks.map((token, idx) => {
        const act = acts[idx] || 0
        const bg = activationColorHex(act, maxAct)
        const { display, isSpace } = displayToken(token)
        const hasActivation = act > 0
        return (
          <span
            key={idx}
            style={{
              ...styles.token,
              ...(isSpace ? styles.spaceToken : {}),
              backgroundColor: bg,
              ...(hasActivation ? { color: '#000' } : {}),
            }}
            onMouseEnter={(e) => handleMouseEnter(e, token, idx, act)}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            {display}
          </span>
        )
      })}
      {tooltip && (
        <span style={{ ...styles.tooltip, left: tooltip.x, top: tooltip.y }}>
          {tooltip.text}
        </span>
      )}
    </div>
  )
}
