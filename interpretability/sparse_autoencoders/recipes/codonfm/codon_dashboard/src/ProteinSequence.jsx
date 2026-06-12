import React, { useState, useEffect, useRef } from 'react'
import { codonToAA, parseCodons } from './utils'

function activationColorHex(value, maxValue) {
  if (maxValue <= 0 || value <= 0) return 'transparent'
  const n = Math.min(value / maxValue, 1)
  const r = Math.round(255 - n * 137)
  const g = Math.round(255 - n * 70)
  const b = Math.round(255 * (1 - n))
  const toHex = (c) => c.toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

const CODON_WIDTH = 30

const styles = {
  container: {
    fontFamily: 'Monaco, Menlo, "Courier New", monospace',
    fontSize: '11px',
    lineHeight: '1.2',
    overflowX: 'auto',
    position: 'relative',
  },
  codonRow: {
    display: 'inline-flex',
    whiteSpace: 'nowrap',
  },
  codonBlock: {
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    cursor: 'default',
    borderRadius: '2px',
    padding: '1px 2px',
    marginRight: '1px',
    minWidth: `${CODON_WIDTH - 1}px`,
  },
  padBlock: {
    display: 'inline-flex',
    flexDirection: 'column',
    alignItems: 'center',
    borderRadius: '2px',
    padding: '1px 2px',
    marginRight: '1px',
    minWidth: `${CODON_WIDTH - 1}px`,
    background: 'var(--density-bar-bg)',
  },
  padText: {
    fontSize: '10px',
    color: 'var(--text-muted)',
  },
  codonText: {
    fontSize: '10px',
    letterSpacing: '0.5px',
    color: 'var(--text)',
  },
  aaText: {
    fontSize: '9px',
    color: 'var(--text-secondary)',
    marginTop: '1px',
  },
  idxText: {
    fontSize: '7px',
    color: 'var(--text-tertiary)',
    marginTop: '0px',
    lineHeight: '1',
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

// Show index under every codon
const INDEX_INTERVAL = 1

export default function ProteinSequence({
  sequence, activations, maxActivation,
  alignMode, alignAnchor, totalLength,
  scrollGroupRef,
}) {
  const [tooltip, setTooltip] = useState(null)
  const scrollRef = useRef(null)
  const anchorRef = useRef(null)

  const codons = parseCodons(sequence)
  const acts = activations ? activations.slice(0, codons.length) : []
  const maxAct = maxActivation || Math.max(...acts, 0.001)

  // Compute local anchor index
  let localAnchor = 0
  if (alignMode === 'first_activation') {
    localAnchor = acts.findIndex(a => a > 0)
    if (localAnchor < 0) localAnchor = 0
  } else if (alignMode === 'max_activation') {
    let maxVal = -1
    acts.forEach((a, i) => { if (a > maxVal) { maxVal = a; localAnchor = i } })
  }

  // Padding
  const isAligned = alignMode && alignMode !== 'start' && alignAnchor != null
  const leftPad = isAligned ? Math.max(0, alignAnchor - localAnchor) : 0
  const rightPad = (totalLength != null)
    ? Math.max(0, totalLength - leftPad - codons.length)
    : 0

  // Scroll to anchor when alignMode changes
  useEffect(() => {
    if (isAligned && anchorRef.current && scrollRef.current) {
      anchorRef.current.scrollIntoView({ behavior: 'instant', inline: 'center', block: 'nearest' })
    }
  }, [alignMode, alignAnchor])

  // Synchronized scrolling across sequences in the same card
  useEffect(() => {
    const el = scrollRef.current
    if (!el || !scrollGroupRef) return

    // Register this element in the group
    if (!scrollGroupRef.current) scrollGroupRef.current = []
    const group = scrollGroupRef.current
    if (!group.includes(el)) group.push(el)

    let isSyncing = false
    const handleScroll = () => {
      if (isSyncing) return
      isSyncing = true
      const scrollLeft = el.scrollLeft
      for (const other of group) {
        if (other !== el) other.scrollLeft = scrollLeft
      }
      isSyncing = false
    }

    el.addEventListener('scroll', handleScroll)
    return () => {
      el.removeEventListener('scroll', handleScroll)
      const idx = group.indexOf(el)
      if (idx !== -1) group.splice(idx, 1)
    }
  }, [scrollGroupRef])

  if (!sequence || sequence.length === 0) {
    return <span style={{ color: 'var(--text-muted)' }}>No sequence</span>
  }

  const handleMouseEnter = (e, codon, aa, idx, act) => {
    setTooltip({
      x: e.clientX + 10,
      y: e.clientY - 25,
      text: `${codon} (${aa}) pos ${idx + 1} — activation: ${act.toFixed(4)}`,
    })
  }

  const handleMouseMove = (e) => {
    if (tooltip) {
      setTooltip((prev) => prev ? { ...prev, x: e.clientX + 10, y: e.clientY - 25 } : null)
    }
  }

  const handleMouseLeave = () => {
    setTooltip(null)
  }

  const shouldShowIdx = () => true

  return (
    <div style={styles.container} ref={scrollRef}>
      <div style={styles.codonRow}>
        {/* Left padding */}
        {Array.from({ length: leftPad }, (_, i) => (
          <span key={`lpad-${i}`} style={styles.padBlock}>
            <span style={styles.padText}>&middot;</span>
            <span style={{ ...styles.aaText, color: 'var(--text-muted)' }}>&middot;</span>
            <span style={styles.idxText}>&nbsp;</span>
          </span>
        ))}

        {/* Actual codons */}
        {codons.map((codon, idx) => {
          const act = acts[idx] || 0
          const bg = activationColorHex(act, maxAct)
          const aa = codonToAA(codon)
          const isAnchor = isAligned && idx === localAnchor
          const hasActivation = act > 0
          const activeTextColor = hasActivation ? '#000' : undefined
          return (
            <span
              key={idx}
              ref={isAnchor ? anchorRef : null}
              style={{
                ...styles.codonBlock,
                backgroundColor: bg,
                ...(isAnchor ? { outline: '2px solid #76b900', outlineOffset: '-1px' } : {}),
              }}
              onMouseEnter={(e) => handleMouseEnter(e, codon, aa, idx, act)}
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            >
              <span style={{ ...styles.codonText, ...(activeTextColor && { color: activeTextColor }) }}>{codon}</span>
              <span style={{ ...styles.aaText, ...(activeTextColor && { color: activeTextColor }) }}>{aa}</span>
              <span style={{ ...styles.idxText, ...(activeTextColor && { color: '#333' }) }}>{shouldShowIdx(idx) ? idx + 1 : '\u00A0'}</span>
            </span>
          )
        })}

        {/* Right padding */}
        {Array.from({ length: rightPad }, (_, i) => (
          <span key={`rpad-${i}`} style={styles.padBlock}>
            <span style={styles.padText}>&middot;</span>
            <span style={{ ...styles.aaText, color: 'var(--text-muted)' }}>&middot;</span>
            <span style={styles.idxText}>&nbsp;</span>
          </span>
        ))}
      </div>
      {tooltip && (
        <span style={{ ...styles.tooltip, left: tooltip.x, top: tooltip.y }}>
          {tooltip.text}
        </span>
      )}
    </div>
  )
}

/**
 * Compute alignment info for a set of examples.
 */
export function computeAlignInfo(examples, alignMode) {
  if (!examples || examples.length === 0) return { anchor: 0, totalLength: 0 }

  if (alignMode === 'start') {
    // No alignment padding, just pad to longest sequence
    const maxLen = Math.max(...examples.map(ex => (ex.activations || []).length))
    return { anchor: 0, totalLength: maxLen }
  }

  let maxAnchor = 0
  for (const ex of examples) {
    const acts = ex.activations || []
    let anchor = 0
    if (alignMode === 'first_activation') {
      anchor = acts.findIndex(a => a > 0)
      if (anchor < 0) anchor = 0
    } else if (alignMode === 'max_activation') {
      let maxVal = -1
      acts.forEach((a, i) => { if (a > maxVal) { maxVal = a; anchor = i } })
    }
    if (anchor > maxAnchor) maxAnchor = anchor
  }

  // Compute totalLength with final maxAnchor
  let totalLength = 0
  for (const ex of examples) {
    const acts = ex.activations || []
    let anchor = 0
    if (alignMode === 'first_activation') {
      anchor = acts.findIndex(a => a > 0)
      if (anchor < 0) anchor = 0
    } else if (alignMode === 'max_activation') {
      let maxVal = -1
      acts.forEach((a, i) => { if (a > maxVal) { maxVal = a; anchor = i } })
    }
    const leftPad = maxAnchor - anchor
    const thisTotal = leftPad + acts.length
    if (thisTotal > totalLength) totalLength = thisTotal
  }

  return { anchor: maxAnchor, totalLength }
}
