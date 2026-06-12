import React from 'react'

// Interpolate between white and NVIDIA green (#76b900) based on activation
function getTokenStyle(activation, maxActivation) {
  if (!activation || activation <= 0) {
    return { background: 'transparent' }
  }

  // Normalize to 0-1 range
  const normalized = Math.min(activation / (maxActivation || 1), 1)

  // Interpolate from white (255,255,255) to NVIDIA green (118,185,0)
  const r = Math.round(255 - normalized * (255 - 118))  // 255 -> 118
  const g = Math.round(255 - normalized * (255 - 185))  // 255 -> 185
  const b = Math.round(255 - normalized * (255 - 0))    // 255 -> 0

  return {
    background: `rgb(${r}, ${g}, ${b})`,
    borderRadius: '2px',
  }
}

const styles = {
  container: {
    fontFamily: 'monospace',
    fontSize: '12px',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  token: {
    padding: '1px 2px',
    margin: '0 -1px',
    display: 'inline',
    cursor: 'default',
  },
  tooltip: {
    position: 'absolute',
    background: '#333',
    color: '#fff',
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '10px',
    fontFamily: 'monospace',
    zIndex: 1000,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
  },
}

function Token({ token, activation, maxActivation }) {
  const [showTooltip, setShowTooltip] = React.useState(false)
  const [tooltipPos, setTooltipPos] = React.useState({ x: 0, y: 0 })

  const handleMouseEnter = (e) => {
    if (activation > 0) {
      setShowTooltip(true)
      setTooltipPos({ x: e.clientX + 10, y: e.clientY - 25 })
    }
  }

  const handleMouseLeave = () => {
    setShowTooltip(false)
  }

  const handleMouseMove = (e) => {
    if (showTooltip) {
      setTooltipPos({ x: e.clientX + 10, y: e.clientY - 25 })
    }
  }

  // Handle special characters for display
  let displayToken = token
  if (token === '\n') displayToken = '\u21b5\n'
  if (token === '\t') displayToken = '\u2192\t'

  return (
    <>
      <span
        style={{ ...styles.token, ...getTokenStyle(activation, maxActivation) }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onMouseMove={handleMouseMove}
      >
        {displayToken}
      </span>
      {showTooltip && (
        <span
          style={{
            ...styles.tooltip,
            position: 'fixed',
            left: tooltipPos.x,
            top: tooltipPos.y,
          }}
        >
          {activation.toFixed(3)}
        </span>
      )}
    </>
  )
}

export default function TokenHighlight({ tokens }) {
  if (!tokens || tokens.length === 0) {
    return <span style={{ color: '#999' }}>No tokens</span>
  }

  // Find max activation for normalization
  const maxActivation = Math.max(...tokens.map(t => t.activation || 0), 0.001)

  return (
    <div style={styles.container}>
      {tokens.map((t, i) => (
        <Token
          key={i}
          token={t.token}
          activation={t.activation || 0}
          maxActivation={maxActivation}
        />
      ))}
    </div>
  )
}
