import React from 'react'
import { displayToken } from './utils'

/**
 * Decoder logits for a text-LM SAE feature.
 *
 * The codon dashboard grouped 64 codons by amino acid; here the vocabulary is
 * the language model's token vocabulary, so we just show the top promoted and
 * suppressed tokens as ranked chips with a bar proportional to |logit|.
 *
 * `logits` shape: { top_positive: [[token, value], ...], top_negative: [[token, value], ...] }
 */
export default function TokenLogits({ logits, limit = 12, compact = false }) {
  if (!logits) return null

  const positive = (logits.top_positive || []).slice(0, limit)
  const negative = (logits.top_negative || []).slice(0, limit)
  if (positive.length === 0 && negative.length === 0) return null

  const maxAbs = Math.max(
    ...positive.map(([, v]) => Math.abs(v)),
    ...negative.map(([, v]) => Math.abs(v)),
    0.001,
  )

  const fontSize = compact ? '10px' : '11px'

  const Column = ({ title, entries, color }) => (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: '9px', fontWeight: '600', color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '4px' }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
        {entries.map(([token, val], i) => {
          const { display } = displayToken(token)
          const w = Math.max(4, (Math.abs(val) / maxAbs) * 100)
          return (
            <div key={i} style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div
                style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${w}%`, background: color, opacity: 0.18, borderRadius: '2px',
                }}
              />
              <span style={{
                position: 'relative', fontFamily: 'monospace', fontSize, color: 'var(--text)',
                whiteSpace: 'pre', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1, padding: '1px 4px',
              }} title={JSON.stringify(token)}>
                {display === '' ? '∅' : display}
              </span>
              <span style={{ position: 'relative', fontFamily: 'monospace', fontSize: '9px', color: 'var(--text-muted)', flexShrink: 0 }}>
                {val > 0 ? '+' : ''}{val.toFixed(2)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <div style={{ display: 'flex', gap: '14px' }}>
      <Column title="Promoted" entries={positive} color="#76b900" />
      <Column title="Suppressed" entries={negative} color="#e57373" />
    </div>
  )
}
