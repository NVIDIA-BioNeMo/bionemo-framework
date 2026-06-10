import React, { useEffect } from 'react'
import ReactDOM from 'react-dom'
import SequenceView from './SequenceView'
import { getRegionLabel } from './utils'

const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.5)',
    zIndex: 9999,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modal: {
    background: '#fff',
    borderRadius: '12px',
    width: '90vw',
    maxWidth: '1000px',
    maxHeight: '85vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
    position: 'relative',
  },
  closeBtn: {
    position: 'absolute',
    top: '12px',
    right: '12px',
    zIndex: 10,
    background: 'rgba(255,255,255,0.9)',
    border: '1px solid #ddd',
    borderRadius: '50%',
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    fontSize: '16px',
    color: '#555',
  },
  body: {
    flex: 1,
    padding: '32px',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    flexWrap: 'wrap',
  },
  regionLabel: {
    fontSize: '18px',
    fontWeight: '700',
    fontFamily: 'monospace',
    color: '#222',
  },
  statsRow: {
    display: 'flex',
    gap: '20px',
    flexWrap: 'wrap',
  },
  statBox: {
    padding: '10px 14px',
    background: '#f9fafb',
    borderRadius: '8px',
    border: '1px solid #eee',
  },
  statLabel: {
    fontSize: '10px',
    color: '#888',
    textTransform: 'uppercase',
    marginBottom: '2px',
  },
  statValue: {
    fontSize: '14px',
    fontWeight: '600',
    fontFamily: 'monospace',
    color: '#333',
  },
  sectionLabel: {
    fontSize: '11px',
    color: '#888',
    textTransform: 'uppercase',
    fontWeight: '500',
  },
  sequenceBox: {
    background: '#fafafa',
    border: '1px solid #eee',
    borderRadius: '8px',
    padding: '12px',
    maxHeight: '300px',
    overflowY: 'auto',
  },
}

export default function RegionDetailModal({ region, onClose }) {
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  const label = getRegionLabel(region)
  const sequenceLength = (region.sequence || '').length

  const modal = (
    <div style={styles.backdrop} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        <div style={styles.closeBtn} onClick={onClose}>x</div>

        <div style={styles.body}>
          <div style={styles.header}>
            <span style={styles.regionLabel}>{label}</span>
          </div>

          <div style={styles.statsRow}>
            <div style={styles.statBox}>
              <div style={styles.statLabel}>Max Activation</div>
              <div style={styles.statValue}>{(region.max_activation || 0).toFixed(4)}</div>
            </div>
            <div style={styles.statBox}>
              <div style={styles.statLabel}>Sequence Length</div>
              <div style={styles.statValue}>{sequenceLength} bp</div>
            </div>
            {region.best_annotation && (
              <div style={styles.statBox}>
                <div style={styles.statLabel}>Annotation</div>
                <div style={styles.statValue}>{region.best_annotation}</div>
              </div>
            )}
          </div>

          <div>
            <div style={styles.sectionLabel}>Sequence (activation highlighted)</div>
            <div style={styles.sequenceBox}>
              <SequenceView
                sequence={region.sequence}
                activations={region.activations}
                maxActivation={region.max_activation}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  return ReactDOM.createPortal(modal, document.body)
}
