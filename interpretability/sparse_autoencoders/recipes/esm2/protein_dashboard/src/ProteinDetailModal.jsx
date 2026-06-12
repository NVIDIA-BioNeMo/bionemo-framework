import React, { useEffect, useRef, useState } from 'react'
import ReactDOM from 'react-dom'
import ProteinSequence from './ProteinSequence'
import { getAccession, uniprotUrl } from './utils'

const styles = {
  backdrop: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.45)',
    zIndex: 9999,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modal: {
    background: 'var(--bg-card)',
    borderRadius: '10px',
    width: '90vw',
    maxWidth: '1200px',
    height: '80vh',
    maxHeight: '800px',
    display: 'flex',
    overflow: 'hidden',
    boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
    position: 'relative',
  },
  closeBtn: {
    position: 'absolute',
    top: '12px',
    right: '12px',
    zIndex: 10,
    background: 'var(--bg-card)',
    border: '1px solid var(--border-input)',
    borderRadius: '50%',
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    fontSize: '16px',
    color: 'var(--text-secondary)',
  },
  leftPanel: {
    flex: '0 0 60%',
    position: 'relative',
    background: 'var(--bg)',
    borderRight: '1px solid var(--border-light)',
  },
  viewer: {
    width: '100%',
    height: '100%',
    position: 'relative',
  },
  viewerLoading: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'var(--text-muted)',
    fontSize: '13px',
  },
  viewerError: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#e57373',
    fontSize: '13px',
  },
  rightPanel: {
    flex: 1,
    padding: '28px 32px',
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
  proteinId: {
    fontSize: '18px',
    fontWeight: '700',
    fontFamily: 'monospace',
    color: 'var(--text-heading)',
  },
  uniprotBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    padding: '4px 10px',
    fontSize: '12px',
    color: 'var(--link)',
    background: 'var(--bg-card-expanded)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    textDecoration: 'none',
    fontWeight: '500',
  },
  statsRow: {
    display: 'flex',
    gap: '20px',
  },
  statBox: {
    padding: '10px 14px',
    background: 'var(--bg-card-expanded)',
    borderRadius: '8px',
    border: '1px solid var(--border-light)',
  },
  statLabel: {
    fontSize: '10px',
    color: 'var(--text-tertiary)',
    textTransform: 'uppercase',
    marginBottom: '2px',
  },
  statValue: {
    fontSize: '14px',
    fontWeight: '600',
    fontFamily: 'monospace',
    color: 'var(--text)',
  },
  sectionLabel: {
    fontSize: '11px',
    color: 'var(--text-tertiary)',
    textTransform: 'uppercase',
    fontWeight: '500',
  },
  sequenceBox: {
    background: 'var(--bg-card-expanded)',
    border: '1px solid var(--border-light)',
    borderRadius: '8px',
    padding: '12px',
    maxHeight: '300px',
    overflowY: 'auto',
  },
}

export default function ProteinDetailModal({ protein, onClose, darkMode }) {
  const wrapperRef = useRef(null)
  const molContainerRef = useRef(null)
  const pluginRef = useRef(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const accession = getAccession(protein.protein_id, protein.alphafold_id)
  const acts = protein.activations ? protein.activations.slice(0, (protein.sequence || '').length) : []

  // Close on ESC
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [onClose])

  // Init Mol* viewer
  useEffect(() => {
    let disposed = false

    async function init() {
      if (!wrapperRef.current) return
      setLoading(true)
      setError(null)

      if (molContainerRef.current) {
        try { wrapperRef.current.removeChild(molContainerRef.current) } catch {}
        molContainerRef.current = null
      }

      const molDiv = document.createElement('div')
      molDiv.style.width = '100%'
      molDiv.style.height = '100%'
      molDiv.style.position = 'relative'
      wrapperRef.current.appendChild(molDiv)
      molContainerRef.current = molDiv

      try {
        const molPlugin = await import('molstar/lib/mol-plugin/context')
        const molSpec = await import('molstar/lib/mol-plugin/spec')
        const molConfig = await import('molstar/lib/mol-plugin/config')
        const molColor = await import('molstar/lib/mol-util/color')

        if (disposed) return

        const spec = molSpec.DefaultPluginSpec()
        spec.config = spec.config || []
        spec.config.push([molConfig.PluginConfig.Viewport.ShowExpand, false])
        spec.config.push([molConfig.PluginConfig.Viewport.ShowControls, true])
        spec.config.push([molConfig.PluginConfig.Viewport.ShowSelectionMode, false])
        spec.config.push([molConfig.PluginConfig.Viewport.ShowAnimation, false])

        const plugin = new molPlugin.PluginContext(spec)
        await plugin.init()

        if (disposed) { plugin.dispose(); return }

        const canvas = document.createElement('canvas')
        canvas.style.width = '100%'
        canvas.style.height = '100%'
        molDiv.appendChild(canvas)

        try { plugin.initViewer(canvas, molDiv) } catch {}

        // Set canvas background based on dark mode
        try {
          const bgColor = darkMode ? 0x000000 : 0xffffff
          plugin.canvas3d?.setProps({
            renderer: { backgroundColor: bgColor },
          })
        } catch { /* older Mol* versions may not support this */ }

        pluginRef.current = plugin

        // Custom activation color theme
        const themeName = `activation-detail-${protein.protein_id}`
        const Color = molColor.Color
        const maxAct = protein.max_activation || 0

        const colorFn = (location) => {
          try {
            if (location && location.unit && location.element !== undefined) {
              const unit = location.unit
              if (unit.model && unit.model.atomicHierarchy) {
                const { residueAtomSegments, residues } = unit.model.atomicHierarchy
                const rI = residueAtomSegments.index[location.element]
                const seqId = residues.auth_seq_id.value(rI)
                const idx = seqId - 1
                if (idx >= 0 && idx < acts.length) {
                  const act = acts[idx]
                  const n = maxAct > 0 ? Math.min(act / maxAct, 1) : 0
                  const r = Math.round(255 - n * 137)  // 255 -> 118
                  const g = Math.round(255 - n * 70)   // 255 -> 185
                  const b = Math.round(255 * (1 - n))  // 255 -> 0
                  return Color.fromRgb(r, g, b)
                }
              }
            }
          } catch {}
          return darkMode ? Color.fromRgb(80, 80, 80) : Color.fromRgb(200, 200, 200)
        }

        const colorThemeProvider = {
          name: themeName,
          label: themeName,
          category: 'Custom',
          factory: (_ctx, props) => ({
            factory: colorThemeProvider,
            granularity: 'group',
            props,
            description: '',
            color: colorFn,
            legend: undefined,
          }),
          getParams: () => ({}),
          defaultValues: {},
          isApplicable: () => true,
        }

        plugin.representation.structure.themes.colorThemeRegistry.add(colorThemeProvider)

        // Load AlphaFold CIF with version fallback
        let data = null
        for (const version of [6, 4, 3]) {
          const cifUrl = `https://alphafold.ebi.ac.uk/files/AF-${accession}-F1-model_v${version}.cif`
          try {
            data = await plugin.builders.data.download(
              { url: cifUrl, isBinary: false },
              { state: { isGhost: true } },
            )
            break
          } catch {}
        }

        if (!data) throw new Error('Structure not found')

        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'mmcif')
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default')

        const structures = plugin.managers.structure.hierarchy.current.structures
        for (const s of structures) {
          for (const c of s.components) {
            await plugin.managers.structure.component.updateRepresentationsTheme(
              [c],
              { color: themeName },
            )
          }
        }

        if (disposed) { plugin.dispose(); return }
        setLoading(false)
      } catch (err) {
        if (!disposed) {
          setError(err instanceof Error ? err.message : 'Failed to load')
          setLoading(false)
        }
      }
    }

    init()

    return () => {
      disposed = true
      if (pluginRef.current) {
        pluginRef.current.dispose()
        pluginRef.current = null
      }
      if (molContainerRef.current && wrapperRef.current) {
        try { wrapperRef.current.removeChild(molContainerRef.current) } catch {}
        molContainerRef.current = null
      }
    }
  }, [protein.protein_id, accession])

  const modal = (
    <div style={styles.backdrop} onClick={onClose}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        <div style={styles.closeBtn} onClick={onClose}>x</div>

        {/* Left: Mol* viewer */}
        <div style={styles.leftPanel}>
          <div ref={wrapperRef} style={styles.viewer}>
            {loading && <div style={styles.viewerLoading}>Loading structure...</div>}
            {error && <div style={styles.viewerError}>{error}</div>}
          </div>
        </div>

        {/* Right: Protein info */}
        <div style={styles.rightPanel}>
          <div style={styles.header}>
            <span style={styles.proteinId}>{protein.protein_id}</span>
            <a
              href={uniprotUrl(accession)}
              target="_blank"
              rel="noopener noreferrer"
              style={styles.uniprotBtn}
            >
              UniProt ↗
            </a>
          </div>

          <div style={styles.statsRow}>
            <div style={styles.statBox}>
              <div style={styles.statLabel}>Max Activation</div>
              <div style={styles.statValue}>{(protein.max_activation || 0).toFixed(4)}</div>
            </div>
            <div style={styles.statBox}>
              <div style={styles.statLabel}>Sequence Length</div>
              <div style={styles.statValue}>{(protein.sequence || '').length}</div>
            </div>
            {protein.best_annotation && (
              <div style={styles.statBox}>
                <div style={styles.statLabel}>Annotation</div>
                <div style={styles.statValue}>{protein.best_annotation}</div>
              </div>
            )}
          </div>

          <div>
            <div style={styles.sectionLabel}>Sequence (activation highlighted)</div>
            <div style={styles.sequenceBox}>
              <ProteinSequence
                sequence={protein.sequence}
                activations={protein.activations}
                maxActivation={protein.max_activation}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  return ReactDOM.createPortal(modal, document.body)
}
