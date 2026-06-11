import React, { useEffect, useRef, useState } from 'react'
import { getAccession } from './utils'

const styles = {
  container: {
    background: '#fafafa',
    borderRadius: '6px',
    border: '1px solid #eee',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
  },
  viewer: {
    position: 'relative',
    height: '180px',
    width: '100%',
  },
  expandBtn: {
    position: 'absolute',
    top: '6px',
    right: '6px',
    zIndex: 20,
    background: 'rgba(255,255,255,0.85)',
    border: '1px solid #ddd',
    borderRadius: '4px',
    width: '24px',
    height: '24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: 'pointer',
    fontSize: '12px',
    color: '#555',
    opacity: 0,
    transition: 'opacity 0.15s',
    pointerEvents: 'auto',
  },
  label: {
    padding: '4px 8px',
    fontSize: '10px',
    fontFamily: 'monospace',
    color: '#555',
    borderTop: '1px solid #eee',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  proteinId: {
    fontWeight: '600',
    color: '#2563eb',
  },
  activation: {
    color: '#999',
  },
  loading: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#f9f9f9',
    zIndex: 10,
    pointerEvents: 'none',
    fontSize: '10px',
    color: '#aaa',
  },
  error: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#f9f9f9',
    zIndex: 10,
    fontSize: '10px',
    color: '#e57373',
  },
}

export default function MolstarThumbnail({ proteinId, sequence, activations, maxActivation, onExpand }) {
  const wrapperRef = useRef(null)
  const molContainerRef = useRef(null)
  const pluginRef = useRef(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [hovered, setHovered] = useState(false)

  const uniprotId = getAccession(proteinId)

  // Trim activations to sequence length
  const acts = activations ? activations.slice(0, (sequence || '').length) : []

  useEffect(() => {
    let disposed = false

    async function init() {
      if (!wrapperRef.current) return
      setLoading(true)
      setError(null)

      // Clean up previous Mol* container
      if (molContainerRef.current) {
        try {
          wrapperRef.current.removeChild(molContainerRef.current)
        } catch { /* already removed */ }
        molContainerRef.current = null
      }

      // Create imperative container (outside React's DOM tree)
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
        spec.config.push([molConfig.PluginConfig.Viewport.ShowControls, false])
        spec.config.push([molConfig.PluginConfig.Viewport.ShowSelectionMode, false])
        spec.config.push([molConfig.PluginConfig.Viewport.ShowAnimation, false])

        const plugin = new molPlugin.PluginContext(spec)
        await plugin.init()

        if (disposed) {
          plugin.dispose()
          return
        }

        const canvas = document.createElement('canvas')
        canvas.style.width = '100%'
        canvas.style.height = '100%'
        molDiv.appendChild(canvas)

        try {
          plugin.initViewer(canvas, molDiv)
        } catch { /* fallback for different Mol* versions */ }

        pluginRef.current = plugin

        // Custom activation color theme
        const themeName = `activation-thumb-${proteinId}`
        const Color = molColor.Color

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
                  const n = maxActivation > 0 ? Math.min(act / maxActivation, 1) : 0
                  const r = Math.round(255 - n * 137)  // 255 -> 118
                  const g = Math.round(255 - n * 70)   // 255 -> 185
                  const b = Math.round(255 * (1 - n))  // 255 -> 0
                  return Color.fromRgb(r, g, b)
                }
              }
            }
          } catch { /* fallback */ }
          return Color.fromRgb(200, 200, 200)
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
          const cifUrl = `https://alphafold.ebi.ac.uk/files/AF-${uniprotId}-F1-model_v${version}.cif`
          try {
            data = await plugin.builders.data.download(
              { url: cifUrl, isBinary: false },
              { state: { isGhost: true } },
            )
            break
          } catch { /* try next version */ }
        }

        if (!data) {
          throw new Error('Structure not found')
        }

        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'mmcif')
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default')

        // Apply activation color theme
        const structures = plugin.managers.structure.hierarchy.current.structures
        for (const s of structures) {
          for (const c of s.components) {
            await plugin.managers.structure.component.updateRepresentationsTheme(
              [c],
              { color: themeName },
            )
          }
        }

        if (disposed) {
          plugin.dispose()
          return
        }
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
        try {
          wrapperRef.current.removeChild(molContainerRef.current)
        } catch { /* already removed */ }
        molContainerRef.current = null
      }
    }
  }, [proteinId, uniprotId])

  return (
    <div
      style={styles.container}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {onExpand && (
        <div
          style={{ ...styles.expandBtn, opacity: hovered ? 1 : 0 }}
          onClick={(e) => { e.stopPropagation(); onExpand() }}
          title="View detail"
        >
          ↗
        </div>
      )}
      <div ref={wrapperRef} style={styles.viewer}>
        {loading && (
          <div style={styles.loading}>Loading...</div>
        )}
        {error && (
          <div style={styles.error}>No structure</div>
        )}
      </div>
      <div style={styles.label}>
        <span style={styles.proteinId}>{proteinId}</span>
        <span style={styles.activation}>max: {(maxActivation || 0).toFixed(3)}</span>
      </div>
    </div>
  )
}
