import React, { useEffect, useMemo, useRef, useState } from 'react'
import { UMAP } from 'umap-js'

// 500-gene UMAP view with feature-conditional re-embedding.
//
// Loads three precomputed files from /gene_umap/ (in the dashboard's public/):
//   - G.bin: float32 [n_genes * n_features] raw bytes
//   - genes_meta.json: per-gene metadata + base UMAP coords + HDBSCAN cluster
//   - feature_stats.json: features with n_firing >= 10 (clickable list)
//
// Interactions:
//   - Click a feature -> recolor every gene point by G[gene_idx, feature_idx].
//   - Click "Reorganize" -> re-run UMAP client-side on feature-weighted vectors
//     (G[i, :] * (1 + lambda * indicator(feature == clicked))). Animate transition.
//   - Hover gene -> tooltip with symbol, species, cluster, top 5 firing features.

const LAMBDA = 5.0
const REORG_ANIM_MS = 800

// Tableau-10-ish palette used for HDBSCAN cluster coloring. Index -1 (noise)
// maps to grey at the end.
const CLUSTER_PALETTE = [
  '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
  '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#499894',
  '#D37295', '#86BCB6', '#FABFD2', '#D7B5A6', '#BA9582',
  '#A0CBE8', '#FFBE7D', '#8CD17D', '#B6992D', '#79706E',
  '#D4A6C8',
]
const NOISE_COLOR = '#BAB0AC'


function colorForCluster(cid) {
  if (cid == null || cid < 0) return NOISE_COLOR
  return CLUSTER_PALETTE[cid % CLUSTER_PALETTE.length]
}


// Linear viridis-ish ramp for activation-strength coloring.
function colorForActivation(t) {
  // t in [0, 1]. Dark blue at 0 -> yellow at 1.
  const stops = [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ]
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1]
      const [t1, c1] = stops[i]
      const u = (t - t0) / (t1 - t0)
      const r = Math.round(c0[0] + u * (c1[0] - c0[0]))
      const g = Math.round(c0[1] + u * (c1[1] - c0[1]))
      const b = Math.round(c0[2] + u * (c1[2] - c0[2]))
      return `rgb(${r}, ${g}, ${b})`
    }
  }
  const [, c] = stops[stops.length - 1]
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}


// Load the precomputed gene-UMAP bundle: G.bin (binary float32), genes_meta.json,
// feature_stats.json. Returns { G, genes, featureStats, n_genes, n_features }.
async function loadGeneUMAPBundle(baseURL = '/gene_umap') {
  const [gMetaResp, fStatsResp, gBinResp] = await Promise.all([
    fetch(`${baseURL}/genes_meta.json`),
    fetch(`${baseURL}/feature_stats.json`),
    fetch(`${baseURL}/G.bin`),
  ])
  const genesMeta = await gMetaResp.json()
  const featureStats = await fStatsResp.json()
  const gBuffer = await gBinResp.arrayBuffer()
  // float32 little-endian (assumes precompute ran on a little-endian host;
  // x86/ARM both fit).
  const G = new Float32Array(gBuffer)
  return {
    G,
    genes: genesMeta.genes,
    n_genes: genesMeta.n_genes,
    n_features: genesMeta.n_features,
    featureStats,
  }
}


export default function GeneUMAPView({ height = 600, bundleURL = '/gene_umap' }) {
  const [bundle, setBundle] = useState(null)
  const [error, setError] = useState(null)
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [reorgCoords, setReorgCoords] = useState(null)   // null = use base coords
  const [reorgFeatureId, setReorgFeatureId] = useState(null)
  const [reorgRunning, setReorgRunning] = useState(false)
  const [animFrame, setAnimFrame] = useState(1.0)        // 0 -> 1 transition factor
  const [hoverIdx, setHoverIdx] = useState(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    loadGeneUMAPBundle(bundleURL)
      .then((b) => setBundle(b))
      .catch((e) => setError(e.message))
  }, [bundleURL])

  // Build the per-point color array based on current selection mode.
  const pointColors = useMemo(() => {
    if (!bundle) return null
    const { G, n_features, genes } = bundle
    if (selectedFeature == null) {
      return genes.map((g) => colorForCluster(g.cluster_id))
    }
    // Feature-strength coloring.
    let maxAct = 0
    for (let i = 0; i < genes.length; i++) {
      const v = G[i * n_features + selectedFeature]
      if (v > maxAct) maxAct = v
    }
    return genes.map((g, i) => {
      const v = G[i * n_features + selectedFeature]
      if (maxAct <= 0) return NOISE_COLOR
      return colorForActivation(v / maxAct)
    })
  }, [bundle, selectedFeature])

  // Resolve current displayed coords with smooth interpolation between base
  // and reorg coords during the transition window.
  const displayCoords = useMemo(() => {
    if (!bundle) return null
    const base = bundle.genes.map((g) => [g.x, g.y])
    if (reorgCoords == null) return base
    return base.map((b, i) => {
      const r = reorgCoords[i]
      const t = animFrame
      return [b[0] + (r[0] - b[0]) * t, b[1] + (r[1] - b[1]) * t]
    })
  }, [bundle, reorgCoords, animFrame])

  // Draw to canvas whenever coords or colors change.
  useEffect(() => {
    if (!bundle || !displayCoords || !pointColors) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    ctx.clearRect(0, 0, w, h)
    // Scale displayCoords -> canvas pixels.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const [x, y] of displayCoords) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    const pad = 30
    const sx = (w - 2 * pad) / Math.max(1e-9, maxX - minX)
    const sy = (h - 2 * pad) / Math.max(1e-9, maxY - minY)
    const s = Math.min(sx, sy)
    const ox = pad + ((w - 2 * pad) - s * (maxX - minX)) / 2
    const oy = pad + ((h - 2 * pad) - s * (maxY - minY)) / 2
    for (let i = 0; i < displayCoords.length; i++) {
      const [x, y] = displayCoords[i]
      const px = ox + (x - minX) * s
      const py = oy + (y - minY) * s
      ctx.fillStyle = pointColors[i]
      ctx.globalAlpha = hoverIdx != null && i !== hoverIdx ? 0.4 : 1.0
      ctx.beginPath()
      ctx.arc(px, py, hoverIdx === i ? 6 : 4, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.globalAlpha = 1.0
  }, [bundle, displayCoords, pointColors, hoverIdx])

  // Animate from base coords -> new reorg coords with requestAnimationFrame.
  useEffect(() => {
    if (reorgCoords == null) {
      setAnimFrame(1.0)
      return
    }
    let raf
    const start = performance.now()
    const tick = (now) => {
      const t = Math.min(1, (now - start) / REORG_ANIM_MS)
      // Ease-in-out cubic.
      const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
      setAnimFrame(eased)
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    setAnimFrame(0)
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [reorgCoords])

  const handleClickFeature = (fid) => {
    setSelectedFeature(fid === selectedFeature ? null : fid)
  }

  const handleReorganize = async () => {
    if (!bundle || selectedFeature == null) return
    setReorgRunning(true)
    setReorgFeatureId(selectedFeature)
    // Build feature-weighted vectors (umap-js doesn't accept precomputed
    // distances, so encode the weighting into the vector itself).
    const { G, n_features, n_genes } = bundle
    const vecs = new Array(n_genes)
    for (let i = 0; i < n_genes; i++) {
      const row = new Float32Array(n_features)
      const base = i * n_features
      for (let f = 0; f < n_features; f++) row[f] = G[base + f]
      // Amplify the selected feature: row[f] *= (1 + LAMBDA) if f == selected
      // This pulls points with high activation of the clicked feature closer.
      row[selectedFeature] *= 1 + LAMBDA
      vecs[i] = Array.from(row)
    }
    // Defer to next tick so the spinner renders before the heavy compute.
    await new Promise((r) => setTimeout(r, 16))
    const reducer = new UMAP({ nComponents: 2, nNeighbors: 15, minDist: 0.1 })
    const coords = reducer.fit(vecs)
    setReorgCoords(coords)
    setReorgRunning(false)
  }

  const handleResetLayout = () => {
    setReorgCoords(null)
    setReorgFeatureId(null)
  }

  if (error) {
    return <div style={styles.error}>Failed to load gene UMAP bundle: {error}</div>
  }
  if (!bundle) {
    return <div style={styles.loading}>Loading 500-gene UMAP…</div>
  }

  return (
    <div style={styles.container}>
      <div style={styles.toolbar}>
        <span style={styles.title}>Gene UMAP — {bundle.n_genes} genes × {bundle.n_features} features</span>
        <div style={styles.toolbarActions}>
          <button
            disabled={selectedFeature == null || reorgRunning}
            onClick={handleReorganize}
            style={selectedFeature == null ? styles.btnDisabled : styles.btnPrimary}
          >
            {reorgRunning ? 'Reorganizing…' : 'Reorganize by feature'}
          </button>
          {reorgCoords && (
            <button onClick={handleResetLayout} style={styles.btn}>
              Reset layout
            </button>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '12px', height: `${height}px` }}>
        <div style={styles.canvasWrap}>
          <canvas
            ref={canvasRef}
            width={700}
            height={height - 20}
            style={{ width: '100%', height: '100%', cursor: 'crosshair' }}
            onMouseMove={(e) => {
              if (!displayCoords) return
              const rect = e.currentTarget.getBoundingClientRect()
              const mx = ((e.clientX - rect.left) / rect.width) * 700
              const my = ((e.clientY - rect.top) / rect.height) * (height - 20)
              // Re-derive transform (same as draw effect).
              let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
              for (const [x, y] of displayCoords) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
              }
              const pad = 30
              const w = 700, h = height - 20
              const sx = (w - 2 * pad) / Math.max(1e-9, maxX - minX)
              const sy = (h - 2 * pad) / Math.max(1e-9, maxY - minY)
              const s = Math.min(sx, sy)
              const ox = pad + ((w - 2 * pad) - s * (maxX - minX)) / 2
              const oy = pad + ((h - 2 * pad) - s * (maxY - minY)) / 2
              let best = null
              let bestD2 = 100  // pixel² threshold for "hover"
              for (let i = 0; i < displayCoords.length; i++) {
                const [x, y] = displayCoords[i]
                const px = ox + (x - minX) * s
                const py = oy + (y - minY) * s
                const d2 = (px - mx) ** 2 + (py - my) ** 2
                if (d2 < bestD2) {
                  bestD2 = d2
                  best = i
                }
              }
              setHoverIdx(best)
            }}
            onMouseLeave={() => setHoverIdx(null)}
          />
          {hoverIdx != null && bundle.genes[hoverIdx] && (
            <HoverTooltip gene={bundle.genes[hoverIdx]} G={bundle.G} idx={hoverIdx} bundle={bundle} />
          )}
          {reorgCoords && reorgFeatureId != null && (
            <div style={styles.reorgBadge}>
              layout: feature {reorgFeatureId} emphasized
            </div>
          )}
        </div>

        <div style={styles.sidebar}>
          <div style={styles.sidebarTitle}>
            Features (n_firing ≥ 10) · {bundle.featureStats.length}
          </div>
          <div style={styles.featureList}>
            {bundle.featureStats.slice(0, 80).map((fs) => (
              <div
                key={fs.feature_id}
                onClick={() => handleClickFeature(fs.feature_id)}
                style={{
                  ...styles.featureRow,
                  background:
                    selectedFeature === fs.feature_id ? 'var(--bg-card-expanded)' : 'transparent',
                  borderLeft:
                    selectedFeature === fs.feature_id
                      ? '3px solid var(--accent)'
                      : '3px solid transparent',
                }}
              >
                <span style={styles.featureId}>#{fs.feature_id}</span>
                <span style={styles.featureCount}>
                  {fs.n_firing} genes · μ={fs.mean_act_when_firing.toFixed(2)}
                </span>
              </div>
            ))}
            {bundle.featureStats.length > 80 && (
              <div style={styles.featureMore}>… + {bundle.featureStats.length - 80} more</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


function HoverTooltip({ gene, G, idx, bundle }) {
  // Compute top 5 firing features for this gene (no per-gene cache for brevity).
  const top5 = useMemo(() => {
    const base = idx * bundle.n_features
    const pairs = []
    for (let f = 0; f < bundle.n_features; f++) {
      const v = G[base + f]
      if (v > 0) pairs.push([f, v])
    }
    pairs.sort((a, b) => b[1] - a[1])
    return pairs.slice(0, 5)
  }, [G, idx, bundle])
  return (
    <div style={styles.tooltip}>
      <div style={styles.tooltipTitle}>{gene.gene_symbol}</div>
      <div style={styles.tooltipMeta}>{gene.species}</div>
      <div style={styles.tooltipMeta}>
        cluster: {gene.cluster_id < 0 ? 'noise' : gene.cluster_id}
      </div>
      <div style={styles.tooltipSection}>top 5 features</div>
      {top5.map(([f, v]) => (
        <div key={f} style={styles.tooltipRow}>
          <span>#{f}</span>
          <span style={styles.tooltipAct}>{v.toFixed(3)}</span>
        </div>
      ))}
    </div>
  )
}


const styles = {
  container: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: '8px',
    padding: '12px',
  },
  toolbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px',
  },
  title: {
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--text-heading)',
  },
  toolbarActions: { display: 'flex', gap: '8px' },
  btn: {
    padding: '4px 12px',
    border: '1px solid var(--border-input)',
    borderRadius: '4px',
    background: 'var(--bg-input)',
    fontSize: '11px',
    cursor: 'pointer',
    color: 'var(--text-secondary)',
  },
  btnPrimary: {
    padding: '4px 12px',
    border: '1px solid var(--accent)',
    borderRadius: '4px',
    background: 'var(--accent)',
    fontSize: '11px',
    cursor: 'pointer',
    color: 'white',
    fontWeight: 500,
  },
  btnDisabled: {
    padding: '4px 12px',
    border: '1px solid var(--border-light)',
    borderRadius: '4px',
    background: 'var(--bg-input)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    cursor: 'not-allowed',
  },
  canvasWrap: {
    flex: 1,
    position: 'relative',
    background: 'var(--bg-card-expanded)',
    border: '1px solid var(--border-light)',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  reorgBadge: {
    position: 'absolute',
    top: '8px',
    left: '8px',
    background: 'rgba(0,0,0,0.6)',
    color: '#fff',
    padding: '3px 8px',
    borderRadius: '3px',
    fontSize: '10px',
    fontFamily: 'monospace',
  },
  sidebar: {
    width: '240px',
    flexShrink: 0,
    display: 'flex',
    flexDirection: 'column',
    background: 'var(--bg-card-expanded)',
    border: '1px solid var(--border-light)',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  sidebarTitle: {
    fontSize: '10px',
    fontWeight: 600,
    textTransform: 'uppercase',
    color: 'var(--text-tertiary)',
    padding: '8px 12px',
    borderBottom: '1px solid var(--border-light)',
  },
  featureList: {
    overflow: 'auto',
    flex: 1,
    fontSize: '11px',
    fontFamily: 'monospace',
  },
  featureRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '4px 9px',
    cursor: 'pointer',
    userSelect: 'none',
  },
  featureId: { color: 'var(--text-heading)' },
  featureCount: { color: 'var(--text-muted)' },
  featureMore: {
    padding: '6px 9px',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
    fontSize: '10px',
  },
  tooltip: {
    position: 'absolute',
    pointerEvents: 'none',
    top: '10px',
    right: '10px',
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    padding: '8px 10px',
    fontSize: '11px',
    fontFamily: 'system-ui, sans-serif',
    minWidth: '180px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
    color: 'var(--text)',
  },
  tooltipTitle: { fontWeight: 600, fontSize: '13px', color: 'var(--text-heading)' },
  tooltipMeta: { color: 'var(--text-secondary)', fontSize: '11px' },
  tooltipSection: {
    fontSize: '10px',
    textTransform: 'uppercase',
    color: 'var(--text-tertiary)',
    marginTop: '8px',
    paddingTop: '6px',
    borderTop: '1px solid var(--border-light)',
  },
  tooltipRow: {
    display: 'flex',
    justifyContent: 'space-between',
    fontFamily: 'monospace',
    fontSize: '11px',
  },
  tooltipAct: { color: 'var(--text-secondary)' },
  loading: {
    padding: '40px',
    textAlign: 'center',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
  },
  error: {
    padding: '20px',
    color: '#c34',
    background: '#fee',
    borderRadius: '4px',
    fontFamily: 'monospace',
    fontSize: '12px',
  },
}
