import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import * as vg from '@uwdata/vgplot'
import { wasmConnector, MosaicClient } from '@uwdata/mosaic-core'
import { Query, sql, literal } from '@uwdata/mosaic-sql'
import FeatureCard from './FeatureCard'
import EmbeddingView from './EmbeddingView'
import Histogram from './Histogram'

function InfoButton({ text }) {
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const handleClick = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  return (
    <span ref={wrapperRef} style={{ position: 'relative', display: 'inline-block', marginLeft: '5px' }}>
      <span
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '15px',
          height: '15px',
          borderRadius: '50%',
          border: '1px solid #bbb',
          fontSize: '10px',
          fontWeight: '600',
          color: '#888',
          cursor: 'pointer',
          userSelect: 'none',
          lineHeight: 1,
        }}
      >
        i
      </span>
      {open && (
        <div style={{
          position: 'absolute',
          bottom: '22px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '240px',
          background: '#fff',
          border: '1px solid #ddd',
          borderRadius: '6px',
          padding: '10px 12px',
          fontSize: '12px',
          fontWeight: 'normal',
          color: '#444',
          lineHeight: '1.5',
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
          zIndex: 10,
        }}>
          {text}
        </div>
      )}
    </span>
  )
}

const styles = {
  container: {
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    padding: '16px',
    gap: '12px',
    overflow: 'hidden',
  },
  header: {
    flexShrink: 0,
  },
  title: {
    fontSize: '22px',
    fontWeight: '600',
    marginBottom: '2px',
  },
  subtitle: {
    color: '#666',
    fontSize: '13px',
  },
  mainContent: {
    flex: 1,
    display: 'grid',
    gridTemplateColumns: '60% 40%',
    gap: '16px',
    minHeight: 0,
    overflow: 'hidden',
  },
  leftPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    minHeight: 0,
    overflow: 'hidden',
  },
  embeddingPanel: {
    flex: 1,
    background: '#fff',
    borderRadius: '8px',
    border: '1px solid #e0e0e0',
    padding: '12px',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '300px',
    overflow: 'hidden',
  },
  panelHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
    flexShrink: 0,
  },
  panelTitle: {
    fontSize: '14px',
    fontWeight: '600',
  },
  embeddingContainer: {
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
  },
  histogramRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px',
    flexShrink: 0,
  },
  histogramPanel: {
    background: '#fff',
    borderRadius: '8px',
    border: '1px solid #e0e0e0',
    padding: '12px',
  },
  rightPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    minHeight: 0,
    height: '100%',
  },
  searchBar: {
    display: 'flex',
    gap: '8px',
    flexShrink: 0,
  },
  searchInput: {
    flex: 1,
    padding: '8px 12px',
    fontSize: '13px',
    border: '1px solid #ddd',
    borderRadius: '6px',
    outline: 'none',
  },
  sortSelect: {
    padding: '8px 12px',
    fontSize: '13px',
    border: '1px solid #ddd',
    borderRadius: '6px',
    background: 'white',
    cursor: 'pointer',
  },
  stats: {
    padding: '4px 0',
    fontSize: '12px',
    color: '#666',
    flexShrink: 0,
  },
  featureList: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    paddingRight: '8px',
    minHeight: 0,
  },
  loading: {
    textAlign: 'center',
    padding: '40px',
    color: '#666',
  },
  error: {
    textAlign: 'center',
    padding: '40px',
    color: '#c00',
  },
  colorSelect: {
    padding: '4px 8px',
    fontSize: '12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    background: 'white',
    cursor: 'pointer',
  },
  clearButton: {
    padding: '4px 12px',
    fontSize: '12px',
    border: '2px solid #76b900',
    borderRadius: '4px',
    background: 'white',
    color: '#76b900',
    fontWeight: '600',
    cursor: 'pointer',
  },
}

export default function App({ title = "SAE Feature Explorer", subtitle = "Explore sparse autoencoder features with UMAP embedding and crossfiltering" }) {
  const [features, setFeatures] = useState([])
  const [loading, setLoading] = useState(true)
  const [loadingProgress, setLoadingProgress] = useState({ step: 0, total: 4, message: 'Starting up...' })
  const [error, setError] = useState(null)
  const [sortBy, setSortBy] = useState('frequency')
  const [selectedFeatureIds, setSelectedFeatureIds] = useState(null) // null = all selected
  const [mosaicReady, setMosaicReady] = useState(false)
  const [categoryColumns, setCategoryColumns] = useState([])
  const [selectedCategory, setSelectedCategory] = useState('none')
  const [clickedFeatureId, setClickedFeatureId] = useState(null)
  const [clusterLabels, setClusterLabels] = useState(null)

  const brushRef = useRef(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [cardResetKey, setCardResetKey] = useState(0)
  const [plotResetKey, setPlotResetKey] = useState(0)
  const [viewportState, setViewportState] = useState(null) // null = default/fit view
  const featureRefs = useRef({})
  const featureListRef = useRef(null)
  const searchSource = useRef({ source: 'search' })

  // Lazy-load examples for a single feature from DuckDB (feature_examples VIEW)
  const loadExamplesForFeature = useCallback(async (featureId) => {
    const result = await vg.coordinator().query(
      `SELECT * FROM feature_examples WHERE feature_id = ${featureId} ORDER BY example_rank`
    )
    return result.toArray().map(row => ({
      tokens: JSON.parse(row.tokens_json),
      max_activation: row.max_activation,
    }))
  }, [])

  // Handle click on a feature in the UMAP (or null for empty canvas click)
  const animationRef = useRef(null)
  const currentViewportRef = useRef(null)
  const initialViewportRef = useRef(null)

  // Handle viewport changes from the UMAP component
  const handleViewportChange = useCallback((vp) => {
    // Capture initial viewport on first report
    if (!initialViewportRef.current && vp) {
      initialViewportRef.current = { ...vp }
    }
    // Always track current viewport (but not during our own animations)
    if (!animationRef.current) {
      currentViewportRef.current = vp
    }
  }, [])

  // Easing functions
  const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4)
  const easeInOutCubic = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
  const easeInOutQuad = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2

  // Smooth zoom-in with "fly-to" trajectory (zoom out -> pan -> zoom in)
  const zoomToPoint = useCallback((x, y) => {
    if (x == null || y == null) return

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    const start = currentViewportRef.current || initialViewportRef.current || { x: 0, y: 0, scale: 1 }
    const targetScale = 4
    const duration = 800
    const startTime = performance.now()

    // Calculate how far we need to pan (in data space)
    const panDistance = Math.sqrt(Math.pow(x - start.x, 2) + Math.pow(y - start.y, 2))

    // Determine the "cruise altitude" - how much to zoom out during the pan
    // Zoom out more for longer distances, less for short distances
    const minScale = Math.min(start.scale, 0.8) // Never zoom out below 0.8
    const maxZoomOut = Math.max(0, start.scale - minScale)
    const zoomOutAmount = Math.min(maxZoomOut, panDistance * 0.1) // Scale zoom-out with distance
    const cruiseScale = start.scale - zoomOutAmount

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime
      const t = Math.min(elapsed / duration, 1)

      // Use smooth ease-in-out for the overall progress
      const smoothT = easeInOutCubic(t)

      // Pan follows the smooth progress
      const panT = smoothT

      // Zoom follows a "U-shaped" profile:
      // - First half: ease from start.scale down to cruiseScale (or stay flat if already low)
      // - Second half: ease from cruiseScale up to targetScale
      let zoomScale
      if (t < 0.4) {
        // First 40%: zoom out slightly (ease-out)
        const zoomOutT = t / 0.4
        const easeOut = 1 - Math.pow(1 - zoomOutT, 2)
        zoomScale = start.scale + (cruiseScale - start.scale) * easeOut
      } else if (t < 0.6) {
        // Middle 20%: hold at cruise altitude
        zoomScale = cruiseScale
      } else {
        // Last 40%: zoom in to target (ease-in then ease-out)
        const zoomInT = (t - 0.6) / 0.4
        const easeInOut = easeInOutQuad(zoomInT)
        zoomScale = cruiseScale + (targetScale - cruiseScale) * easeInOut
      }

      const newViewport = {
        x: start.x + (x - start.x) * panT,
        y: start.y + (y - start.y) * panT,
        scale: zoomScale
      }

      setViewportState(newViewport)

      if (t < 1) {
        animationRef.current = requestAnimationFrame(animate)
      } else {
        currentViewportRef.current = { x, y, scale: targetScale }
        animationRef.current = null
      }
    }

    animationRef.current = requestAnimationFrame(animate)
  }, [])

  // Smooth zoom-out: zoom out first, then pan back
  const resetViewport = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    const start = currentViewportRef.current || { x: 0, y: 0, scale: 1 }
    const target = initialViewportRef.current || { x: 0, y: 0, scale: 1 }
    const duration = 600
    const startTime = performance.now()

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime
      const t = Math.min(elapsed / duration, 1)

      // Zoom out fast at start (ease-out)
      const zoomT = easeOutQuart(t)

      // Pan eases in-out
      const panT = easeInOutCubic(t)

      const newViewport = {
        x: start.x + (target.x - start.x) * panT,
        y: start.y + (target.y - start.y) * panT,
        scale: start.scale + (target.scale - start.scale) * zoomT
      }

      setViewportState(newViewport)

      if (t < 1) {
        animationRef.current = requestAnimationFrame(animate)
      } else {
        currentViewportRef.current = { ...target }
        animationRef.current = null
      }
    }

    animationRef.current = requestAnimationFrame(animate)
  }, [])

  // Handle click on a feature in the UMAP (with coordinates for zooming)
  const handleFeatureClick = useCallback((featureId, x, y) => {
    console.log('Feature clicked in UMAP:', featureId, x, y)

    setClickedFeatureId(featureId)

    // If clicking on empty canvas (featureId is null), reset viewport and return
    if (featureId == null) {
      resetViewport()
      return
    }

    // Zoom to the clicked point
    zoomToPoint(x, y)

    // Scroll to the feature card after a short delay to allow render
    setTimeout(() => {
      const ref = featureRefs.current[featureId]
      if (ref) {
        ref.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 50)
  }, [zoomToPoint, resetViewport])

  // Handle click on a feature card (highlights point in UMAP and zooms to it)
  const handleCardClick = useCallback(async (featureId, isExpanding) => {
    console.log('Feature card clicked:', featureId, isExpanding ? 'expanding' : 'collapsing')

    // If collapsing, zoom back out and clear highlight
    if (!isExpanding) {
      setClickedFeatureId(null)
      resetViewport()
      return
    }

    setClickedFeatureId(featureId)

    // Query for the feature's coordinates and zoom to it
    try {
      const result = await vg.coordinator().query(`
        SELECT x, y FROM features WHERE feature_id = ${featureId} LIMIT 1
      `)
      const rows = result.toArray()
      if (rows.length > 0) {
        const { x, y } = rows[0]
        zoomToPoint(x, y)
      }
    } catch (err) {
      console.warn('Could not get feature coordinates:', err)
    }
  }, [zoomToPoint, resetViewport])

  // Initialize Mosaic and load data
  useEffect(() => {
    async function init() {
      try {
        // Step 1: Initialize DuckDB-WASM
        setLoadingProgress({ step: 1, total: 4, message: 'Initializing database engine...' })
        const wasm = wasmConnector()
        vg.coordinator().databaseConnector(wasm)

        // Step 2: Load parquet data
        setLoadingProgress({ step: 2, total: 4, message: 'Loading embedding data...' })
        const urlParams = new URLSearchParams(window.location.search)
        const dataPath = urlParams.get('data') || '/features_atlas.parquet'
        const parquetUrl = dataPath.startsWith('http')
          ? dataPath
          : new URL(dataPath, window.location.origin).href

        console.log(`Loading parquet from: ${parquetUrl}`)

        await vg.coordinator().exec(`
          CREATE TABLE features AS
          SELECT * FROM read_parquet('${parquetUrl}')
        `)

        // HDBSCAN assigns -1 to noise points; embedding-atlas casts category
        // columns to UTINYINT which can't hold negatives.  Remap to NULL.
        try {
          await vg.coordinator().exec(`
            UPDATE features SET cluster_id = NULL WHERE cluster_id < 0
          `)
        } catch (e) {
          // cluster_id column may not exist — that's fine
        }

        // Step 3: Process columns and categories
        setLoadingProgress({ step: 3, total: 4, message: 'Processing columns...' })
        const schemaResult = await vg.coordinator().query(`
          SELECT column_name, column_type
          FROM (DESCRIBE features)
        `)

        const columns = schemaResult.toArray().map(row => ({
          name: row.column_name,
          type: row.column_type
        }))

        const detectedCategories = []
        const sequentialColumns = []

        for (const col of columns) {
          if (['x', 'y', 'feature_id', 'top_example_idx'].includes(col.name)) continue

          if (col.type === 'VARCHAR') {
            const cardinalityResult = await vg.coordinator().query(`
              SELECT COUNT(DISTINCT "${col.name}") as n_unique FROM features WHERE "${col.name}" IS NOT NULL
            `)
            const nUnique = cardinalityResult.toArray()[0]?.n_unique ?? 0
            if (nUnique > 0 && nUnique <= 50) {
              detectedCategories.push({ name: col.name, type: 'string', nUnique })
            }
          } else if (col.type === 'BIGINT' || col.type === 'INTEGER') {
            if (col.name.includes('cluster') || col.name.includes('category') || col.name.includes('group')) {
              const cardinalityResult = await vg.coordinator().query(`
                SELECT COUNT(DISTINCT "${col.name}") as n_unique FROM features WHERE "${col.name}" IS NOT NULL
              `)
              const nUnique = cardinalityResult.toArray()[0]?.n_unique ?? 0
              if (nUnique > 0 && nUnique <= 50) {
                detectedCategories.push({ name: col.name, type: 'integer', nUnique })
              }
            }
          } else if (col.type === 'DOUBLE' || col.type === 'FLOAT') {
            // Numeric columns for sequential coloring
            if (['log_frequency', 'max_activation', 'activation_freq', 'frequency'].includes(col.name)) {
              sequentialColumns.push({ name: col.name, type: 'sequential' })
            }
          }
        }

        // Create integer-encoded versions of string category columns
        for (const col of detectedCategories) {
          if (col.type === 'string') {
            await vg.coordinator().exec(`
              CREATE OR REPLACE TABLE features AS
              SELECT *,
                     CASE WHEN "${col.name}" IS NULL THEN NULL
                          ELSE DENSE_RANK() OVER (ORDER BY "${col.name}") - 1
                     END AS "${col.name}_cat"
              FROM features
            `)
          }
        }

        // Create binned versions of sequential columns (10 bins)
        const NUM_BINS = 10
        for (const col of sequentialColumns) {
          await vg.coordinator().exec(`
            CREATE OR REPLACE TABLE features AS
            SELECT *,
                   CASE WHEN "${col.name}" IS NULL THEN NULL
                        ELSE LEAST(${NUM_BINS - 1}, CAST(
                          (("${col.name}" - (SELECT MIN("${col.name}") FROM features)) /
                           NULLIF((SELECT MAX("${col.name}") - MIN("${col.name}") FROM features), 0)) * ${NUM_BINS}
                        AS INTEGER))
                   END AS "${col.name}_bin"
            FROM features
          `)
          detectedCategories.push({ name: col.name, type: 'sequential', nUnique: NUM_BINS })
        }

        setCategoryColumns(detectedCategories)

        // Create crossfilter selection
        brushRef.current = vg.Selection.crossfilter()


        // Step 4: Load feature metadata from parquet via DuckDB
        setLoadingProgress({ step: 4, total: 4, message: 'Loading feature metadata...' })
        const metaUrl = new URL('/feature_metadata.parquet', window.location.origin).href
        const examplesUrl = new URL('/feature_examples.parquet', window.location.origin).href

        await vg.coordinator().exec(`
          CREATE TABLE IF NOT EXISTS feature_metadata AS
          SELECT * FROM read_parquet('${metaUrl}')
        `)
        await vg.coordinator().exec(`
          CREATE VIEW IF NOT EXISTS feature_examples AS
          SELECT * FROM read_parquet('${examplesUrl}')
        `)

        const metaResult = await vg.coordinator().query(`SELECT * FROM feature_metadata ORDER BY feature_id`)
        const loadedFeatures = metaResult.toArray().map(row => ({
          feature_id: row.feature_id,
          description: row.description,
          activation_freq: row.activation_freq,
          max_activation: row.max_activation,
          top_positive_logits: row.top_positive_logits_json ? JSON.parse(row.top_positive_logits_json) : [],
          top_negative_logits: row.top_negative_logits_json ? JSON.parse(row.top_negative_logits_json) : [],
        }))
        setFeatures(loadedFeatures)

        // Load cluster labels (non-fatal if missing)
        try {
          const labelsRes = await fetch('./cluster_labels.json')
          if (labelsRes.ok) {
            const labelsData = await labelsRes.json()
            setClusterLabels(labelsData)
            console.log(`Loaded ${labelsData.length} cluster labels`)
          }
        } catch (labelErr) {
          console.log('No cluster labels found (optional)')
        }

        setMosaicReady(true)
        setLoading(false)

        console.log('Mosaic initialized successfully!')
        console.log(`Loaded ${loadedFeatures.length} features (metadata). Examples loaded on demand via DuckDB.`)
      } catch (err) {
        console.error('Init error:', err)
        setError(err.message)
        setLoading(false)
      }
    }

    init()
  }, [])

  // Create a Mosaic client that receives filtered feature IDs
  useEffect(() => {
    if (!mosaicReady || !brushRef.current) return

    const coordinator = vg.coordinator()
    const selection = brushRef.current
    const totalFeatures = features.length

    // Create a class that extends MosaicClient
    class FeatureFilterClient extends MosaicClient {
      constructor(filterBy) {
        super(filterBy)
      }

      query(filter = []) {
        // Use Mosaic's Query builder
        const q = Query
          .select({ feature_id: 'feature_id' })
          .distinct()
          .from('features')

        // Apply filter if present
        if (filter.length > 0) {
          q.where(filter)
        }

        return q
      }

      queryResult(data) {
        console.log('FeatureFilterClient received data:', data)
        try {
          let ids = new Set()
          if (data && typeof data.getChild === 'function') {
            const col = data.getChild('feature_id')
            if (col) {
              for (let i = 0; i < col.length; i++) {
                ids.add(col.get(i))
              }
            }
          } else if (data && data.toArray) {
            ids = new Set(data.toArray().map(r => r.feature_id))
          }
          console.log('Filtered to', ids.size, 'of', totalFeatures, 'features')
          setSelectedFeatureIds(ids.size > 0 && ids.size < totalFeatures ? ids : null)
        } catch (err) {
          console.error('Error processing result:', err)
        }
      }

      // Required by Mosaic for selection updates
      update() {
        return this
      }

      queryError(err) {
        console.error('FeatureFilterClient error:', err)
      }
    }

    const client = new FeatureFilterClient(selection)

    // Delay connection slightly to ensure Mosaic is fully ready
    const timeoutId = setTimeout(() => {
      try {
        coordinator.connect(client)
      } catch (err) {
        console.warn('Error connecting FeatureFilterClient:', err)
      }
    }, 0)

    return () => {
      clearTimeout(timeoutId)
      try {
        coordinator.disconnect(client)
      } catch (err) {
        // Ignore disconnect errors
      }
    }
  }, [mosaicReady, features.length])

  // Clear ALL selections (search, histograms, UMAP, clicked feature)
  const handleClearSelection = useCallback(() => {
    if (brushRef.current) {
      const selection = brushRef.current
      // Clear each clause by updating with null predicate for each source
      const clauses = selection.clauses || []
      for (const clause of clauses) {
        if (clause.source) {
          try {
            selection.update({ source: clause.source, predicate: null, value: null })
          } catch (e) {
            // Ignore errors from clearing
          }
        }
      }
      // Also clear the search clause specifically
      if (searchSource.current) {
        try {
          selection.update({ source: searchSource.current, predicate: null, value: null })
        } catch (e) {
          // Ignore
        }
      }
    }
    setSelectedFeatureIds(null)
    setSearchTerm('')
    setClickedFeatureId(null)
    // Reset viewport - will be handled by remount with plotResetKey
    setViewportState(null)
    // Clear initial viewport ref so it gets recaptured after remount
    initialViewportRef.current = null
    currentViewportRef.current = null
    // Reset all cards to collapsed state
    setCardResetKey(k => k + 1)
    // Reset histograms and UMAP to clear brush visuals
    setPlotResetKey(k => k + 1)
  }, [])

  // Handle search - updates both Mosaic crossfilter (for UMAP/histograms) and local state (for cards)
  const handleSearchChange = useCallback((e) => {
    const term = e.target.value
    setSearchTerm(term)

    // Also update Mosaic crossfilter so UMAP and histograms filter
    if (brushRef.current) {
      const selection = brushRef.current

      try {
        if (term.trim()) {
          // Build predicate using sql template - ILIKE for case-insensitive search
          const pattern = literal('%' + term.trim() + '%')
          const predicate = sql`label ILIKE ${pattern}`

          selection.update({
            source: searchSource.current,
            predicate: predicate,
            value: term.trim()
          })
        } else {
          // Clear search by removing the clause
          selection.update({
            source: searchSource.current,
            predicate: null,
            value: null
          })
        }
      } catch (err) {
        console.warn('Search update error:', err)
      }
    }
  }, [])

  // Filter and sort features
  const filteredFeatures = useMemo(() => {
    let result = features

    // Filter by Mosaic selection (includes UMAP brush)
    if (selectedFeatureIds !== null) {
      result = result.filter(f => selectedFeatureIds.has(f.feature_id))
    }

    // Also filter by search term client-side (searches metadata fields)
    if (searchTerm.trim()) {
      const q = searchTerm.toLowerCase()
      result = result.filter(f =>
        f.description?.toLowerCase().includes(q) ||
        f.feature_id.toString().includes(q)
      )
    }

    // Sort
    if (sortBy === 'frequency') {
      result = [...result].sort((a, b) => (b.activation_freq || 0) - (a.activation_freq || 0))
    } else if (sortBy === 'max_activation') {
      result = [...result].sort((a, b) => (b.max_activation || 0) - (a.max_activation || 0))
    } else if (sortBy === 'feature_id') {
      result = [...result].sort((a, b) => a.feature_id - b.feature_id)
    }

    return result
  }, [features, sortBy, selectedFeatureIds, searchTerm])

  if (loading) {
    const pct = Math.round(((loadingProgress.step - 1) / loadingProgress.total) * 100)
    return (
      <div style={styles.loading}>
        <div style={{ marginBottom: '16px', fontSize: '15px' }}>Loading dashboard...</div>
        <div style={{
          width: '280px',
          height: '6px',
          background: '#e0e0e0',
          borderRadius: '3px',
          overflow: 'hidden',
          margin: '0 auto 12px',
        }}>
          <div style={{
            width: `${pct}%`,
            height: '100%',
            background: '#76b900',
            borderRadius: '3px',
            transition: 'width 0.3s ease',
          }} />
        </div>
        <div style={{ fontSize: '13px', color: '#888' }}>{loadingProgress.message}</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={styles.error}>
        <p>Error: {error}</p>
        <p style={{ marginTop: '10px', fontSize: '14px' }}>
          Make sure features_atlas.parquet, feature_metadata.parquet, and feature_examples.parquet exist in the public/ folder.
        </p>
      </div>
    )
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>{title}</h1>
        <p style={styles.subtitle}>{subtitle}</p>
      </div>

      <div style={styles.mainContent}>
        <div style={styles.leftPanel}>
          <div style={styles.embeddingPanel}>
            <div style={styles.panelHeader}>
              <span style={styles.panelTitle}>
                UMAP Embedding
                <InfoButton text="Each point is a learned SAE feature, positioned by its decoder weight vector projected into 2D via UMAP. Features that are close together respond to similar patterns in the input space. Clusters reveal the structure of the representation the model has learned." />
              </span>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <label style={{ fontSize: '12px', color: '#666' }}>Color by:</label>
                <select
                  value={selectedCategory}
                  onChange={e => setSelectedCategory(e.target.value)}
                  style={styles.colorSelect}
                >
                  <option value="none">None (uniform)</option>
                  {categoryColumns.map(col => (
                    <option key={col.name} value={col.name}>
                      {col.name} {col.type === 'sequential' ? '(sequential)' : `(${col.nUnique} values)`}
                    </option>
                  ))}
                </select>
                <button onClick={handleClearSelection} style={styles.clearButton}>
                  Clear Selection
                </button>
              </div>
            </div>
            <div style={styles.embeddingContainer}>
              {mosaicReady && (
                <EmbeddingView
                  key={`umap-${plotResetKey}`}
                  brush={brushRef.current}
                  categoryColumn={selectedCategory}
                  categoryColumns={categoryColumns}
                  onFeatureClick={handleFeatureClick}
                  highlightedFeatureId={clickedFeatureId}
                  viewportState={viewportState}
                  onViewportChange={handleViewportChange}
                  labels={clusterLabels}
                />
              )}
            </div>
          </div>

          <div style={styles.histogramRow}>
            <div style={styles.histogramPanel}>
              <span style={styles.panelTitle}>
                Log Frequency
                <InfoButton text="Distribution of feature activation frequencies on a log scale. A good SAE typically looks log-normal here. Features on the left fire rarely; features on the right fire often. Drag to brush and filter the UMAP and feature list." />
              </span>
              <div style={{ fontSize: '11px', color: '#999', marginTop: '1px' }}>How often each feature fires across inputs</div>
              {mosaicReady && (
                <Histogram
                  key={`hist-freq-${plotResetKey}`}
                  brush={brushRef.current}
                  column="log_frequency"
                />
              )}
            </div>
            <div style={styles.histogramPanel}>
              <span style={styles.panelTitle}>
                Max Activation
                <InfoButton text="Distribution of peak activation values across features. Higher values mean the feature responds more strongly to its preferred input. Drag to brush and filter the UMAP and feature list." />
              </span>
              <div style={{ fontSize: '11px', color: '#999', marginTop: '1px' }}>Strongest activation observed per feature</div>
              {mosaicReady && (
                <Histogram
                  key={`hist-max-${plotResetKey}`}
                  brush={brushRef.current}
                  column="max_activation"
                />
              )}
            </div>
          </div>
        </div>

        <div style={styles.rightPanel}>
          <div style={styles.searchBar}>
            <input
              type="text"
              placeholder="Search features..."
              value={searchTerm}
              onChange={handleSearchChange}
              style={styles.searchInput}
            />
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value)}
              style={styles.sortSelect}
            >
              <option value="frequency">By Frequency</option>
              <option value="max_activation">By Max Activation</option>
              <option value="feature_id">By Feature ID</option>
            </select>
          </div>

          <div style={styles.stats}>
            Showing {filteredFeatures.length} of {features.length} features
            {selectedFeatureIds !== null && ` (${selectedFeatureIds.size} selected in UMAP)`}
          </div>

          <div ref={featureListRef} style={styles.featureList}>
            {(() => {
              const visibleFeatures = filteredFeatures.slice(0, 100)
              const clickedIsVisible = clickedFeatureId != null &&
                visibleFeatures.some(f => Number(f.feature_id) === Number(clickedFeatureId))
              const clickedFeature = clickedFeatureId != null && !clickedIsVisible
                ? features.find(f => Number(f.feature_id) === Number(clickedFeatureId))
                : null

              return (
                <>
                  {/* Only render clicked feature at top if NOT already in visible list */}
                  {clickedFeature && (
                    <FeatureCard
                      key={`clicked-${clickedFeature.feature_id}-${cardResetKey}`}
                      ref={el => { featureRefs.current[clickedFeature.feature_id] = el }}
                      feature={clickedFeature}
                      isHighlighted={true}
                      forceExpanded={true}
                      onClick={handleCardClick}
                      loadExamples={loadExamplesForFeature}
                    />
                  )}
                  {visibleFeatures.map(feature => (
                    <FeatureCard
                      key={`${feature.feature_id}-${cardResetKey}`}
                      ref={el => { featureRefs.current[feature.feature_id] = el }}
                      feature={feature}
                      isHighlighted={Number(clickedFeatureId) === Number(feature.feature_id)}
                      forceExpanded={Number(clickedFeatureId) === Number(feature.feature_id)}
                      onClick={handleCardClick}
                      loadExamples={loadExamplesForFeature}
                    />
                  ))}
                </>
              )
            })()}
            {filteredFeatures.length > 100 && (
              <div style={{ textAlign: 'center', padding: '12px', color: '#666', fontSize: '13px' }}>
                Showing first 100 results. Refine your selection to see more.
              </div>
            )}
            {filteredFeatures.length === 0 && clickedFeatureId == null && (
              <div style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
                No features match your selection.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
