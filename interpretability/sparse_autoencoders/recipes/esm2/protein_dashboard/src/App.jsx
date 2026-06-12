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
          border: '1px solid var(--border-input)',
          fontSize: '10px',
          fontWeight: '600',
          color: 'var(--text-tertiary)',
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
          background: 'var(--bg-card)',
          border: '1px solid var(--border-input)',
          borderRadius: '6px',
          padding: '10px 12px',
          fontSize: '12px',
          fontWeight: 'normal',
          color: 'var(--text-secondary)',
          lineHeight: '1.5',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
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
    gap: '4px',
    overflow: 'hidden',
    background: 'var(--bg)',
    color: 'var(--text)',
  },
  header: {
    flexShrink: 0,
  },
  title: {
    fontSize: '22px',
    fontWeight: '600',
    marginBottom: '2px',
    color: 'var(--text-heading)',
  },
  subtitle: {
    color: 'var(--text-secondary)',
    fontSize: '13px',
    margin: 0,
  },
  mainContent: {
    flex: 1,
    display: 'grid',
    gridTemplateColumns: '3fr 2fr',
    gap: '16px',
    minHeight: 0,
    overflow: 'hidden',
  },
  leftPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    minHeight: 0,
    minWidth: 0,
    overflow: 'hidden',
  },
  embeddingPanel: {
    flex: 1,
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    padding: '12px',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '300px',
    minWidth: 0,
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
    minHeight: '100px',
    marginBottom: '4px',
  },
  histogramPanel: {
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    padding: '8px',
    overflow: 'hidden',
  },
  rightPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    minHeight: 0,
    height: '100%',
    overflow: 'hidden',
  },
  searchBar: {
    display: 'flex',
    gap: '8px',
    flexShrink: 0,
  },
  searchInput: {
    flex: 0.81,
    padding: '8px 12px',
    fontSize: '13px',
    border: '1px solid var(--border-input)',
    borderRadius: '6px',
    outline: 'none',
    background: 'var(--bg-input)',
    color: 'var(--text)',
  },
  sortSelect: {
    padding: '8px 12px',
    fontSize: '13px',
    border: '1px solid var(--border-input)',
    borderRadius: '6px',
    background: 'var(--bg-input)',
    color: 'var(--text)',
    cursor: 'pointer',
  },
  stats: {
    padding: '4px 0',
    fontSize: '12px',
    color: 'var(--text-secondary)',
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
    color: 'var(--text-secondary)',
  },
  error: {
    textAlign: 'center',
    padding: '40px',
    color: '#c00',
  },
  colorSelect: {
    padding: '4px 8px',
    fontSize: '12px',
    border: '1px solid var(--border-input)',
    borderRadius: '4px',
    background: 'var(--bg-input)',
    color: 'var(--text)',
    cursor: 'pointer',
  },
  clearButton: {
    padding: '4px 12px',
    fontSize: '12px',
    border: '2px solid var(--accent)',
    borderRadius: '4px',
    background: 'transparent',
    color: 'var(--accent)',
    fontWeight: '600',
    cursor: 'pointer',
  },
  darkModeBtn: {
    padding: '4px 10px',
    fontSize: '16px',
    border: '1px solid var(--border-input)',
    borderRadius: '6px',
    background: 'var(--bg-input)',
    color: 'var(--text)',
    cursor: 'pointer',
    lineHeight: 1,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
}

export default function App({ title = "ESM2 Sparse Autoencoder Feature Explorer", subtitle = "Explore sparse autoencoder features with UMAP embedding and crossfiltering" }) {
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
  const [vocabLogits, setVocabLogits] = useState(null)
  const [darkMode, setDarkMode] = useState(true)
  const [histMetric1, setHistMetric1] = useState('log_frequency')
  const [histMetric2, setHistMetric2] = useState('max_activation')

  const brushRef = useRef(null)

  const [searchTerm, setSearchTerm] = useState('')
  const [cardResetKey, setCardResetKey] = useState(0)
  const [plotResetKey, setPlotResetKey] = useState(0)
  const [viewportState, setViewportState] = useState(null) // null = default/fit view
  const featureRefs = useRef({})
  const featureListRef = useRef(null)
  const searchSource = useRef({ source: 'search' })

  // Dark mode toggle
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  // Sync second histogram with color-by selection
  useEffect(() => {
    if (selectedCategory && selectedCategory !== 'none') {
      setHistMetric2(selectedCategory)
    }
  }, [selectedCategory])

  // Lazy-load examples for a single feature from DuckDB (feature_examples VIEW)
  const loadExamplesForFeature = useCallback(async (featureId) => {
    const result = await vg.coordinator().query(
      `SELECT * FROM feature_examples WHERE feature_id = ${featureId} ORDER BY example_rank`
    )
    return result.toArray().map(row => ({
      protein_id: row.protein_id,
      alphafold_id: row.alphafold_id,
      sequence: row.sequence,
      activations: Array.from(row.activations),
      max_activation: row.max_activation,
      best_annotation: row.best_annotation,
    }))
  }, [])

  const currentViewportRef = useRef(null)

  // Handle viewport changes from the UMAP component
  const handleViewportChange = useCallback((vp) => {
    currentViewportRef.current = vp
  }, [])

  // Handle click on a feature in the UMAP (highlight + scroll, no zoom)
  const handleFeatureClick = useCallback((featureId, x, y) => {
    setClickedFeatureId(featureId)

    if (featureId == null) return

    // Scroll to the feature card after a short delay to allow render
    setTimeout(() => {
      const ref = featureRefs.current[featureId]
      if (ref) {
        ref.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 50)
  }, [])

  // Handle click on a feature card (highlights point in UMAP)
  const handleCardClick = useCallback((featureId, isExpanding) => {
    if (!isExpanding) {
      setClickedFeatureId(null)
      return
    }

    setClickedFeatureId(featureId)
  }, [])

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
          best_f1: row.best_f1,
          best_annotation: row.best_annotation,
        }))
        setFeatures(loadedFeatures)

        // Derive annotation_type (top-level category) from best_annotation
        try {
          await vg.coordinator().exec(`
            CREATE OR REPLACE TABLE features AS
            SELECT f.*,
                   CASE
                     WHEN m.best_annotation IS NULL OR m.best_annotation = '' OR m.best_annotation = 'None' THEN 'unlabeled'
                     WHEN CONTAINS(m.best_annotation, ':') THEN SPLIT_PART(m.best_annotation, ':', 1)
                     ELSE m.best_annotation
                   END AS annotation_type
            FROM features f
            LEFT JOIN feature_metadata m ON f.feature_id = m.feature_id
          `)
          // Add integer-encoded version for embedding-atlas
          await vg.coordinator().exec(`
            CREATE OR REPLACE TABLE features AS
            SELECT *,
                   DENSE_RANK() OVER (ORDER BY annotation_type) - 1 AS annotation_type_cat
            FROM features
          `)
          const cardResult = await vg.coordinator().query(`
            SELECT COUNT(DISTINCT annotation_type) as n_unique FROM features WHERE annotation_type IS NOT NULL
          `)
          const nUnique = cardResult.toArray()[0]?.n_unique ?? 0
          if (nUnique > 0 && nUnique <= 50) {
            detectedCategories.push({ name: 'annotation_type', type: 'string', nUnique })
            setCategoryColumns([...detectedCategories])
            // Default color-by to annotation_type
            setSelectedCategory('annotation_type')
          }
        } catch (err) {
          console.warn('Could not create annotation_type column:', err)
        }

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

        // Load vocab logits (non-fatal if missing)
        try {
          const logitsRes = await fetch('./vocab_logits.json')
          if (logitsRes.ok) {
            const logitsData = await logitsRes.json()
            setVocabLogits(logitsData)
            console.log(`Loaded vocab logits for ${Object.keys(logitsData).length} features`)
          }
        } catch (e) {
          console.log('No vocab logits found (optional)')
        }

        // Pre-cache feature coordinates for instant zoom
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
        f.feature_id.toString().includes(q) ||
        f.best_annotation?.toLowerCase().includes(q)
      )
    }

    // Sort
    if (sortBy === 'frequency') {
      result = [...result].sort((a, b) => (b.activation_freq || 0) - (a.activation_freq || 0))
    } else if (sortBy === 'max_activation') {
      result = [...result].sort((a, b) => (b.max_activation || 0) - (a.max_activation || 0))
    } else if (sortBy === 'best_f1') {
      result = [...result].sort((a, b) => (b.best_f1 || 0) - (a.best_f1 || 0))
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
          background: 'var(--loading-bar-bg)',
          borderRadius: '3px',
          overflow: 'hidden',
          margin: '0 auto 12px',
        }}>
          <div style={{
            width: `${pct}%`,
            height: '100%',
            background: 'var(--accent)',
            borderRadius: '3px',
            transition: 'width 0.3s ease',
          }} />
        </div>
        <div style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>{loadingProgress.message}</div>
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
      <div style={{ ...styles.header, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={styles.title}>{title}</h1>
          <p style={styles.subtitle}>{subtitle}</p>
        </div>
        <button
          onClick={() => setDarkMode(d => !d)}
          style={styles.darkModeBtn}
          title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {darkMode ? '\u2600' : '\u263E'}
        </button>
      </div>

      <div style={styles.mainContent}>
        <div style={styles.leftPanel}>
          <div style={styles.embeddingPanel}>
            <div style={styles.panelHeader}>
              <span style={styles.panelTitle}>
                Decoder UMAP
                <InfoButton text="Each point is a learned SAE feature, positioned by its decoder weight vector projected into 2D via UMAP. Features that are close together respond to similar patterns in the input space. Clusters reveal the structure of the representation the model has learned." />
              </span>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Color by:</label>
                <select
                  value={selectedCategory}
                  onChange={e => setSelectedCategory(e.target.value)}
                  style={styles.colorSelect}
                >
                  <option value="none">None</option>
                  {categoryColumns.map(col => (
                    <option key={col.name} value={col.name}>
                      {col.name.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
                <button onClick={handleClearSelection} style={styles.clearButton}>
                  Clear Selection
                </button>
              </div>
            </div>
            <div style={{ ...styles.embeddingContainer, position: 'relative' }}>
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
                  darkMode={darkMode}
                />
              )}
              {selectedCategory && selectedCategory !== 'none' && (() => {
                const colInfo = categoryColumns.find(c => c.name === selectedCategory)
                if (!colInfo) return null

                if (colInfo.type === 'sequential') {
                  const colors = [
                    "#c359ef", "#9525C6", "#0046a4", "#0074DF", "#3f8500",
                    "#76B900", "#ef9100", "#F9C500", "#ff8181", "#EF2020"
                  ]
                  const vals = features
                    .map(f => f[selectedCategory])
                    .filter(v => v != null && !isNaN(v))
                  const minVal = vals.length > 0 ? Math.min(...vals) : 0
                  const maxVal = vals.length > 0 ? Math.max(...vals) : 1
                  const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3)
                  return (
                    <div style={{
                      position: 'absolute', right: '12px', top: '12%', bottom: '12%',
                      display: 'flex', flexDirection: 'column', alignItems: 'center',
                      gap: '2px', pointerEvents: 'none',
                    }}>
                      <span style={{ fontSize: '9px', color: 'var(--text-secondary)', fontWeight: '600' }}>{fmt(maxVal)}</span>
                      <div style={{
                        flex: 1, width: '12px', borderRadius: '3px',
                        background: `linear-gradient(to bottom, ${[...colors].reverse().join(', ')})`,
                      }} />
                      <span style={{ fontSize: '9px', color: 'var(--text-secondary)', fontWeight: '600' }}>{fmt(minVal)}</span>
                      <span style={{
                        fontSize: '8px', color: 'var(--text-muted)', maxWidth: '60px', textAlign: 'center',
                        lineHeight: '1.2', marginTop: '2px',
                      }}>
                        {selectedCategory.replace(/_/g, ' ')}
                      </span>
                    </div>
                  )
                }

                // Categorical legend (string or integer types)
                const catColors = [
                  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                  "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
                ]
                // Get distinct values sorted alphabetically (matches DENSE_RANK ORDER BY)
                const distinctVals = [...new Set(
                  features.map(f => f.best_annotation)
                )]
                // Derive category labels the same way as SQL
                const categoryLabels = [...new Set(
                  distinctVals.map(v => {
                    if (v == null || v === '' || v === 'None') return 'unlabeled'
                    if (v.includes(':')) return v.split(':')[0]
                    return v
                  })
                )].sort()

                return (
                  <div style={{
                    position: 'absolute', right: '10px', top: '8px',
                    display: 'flex', flexDirection: 'column',
                    gap: '3px', pointerEvents: 'none',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    borderRadius: '6px',
                    padding: '8px 10px',
                    opacity: 0.9,
                  }}>
                    <span style={{
                      fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase',
                      fontWeight: '600', marginBottom: '2px',
                    }}>
                      {selectedCategory.replace(/_/g, ' ')}
                    </span>
                    {categoryLabels.map((label, i) => (
                      <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <div style={{
                          width: '8px', height: '8px', borderRadius: '50%',
                          background: catColors[i % catColors.length], flexShrink: 0,
                        }} />
                        <span style={{ fontSize: '10px', color: 'var(--text-secondary)', lineHeight: '1.2' }}>
                          {label}
                        </span>
                      </div>
                    ))}
                  </div>
                )
              })()}
            </div>
          </div>

          <div style={styles.histogramRow}>
            {[
              { value: histMetric1, setter: setHistMetric1 },
              { value: histMetric2, setter: setHistMetric2 },
            ].map(({ value, setter }, i) => (
              <div key={i} style={styles.histogramPanel}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <select
                    value={value}
                    onChange={e => setter(e.target.value)}
                    style={{
                      padding: '2px 6px',
                      fontSize: '11px',
                      border: '1px solid var(--border-input)',
                      borderRadius: '4px',
                      background: 'var(--bg-input)',
                      color: 'var(--text)',
                      cursor: 'pointer',
                    }}
                  >
                    <option value="log_frequency">Log Frequency</option>
                    <option value="max_activation">Max Activation</option>
                    <option value="activation_freq">Activation Frequency</option>
                    {categoryColumns
                      .filter(c => c.type === 'sequential')
                      .map(col => (
                        <option key={col.name} value={col.name}>
                          {col.name.replace(/_/g, ' ')}
                        </option>
                      ))}
                    {categoryColumns
                      .filter(c => c.type === 'string' || c.type === 'integer')
                      .map(col => (
                        <option key={col.name} value={col.name}>
                          {col.name.replace(/_/g, ' ')}
                        </option>
                      ))}
                  </select>
                </div>
                {mosaicReady && value && value !== 'none' && (
                  <Histogram
                    key={`hist-${i}-${value}-${plotResetKey}`}
                    brush={brushRef.current}
                    column={value}
                    categoryColumns={categoryColumns}
                  />
                )}
              </div>
            ))}
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
              <option value="best_f1">By F1 Score</option>
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
                      vocabLogits={vocabLogits}
                      darkMode={darkMode}
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
                      vocabLogits={vocabLogits}
                      darkMode={darkMode}
                    />
                  ))}
                </>
              )
            })()}
            {filteredFeatures.length > 100 && (
              <div style={{ textAlign: 'center', padding: '12px', color: 'var(--text-secondary)', fontSize: '13px' }}>
                Showing first 100 results. Refine your selection to see more.
              </div>
            )}
            {filteredFeatures.length === 0 && clickedFeatureId == null && (
              <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)' }}>
                No features match your selection.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
