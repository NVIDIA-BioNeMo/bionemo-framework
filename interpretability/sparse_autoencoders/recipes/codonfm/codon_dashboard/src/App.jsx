import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import * as vg from '@uwdata/vgplot'
import { wasmConnector, MosaicClient } from '@uwdata/mosaic-core'
import { Query, sql, literal } from '@uwdata/mosaic-sql'
import FeatureCard from './FeatureCard'
import FeatureList from './FeatureList'
import EmbeddingView from './EmbeddingView'
import Histogram from './Histogram'
import InfoButton from './InfoButton'
import { Sun, Moon } from 'lucide-react'

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
  embeddingContainer: {
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
  },
  histogramRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 1fr',
    gap: '12px',
    flexShrink: 0,
    height: '100px',
    marginBottom: '4px',
  },
  histogramPanel: {
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    padding: '8px',
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
    color: 'var(--text-heading)',
  },
  rightPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    minHeight: 0,
    minWidth: 0,
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
}

export default function App({ title = "SAE Feature Explorer", subtitle = "Explore sparse autoencoder features with UMAP embedding and crossfiltering" }) {
  const [darkMode, setDarkMode] = useState(true)

  // Toggle dark class on document root
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode)
  }, [darkMode])

  const [features, setFeatures] = useState([])
  const [loading, setLoading] = useState(true)
  const [loadingProgress, setLoadingProgress] = useState({ step: 0, total: 4, message: 'Starting up...' })
  const [error, setError] = useState(null)
  const [sortBy, setSortBy] = useState('frequency')
  const [selectedFeatureIds, setSelectedFeatureIds] = useState(null) // null = all selected
  const [mosaicReady, setMosaicReady] = useState(false)
  const [categoryColumns, setCategoryColumns] = useState([])
  const [selectedCategory, setSelectedCategory] = useState('mean_variant_1bcdwt')
  const [hiddenCategories, setHiddenCategories] = useState(new Set())
  const [clickedFeatureId, setClickedFeatureId] = useState(null)
  const [clusterLabels, setClusterLabels] = useState(null)
  const [vocabLogits, setVocabLogits] = useState(null)
  const [featureAnalysis, setFeatureAnalysis] = useState(null)

  const brushRef = useRef(null)
  const [showGuideModal, setShowGuideModal] = useState(false)
  const [showMetricsModal, setShowMetricsModal] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [cardResetKey, setCardResetKey] = useState(0)
  const [plotResetKey, setPlotResetKey] = useState(0)
  const [viewportState, setViewportState] = useState(null) // null = let embedding-atlas auto-fit on first load
  const [displayedCardCount, setDisplayedCardCount] = useState(20) // Pagination: start with 20 cards
  const [showEditedOnly, setShowEditedOnly] = useState(false) // Filter for edited features only
  const [histMetric1, setHistMetric1] = useState('log_frequency')
  const [histMetric2, setHistMetric2] = useState('max_activation')
  const [histMetric3, setHistMetric3] = useState('mean_variant_1bcdwt') // tracks color-by selection
  const featureRefs = useRef({})
  const featureListRef = useRef(null)
  const endOfListRef = useRef(null)
  const searchSource = useRef({ source: 'search' })
  const editedSource = useRef({ source: 'edited' })
  const legendSource = useRef({ source: 'legend' })
  const loadingMoreRef = useRef(false)

  // Lazy-load examples for a single feature from DuckDB (feature_examples VIEW)
  const loadExamplesForFeature = useCallback(async (featureId) => {
    const result = await vg.coordinator().query(
      `SELECT * FROM feature_examples WHERE feature_id = ${featureId} ORDER BY example_rank`
    )
    return result.toArray().map(row => ({
      protein_id: row.protein_id,
      sequence: row.sequence,
      activations: Array.from(row.activations),
      max_activation: row.max_activation,
      best_annotation: row.best_annotation,
    }))
  }, [])

  // Intersection Observer for infinite scroll pagination
  useEffect(() => {
    const sentinel = endOfListRef.current
    const scrollContainer = featureListRef.current
    if (!sentinel || !scrollContainer) return

    const observer = new IntersectionObserver(
      entries => {
        console.log('[scroll] sentinel intersecting:', entries[0].isIntersecting, 'loadingMore:', loadingMoreRef.current)
        if (entries[0].isIntersecting && !loadingMoreRef.current) {
          loadingMoreRef.current = true
          setDisplayedCardCount(prev => prev + 20)
          // Reset flag after a delay to allow next batch
          setTimeout(() => {
            loadingMoreRef.current = false
          }, 300)
        }
      },
      { root: scrollContainer, threshold: 0.1, rootMargin: '200px' }
    )

    observer.observe(sentinel)

    return () => {
      observer.disconnect()
    }
  }, [mosaicReady])

  // Handle click on a feature in the UMAP (or null for empty canvas click)
  const animationRef = useRef(null)
  const currentViewportRef = useRef(null)
  const initialViewportRef = useRef(null)

  // Handle viewport changes from the UMAP component
  const handleViewportChange = useCallback((vp) => {
    // Capture initial viewport on first report, slightly zoomed out so all points fit
    if (!initialViewportRef.current && vp) {
      initialViewportRef.current = { ...vp, scale: vp.scale * 0.5 }
      setViewportState(initialViewportRef.current)
      currentViewportRef.current = { ...initialViewportRef.current }
    }
    // Clamp zoom to max scale of 5
    if (vp && vp.scale > 5) {
      const clamped = { ...vp, scale: 5 }
      setViewportState(clamped)
      currentViewportRef.current = clamped
      return
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
    const targetScale = 4 // capped below max zoom of 5
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

    setClickedFeatureId(featureId)

    if (featureId == null) return

    // Scroll to the feature card
    setTimeout(() => {
      const ref = featureRefs.current[featureId]
      if (ref) {
        ref.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 50)
  }, [])

  // Handle click on a feature card (highlights point in UMAP, no zoom)
  const handleCardClick = useCallback(async (featureId, isExpanding) => {

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
            const isGsea = col.name.startsWith('gsea_')
            const maxUnique = isGsea ? Infinity : 50
            const cardinalityResult = await vg.coordinator().query(`
              SELECT COUNT(DISTINCT "${col.name}") as n_unique FROM features WHERE "${col.name}" IS NOT NULL AND "${col.name}" != 'unlabeled'
            `)
            const nUnique = cardinalityResult.toArray()[0]?.n_unique ?? 0
            if (nUnique > 0 && nUnique <= maxUnique) {
              // For high-cardinality GSEA columns, collapse to top 20 + "other"
              if (isGsea && nUnique > 20) {
                await vg.coordinator().exec(`
                  CREATE OR REPLACE TABLE features AS
                  SELECT * REPLACE (
                    CASE
                      WHEN "${col.name}" IS NULL OR "${col.name}" = 'unlabeled' THEN 'unlabeled'
                      WHEN "${col.name}" IN (
                        SELECT "${col.name}" FROM features
                        WHERE "${col.name}" IS NOT NULL AND "${col.name}" != 'unlabeled'
                        GROUP BY "${col.name}" ORDER BY COUNT(*) DESC LIMIT 20
                      ) THEN "${col.name}"
                      ELSE 'other'
                    END AS "${col.name}"
                  ) FROM features
                `)
                detectedCategories.push({ name: col.name, type: 'string', nUnique: 22 })
              } else {
                detectedCategories.push({ name: col.name, type: 'string', nUnique })
              }
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
            if (['log_frequency', 'max_activation', 'activation_freq', 'frequency',
                 'mean_variant_1bcdwt',
                 'high_score_fraction', 'clinvar_fraction',
                 'mean_phylop', 'mean_variant_delta', 'mean_site_delta', 'mean_local_delta',
                 'high_score_delta', 'low_score_delta',
                 'gc_mean', 'gc_std',
                 'trinuc_entropy', 'trinuc_dominant_frac',
                 'pli_mean_pli', 'pli_frac_constrained', 'pli_max_pli',
                 'codon_cai', 'codon_tai', 'codon_rscu',
                 'gene_entropy', 'gene_n_unique', 'gene_dominant_frac',
            ].includes(col.name)) {
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

        // Load features from the features table (which has labels + category columns)
        const categorySelectCols = detectedCategories
          .filter(c => c.type === 'string' || c.type === 'integer')
          .map(c => `"${c.name}"`)
          .join(', ')
        const extraSelect = categorySelectCols ? `, ${categorySelectCols}` : ''
        const featuresResult = await vg.coordinator().query(`
          SELECT
            feature_id,
            label,
            activation_freq,
            max_activation,
            x,
            y
            ${extraSelect}
          FROM features
          ORDER BY feature_id
        `)
        const loadedFeatures = featuresResult.toArray().map(row => {
          const f = {
            feature_id: row.feature_id,
            label: row.label,
            description: row.label,
            activation_freq: row.activation_freq,
            max_activation: row.max_activation,
            x: row.x,
            y: row.y,
          }
          for (const col of detectedCategories) {
            if (col.type === 'string' || col.type === 'integer') {
              f[col.name] = row[col.name]
            }
          }
          return f
        })
        setFeatures(loadedFeatures)

        // Generate cluster labels from DuckDB (non-fatal if cluster_id doesn't exist)
        try {
          const clusterResult = await vg.coordinator().query(`
            SELECT
              cluster_id,
              AVG(x) as cx,
              AVG(y) as cy,
              MODE(label) as top_label,
              COUNT(*) as n
            FROM features
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
            ORDER BY n DESC
          `)
          const labels = clusterResult.toArray()
            .filter(row => row.top_label && !row.top_label.startsWith('Feature '))
            .map((row, i) => ({
              x: Number(row.cx),
              y: Number(row.cy),
              text: row.top_label.length > 40 ? row.top_label.slice(0, 40) + '...' : row.top_label,
              priority: row.n,
              level: 0,
            }))
          console.log('[cluster labels] generated:', labels.length, labels.slice(0, 5))
          if (labels.length > 0) {
            setClusterLabels(labels)
          }
        } catch (e) {
          console.log('[cluster labels] query failed:', e.message)
        }

        // Load cluster labels from file (overrides computed ones if present)
        try {
          const labelsRes = await fetch('./cluster_labels.json')
          if (labelsRes.ok) {
            const labelsData = await labelsRes.json()
            setClusterLabels(labelsData)
          }
        } catch (labelErr) {
        }

        // Load vocab logits (non-fatal if missing)
        try {
          const logitsRes = await fetch('./vocab_logits.json')
          if (logitsRes.ok) {
            const logitsData = await logitsRes.json()
            setVocabLogits(logitsData)
          }
        } catch (e) {
        }

        // Load feature analysis (non-fatal if missing)
        try {
          const analysisRes = await fetch('./feature_analysis.json')
          if (analysisRes.ok) {
            const analysisData = await analysisRes.json()
            setFeatureAnalysis(analysisData)
          }
        } catch (e) {
        }

        setMosaicReady(true)
        setLoading(false)

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
        this._isConnected = true
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
        if (!this._isConnected) return

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
        if (this._isConnected) {
          console.error('FeatureFilterClient error:', err)
        }
      }

      disconnect() {
        this._isConnected = false
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
        client.disconnect()
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
    setHiddenCategories(new Set())
    // Reset viewport to the auto-fit view captured on first load
    if (initialViewportRef.current) {
      setViewportState({ ...initialViewportRef.current })
      currentViewportRef.current = { ...initialViewportRef.current }
    } else {
      setViewportState(null)
      currentViewportRef.current = null
    }
    // Reset all cards to collapsed state
    setCardResetKey(k => k + 1)
    // Reset histograms and UMAP to clear brush visuals
    setPlotResetKey(k => k + 1)
  }, [])

  // Export all edited features to CSV with full data
  const handleExportEdited = useCallback(() => {
    // Get all edited features
    const editedFeatures = features.filter(f => localStorage.getItem(`featureTitle_${f.feature_id}`) !== null)

    if (editedFeatures.length === 0) {
      alert('No edited features to export')
      return
    }

    const lines = []
    const escapeCsv = (str) => `"${(str || '').toString().replace(/"/g, '""')}"`

    // Codon mapping for amino acids
    const CODON_AA = {
      'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
      'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
      'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
      'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
      'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
      'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
      'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
      'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
    }

    editedFeatures.forEach((f, idx) => {
      const userTitle = localStorage.getItem(`featureTitle_${f.feature_id}`)
      const label = f.label || `Feature ${f.feature_id}`

      // Add separator for readability
      if (idx > 0) lines.push('')

      // Feature metadata
      lines.push(`=== FEATURE ${f.feature_id} ===`)
      lines.push(`Feature ID,${f.feature_id}`)
      lines.push(`Original Label,${escapeCsv(label)}`)
      lines.push(`Your Title,${escapeCsv(userTitle)}`)
      lines.push(`Activation Frequency,${(f.activation_freq || 0).toFixed(6)}`)
      lines.push(`Max Activation,${(f.max_activation || 0).toFixed(4)}`)
      lines.push('')

      // Vocab logits
      const logits = vocabLogits?.[String(f.feature_id)]
      if (logits) {
        lines.push('TOP PROMOTED CODONS')
        lines.push('Codon,Amino Acid,Logit Value')
        ;(logits.top_positive || []).forEach(([codon, val]) => {
          lines.push(`${codon},${CODON_AA[codon] || '?'},${val.toFixed(4)}`)
        })
        lines.push('')

        lines.push('TOP SUPPRESSED CODONS')
        lines.push('Codon,Amino Acid,Logit Value')
        ;(logits.top_negative || []).forEach(([codon, val]) => {
          lines.push(`${codon},${CODON_AA[codon] || '?'},${val.toFixed(4)}`)
        })
        lines.push('')
      }

      // Feature analysis
      const analysis = featureAnalysis?.[String(f.feature_id)]
      if (analysis?.codon_annotations) {
        lines.push('CODON ANNOTATIONS')
        const ann = analysis.codon_annotations
        if (ann.amino_acid) {
          lines.push(`Amino Acid,${ann.amino_acid.aa}`)
          lines.push(`AA Frequency,${(ann.amino_acid.fraction * 100).toFixed(1)}%`)
        }
        if (ann.codon_usage) {
          lines.push(`Codon Usage,${ann.codon_usage.bias}`)
        }
        if (ann.wobble) {
          lines.push(`Wobble Position,${ann.wobble.preference}`)
        }
        if (ann.cpg) {
          lines.push(`CpG Context,${ann.cpg.fraction}`)
        }
        lines.push('')
      }
    })

    // Create and download file
    const csv = lines.join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `edited_features_${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [features, vocabLogits, featureAnalysis])

  // Update Mosaic crossfilter when "Edited Only" toggle changes
  useEffect(() => {
    if (!brushRef.current || !mosaicReady) return

    const selection = brushRef.current

    if (showEditedOnly) {
      // Get all edited feature IDs from localStorage
      const editedIds = features
        .filter(f => localStorage.getItem(`featureTitle_${f.feature_id}`) !== null)
        .map(f => f.feature_id)

      if (editedIds.length > 0) {
        // Create predicate: feature_id IN (id1, id2, id3, ...)
        const idsStr = editedIds.join(',')
        // Use raw SQL string, not literal() which would quote it as a string
        const predicateSql = `feature_id IN (${idsStr})`

        try {
          selection.update({
            source: editedSource.current,
            predicate: predicateSql,
            value: 'edited'
          })
        } catch (err) {
          console.warn('Error updating edited filter:', err)
        }
      }
    } else {
      // Clear the edited filter
      try {
        selection.update({
          source: editedSource.current,
          predicate: null,
          value: null
        })
      } catch (err) {
        console.warn('Error clearing edited filter:', err)
      }
    }
  }, [showEditedOnly, mosaicReady, features])

  // Update Mosaic crossfilter when legend selection changes
  useEffect(() => {
    if (!brushRef.current || !mosaicReady) return

    const selection = brushRef.current

    if (hiddenCategories.size > 0 && selectedCategory && selectedCategory !== 'none') {
      const colInfo = categoryColumns.find(c => c.name === selectedCategory)
      if (colInfo && (colInfo.type === 'string' || colInfo.type === 'integer')) {
        const values = Array.from(hiddenCategories).map(v => `'${v.replace(/'/g, "''")}'`).join(',')
        const predicateSql = `"${selectedCategory}" IN (${values})`

        try {
          selection.update({
            source: legendSource.current,
            predicate: predicateSql,
            value: Array.from(hiddenCategories).join(',')
          })
        } catch (err) {
          console.warn('Legend filter update failed:', err)
        }
      }
    } else {
      try {
        selection.update({
          source: legendSource.current,
          predicate: null,
          value: null
        })
      } catch (err) {
        // Ignore
      }
    }
  }, [hiddenCategories, selectedCategory, mosaicReady, categoryColumns])

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

    // Filter by edited features only
    if (showEditedOnly) {
      result = result.filter(f => localStorage.getItem(`featureTitle_${f.feature_id}`) !== null)
    }

    // Helper: unlabeled features sort last
    const isUnlabeled = (f) => {
      const lbl = (f.label || f.description || '').toLowerCase()
      return !lbl || lbl.startsWith('feature ') || lbl.includes('common codons')
    }

    // Sort (labeled features first, then by chosen metric)
    if (sortBy === 'frequency') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (b.activation_freq || 0) - (a.activation_freq || 0))
    } else if (sortBy === 'max_activation') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (b.max_activation || 0) - (a.max_activation || 0))
    } else if (sortBy === 'feature_id') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || a.feature_id - b.feature_id)
    } else if (sortBy === 'high_score_fraction') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (b.high_score_fraction || 0) - (a.high_score_fraction || 0))
    } else if (sortBy === 'mean_variant_delta') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || Math.abs(b.mean_variant_delta || 0) - Math.abs(a.mean_variant_delta || 0))
    } else if (sortBy === 'mean_site_delta') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || Math.abs(b.mean_site_delta || 0) - Math.abs(a.mean_site_delta || 0))
    } else if (sortBy === 'mean_local_delta') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || Math.abs(b.mean_local_delta || 0) - Math.abs(a.mean_local_delta || 0))
    } else if (sortBy === 'clinvar_fraction') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (b.clinvar_fraction || 0) - (a.clinvar_fraction || 0))
    } else if (sortBy === 'mean_phylop') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (b.mean_phylop || 0) - (a.mean_phylop || 0))
    } else if (sortBy === 'gc_mean') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || Math.abs((b.gc_mean || 0.5) - 0.5) - Math.abs((a.gc_mean || 0.5) - 0.5))
    } else if (sortBy === 'trinuc_entropy') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (a.trinuc_entropy ?? 99) - (b.trinuc_entropy ?? 99))
    } else if (sortBy === 'gene_entropy') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (a.gene_entropy ?? 99) - (b.gene_entropy ?? 99))
    } else if (sortBy === 'gene_n_unique') {
      result = [...result].sort((a, b) => isUnlabeled(a) - isUnlabeled(b) || (a.gene_n_unique || 999) - (b.gene_n_unique || 999))
    }

    return result
  }, [features, sortBy, selectedFeatureIds, searchTerm, showEditedOnly])

  // Reset pagination when filters change
  useEffect(() => {
    setDisplayedCardCount(20)
    loadingMoreRef.current = false
  }, [searchTerm, sortBy, selectedFeatureIds, showEditedOnly])

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
            background: '#76b900',
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
          <p style={styles.subtitle}>CodonFM Sparse AutoEncoder</p>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            onClick={handleExportEdited}
            title="Export all edited features to CSV"
            style={{
              padding: '8px 14px',
              fontSize: '12px',
              border: '1px solid var(--border-input)',
              borderRadius: '6px',
              background: 'var(--bg-input)',
              cursor: 'pointer',
              color: 'var(--text-secondary)',
              whiteSpace: 'nowrap',
              fontWeight: '500',
            }}
          >
            Export Edited
          </button>
          <button
            onClick={() => setDarkMode(d => !d)}
            title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            style={{
              padding: '6px',
              border: '1px solid var(--border-input)',
              borderRadius: '6px',
              background: 'var(--bg-input)',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--text-secondary)',
            }}
          >
            {darkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>

      <div style={styles.mainContent}>
        <div style={styles.leftPanel}>
          <div style={styles.embeddingPanel}>
            <div style={styles.panelHeader}>
              <span style={styles.panelTitle}>
                Decoder UMAP
              </span>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Color by:</label>
                <select
                  value={selectedCategory}
                  onChange={e => {
                    const val = e.target.value
                    setSelectedCategory(val)
                    setHiddenCategories(new Set())
                    setHistMetric3(val)
                    setClickedFeatureId(null)
                    setCardResetKey(k => k + 1)
                  }}
                  style={styles.colorSelect}
                >
                  <option value="none">None</option>
                  {categoryColumns.map(col => (
                    <option key={col.name} value={col.name}>
                      {col.name.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
                <span
                  onClick={() => setShowMetricsModal(true)}
                  style={{
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    width: '15px', height: '15px', borderRadius: '50%', border: '1px solid var(--border-input)',
                    fontSize: '10px', fontWeight: '600', color: 'var(--text-tertiary)', cursor: 'pointer',
                    userSelect: 'none', lineHeight: 1, flexShrink: 0,
                  }}
                >i</span>
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
                  features={features}
                  selectedCategory={selectedCategory}
                  darkMode={darkMode}
                  hiddenCategories={hiddenCategories}
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

                if (colInfo.type === 'string' || colInfo.type === 'integer') {
                  const catColors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
                  ]
                  // Count occurrences of each category value, sorted alphabetically
                  // (matching DENSE_RANK ORDER BY which is alphabetical)
                  const counts = {}
                  for (const f of features) {
                    const val = f[selectedCategory]
                    if (val != null && val !== '') {
                      counts[val] = (counts[val] || 0) + 1
                    }
                  }
                  // Sort alphabetically to match dense_rank ordering
                  const sortedCategories = Object.keys(counts).sort()
                  return (
                    <div style={{
                      position: 'absolute', right: '8px', top: '8px',
                      maxHeight: 'calc(100% - 16px)', overflowY: 'auto',
                      background: 'var(--bg-card)', border: '1px solid var(--border-card)',
                      borderRadius: '6px', padding: '6px 8px',
                      fontSize: '10px', lineHeight: '1.4',
                      pointerEvents: 'auto', minWidth: '120px', maxWidth: '200px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                    }}>
                      <div style={{
                        fontWeight: '600', fontSize: '10px', color: 'var(--text-secondary)',
                        marginBottom: '4px', borderBottom: '1px solid var(--border-card)', paddingBottom: '3px',
                      }}>
                        {selectedCategory.replace(/_/g, ' ').replace('gsea ', '')}
                      </div>
                      {sortedCategories.map((cat, i) => {
                        const hasFilter = hiddenCategories.size > 0
                        const isHidden = hasFilter && !hiddenCategories.has(cat)
                        return (
                          <div
                            key={cat}
                            onClick={(e) => {
                              if (e.metaKey || e.ctrlKey) {
                                // Cmd/Ctrl+click: toggle this category in the selection
                                setHiddenCategories(prev => {
                                  const next = new Set(prev)
                                  if (next.has(cat)) {
                                    next.delete(cat)
                                    // If nothing left selected, clear filter
                                    return next.size === 0 ? new Set() : next
                                  } else {
                                    next.add(cat)
                                    return next
                                  }
                                })
                              } else {
                                // Regular click: solo this category (or clear if already solo'd)
                                setHiddenCategories(prev => {
                                  if (prev.size === 1 && prev.has(cat)) return new Set()
                                  return new Set([cat])
                                })
                              }
                            }}
                            style={{
                              display: 'flex', alignItems: 'center', gap: '5px', padding: '2px 0',
                              cursor: 'pointer', opacity: isHidden ? 0.15 : 1,
                              userSelect: 'none',
                            }}
                          >
                            <span style={{
                              width: '8px', height: '8px', borderRadius: '2px', flexShrink: 0,
                              background: isHidden ? '#888' : catColors[i % catColors.length],
                            }} />
                            <span style={{
                              color: 'var(--text-primary)', overflow: 'hidden',
                              textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                              textDecoration: isHidden ? 'line-through' : 'none',
                            }} title={cat}>
                              {cat}
                            </span>
                            <span style={{ color: 'var(--text-muted)', fontSize: '9px', flexShrink: 0 }}>
                              {counts[cat]}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  )
                }

                return null
              })()}
            </div>
          </div>

          <div style={styles.histogramRow}>
            {[
              { value: histMetric1, setter: setHistMetric1 },
              { value: histMetric2, setter: setHistMetric2 },
              { value: histMetric3, setter: setHistMetric3 },
            ].map(({ value, setter }, i) => (
              <div key={i} style={styles.histogramPanel}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <select
                    value={value}
                    onChange={e => setter(e.target.value)}
                    style={{
                      padding: '4px 8px',
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
                  </select>
                </div>
                {mosaicReady && value && value !== 'none' && (
                  <Histogram
                    key={`hist-${i}-${value}-${plotResetKey}`}
                    brush={brushRef.current}
                    column={value}
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
              <option value="feature_id">By Feature ID</option>
              <option value="high_score_fraction">By High Score Fraction</option>
              <option value="mean_variant_delta">By Variant Delta</option>
              <option value="mean_site_delta">By Site Delta</option>
              <option value="mean_local_delta">By Local Delta</option>
              <option value="clinvar_fraction">By ClinVar Fraction</option>
              <option value="mean_phylop">By PhyloP</option>
              <option value="gc_mean">By GC Bias</option>
              <option value="trinuc_entropy">By Trinuc Specificity</option>
              <option value="gene_entropy">By Gene Specificity</option>
              <option value="gene_n_unique">By Gene Specificity (count)</option>
            </select>
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', color: '#666', cursor: 'pointer', whiteSpace: 'nowrap' }}>
              <input
                type="checkbox"
                checked={showEditedOnly}
                onChange={() => setShowEditedOnly(!showEditedOnly)}
                style={{ cursor: 'pointer' }}
              />
              Edited Only
            </label>
          </div>

          <div style={{ ...styles.stats, display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span>
              Showing {filteredFeatures.length} of {features.length} features
              {selectedFeatureIds !== null && ` (${selectedFeatureIds.size} selected in UMAP)`}
            </span>
            <span
              onClick={() => setShowGuideModal(true)}
              style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: '15px', height: '15px', borderRadius: '50%', border: '1px solid #bbb',
                fontSize: '10px', fontWeight: '600', color: '#888', cursor: 'pointer',
                userSelect: 'none', lineHeight: 1, flexShrink: 0,
              }}
            >i</span>
          </div>

          <FeatureList
            filteredFeatures={filteredFeatures}
            displayedCardCount={displayedCardCount}
            clickedFeatureId={clickedFeatureId}
            features={features}
            cardResetKey={cardResetKey}
            handleCardClick={handleCardClick}
            loadExamples={loadExamplesForFeature}
            vocabLogits={vocabLogits}
            featureAnalysis={featureAnalysis}
            featureListRef={featureListRef}
            endOfListRef={endOfListRef}
            featureRefs={featureRefs}
          />
        </div>
      </div>

      {showGuideModal && (
        <div
          onClick={() => setShowGuideModal(false)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: 'var(--bg-card)', borderRadius: '10px', maxWidth: '560px', width: '90%',
              maxHeight: '80vh', overflowY: 'auto', padding: '28px 32px',
              boxShadow: '0 8px 30px rgba(0,0,0,0.2)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>Feature Card Guide</h2>
              <span
                onClick={() => setShowGuideModal(false)}
                style={{ cursor: 'pointer', fontSize: '20px', color: '#999', lineHeight: 1 }}
              >&times;</span>
            </div>

            <div style={{ fontSize: '13px', color: 'var(--text)', lineHeight: '1.7' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', marginTop: '0', marginBottom: '6px' }}>Decoder Logits</h3>
              <p style={{ margin: '0 0 16px' }}>
                The decoder logits histogram shows the projection of each feature's learned decoder weight vector through the language model's prediction head, with the mean logit vector subtracted across all features. This mean-centering removes the model's shared baseline bias toward common codons (e.g. GCC), so values reflect what each feature <em>specifically</em> promotes or suppresses relative to the average feature. Each bar represents a codon. Green bars indicate codons the feature promotes above baseline; red bars indicate codons it suppresses below baseline. Gray bars have no feature-specific effect. This tells you what the feature pushes the model to output — not what activates it. Stop codons (TAA, TAG, TGA) are excluded because the model was trained on coding sequences where internal stops almost never appear, so all features uniformly suppress them.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Top Activating Sequences</h3>
              <p style={{ margin: '0 0 16px' }}>
                These are the protein-coding sequences where this feature fires most strongly. Each codon is colored by its activation value — brighter highlights mean the feature responds more strongly at that position. This shows what <em>inputs trigger</em> the feature, which is conceptually distinct from decoder logits. A feature can activate strongly on a particular codon (e.g., lysine codons) without promoting that same codon in the output — it may instead influence downstream or contextual predictions.
              </p>

            </div>
          </div>
        </div>
      )}

      {showMetricsModal && (
        <div
          onClick={() => setShowMetricsModal(false)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: 'var(--bg-card)', borderRadius: '10px', maxWidth: '620px', width: '90%',
              maxHeight: '80vh', overflowY: 'auto', padding: '28px 32px',
              boxShadow: '0 8px 30px rgba(0,0,0,0.2)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>Variant Analysis Metrics</h2>
              <span
                onClick={() => setShowMetricsModal(false)}
                style={{ cursor: 'pointer', fontSize: '20px', color: '#999', lineHeight: 1 }}
              >&times;</span>
            </div>

            <div style={{ fontSize: '13px', color: 'var(--text)', lineHeight: '1.7' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', marginTop: '0', marginBottom: '6px' }}>Mean Variant Score (per model)</h3>
              <p style={{ margin: '0 0 16px' }}>
                For each feature, the average model effect score across variant sequences where the feature fires. Computed for the <code>1b_cdwt</code> model score column. A high value means the feature preferentially activates on variants that model predicts to be functionally impactful.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>High Score Fraction</h3>
              <p style={{ margin: '0 0 16px' }}>
                Variants are split at the median model score. Among variants where a feature fires, what fraction are high-scoring? A value of 0.5 means no preference. Above 0.5 means the feature disproportionately fires on high-impact variants. Robust to outliers — measures distributional preference rather than average.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>ClinVar Fraction</h3>
              <p style={{ margin: '0 0 16px' }}>
                Among variant sequences where the feature fires, the fraction from ClinVar vs COSMIC. ClinVar variants are germline (inherited, Mendelian disease). COSMIC variants are somatic (cancer mutations). High ClinVar fraction means the feature responds to germline disease patterns; low means it prefers somatic cancer mutation patterns.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Mean PhyloP</h3>
              <p style={{ margin: '0 0 16px' }}>
                Average evolutionary conservation score (PhyloP) across sequences where the feature fires. High values indicate conserved positions (functionally important). Negative values indicate rapidly evolving regions. Features with high mean PhyloP capture evolutionarily constrained patterns.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Mean Variant Delta</h3>
              <p style={{ margin: '0 0 16px' }}>
                For each gene, the difference in max feature activation between the variant and reference sequence: <code>max_act(variant) &minus; max_act(ref)</code>, averaged across all variant-ref pairs. Positive means the mutation increases feature activation; negative means it suppresses it. Near zero means the feature responds to the gene background, not the specific mutation. This controls for gene identity.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Mean Site Delta</h3>
              <p style={{ margin: '0 0 16px' }}>
                Like mean variant delta, but measured only at the exact codon position where the mutation occurs: <code>activation_f(variant, pos) &minus; activation_f(ref, pos)</code>. This captures direct effects — the feature responding to the changed codon itself. Compare with mean variant delta: a large variant delta but small site delta means the feature captures indirect/distal effects of the mutation (e.g., changes to predicted protein folding context), not the local codon change.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Mean Local Delta</h3>
              <p style={{ margin: '0 0 16px' }}>
                Like variant delta, but using the max activation within a <strong>3-codon window</strong> around the variant site instead of the full sequence. Captures local effects of the mutation: <code>max(window_variant) &minus; max(window_ref)</code>. A large local delta with a small global delta means the mutation's effect is localized. Compare with site delta (exact position only) and variant delta (full sequence).
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>GC Content (mean, std)</h3>
              <p style={{ margin: '0 0 16px' }}>
                Mean and standard deviation of GC content across all sequences where the feature fires. Features with extreme GC mean (far from ~0.5) are GC-biased. Features with low GC std activate only on sequences with similar GC content — suggesting sensitivity to nucleotide composition rather than specific codon patterns.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Trinuc Entropy</h3>
              <p style={{ margin: '0 0 16px' }}>
                Shannon entropy (in bits) of the trinucleotide context distribution among variant sequences where the feature fires. Low entropy means the feature concentrates on specific mutation contexts (e.g., <code>C[C&gt;T]G</code> for CpG transitions). High entropy means it fires across diverse mutation types. The dominant fraction shows what fraction of activations come from the most common trinuc context.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Gene Distribution</h3>
              <p style={{ margin: '0 0 16px' }}>
                Shannon entropy of the gene distribution among sequences where the feature fires. Low entropy means the feature is gene-specific — it concentrates on a few genes. High entropy means it fires broadly. <code>gene_n_unique</code> is the number of distinct genes. <code>gene_dominant_frac</code> is the fraction from the most common gene. A feature with low entropy and high dominant fraction has learned something specific to one gene family.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>High Score Delta</h3>
              <p style={{ margin: '0 0 16px' }}>
                Same as mean variant delta, but averaged only over variants with model scores above the median. Shows how the feature responds specifically to high-impact mutations. Compare with low score delta: if <code>high_score_delta &gt;&gt; low_score_delta</code>, the feature selectively detects impactful mutations.
              </p>

              <h3 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '6px' }}>Low Score Delta</h3>
              <p style={{ margin: '0 0 0' }}>
                Same as mean variant delta, but averaged only over variants with model scores below the median. Features where high score delta and low score delta differ significantly have learned to discriminate mutation severity. Features where both are similar just detect that a mutation occurred without distinguishing impact.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
