import React, { useEffect, useRef } from 'react'
import { EmbeddingViewMosaic } from 'embedding-atlas'

// Color palette for categories (D3 category10 + extended)
const CATEGORY_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
  "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

// Sequential color palette (NVIDIA brand)
const SEQUENTIAL_COLORS = [
  "#c359ef", "#9525C6", "#0046a4", "#0074DF", "#3f8500",
  "#76B900", "#ef9100", "#F9C500", "#ff8181", "#EF2020"
]

// Default color for uniform coloring (NVIDIA green)
const DEFAULT_COLOR = "#76b900"

// Custom tooltip renderer
class FeatureTooltip {
  constructor(node, props) {
    this.node = node
    this.inner = document.createElement("div")
    this.inner.style.cssText = `
      background: var(--bg-card);
      border: 1px solid var(--border-input);
      border-radius: 4px;
      padding: 8px 12px;
      font-family: 'NVIDIA Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 13px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      max-width: 300px;
      color: var(--text);
    `
    this.node.appendChild(this.inner)
    this.update(props)
  }

  update(props) {
    const { tooltip } = props
    if (!tooltip) {
      this.inner.innerHTML = ""
      return
    }
    const featureId = tooltip.identifier ?? ""
    const label = tooltip.fields?.label ?? tooltip.text ?? ""
    const logFreq = tooltip.fields?.log_frequency
    const maxAct = tooltip.fields?.max_activation
    const colorField = tooltip.fields?.color_field

    this.inner.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 4px; color: var(--text-heading);">Feature #${featureId}</div>
      <div style="color: var(--text-secondary); margin-bottom: 4px; font-size: 12px;">${label}</div>
      ${colorField ? `<div style="color: var(--text-tertiary); font-size: 11px;">Category: ${colorField}</div>` : ""}
      ${logFreq !== undefined ? `<div style="color: var(--text-tertiary); font-size: 11px;">Log Frequency: ${logFreq.toFixed(3)}</div>` : ""}
      ${maxAct !== undefined ? `<div style="color: var(--text-tertiary); font-size: 11px;">Max Activation: ${maxAct.toFixed(2)}</div>` : ""}
    `
  }

  destroy() {
    this.inner.remove()
  }
}

export default function EmbeddingView({ brush, categoryColumn, categoryColumns, onFeatureClick, highlightedFeatureId, viewportState, onViewportChange, labels, darkMode }) {
  const containerRef = useRef(null)
  const viewRef = useRef(null)
  const onFeatureClickRef = useRef(onFeatureClick)
  const onViewportChangeRef = useRef(onViewportChange)

  // Keep the callback refs updated
  useEffect(() => {
    onFeatureClickRef.current = onFeatureClick
  }, [onFeatureClick])

  useEffect(() => {
    onViewportChangeRef.current = onViewportChange
  }, [onViewportChange])

  // Update selection and tooltip when highlightedFeatureId changes
  useEffect(() => {
    if (viewRef.current) {
      viewRef.current.update({
        selection: highlightedFeatureId != null ? [highlightedFeatureId] : null,
        tooltip: highlightedFeatureId != null ? highlightedFeatureId : null
      })
    }
  }, [highlightedFeatureId])

  // Update viewport when viewportState changes
  useEffect(() => {
    if (viewRef.current) {
      viewRef.current.update({
        viewportState: viewportState
      })
    }
  }, [viewportState])

  // Update labels when they change
  useEffect(() => {
    if (viewRef.current) {
      viewRef.current.update({
        labels: labels || null
      })
    }
  }, [labels])

  useEffect(() => {
    if (!containerRef.current || !brush) return

    // Clear previous view
    if (viewRef.current) {
      containerRef.current.innerHTML = ''
    }

    // Determine category column and colors
    let categoryColName = null
    let colors = Array(50).fill(DEFAULT_COLOR)
    let additionalFields = {
      label: "label",
      log_frequency: "log_frequency",
      max_activation: "max_activation",
    }

    if (categoryColumn && categoryColumn !== "none") {
      const colInfo = categoryColumns?.find(c => c.name === categoryColumn)
      if (colInfo) {
        if (colInfo.type === 'sequential') {
          // Sequential column - use binned version and sequential colors
          categoryColName = `${categoryColumn}_bin`
          colors = SEQUENTIAL_COLORS
        } else if (colInfo.type === 'string') {
          // Categorical string column
          categoryColName = `${categoryColumn}_cat`
          colors = CATEGORY_COLORS.slice(0, Math.max(colInfo.nUnique, 10))
        } else {
          // Integer categorical column
          categoryColName = categoryColumn
          colors = CATEGORY_COLORS.slice(0, Math.max(colInfo.nUnique, 10))
        }
        additionalFields.color_field = categoryColumn
      }
    }

    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight

    try {
      viewRef.current = new EmbeddingViewMosaic(
        containerRef.current,
        {
          table: "features",
          x: "x",
          y: "y",
          category: categoryColName,
          text: "label",
          identifier: "feature_id",
          filter: brush,
          rangeSelection: brush,
          selection: highlightedFeatureId != null ? [highlightedFeatureId] : null,
          viewportState: viewportState,
          categoryColors: colors,
          width: width,
          height: height,
          labels: labels || null,
          config: {
            mode: "points",
            colorScheme: darkMode ? "dark" : "light",
            autoLabelEnabled: false,
          },
          theme: {
            brandingLink: {
              text: "NVIDIA BioNeMo",
              href: "https://github.com/NVIDIA/bionemo-framework",
            },
          },
          additionalFields: additionalFields,
          customTooltip: FeatureTooltip,
          onSelection: (selection) => {
            // selection is DataPoint[] | null
            if (!onFeatureClickRef.current) return

            if (selection && selection.length > 0) {
              // Get the last clicked point (most recent selection)
              const lastPoint = selection[selection.length - 1]
              const featureId = lastPoint?.identifier ?? lastPoint
              const x = lastPoint?.x
              const y = lastPoint?.y
              if (featureId != null) {
                onFeatureClickRef.current(featureId, x, y)
              }
            } else {
              // Clicked on empty canvas - clear selection
              onFeatureClickRef.current(null)
            }
          },
          onViewportState: (vp) => {
            if (onViewportChangeRef.current && vp) {
              onViewportChangeRef.current(vp)
            }
          },
        }
      )
    } catch (err) {
      console.warn('Error creating EmbeddingViewMosaic:', err)
    }

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [brush, categoryColumn, categoryColumns, darkMode])

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (viewRef.current && containerRef.current) {
        const width = containerRef.current.clientWidth
        const height = containerRef.current.clientHeight
        viewRef.current.update({ width, height })
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: '300px' }}
    />
  )
}
