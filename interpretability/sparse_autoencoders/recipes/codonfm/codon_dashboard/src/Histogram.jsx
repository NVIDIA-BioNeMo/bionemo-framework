import React, { useEffect, useRef } from 'react'
import * as vg from '@uwdata/vgplot'

const FILL_COLOR = "#76b900"

function injectAxisLine(plot, marginLeft, marginRight, marginBottom, height, axisColor) {
  const svg = plot.tagName === 'svg' ? plot : plot.querySelector?.('svg')
  if (!svg) return
  // Remove any previously injected line
  svg.querySelectorAll('.x-axis-line').forEach(el => el.remove())
  const svgWidth = svg.getAttribute('width') || svg.clientWidth
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
  line.classList.add('x-axis-line')
  line.setAttribute('x1', marginLeft)
  line.setAttribute('x2', svgWidth - marginRight)
  line.setAttribute('y1', height - marginBottom)
  line.setAttribute('y2', height - marginBottom)
  line.setAttribute('stroke', axisColor)
  line.setAttribute('stroke-width', '1')
  svg.appendChild(line)
}

export default function Histogram({ brush, column, label }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !brush) return

    // Clear previous content
    containerRef.current.innerHTML = ''

    const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--density-bar-bg').trim() || '#e0e0e0'
    const axisColor = getComputedStyle(document.documentElement).getPropertyValue('--text-tertiary').trim() || '#888'
    const width = containerRef.current.clientWidth - 20
    const height = 50
    const marginLeft = 45
    const marginBottom = 20
    const marginRight = 10
    const marginTop = 5

    const plot = vg.plot(
      // Background histogram: full data (no filterBy)
      vg.rectY(
        vg.from("features"),
        { x: vg.bin(column), y: vg.count(), fill: bgColor, inset: 1 }
      ),
      // Foreground histogram: filtered data
      vg.rectY(
        vg.from("features", { filterBy: brush }),
        { x: vg.bin(column), y: vg.count(), fill: FILL_COLOR, inset: 1 }
      ),
      vg.intervalX({ as: brush }),
      vg.xLabel(null),
      vg.yLabel(null),
      vg.width(width),
      vg.height(height),
      vg.marginLeft(marginLeft),
      vg.marginBottom(marginBottom),
      vg.marginTop(marginTop),
      vg.marginRight(marginRight)
    )

    containerRef.current.appendChild(plot)

    // Inject axis line into the SVG directly (immune to container resize)
    // Use a short delay to ensure the SVG is rendered
    const timer = setTimeout(() => {
      injectAxisLine(plot, marginLeft, marginRight, marginBottom, height, axisColor)
    }, 50)

    return () => {
      clearTimeout(timer)
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [brush, column, label])

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', minHeight: '90px', marginTop: '4px' }}
    />
  )
}
