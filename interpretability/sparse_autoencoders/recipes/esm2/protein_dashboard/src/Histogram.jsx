import React, { useEffect, useRef } from 'react'
import * as vg from '@uwdata/vgplot'

const FILL_COLOR = "#76b900"

function injectAxisLine(plot, marginLeft, marginRight, marginBottom, height, axisColor) {
  const svg = plot.tagName === 'svg' ? plot : plot.querySelector?.('svg')
  if (!svg) return
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

export default function Histogram({ brush, column, label, categoryColumns }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !brush) return

    // Clear previous content
    containerRef.current.innerHTML = ''

    const computedBg = getComputedStyle(document.documentElement).getPropertyValue('--density-bar-bg').trim() || '#e0e0e0'
    const axisColor = getComputedStyle(document.documentElement).getPropertyValue('--text-tertiary').trim() || '#888'
    const width = containerRef.current.clientWidth - 20
    const height = 50
    const marginLeft = 45
    const marginBottom = 20
    const marginRight = 10
    const marginTop = 5

    // Check if this column is categorical
    const colInfo = categoryColumns?.find(c => c.name === column)
    const isCategorical = colInfo && (colInfo.type === 'string' || colInfo.type === 'integer')

    let plot
    if (isCategorical) {
      const catHeight = 80
      const catMarginBottom = 50
      plot = vg.plot(
        vg.barY(
          vg.from("features"),
          { x: column, y: vg.count(), fill: computedBg, inset: 1 }
        ),
        vg.barY(
          vg.from("features", { filterBy: brush }),
          { x: column, y: vg.count(), fill: FILL_COLOR, inset: 1 }
        ),
        vg.toggleX({ as: brush }),
        vg.xLabel(null),
        vg.yLabel(null),
        vg.xTickRotate(-45),
        vg.xTickSize(3),
        vg.style({ fontSize: '9px' }),
        vg.width(width),
        vg.height(catHeight),
        vg.marginLeft(marginLeft),
        vg.marginBottom(catMarginBottom),
        vg.marginTop(marginTop),
        vg.marginRight(marginRight)
      )
    } else {
      // Numeric histogram: binned rectY
      plot = vg.plot(
        vg.rectY(
          vg.from("features"),
          { x: vg.bin(column), y: vg.count(), fill: computedBg, inset: 1 }
        ),
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
    }

    containerRef.current.appendChild(plot)

    const timer = setTimeout(() => {
      injectAxisLine(plot, marginLeft, marginRight, marginBottom, height, axisColor)
    }, 50)

    return () => {
      clearTimeout(timer)
      if (containerRef.current) {
        containerRef.current.innerHTML = ''
      }
    }
  }, [brush, column, label, categoryColumns])

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', marginTop: '2px' }}
    />
  )
}
