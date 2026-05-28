import React, { useState } from 'react'
import App from './App'
import SteeringExplorer from './SteeringExplorer'

// Hit http://localhost:5176/#preview to see the tabbed preview. The plain `/`
// URL still renders the unchanged main dashboard.

const TABS = [
  { id: 'main', label: 'Main dashboard (features + atlas + WebLogos)' },
  { id: 'steering', label: 'Steering explorer (mock slider + heatmap)' },
]

const styles = {
  container: {
    fontFamily: 'system-ui, sans-serif',
    color: 'var(--text, #222)',
    background: 'var(--bg, #fafafa)',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  tabBar: {
    display: 'flex',
    gap: '4px',
    padding: '8px 16px',
    background: 'var(--bg-card, #fff)',
    borderBottom: '1px solid var(--border, #ddd)',
    flexShrink: 0,
  },
  tab: (active) => ({
    padding: '6px 14px',
    border: '1px solid',
    borderColor: active ? 'var(--accent, #76b900)' : 'var(--border, #ddd)',
    background: active ? 'var(--bg-card-expanded, #f0f8e8)' : '#fff',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '12px',
    fontWeight: active ? 600 : 400,
    color: active ? 'var(--accent, #76b900)' : 'var(--text-secondary, #555)',
  }),
  tabContent: {
    flex: 1,
    overflow: 'auto',
    background: 'var(--bg, #fafafa)',
  },
  wrap: { padding: '24px' },
  title: { fontSize: '20px', fontWeight: 600, marginBottom: '4px' },
  subtitle: { fontSize: '12px', color: 'var(--text-secondary, #666)', marginBottom: '16px' },
}


export default function Preview() {
  const [tab, setTab] = useState('main')

  return (
    <div style={styles.container}>
      <div style={styles.tabBar}>
        {TABS.map((t) => (
          <button key={t.id} style={styles.tab(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={styles.tabContent}>
        {tab === 'main' && (
          // Existing dashboard: feature catalog, UMAP atlas, FeatureCard
          // expansions with WebLogo PNGs, histograms. Untouched.
          <div style={{ height: '100%' }}>
            <App />
          </div>
        )}

        {tab === 'steering' && (
          <div style={styles.wrap}>
            <div style={styles.title}>Steering explorer</div>
            <div style={styles.subtitle}>
              Slide the <b>clamp</b> control to see per-position P(A/C/G/T) shifts across an entire
              200 bp sequence. All data is algorithmically generated mock — when the real
              steering backend lands, the same UI swaps in live results.
            </div>
            <SteeringExplorer />
          </div>
        )}
      </div>
    </div>
  )
}
