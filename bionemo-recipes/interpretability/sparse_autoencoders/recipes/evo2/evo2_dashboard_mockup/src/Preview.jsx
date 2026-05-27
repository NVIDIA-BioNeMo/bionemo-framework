import React, { useState } from 'react'
import App from './App'
import ColoredSequence from './ColoredSequence'
import GeneUMAPView from './GeneUMAPView'

// Hit http://localhost:5176/#preview to see all three views side by side.
// Tabs switch between the existing dashboard ("Main") and the two new
// components (ColoredSequence, GeneUMAPView) without restructuring the
// existing layout.

const MOCK_500BP = (
  'ATGCGCAATCGTAGCTTAGCATCGATCGTAGCTATCGATCGTACGTACGTAGCTAGCTAGCTAGCTAGCAATCGTAGCATCGTAG' +
  'CTAGCATCGTAGCTAGCTACGTACGTAGCTAGCTAGCTAATCGGGGGGGGGGGCATCGCGCGCGCGCGCGCATCGTAGCTAGCTA' +
  'GCATCGTAGCTAGCATGCTAGCATGCTAGCTAGCTAGCATCGATGCTAGCATGCTAGCATGCTAGCATCGTAGCATCGTAGCATC' +
  'GTAGCTAGCTAATCGATCGTAGCTAGCATCGATCGTAGCTAGCAATCGTAGCTAGCTAGCTAGCATCGTAGCTAGCTAGCTAGCT' +
  'AGCTAGCATCGTAGCATGCTAGCATGCTAGCATCGTAGCTAGCATGCTAGCAATCGGGGGCATCGCGCGCGCGCATCGTAGCATC' +
  'GTAGCTAGCTAGCTAGCATCGATCGTAGCATCGTAGCAATCGTAGCATCGATCGAATCGTAGCAATCGTAGCATCGTACGTACGT' +
  'AGCTAGCTAGCTAATCGATCGATCGTAGCATCGTACGTACGTACGT'
).slice(0, 500)

const FAKE_FEATURE_CATALOG = {
  101: { label: 'TATA-box-like' },
  207: { label: 'start-codon (ATG)' },
  314: { label: 'GC-rich exon' },
  422: { label: 'splice donor' },
  588: { label: 'polyA signal' },
}

const TABS = [
  { id: 'main', label: 'Main dashboard (features + atlas + WebLogos)' },
  { id: 'sequence', label: 'ColoredSequence (mock 500bp)' },
  { id: 'genes', label: 'Gene UMAP (500 genes, precomputed)' },
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
  sequenceWrap: { padding: '24px' },
  genesWrap: { padding: '24px' },
  title: { fontSize: '20px', fontWeight: 600, marginBottom: '4px' },
  subtitle: { fontSize: '12px', color: 'var(--text-secondary, #666)', marginBottom: '16px' },
  toggleRow: {
    display: 'flex',
    gap: '8px',
    marginBottom: '12px',
    alignItems: 'center',
    fontSize: '12px',
  },
  inputRow: {
    marginBottom: '12px',
    padding: '10px 12px',
    background: 'var(--bg-card, #fff)',
    border: '1px solid var(--border, #ddd)',
    borderRadius: '6px',
  },
  seqInput: {
    width: '100%',
    fontFamily: 'monospace',
    fontSize: '12px',
    padding: '8px',
    border: '1px solid var(--border, #ddd)',
    borderRadius: '4px',
    resize: 'vertical',
    boxSizing: 'border-box',
  },
  inputActions: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    marginTop: '8px',
    fontSize: '11px',
  },
  btnPrimary: {
    padding: '5px 14px',
    border: '1px solid var(--accent, #76b900)',
    background: 'var(--accent, #76b900)',
    color: '#fff',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    fontWeight: 600,
  },
  btnSecondary: {
    padding: '5px 12px',
    border: '1px solid var(--border, #ddd)',
    background: '#fff',
    color: 'var(--text-secondary, #555)',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
  },
  seqMeta: {
    marginLeft: 'auto',
    color: 'var(--text-secondary, #666)',
    fontSize: '11px',
    fontFamily: 'monospace',
  },
  toggle: (active) => ({
    padding: '4px 10px',
    border: `1px solid ${active ? 'var(--accent, #76b900)' : 'var(--border, #ddd)'}`,
    background: active ? 'var(--bg-card-expanded, #f0f8e8)' : '#fff',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '11px',
    fontWeight: active ? 600 : 400,
    color: active ? 'var(--accent, #76b900)' : 'var(--text, #555)',
  }),
}


export default function Preview() {
  const [tab, setTab] = useState('main')
  const [mode, setMode] = useState('top')
  const [singleFeatureId, setSingleFeatureId] = useState(101)
  const [pastedSeq, setPastedSeq] = useState('')        // user-entered sequence, raw
  const [activeSeq, setActiveSeq] = useState(MOCK_500BP) // sequence currently rendered

  // Strip whitespace + newlines + numbers (typical when pasting from FASTA),
  // uppercase, then keep only A/C/G/T/N. Falls back to mock if input is empty.
  const cleanSeq = (raw) => {
    const cleaned = (raw || '').toUpperCase().replace(/[^ACGTN]/g, '')
    return cleaned.length > 0 ? cleaned : MOCK_500BP
  }

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
          // The full existing dashboard — feature catalog, UMAP atlas, FeatureCard
          // expansions with WebLogo PNGs, histograms. Untouched.
          <div style={{ height: '100%' }}>
            <App />
          </div>
        )}

        {tab === 'sequence' && (
          <div style={styles.sequenceWrap}>
            <div style={styles.title}>ColoredSequence</div>
            <div style={styles.subtitle}>
              Paste a DNA sequence below — each base background-colored by its top-firing
              SAE feature. <b>Mock activations</b> (real backend not wired yet); swap the
              <code> analysis</code> prop with a real <code>/analyze</code> response to use live inference.
              Defaults to a 500 bp example if input is empty.
            </div>

            <div style={styles.inputRow}>
              <textarea
                value={pastedSeq}
                onChange={(e) => setPastedSeq(e.target.value)}
                placeholder="Paste DNA sequence here (FASTA okay — whitespace and headers will be stripped). Example: ATGCGCAATCGT..."
                style={styles.seqInput}
                rows={3}
              />
              <div style={styles.inputActions}>
                <button
                  style={styles.btnPrimary}
                  onClick={() => setActiveSeq(cleanSeq(pastedSeq))}
                >
                  Visualize
                </button>
                <button
                  style={styles.btnSecondary}
                  onClick={() => {
                    setPastedSeq('')
                    setActiveSeq(MOCK_500BP)
                  }}
                >
                  Reset to 500 bp mock
                </button>
                <span style={styles.seqMeta}>
                  rendering: <b>{activeSeq.length}</b> bp
                </span>
              </div>
            </div>

            <div style={styles.toggleRow}>
              <span>Mode:</span>
              <button style={styles.toggle(mode === 'top')} onClick={() => setMode('top')}>
                Top feature
              </button>
              <button style={styles.toggle(mode === 'single')} onClick={() => setMode('single')}>
                Single feature
              </button>
              {mode === 'single' && (
                <select
                  value={singleFeatureId}
                  onChange={(e) => setSingleFeatureId(Number(e.target.value))}
                  style={{ marginLeft: '8px', fontSize: '11px', padding: '3px 6px' }}
                >
                  {[101, 207, 314, 422, 588].map((f) => (
                    <option key={f} value={f}>
                      {FAKE_FEATURE_CATALOG[f]?.label || `feature_${f}`}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <ColoredSequence
              sequence={activeSeq}
              featureCatalog={FAKE_FEATURE_CATALOG}
              mode={mode}
              singleFeatureId={mode === 'single' ? singleFeatureId : null}
              onBaseClick={(info) => console.log('clicked base', info)}
            />
          </div>
        )}

        {tab === 'genes' && (
          <div style={styles.genesWrap}>
            <div style={styles.title}>Gene UMAP — 500 genes</div>
            <div style={styles.subtitle}>
              Each point is one gene, positioned by its mean SAE feature vector (layer 20
              activations on Evo2 1B → trained TopK SAE → mean across positions). Cluster
              coloring comes from HDBSCAN on the base UMAP. Click a feature in the
              sidebar to recolor by activation strength; click <b>Reorganize</b> to re-run
              UMAP with that feature emphasized (~2–5 sec, runs in browser).
            </div>
            <GeneUMAPView height={620} />
          </div>
        )}
      </div>
    </div>
  )
}
