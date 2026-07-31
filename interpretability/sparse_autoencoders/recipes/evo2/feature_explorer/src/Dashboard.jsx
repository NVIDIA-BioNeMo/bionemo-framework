import React, { useEffect, useMemo, useState } from 'react'
import App from './App'
import GenerativeSteering from './GenerativeSteering'
import SequenceInspector from './SequenceInspector'
import SequenceUMAPView, { EMBEDDINGS_URL } from './SequenceUMAPView'
import { Sun, Moon } from 'lucide-react'
import { useHealth } from './backend'

// Four-tab shell with graceful degradation when there's no live backend:
//   offline:true     -> always available (reads static files only)
//   offline:'bundle' -> available offline IFF a precomputed embeddings bundle is present
//   offline:false    -> needs the live model (server.py); hidden when the backend is offline
const ALL_TABS = [
  {
    id: 'atlas', label: 'Feature atlas', offline: true, // static parquet
    desc: 'Browse every feature in the SAE dictionary — its firing rate across the corpus, position in a decoder-space UMAP, top-activating example sequences, and any label. Reads precomputed static files, so it needs no backend or GPU.',
    limits: [
      'Static snapshot — reflects the activation corpus and SAE checkpoint it was built from, not anything you enter live.',
      'The UMAP is a 2-D projection of decoder vectors: nearby points are roughly similar, but distances and axes have no absolute meaning.',
      'Top examples come only from the precompute corpus — a feature may also fire on sequences not shown here.',
      'Labels are human/automatic annotations and are incomplete; most features are unlabeled.',
    ],
  },
  {
    id: 'steering', label: 'Generative steering', offline: false, // needs the live 7B
    desc: 'Generate DNA from a prompt while clamping chosen SAE features on the continuation, and compare against the unsteered baseline. Each request runs the live model end-to-end.',
    limits: [
      'A clamp sets an absolute SAE activation value (capped at ±300); large magnitudes can push the model off-distribution into repetitive or degenerate DNA.',
      'Steering affects only the generated continuation, at the SAE’s layer — the prompt itself is not rewritten.',
      'Prompt + generation must fit the 8192 bp context: an over-length prompt is rejected, and the number of generated tokens is capped to what’s left after the prompt (and its organism tag).',
      'A clamp changes output only if the feature is actually causal for what it appears to encode; a dead or non-causal feature leaves generation unchanged.',
      'One GPU, serialized — concurrent generations queue, and long generations can take a while.',
    ],
  },
  {
    id: 'inspector', label: 'Sequence inspector', offline: false, // needs live encode
    desc: 'Paste a sequence to see per-base SAE feature activations — either the top-k features firing on it or specific feature ids you pick. Each request runs a live encode.',
    limits: [
      'Activations are read at the SAE’s single training layer only — not the whole network.',
      'A phylogenetic tag is prepended to the sequence; its leading positions are shown but excluded from the top-k ranking.',
      'Sequences longer than the model’s 8192 bp context are rejected, not truncated.',
      'Top-k ranks by each feature’s peak activation over the DNA region, so broad-but-weak features can rank below sharp local ones.',
    ],
  },
  {
    id: 'sequmap', label: 'Sequence UMAP', offline: 'bundle', // offline iff dashboard.py embeddings bundle exists
    desc: 'Embed a set of sequences (each pooled into one per-feature SAE vector), UMAP them to 2-D, then color or re-project the layout by any SAE feature. Uses the live backend, or a precomputed bundle when offline.',
    limits: [
      'UMAP is stochastic and only 2-D: the same sequences can land in different spots between runs, and cluster sizes/distances aren’t quantitative.',
      'Each sequence is pooled (mean or max) over its DNA region into one vector — within-sequence position is lost.',
      'Only features firing in at least min_firing sequences are kept, so rare features drop out of the layout.',
      'Up to 1000 sequences per request; sequences past the cap are dropped, and those longer than the 8192 bp context are clamped to it (embedded from their first 8192 bp) — the tab reports the counts. The offline bundle is a fixed precomputed set.',
    ],
  },
]

const TAB_COMPONENTS = { atlas: App, steering: GenerativeSteering, inspector: SequenceInspector, sequmap: SequenceUMAPView }

export default function Dashboard() {
  const [tab, setTab] = useState('atlas')
  const [dark, setDark] = useState(true)
  const health = useHealth()
  const online = health.status === 'ready'
  const [hasBundle, setHasBundle] = useState(false)
  // Keep-alive: remember which tabs have been opened. We mount a tab on first visit and keep it
  // mounted (just hidden) thereafter, so switching tabs never unmounts one. Unmounting is what made
  // the backend appear to "reload" (each tab's useHealth reset to loading) and dropped typed input
  // (a steering prompt/clamps, a loaded UMAP set) on every tab switch.
  const [visited, setVisited] = useState({ atlas: true })
  useEffect(() => { setVisited((v) => (v[tab] ? v : { ...v, [tab]: true })) }, [tab])
  // Once the backend has ever been ready, keep the backend-backed tabs in the bar even if it briefly
  // drops (e.g. an engine restart) — otherwise they'd vanish and bounce you to the atlas for the whole
  // ~30-45s reload. A pure static deploy that was never ready still hides them (nothing to reconnect to).
  const [everReady, setEverReady] = useState(false)
  useEffect(() => { if (health.status === 'ready') setEverReady(true) }, [health.status])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  // Probe for a precomputed Sequence-UMAP bundle so that tab survives without a backend.
  useEffect(() => {
    fetch(EMBEDDINGS_URL, { method: 'HEAD' })
      .then((r) => setHasBundle(r.ok))
      .catch(() => setHasBundle(false))
  }, [])

  const TABS = useMemo(
    () => ALL_TABS.filter((t) => online || everReady || t.offline === true || (t.offline === 'bundle' && hasBundle)),
    [online, everReady, hasBundle],
  )

  // If the active tab is no longer available (backend dropped), fall back to the atlas.
  useEffect(() => {
    if (!TABS.some((t) => t.id === tab)) setTab('atlas')
  }, [TABS, tab])

  return (
    <div style={S.shell}>
      <div style={S.tabBar}>
        <span style={S.brand}>Evo 2 SAE Feature Explorer</span>
        {TABS.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)} style={tab === t.id ? S.tabOn : S.tabOff}>
            {t.label}
          </button>
        ))}
        <button onClick={() => setDark((d) => !d)} style={S.theme} title="Toggle theme">
          {dark ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>

      {(() => {
        const active = ALL_TABS.find((t) => t.id === tab)
        if (!active?.desc) return null
        return (
          <div style={S.desc}>
            <div>{active.desc}</div>
            {active.limits?.length > 0 && (
              <details style={S.limitsBox}>
                <summary style={S.limitsToggle}>Limitations</summary>
                <ul style={S.limits}>
                  {active.limits.map((l, i) => (
                    <li key={i} style={S.limitItem}>{l}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )
      })()}

      <div style={{ ...S.content, overflow: tab === 'atlas' ? 'hidden' : 'auto' }}>
        {ALL_TABS.map((t) => {
          if (!visited[t.id]) return null // not opened yet — don't mount eagerly
          const Comp = TAB_COMPONENTS[t.id]
          // Kept mounted once visited; only the active tab is shown. State + health persist across switches.
          return (
            <div key={t.id} style={{ display: t.id === tab ? 'block' : 'none', height: '100%' }}>
              <Comp dark={dark} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

const S = {
  shell: { height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg)', color: 'var(--text)' },
  tabBar: {
    display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 16px',
    background: 'var(--bg-card)', borderBottom: '1px solid var(--border)', flexShrink: 0,
  },
  brand: { fontSize: '13px', fontWeight: 700, color: 'var(--text-heading)', marginRight: '14px' },
  tabOn: {
    padding: '6px 14px', border: '1px solid var(--accent)', background: 'var(--bg-card-expanded)',
    color: 'var(--accent)', borderRadius: '5px', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
  },
  tabOff: {
    padding: '6px 14px', border: '1px solid var(--border)', background: 'transparent',
    color: 'var(--text-secondary)', borderRadius: '5px', cursor: 'pointer', fontSize: '12px',
  },
  theme: {
    marginLeft: 'auto', display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    width: '30px', height: '30px', border: '1px solid var(--border)', background: 'transparent',
    color: 'var(--text-secondary)', borderRadius: '5px', cursor: 'pointer',
  },
  desc: {
    padding: '7px 16px', fontSize: '12px', lineHeight: 1.4, color: 'var(--text-secondary)',
    background: 'var(--bg-card)', borderBottom: '1px solid var(--border)', flexShrink: 0,
  },
  limitsBox: { marginTop: '4px' },
  limitsToggle: {
    cursor: 'pointer', fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', userSelect: 'none',
  },
  limits: { margin: '4px 0 2px', paddingLeft: '16px' },
  limitItem: { fontSize: '11px', lineHeight: 1.45, color: 'var(--text-secondary)', marginBottom: '2px' },
  content: { flex: 1, minHeight: 0 },
}
