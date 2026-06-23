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
    desc: 'Browse every SAE feature — firing rate, decoder-space UMAP, top-activating example sequences, and labels. Reads precomputed files; no backend needed.',
  },
  {
    id: 'steering', label: 'Generative steering', offline: false, // needs the live 7B
    desc: 'Generate DNA from a prompt while clamping chosen SAE features on the continuation, and compare against the unsteered baseline. Runs the live 7B model.',
  },
  {
    id: 'inspector', label: 'Sequence inspector', offline: false, // needs live encode
    desc: 'Paste a sequence to see per-base SAE feature activations — the top-k firing features or specific ones you pick. Runs a live encode.',
  },
  {
    id: 'sequmap', label: 'Sequence UMAP', offline: 'bundle', // offline iff dashboard.py embeddings bundle exists
    desc: 'Embed a set of sequences, UMAP them, then color or reorganize the layout by an SAE feature. Uses the live backend, or a precomputed bundle when offline.',
  },
]

export default function Dashboard() {
  const [tab, setTab] = useState('atlas')
  const [dark, setDark] = useState(true)
  const health = useHealth()
  const online = health.status === 'ready'
  const [hasBundle, setHasBundle] = useState(false)

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
    () => ALL_TABS.filter((t) => online || t.offline === true || (t.offline === 'bundle' && hasBundle)),
    [online, hasBundle],
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

      {ALL_TABS.find((t) => t.id === tab)?.desc && (
        <div style={S.desc}>{ALL_TABS.find((t) => t.id === tab).desc}</div>
      )}

      <div style={{ ...S.content, overflow: tab === 'atlas' ? 'hidden' : 'auto' }}>
        {tab === 'atlas' && <App />}
        {tab === 'steering' && <GenerativeSteering />}
        {tab === 'inspector' && <SequenceInspector />}
        {tab === 'sequmap' && <SequenceUMAPView />}
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
  content: { flex: 1, minHeight: 0 },
}
