import React, { useEffect, useState } from 'react'
import App from './App'
import GenerativeSteering from './GenerativeSteering'
import SequenceInspector from './SequenceInspector'
import { Sun, Moon } from 'lucide-react'

// Three-tab shell. The Feature atlas is the static-parquet explorer (works
// offline); Generative steering and Sequence inspector both talk to the live
// backend (server.py) through the /api proxy.
const TABS = [
  { id: 'atlas', label: 'Feature atlas' },
  { id: 'steering', label: 'Generative steering' },
  { id: 'inspector', label: 'Sequence inspector' },
]

export default function Dashboard() {
  const [tab, setTab] = useState('atlas')
  const [dark, setDark] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

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

      <div style={{ ...S.content, overflow: tab === 'atlas' ? 'hidden' : 'auto' }}>
        {tab === 'atlas' && <App />}
        {tab === 'steering' && <GenerativeSteering />}
        {tab === 'inspector' && <SequenceInspector />}
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
  content: { flex: 1, minHeight: 0 },
}
