import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import Preview from './Preview'

// Hit `/#preview` to see the new ColoredSequence + GeneUMAPView demo without
// disturbing the production dashboard routing.
const isPreview = typeof window !== 'undefined' && window.location.hash === '#preview'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {isPreview ? <Preview /> : <App />}
  </React.StrictMode>
)
