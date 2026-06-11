import React, { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

export default function InfoButton({ text }) {
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef(null)
  const buttonRef = useRef(null)
  const [pos, setPos] = useState(null)

  useEffect(() => {
    if (!open) return
    const handleClick = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [open])

  useEffect(() => {
    if (open && buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect()
      setPos({
        top: rect.top - 8,
        left: rect.left + rect.width / 2,
      })
    }
  }, [open])

  return (
    <span ref={wrapperRef} style={{ position: 'relative', display: 'inline-block', marginLeft: '5px' }}>
      <span
        ref={buttonRef}
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '15px',
          height: '15px',
          borderRadius: '50%',
          border: '1px solid var(--border-input)',
          fontSize: '10px',
          fontWeight: '600',
          color: 'var(--text-tertiary)',
          cursor: 'pointer',
          userSelect: 'none',
          lineHeight: 1,
        }}
      >
        i
      </span>
      {open && pos && createPortal(
        <div style={{
          position: 'fixed',
          bottom: window.innerHeight - pos.top,
          left: pos.left,
          transform: 'translateX(-50%)',
          width: '240px',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: '6px',
          padding: '10px 12px',
          fontSize: '12px',
          fontWeight: 'normal',
          color: 'var(--text)',
          lineHeight: '1.5',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          zIndex: 10000,
        }}>
          {text}
        </div>,
        document.body
      )}
    </span>
  )
}
