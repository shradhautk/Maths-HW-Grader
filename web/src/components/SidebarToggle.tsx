import { useCallback, useEffect, useRef, useState } from 'react'
import '../styles/sidebar-toggle.css'

const KEY = 'ga:sidebarClosed'

/** Whether the left sidebar is slid away (desktop only); remembered across pages and visits. */
// eslint-disable-next-line react-refresh/only-export-components
export function useSidebarClosed() {
  const [closed, setClosed] = useState(() => {
    try {
      return localStorage.getItem(KEY) === '1'
    } catch {
      return false
    }
  })
  const set = useCallback((value: boolean) => {
    setClosed(value)
    try {
      localStorage.setItem(KEY, value ? '1' : '0')
    } catch {
      // Storage unavailable (private window): the choice just isn't remembered.
    }
  }, [])
  return [closed, set] as const
}

/**
 * The sidebar's close button plus the edge tab that brings it back. Render it
 * inside the sidebar <aside>, which also takes the `is-closed` class.
 */
export default function SidebarToggle({ closed, onChange }: { closed: boolean; onChange: (closed: boolean) => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)
  const openRef = useRef<HTMLButtonElement>(null)
  // After a toggle, keep keyboard focus on the button that undoes it.
  const toggled = useRef(false)
  useEffect(() => {
    if (!toggled.current) return
    toggled.current = false
    ;(closed ? openRef : closeRef).current?.focus()
  }, [closed])

  const change = (value: boolean) => {
    toggled.current = true
    onChange(value)
  }

  return (
    <>
      <button ref={closeRef} type="button" className="side-close-btn" aria-label="Close sidebar" title="Close sidebar" onClick={() => change(true)}>
        <span className="material-icons" aria-hidden>
          close
        </span>
      </button>
      <button ref={openRef} type="button" className="side-open-btn" aria-label="Open sidebar" title="Open sidebar" onClick={() => change(false)}>
        <span className="material-icons" aria-hidden>
          menu
        </span>
      </button>
    </>
  )
}
