import { useCallback, useEffect, useRef, useState } from 'react'

export type AlertType = 'info' | 'success' | 'warning' | 'danger'

export interface AlertItem {
  id: number
  message: string
  type: AlertType
  fading: boolean
}

/** Stack of auto-dismissing alerts (the old UIManager.showAlert pattern). */
export function useAlerts(timeoutMs = 5000) {
  const [alerts, setAlerts] = useState<AlertItem[]>([])
  const nextId = useRef(0)
  const timers = useRef<number[]>([])

  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  const dismiss = useCallback((id: number) => setAlerts((a) => a.filter((x) => x.id !== id)), [])

  const show = useCallback(
    (message: string, type: AlertType = 'info') => {
      const id = nextId.current++
      setAlerts((a) => [...a, { id, message, type, fading: false }])
      timers.current.push(
        window.setTimeout(() => {
          setAlerts((a) => a.map((x) => (x.id === id ? { ...x, fading: true } : x)))
          timers.current.push(window.setTimeout(() => dismiss(id), 500))
        }, timeoutMs),
      )
    },
    [dismiss, timeoutMs],
  )

  const clear = useCallback(() => setAlerts([]), [])

  return { alerts, show, dismiss, clear }
}
