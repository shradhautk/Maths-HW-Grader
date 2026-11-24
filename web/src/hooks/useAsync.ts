import { useCallback, useEffect, useState } from 'react'

/**
 * Runs `load` on mount and whenever `deps` change; `reload(true)` re-runs it
 * bypassing caches (loaders take a `force` flag).
 */
export function useAsync<T>(load: (force: boolean) => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<{ data?: T; error?: Error; loading: boolean }>({ loading: true })
  const [nonce, setNonce] = useState<{ n: number; force: boolean }>({ n: 0, force: false })

  useEffect(() => {
    let live = true
    setState((s) => ({ data: s.data, loading: true }))
    load(nonce.force).then(
      (data) => live && setState({ data, loading: false }),
      (error: Error) => live && setState({ error, loading: false }),
    )
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback((force = true) => setNonce((p) => ({ n: p.n + 1, force })), [])
  return { ...state, reload }
}
