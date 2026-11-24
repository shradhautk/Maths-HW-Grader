import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import { clearSession, loadSession, patchUserData, saveSession, type Session } from '../lib/session'

interface AuthValue {
  session: Session | null
  login: (s: Omit<Session, 'loginTime'>, remember: boolean) => Session
  logout: () => void
  /** Merge fields (e.g. photoUrl, must_change_password) into the stored login data. */
  updateUserData: (patch: Record<string, unknown>) => void
}

const AuthContext = createContext<AuthValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(loadSession)

  const login = useCallback((s: Omit<Session, 'loginTime'>, remember: boolean) => {
    const saved = saveSession(s, remember)
    setSession(saved)
    return saved
  }, [])

  const logout = useCallback(() => {
    clearSession()
    setSession(null)
  }, [])

  const updateUserData = useCallback((patch: Record<string, unknown>) => {
    const next = patchUserData(patch)
    if (next) setSession(next)
  }, [])

  const value = useMemo(() => ({ session, login, logout, updateUserData }), [session, login, logout, updateUserData])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
