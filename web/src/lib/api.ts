import { API, API_BASE_URL } from '../config'
import { apiCache } from './apiCache'
import { clearSession, getToken } from './session'

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

/**
 * Lambda proxy responses sometimes arrive wrapped as
 * { statusCode, headers, body: "<json string>" }; unwrap to the payload.
 */
export function unwrapLambda<T>(data: unknown): T {
  if (data && typeof data === 'object' && 'body' in data) {
    const body = (data as { body: unknown }).body
    if (typeof body === 'string') {
      try {
        return JSON.parse(body) as T
      } catch {
        return body as T
      }
    }
    if (body && typeof body === 'object') return body as T
  }
  return data as T
}

let expiring = false

function handleExpiredSession() {
  if (expiring) return
  expiring = true
  clearSession()
  const here = window.location.pathname + window.location.search
  window.location.assign(`/session-expired?returnTo=${encodeURIComponent(here)}`)
}

/**
 * True when the API route looks undeployed rather than failing inside a Lambda:
 * API Gateway's "Missing Authentication Token" / "Invalid key=value pair" 403s,
 * which carry no CORS headers, so the browser usually reports a network error.
 */
export function isRouteMissing(error: unknown): boolean {
  return (
    error instanceof Error &&
    (/Missing Authentication Token|Invalid key=value pair/i.test(error.message) ||
      (error instanceof TypeError && /fetch|network|load failed/i.test(error.message)))
  )
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
  body?: unknown
  /** Attach the stored bearer token (default true). */
  auth?: boolean
  headers?: Record<string, string>
  /** Return the response JSON as-is instead of unwrapping a Lambda envelope. */
  raw?: boolean
}

/** fetch() against API Gateway: JSON in, unwrapped JSON out, throws ApiError on non-2xx. */
export async function apiRequest<T = unknown>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...opts.headers }
  if (opts.auth !== false) {
    const token = getToken()
    if (token) headers.Authorization = `Bearer ${token}`
  }

  const url = /^https?:\/\//.test(path) ? path : `${API_BASE_URL}${path}`
  const response = await fetch(url, {
    method: opts.method ?? (opts.body === undefined ? 'GET' : 'POST'),
    headers,
    body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
  })

  let data: unknown = null
  try {
    data = await response.json()
  } catch {
    /* empty or non-JSON body */
  }

  if (!response.ok) {
    const payload = unwrapLambda<{ message?: string; error?: string } | null>(data)
    // Every 401 on an authenticated call means the token is expired or invalid
    // (login's own 401 is sent without one), so end the session cleanly.
    if (response.status === 401 && headers.Authorization) handleExpiredSession()
    throw new ApiError(payload?.message || payload?.error || `Request failed (${response.status})`, response.status)
  }
  return opts.raw ? (data as T) : unwrapLambda<T>(data)
}

// ---- Admin Lambda: one "/admin" resource dispatching on `operation` ----

const ADMIN_READ_OPERATIONS = new Set([
  'listUsers',
  'listResetRequests',
  'getPermissionMatrix',
  'listContent',
  'getAnalytics',
  'listTickets',
  'listAuditLogs',
  'getSettings',
])

/** Reads are cached per exact query; any write invalidates every cached admin read. */
export async function adminApiRequest<T = unknown>(
  operation: string,
  payload: Record<string, unknown> = {},
  opts: { forceRefresh?: boolean } = {},
): Promise<T> {
  const isRead = ADMIN_READ_OPERATIONS.has(operation)
  const cacheKey = `admin:${operation}:${JSON.stringify(payload)}`

  if (isRead && !opts.forceRefresh) {
    const cached = apiCache.get<T>(cacheKey)
    if (cached !== undefined) return cached
  }

  const data = await apiRequest<T>(API.admin, { body: { operation, ...payload } })
  if (isRead) apiCache.set(cacheKey, data)
  else apiCache.clearPrefix('admin:')
  return data
}
