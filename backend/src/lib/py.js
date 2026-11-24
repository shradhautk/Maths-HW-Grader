// Small helpers that reproduce the Python Lambdas' semantics where JavaScript
// differs (truthiness, float(), round(), str[:n], tuple sorting, isoformat),
// so the API answers exactly as before.

/** Python truthiness: [], {}, '', 0, None and False are all falsy. */
export function truthy(v) {
  if (Array.isArray(v)) return v.length > 0
  if (v && typeof v === 'object') return Object.keys(v).length > 0
  return Boolean(v)
}

/** `a or b` with Python truthiness. */
export const or = (...values) => {
  for (const v of values) if (truthy(v)) return v
  return values[values.length - 1]
}

/** workflow.py's _num(): float(value), or `fallback` when Python's float() would raise. */
export function num(value, fallback = 0) {
  if (typeof value === 'number') return value
  if (typeof value === 'boolean') return value ? 1 : 0
  if (typeof value === 'string') {
    const s = value.trim().replace(/_/g, '')
    const special = /^([+-]?)(inf|infinity|nan)$/i.exec(s)
    if (special) return special[2].toLowerCase() === 'nan' ? NaN : special[1] === '-' ? -Infinity : Infinity
    return /^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(s) ? Number(s) : fallback
  }
  return fallback
}

/** Python's round(): half to even. */
export function pyRound(x) {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/** str(value)[:n] on code points, like Python slicing. */
export function trunc(value, n) {
  const s = String(value)
  return s.length <= n ? s : Array.from(s).slice(0, n).join('')
}

/** Python str() for the scalar values that come out of DynamoDB / JSON. */
export function pyStr(v) {
  if (v === null || v === undefined) return 'None'
  if (v === true) return 'True'
  if (v === false) return 'False'
  return String(v)
}

/** Python's repr() for the values used in error messages. */
export function pyRepr(v) {
  if (typeof v === 'string') return `'${v.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
  return pyStr(v)
}

/** Compare like Python: numbers numerically, strings by code point, tuples element-wise. */
export function cmp(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const c = cmp(a[i], b[i])
      if (c) return c
    }
    return a.length - b.length
  }
  if (typeof a === 'number' && typeof b === 'number') return a < b ? -1 : a > b ? 1 : 0
  if (typeof a === 'boolean' || typeof b === 'boolean') return cmp(Number(a), Number(b))
  // Mixed types (Python would raise): numbers sort before strings.
  if (typeof a !== typeof b) return typeof a === 'number' ? -1 : 1
  const sa = String(a)
  const sb = String(b)
  return sa < sb ? -1 : sa > sb ? 1 : 0
}

/** sorted(items, key=fn, reverse=...) — stable, Python ordering. */
export function sortBy(items, key, { reverse = false } = {}) {
  const decorated = items.map((item, i) => ({ item, k: key(item), i }))
  decorated.sort((x, y) => (reverse ? cmp(y.k, x.k) : cmp(x.k, y.k)) || x.i - y.i)
  return decorated.map((d) => d.item)
}

/** min(values, default=fallback) with Python ordering. */
export function pyMin(values, fallback) {
  let best
  let found = false
  for (const v of values) {
    if (!found || cmp(v, best) < 0) {
      best = v
      found = true
    }
  }
  return found ? best : fallback
}

export const mean = (values) => {
  const list = Array.from(values)
  return list.length ? list.reduce((a, b) => a + b, 0) / list.length : null
}

export const sum = (values) => Array.from(values).reduce((a, b) => a + b, 0)

/** Python's datetime.isoformat() for a UTC instant: 2026-10-19T09:00:00.123000+00:00 */
export function isoUtc(date = new Date()) {
  const iso = date.toISOString() // 2026-10-19T09:00:00.123Z
  const ms = date.getUTCMilliseconds()
  return ms ? `${iso.slice(0, 19)}.${String(ms).padStart(3, '0')}000+00:00` : `${iso.slice(0, 19)}+00:00`
}

/** Python's datetime.now().isoformat() as the old Lambdas wrote it (naive, UTC on Lambda). */
export function isoNaive(date = new Date()) {
  return isoUtc(date).replace('+00:00', '')
}

/** datetime.date.fromisoformat(s[:10]) — returns 'YYYY-MM-DD' or null when invalid. */
export function isoDate(value) {
  const s = pyStr(value).slice(0, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return null
  const d = new Date(`${s}T00:00:00Z`)
  return Number.isNaN(d.getTime()) || d.toISOString().slice(0, 10) !== s ? null : s
}

export const daysBetween = (a, b) => Math.round((Date.parse(`${a}T00:00:00Z`) - Date.parse(`${b}T00:00:00Z`)) / 86400000)
