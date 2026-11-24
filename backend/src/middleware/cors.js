// CORS for the React app. CORS_ALLOWED_ORIGINS is "*" or an allowlist; the
// API uses bearer tokens (no cookies), so credentials are never allowed.
import config from '../config/index.js'

export function cors() {
  const { origins } = config.cors
  const anyOrigin = origins.includes('*') || (origins.length === 0 && !config.isProduction)
  return (req, res, next) => {
    const origin = req.headers.origin
    if (anyOrigin) {
      res.set('Access-Control-Allow-Origin', '*')
    } else {
      res.vary('Origin')
      if (origin && origins.includes(origin)) res.set('Access-Control-Allow-Origin', origin)
    }
    res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    res.set('Access-Control-Allow-Methods', 'GET, POST, PUT, OPTIONS')
    res.set('Access-Control-Max-Age', '600')
    if (req.method === 'OPTIONS') return res.status(204).end()
    next()
  }
}
