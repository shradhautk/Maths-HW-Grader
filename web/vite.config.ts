import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import react from '@vitejs/plugin-react'
import prefixer from 'postcss-prefix-selector'
import { defineConfig } from 'vite'
import { parse } from 'yaml'

// config/app.yaml (the main config, at the repo root) → __APP_CONFIG__ in the
// bundle. Only the public parts are compiled in: this mode's URLs, the API
// paths, web pages, and each role's label and signup options. Restart
// `npm run dev` after editing the file.
function loadAppConfig(mode: string) {
  const file = path.resolve(process.env.APP_CONFIG_FILE || path.join(__dirname, '..', 'config', 'app.yaml'))
  if (!existsSync(file)) throw new Error(`config/app.yaml not found at ${file} (set APP_CONFIG_FILE to its path)`)
  const doc = parse(readFileSync(file, 'utf8'))
  // Production builds use the selected `deployment:` target's url overrides (e.g. hybrid:
  // the API on Render), merged over urls.production, the same rule as the backend.
  const target = mode === 'production' ? doc?.deployments?.[doc?.deployment ?? 'aws'] : undefined
  const urls = doc?.urls?.[mode] && { ...doc.urls[mode], ...(target?.urls ?? {}) }
  if (!urls?.api) throw new Error(`config/app.yaml: urls.${mode}.api is missing`)
  if (!doc.api || !doc.pages || !doc.roles) throw new Error('config/app.yaml: the api, pages and roles sections are required')
  const roles = Object.fromEntries(
    Object.entries(doc.roles as Record<string, { label: string; signup: string[] }>).map(([name, r]) => [
      name,
      { label: r.label, signup: r.signup },
    ]),
  )
  return { urls: { api: urls.api, upload: urls.upload || '' }, api: doc.api, pages: doc.pages, roles }
}

// The page stylesheets in src/styles/pages/ are the original per-page CSS
// files from the static site, copied verbatim. In a single-page app every
// imported stylesheet is global, so their generic selectors (.btn, .navbar,
// body, :root…) would collide. This scopes each file to a wrapper class
// named after it: src/styles/pages/auth.css -> `.page-auth …`, rendered by
// <PageScope name="auth">. Shared stylesheets several pages used (e.g.
// css/asthetic_style.css) live in src/styles/themes/ and scope to
// `.theme-<name>`, enabled with <PageScope theme="aesthetic">.
const SCOPED_CSS = /[\\/]src[\\/]styles[\\/](pages|themes)[\\/]([\w-]+)\.css/

function scopePageSelector(_prefix: string, selector: string, _prefixed: string, filePath?: string) {
  const match = filePath && SCOPED_CSS.exec(filePath)
  if (!match) return selector
  const scope = `.${match[1] === 'pages' ? 'page' : 'theme'}-${match[2]}`

  // Document-level selectors become the wrapper itself.
  const rootMatch = /^(html\s+body|html|body|:root)(?=$|[\s.:#[>+~])/.exec(selector)
  if (rootMatch) return scope + selector.slice(rootMatch[0].length)
  return `${scope} ${selector}`
}

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  define: {
    __APP_CONFIG__: JSON.stringify(loadAppConfig(mode)),
  },
  css: {
    postcss: {
      plugins: [prefixer({ prefix: '.page', transform: scopePageSelector })],
    },
  },
}))
