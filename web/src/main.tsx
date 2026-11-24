import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { AuthProvider } from './auth/AuthContext'
import './styles/tokens.css'
import './styles/theme.css'
import './index.css'
import './styles/mobile.css'
import './styles/motion.css'

// Privacy-friendly analytics (Plausible: no cookies, no personal data), e.g. to
// count visits from LinkedIn posts. Off unless VITE_PLAUSIBLE_DOMAIN is set at
// build time (e.g. grade-ant.com); it tracks client-side page changes itself.
const plausibleDomain = import.meta.env.VITE_PLAUSIBLE_DOMAIN
if (plausibleDomain) {
  const s = document.createElement('script')
  s.defer = true
  s.dataset.domain = plausibleDomain
  s.src = 'https://plausible.io/js/script.js'
  document.head.appendChild(s)
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <App />
      </AuthProvider>
    </BrowserRouter>
  </StrictMode>,
)
