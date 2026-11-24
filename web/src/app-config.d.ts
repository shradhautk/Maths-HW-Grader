// Shape of __APP_CONFIG__, injected at build time from config/app.yaml (see vite.config.ts).
interface AppConfigRole {
  label: string
  signup: ('self' | 'invite')[]
}

interface AppConfig {
  urls: { api: string; upload: string }
  api: {
    health: string
    healthReady: string
    auth: {
      base: string
      signup: string
      login: string
      logout: string
      forgotPassword: string
      changePassword: string
      me: string
      profilePhoto: string
    }
    upload: string
    grades: string
    corrections: string
    marks: string
    workflow: string
    admin: string
    contact: string
  }
  pages: { firstLogin: string }
  roles: Record<'student' | 'teacher' | 'admin', AppConfigRole>
}

declare const __APP_CONFIG__: AppConfig
