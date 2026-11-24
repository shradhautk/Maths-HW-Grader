// URLs, API paths, pages and roles, all from config/app.yaml at the repo root
// (compiled in at build time, see vite.config.ts). Edit that file, not this one.
//
// One-off overrides per build: VITE_API_BASE_URL / VITE_UPLOAD_BASE_URL
// (e.g. docker build --build-arg VITE_API_BASE_URL=...). Only public URLs
// belong in either place: they end up in the browser bundle.
const app = __APP_CONFIG__

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || app.urls.api

// Where /uploadpdf lives. Unset: the same API (the Node backend, which requires
// the user's token). Set it to keep using the legacy upload Lambda, which is
// called without a token (its CORS setup doesn't allow the Authorization header).
const uploadOverride: string = import.meta.env.VITE_UPLOAD_BASE_URL || app.urls.upload
export const UPLOAD_BASE_URL: string = uploadOverride || API_BASE_URL
export const UPLOAD_SENDS_TOKEN = !uploadOverride

/** API paths (app.yaml api section). */
const auth = app.api.auth
export const API = {
  ...app.api,
  auth: {
    signup: auth.base + auth.signup,
    login: auth.base + auth.login,
    logout: auth.base + auth.logout,
    forgotPassword: auth.base + auth.forgotPassword,
    changePassword: auth.base + auth.changePassword,
    me: auth.base + auth.me,
    profilePhoto: auth.base + auth.profilePhoto,
  },
}

// Assignment workflow resource (wireframes §13–§15, §19, §20).
export const WORKFLOW_PATH: string = import.meta.env.VITE_WORKFLOW_PATH || app.api.workflow

/** Web app pages the backend links to (app.yaml pages section). */
export const PAGES = app.pages

export type RoleName = keyof typeof app.roles

/** Roles in app.yaml order, with their label and how they can join. */
export const ROLES = (Object.keys(app.roles) as RoleName[]).map((name) => ({ name, ...app.roles[name] }))

/** Roles that can sign themselves up on the public signup page. */
export const SELF_SIGNUP_ROLES = ROLES.filter((r) => r.signup.includes('self'))

export const roleLabel = (name: RoleName): string => app.roles[name]?.label ?? name
