# GradeAnt — React app

React 19 + TypeScript + Vite + React Router. It replaces the static HTML pages in the repo root. The API is the Node backend in [`../backend`](../backend); the original Python Lambdas in `lambda_python_files/` serve production until it's deployed.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle in dist/
npm run lint
```

API host: `src/config.ts`, set per mode by Vite env files (documented in `.env.example`):

- `.env.development` (`npm run dev`) points at the local Node backend, `http://localhost:8787`. Start it with `cd ../backend && npm run dev`; the demo logins are in the [root README](../README.md).
- `.env.production` (`npm run build`) points at the deployed API.
- `VITE_UPLOAD_BASE_URL` (optional) sends `/uploadpdf` to the legacy upload Lambda without a token.

Override any of these in `.env.<mode>.local`. `VITE_*` values are compiled into the bundle, so only public URLs belong there.

See [`WIREFRAMES.md`](WIREFRAMES.md) for the site map and how each wireframe screen maps to a route and file.

## Layout

| Path | What |
|---|---|
| `src/App.tsx` | All routes. Role areas are wrapped in `RequireRole` (no session → `/login?returnTo=…`, wrong role → `/unauthorized`). An API 401 ends the session and goes to `/session-expired`. |
| `src/auth/` | `AuthContext` (session in local/sessionStorage, same keys as the old pages) and `RequireRole`. |
| `src/lib/` | `api.ts` (fetch wrapper, Lambda-envelope unwrapping, admin operations), `apiCache.ts`, `session.ts`, `grading.ts`, `upload.ts`, `correction.ts`. |
| `src/components/` | `AppShell` + `Sidebar` (student/teacher), `UploadPage`, `grades/` (score hero, question viewer), `correction/` (OCR correction editor), `PageScope`. |
| `src/pages/` | `public/`, `auth/`, `session/`, `student/`, `teacher/`, `profile/`, `admin/`. |
| `src/styles/` | The original stylesheets, copied as-is (see below). |

## Styling

The original per-page CSS was kept verbatim so pages look the same. In a single-page app all CSS is global, so `vite.config.ts` scopes each file:

- `src/styles/pages/<name>.css` → every selector prefixed with `.page-<name>`
- `src/styles/themes/<name>.css` (shared files like the old `asthetic_style.css`) → `.theme-<name>`
- `body`/`html`/`:root` map to the wrapper itself.

A page opts in with `<PageScope name="<name>" theme={[...]}>`, which renders that wrapper. Import order matters: themes first, then the page stylesheet.

## Routes (old file → new path)

| Old | New |
|---|---|
| `index.html`, `quick_links/*`, `resoures/*`, `legal/*` | `/`, `/features`, `/demo`, `/get-started`, `/about`, `/blog`, `/contact`, `/privacy`, `/terms`, `/cookies` |
| `auth/login.html`, `auth/signup.html` | `/login`, `/signup` |
| `session/*.html` | `/logout`, `/session-expired`, `/unauthorized`, `/multiple-login` |
| `dashboards/student_dashboard_.html`, `student/*` | `/student/dashboard`, `/student/upload`, `/student/grades`, `/student/answer-correction` |
| `dashboards/teacher_dashboard.html`, `teacher/*` | `/teacher/dashboard`, `/teacher/upload`, `/teacher/grades`, `/teacher/student-details`, `/teacher/correction`, `/teacher/question-correction` |
| `userProfiles/*` | `/student/profile`, `/teacher/profile` |
| `admin/*` | `/admin/dashboard`, `/admin/users`, `/admin/reset-requests`, `/admin/content`, `/admin/analytics`, `/admin/tickets`, `/admin/audit-logs`, `/admin/settings`, `/admin/profile`, `/first-login` |

## Deploy

`Dockerfile` builds the app and serves `dist/` with NGINX; `nginx.conf` adds the SPA fallback (so refreshing `/teacher/grades` works) and redirects the main old `.html` URLs.
