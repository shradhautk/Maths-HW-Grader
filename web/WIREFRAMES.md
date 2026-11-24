# GradeAnt web — overview against the wireframes

How the React app in `web/` lines up with the wireframes in
`GradeAnt Website Wireframes (standalone).html` (sections 01–20; the newest
version, superseding `workflows/Website Wireframes.html`) and
`GradeAnt Schema Flow Wireframes.dc.html` (auth/identity model).

Status key: **Built** = the screen exists as a route · **Partial** = the route exists but parts of the wireframe are missing · **Not built** = no screen yet.

## 1. Site map (follows the wireframe's Master Flow Map)

```
PUBLIC ───────────────────────────────────────────────────────────────
  /  Home (01) ─┬─ /features  /demo  /about (18)
                ├─ /blog (06)
                ├─ /get-started  /contact
                ├─ /privacy  /terms  /cookies
                └─ /login (07) ──► role-based routing (08)
                     ├─ /forgot-password   request only → admin approves (07)
                     └─ /first-login       invite link ?token= (17)

TEACHER  /teacher/*                         STUDENT  /student/*
  dashboard (02)                              dashboard (03)
    └─ assignments (13)                         └─ assignments (14) inbox
        ├─ new → :id/setup?step=                    ├─ upload → answer-correction (12)
        │    question › review › method             ├─ grades (feedback)
        │    › solution|rubric › verify (13)        └─ :id/recheck ──► teacher (16)
        ├─ :id/grading  queue › spot-check › publish (15)
        ├─ :id/submissions → student-details (19, 16)
        └─ :id/analytics (19)
  assistant (04) · correction (16) · grades

ADMIN  /admin/*  (05)
  dashboard · users (17) · reset-requests (07) · content · analytics
  tickets · audit-logs · settings · profile (08)

SHARED (all roles)
  /<role>/profile  /<role>/change-password (07, 08)
  /logout  /session-expired  /unauthorized  /multiple-login (09)
```

## 2. Folder tree, annotated with wireframe sections

```
web/
├── index.html                 Vite entry
├── vite.config.ts             CSS scoping plugin (.page-* / .theme-*)
├── nginx.conf, Dockerfile     SPA fallback + old .html redirects
├── public/img/                logos, screenshots, team photos (01, 18)
└── src/
    ├── main.tsx               BrowserRouter + AuthProvider
    ├── App.tsx                every route; role areas wrapped in RequireRole (08)
    ├── config.ts              API base URL
    ├── auth/
    │   ├── AuthContext.tsx    session storage, login/logout (07)
    │   └── RequireRole.tsx    no session → /session-expired, wrong role → /unauthorized (09)
    ├── components/
    │   ├── AppShell.tsx       teacher/student layout
    │   ├── Sidebar.tsx        role-based navigation (08)
    │   ├── UploadPage.tsx     shared upload screen (11, 12, 13, 14)
    │   ├── correction/        CorrectionEditor — OCR text edit (11, 12)
    │   ├── grades/            ScoreHero, QuestionViewer, StudentResult (12, 19)
    │   ├── MathPreview.tsx    MathJax rendering
    │   ├── AlertStack, Spinner, PageScope
    ├── hooks/                 useAlerts, useDisplayName, useMathJax, useStaticPageBehaviors
    ├── lib/                   api, apiCache, session, grading, upload, correction, account, password
    ├── pages/
    │   ├── public/            Home (01) · Features, Demo, About (18) · Blog (06)
    │   │                      GetStarted · Contact · Privacy · Terms · Cookies
    │   ├── auth/              Login, ForgotPassword (07) · Signup · AuthNavbar
    │   ├── session/           SessionPages: Logout, SessionExpired, Unauthorized, MultipleLogin (09)
    │   ├── profile/           Profile, ChangePassword (07, 08)
    │   ├── teacher/           TeacherDashboard (02) · TeacherUpload (11, 13)
    │   │                      TeacherQuestionCorrection (11) · TeacherViewGrade, StudentDetails (19)
    │   │                      TeacherCorrection (16)
    │   ├── student/           StudentDashboard (03) · StudentUpload (12, 14)
    │   │                      StudentAnswerCorrection (12) · StudentGrade (12, 14)
    │   ├── admin/             AdminLayout (05 sidebar) · AdminDashboard · AdminUsers (05, 17)
    │   │                      AdminResetRequests (07) · AdminContent · AdminAnalytics
    │   │                      AdminTickets · AdminAuditLogs · AdminSettings (05)
    │   │                      AdminProfile (08) · FirstLoginSetup (17) · adminShared
    │   └── NotFound.tsx
    └── styles/
        ├── tokens.css         design tokens (orange primary, teal success, sky info)
        ├── mobile.css         responsive rules (10)
        ├── sidebar.css
        ├── pages/*.css        scoped per page
        └── themes/*.css       shared themes (site, admin, account, legal, …)
```

## 3. Wireframe section → route → file

| § | Wireframe screen | Route(s) | File(s) | Status |
|---|---|---|---|---|
| 01 | Marketing homepage (nav: Features · Technology · About · Blog) | `/` | `pages/public/Home.tsx` | Built — no Pricing page (no pricing content exists yet) |
| 02 | Teacher dashboard | `/teacher/dashboard` | `pages/teacher/TeacherDashboard.tsx` | Built (cards link to the new assignment flow) |
| 03 | Student dashboard | `/student/dashboard` | `pages/student/StudentDashboard.tsx` | Built (links to My Assignments) |
| 04 | AI Grading Assistant — docked panel + full assistant | `/teacher/assistant`, docked on `/teacher/student-details` | `components/assistant/AssistantPanel.tsx`, `pages/teacher/Assistant.tsx` | Built (workflow API, Claude on Bedrock) |
| 05 | Admin console | `/admin/*` | `pages/admin/*` | Built |
| 06 | Blog | `/blog` | `pages/public/Blog.tsx` | Built |
| 07 | Login · forgot password · change password · reset queue | `/login`, `/forgot-password`, `/<role>/change-password`, `/admin/reset-requests` | `pages/auth/*`, `pages/profile/ChangePassword.tsx`, `AdminResetRequests.tsx` | Built — `/signup` still exists although the wireframe has no self-signup |
| 08 | Profiles + role navigation | `/<role>/profile` | `Profile.tsx`, `AdminProfile.tsx`, `Sidebar.tsx` | Built |
| 09 | Session states | `/session-expired`, `/unauthorized`, `/logout`, `/multiple-login` | `pages/session/SessionPages.tsx` | Built |
| 10 | Mobile key screens + bottom nav | (all student/teacher pages) | `components/BottomNav.tsx`, `styles/mobile.css`, `styles/ui.css` | Built |
| 11 | Teacher grading workflow | `/teacher/assignments/*`, `/teacher/question-correction` | see §13, §15 | Built |
| 12 | Student assignment workflow | `/student/assignments` → `/student/upload` → `/student/answer-correction` → `/student/grades` | `AssignmentInbox.tsx`, `StudentUpload.tsx`, `StudentAnswerCorrection.tsx`, `StudentGrade.tsx` | Built |
| 13 | Create assignment: question → review Qs (OCR confidence, marks Σ) → method → solution / rubric → verify | `/teacher/assignments/new`, `/teacher/assignments/:id/setup?step=` | `pages/teacher/CreateAssignment.tsx`, `AssignmentList.tsx` | Built (workflow API) |
| 14 | Student submission: receive → submit → track → feedback → re-check | `/student/assignments`, `/student/assignments/:id/recheck` | `AssignmentInbox.tsx`, `Recheck.tsx` | Built (workflow API) |
| 15 | AI grading run: queue → processing → results → spot-check → publish | `/teacher/assignments/:id/grading` | `pages/teacher/GradingRun.tsx` | Built (workflow API) |
| 16 | Manual grading & correction | `/teacher/correction`, `/teacher/student-details` | `TeacherCorrection.tsx`, `StudentDetails.tsx` | Built |
| 17 | Admin onboarding | `/admin/users`, `/first-login` | `AdminUsers.tsx`, `FirstLoginSetup.tsx` | Built |
| 18 | Marketing: "See it in action" · Technology · About | `/`, `/features#technology`, `/about` | `Home.tsx`, `Features.tsx`, `About.tsx` | Built — the video card opens `/demo` (no product video file exists yet) |
| 19 | Submissions roster → student detail → cohort analytics | `/teacher/assignments/:id/submissions`, `/teacher/student-details`, `/teacher/assignments/:id/analytics` | `Submissions.tsx`, `StudentDetails.tsx`, `CohortAnalytics.tsx` | Built — analytics uses the existing grades API; roster uses the workflow API |
| 20 | Status model (QUESTION_PAPERS / SUBMISSIONS, fan-in gate) | all workflow screens | `lib/status.ts`, `components/ui` (`StatusPill`) | Built in the UI — statuses must come from the workflow API |

### Summary

Every wireframe section now has a screen. The new ones (04, 13, 14, 15, 19 roster, 20) are backed
by the `/workflow` API: `backend/src/services/workflow/` (Node, which runs locally with `npm run dev`),
ported from `lambda_python_files/workflow.py`. Against an API without `/workflow` they show "The
assignment workflow service isn't available yet". Still open: a Pricing
page, a product video, and removing `/signup`.

## 3a. Sidebar navigation (from the wireframes)

| Role | Wireframe sidebar | Route |
|---|---|---|
| Teacher (§02, §11) | Dashboard | `/teacher/dashboard` |
| | Assignments | `/teacher/assignments` (+ create, setup, grading run, submissions, per-assignment analytics) |
| | Grading Queue | `/teacher/queue` |
| | Students | `/teacher/students` |
| | Analytics | `/teacher/analytics` |
| | Settings | `/teacher/settings` |
| | *Tools:* Grade lookup · Manual correction (§16) · GradeFlow Assistant (§04) | `/teacher/grades` · `/teacher/correction` · `/teacher/assistant` |
| Student (§12) | Dashboard | `/student/dashboard` |
| | Assignments | `/student/assignments` (+ upload, OCR check, re-check) |
| | Grades | `/student/grades` |
| | Progress | `/student/progress` |
| | Announcements | `/student/announcements` |
| Both (§08) | My Profile · Change Password · Request Password Reset · Logout | user menu at the bottom of the sidebar |

Teachers post announcements from the **dashboard** (the wireframes show no teacher-side screen for them).

## 4. Shared design kit

`styles/tokens.css` now carries the full wireframe token set (colour, grade scale, gradients,
type scale, spacing, radius, shadows, focus ring). `components/ui` + `styles/ui.css` (global,
`ga-` prefixed) hold the wireframe components: `StatusPill`, `Pill`, `GradePill`, `StepProgress`,
`StatTile`, `Card`, `PageHeader`, `Notice`, `LoadError`, `EmptyState`, `ProgressBar`,
`GradeDistribution`, `Spinner`, `FileDrop`. `components/WorkflowPage.tsx` is the page shell.

Signed-in pages share the marketing site's look through `components/app/AppKit.tsx` +
`styles/app.css` (gradient hero, animated stat cards, quick-action tiles, panels, score ring) and
the redesigned `Sidebar`. The dashboards and upload pages are built on it; the grades, correction,
student-details and profile pages keep their markup and get the same look from
`styles/app-bridge.css`.

## 5. Workflow API (`backend/src/services/workflow/`, ported from `lambda_python_files/workflow.py`)

`POST {API_BASE_URL}/workflow` (path overridable with `VITE_WORKFLOW_PATH`), bearer token, body
`{ "operation": "...", ...payload }`, like `/admin`. Types are in `src/lib/workflow.ts`.

| Operation | Payload | Returns | Used by |
|---|---|---|---|
| `listAssignments` | — | `{ assignments: Assignment[] }` | §13 list, §04 |
| `getAssignment` | `assignment_id` | `Assignment` (incl. `qp_status`, `method`, `total_marks`, `grades_status`) | §13, §15, §19 |
| `createAssignment` | `title, class_name, due_at?` | `{ assignment_id, qp_id }` | §13 step 1 (then the existing `/uploadpdf` presign) |
| `confirmQuestions` | `assignment_id` | `{ ok }` | §13 step 2 (questions are read/edited via the existing `/correctanswer`; `ocr_confidence` 0–1 per question is optional) |
| `setGradingMethod` | `assignment_id, method: ai\|standard, persona?, partial_credit?` | `{ qp_status }` | §13 step 3 / AI rubric |
| `getRubric` | `assignment_id` | `{ qp_status, rubric: RubricItem[] }` | §13 review + verify |
| `approveRubric` | `assignment_id, rubric` | `{ qp_status: "rubric_reviewed" }` | §13 (teacher gate input) |
| `startGrading` | `assignment_id` | `{ ok }` | §13 verify |
| `getGradingRun` | `assignment_id` | `{ total, submissions: Submission[] }` | §15 |
| `listFlagged` / `acceptFlagged` | `assignment_id` / `evaluation_id, question_number, subpart?` | `{ items: FlaggedItem[] }` / `{ ok }` | §15 spot-check |
| `publishGrades` | `assignment_id` | `{ notified }` | §15 |
| `listSubmissions` / `remindStudent` | `assignment_id` (+ `student_id`) | `{ submissions, enrolled }` / `{ ok }` | §19 |
| `listStudentAssignments` | — | `{ assignments: StudentAssignment[] }` | §14 |
| `requestRecheck` | `assignment_id, evaluation_id, items: [{question_number, subpart?, note}]` | `{ ok }` | §14 step 5 |
| `askAssistant` | `assignment_id?, student_id?, messages` | `{ reply }` | §04 |
| `getGradingQueue` | — | `{ queue: [{assignment, state, papers, graded, flagged…}], waiting }` | Grading Queue |
| `listStudents` | — | `{ students: [{name, classes, submitted, average, at_risk, latest…}], classes }` | Students |
| `getTeacherAnalytics` | `class_name?` | totals, grade distribution, weekly activity, per-assignment rows, hardest questions, insight, deadlines | Analytics |
| `getTeacherSettings` / `updateTeacherSettings` | — / `settings` | `{ settings: {default_persona, partial_credit, notify_submissions, notify_flagged} }` | Settings (defaults feed `setGradingMethod`) |
| `getProgress` | — | published results, average, by-class averages, due soon, streak, insight | Student Progress |
| `postAnnouncement` / `listAnnouncements` / `deleteAnnouncement` | `class_name, title, message` / — / `announcement_id` | `{ announcement }` / `{ announcements }` / `{ ok }` | Announcements |
