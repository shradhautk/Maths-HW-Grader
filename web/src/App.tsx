import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import RequireRole from './auth/RequireRole'
import { PAGES } from './config'
import NotFound from './pages/NotFound'
import { Logout, MultipleLogin, SessionExpired, Unauthorized } from './pages/session/SessionPages'

// Each page is its own chunk, so a student never downloads the admin console.
const Login = lazy(() => import('./pages/auth/Login'))
const Signup = lazy(() => import('./pages/auth/Signup'))
const ForgotPassword = lazy(() => import('./pages/auth/ForgotPassword'))

const StudentDashboard = lazy(() => import('./pages/student/StudentDashboard'))
const StudentUpload = lazy(() => import('./pages/student/StudentUpload'))
const StudentGrade = lazy(() => import('./pages/student/StudentGrade'))
const StudentAnswerCorrection = lazy(() => import('./pages/student/StudentAnswerCorrection'))

const TeacherDashboard = lazy(() => import('./pages/teacher/TeacherDashboard'))
const TeacherUpload = lazy(() => import('./pages/teacher/TeacherUpload'))
const TeacherViewGrade = lazy(() => import('./pages/teacher/TeacherViewGrade'))
const StudentDetails = lazy(() => import('./pages/teacher/StudentDetails'))
const TeacherCorrection = lazy(() => import('./pages/teacher/TeacherCorrection'))
const TeacherQuestionCorrection = lazy(() => import('./pages/teacher/TeacherQuestionCorrection'))

const AssignmentList = lazy(() => import('./pages/teacher/AssignmentList'))
const CreateAssignment = lazy(() => import('./pages/teacher/CreateAssignment'))
const GradingRun = lazy(() => import('./pages/teacher/GradingRun'))
const Submissions = lazy(() => import('./pages/teacher/Submissions'))
const CohortAnalytics = lazy(() => import('./pages/teacher/CohortAnalytics'))
const Assistant = lazy(() => import('./pages/teacher/Assistant'))
const GradingQueue = lazy(() => import('./pages/teacher/GradingQueue'))
const Students = lazy(() => import('./pages/teacher/Students'))
const Analytics = lazy(() => import('./pages/teacher/Analytics'))
const TeacherSettings = lazy(() => import('./pages/teacher/Settings'))
const Progress = lazy(() => import('./pages/student/Progress'))
const Announcements = lazy(() => import('./pages/student/Announcements'))
const AssignmentInbox = lazy(() => import('./pages/student/AssignmentInbox'))
const Recheck = lazy(() => import('./pages/student/Recheck'))

const Profile = lazy(() => import('./pages/profile/Profile'))
const ChangePassword = lazy(() => import('./pages/profile/ChangePassword'))

const AdminDashboard = lazy(() => import('./pages/admin/AdminDashboard'))
const AdminUsers = lazy(() => import('./pages/admin/AdminUsers'))
const AdminResetRequests = lazy(() => import('./pages/admin/AdminResetRequests'))
const AdminContent = lazy(() => import('./pages/admin/AdminContent'))
const AdminAnalytics = lazy(() => import('./pages/admin/AdminAnalytics'))
const AdminTickets = lazy(() => import('./pages/admin/AdminTickets'))
const AdminAuditLogs = lazy(() => import('./pages/admin/AdminAuditLogs'))
const AdminSettings = lazy(() => import('./pages/admin/AdminSettings'))
const AdminProfile = lazy(() => import('./pages/admin/AdminProfile'))
const FirstLoginSetup = lazy(() => import('./pages/admin/FirstLoginSetup'))

const Home = lazy(() => import('./pages/public/Home'))
const Features = lazy(() => import('./pages/public/Features'))
const Demo = lazy(() => import('./pages/public/Demo'))
const About = lazy(() => import('./pages/public/About'))
const Research = lazy(() => import('./pages/public/Research'))
const Careers = lazy(() => import('./pages/public/Careers'))
const Security = lazy(() => import('./pages/public/Security'))
const RequestPilot = lazy(() => import('./pages/public/RequestPilot'))
const Contact = lazy(() => import('./pages/public/Contact'))
const Privacy = lazy(() => import('./pages/public/Privacy'))
const Terms = lazy(() => import('./pages/public/Terms'))
const Cookies = lazy(() => import('./pages/public/Cookies'))

export default function App() {
  return (
    <Suspense fallback={null}>
      <Routes>
        {/* Public site */}
        <Route path="/" element={<Home />} />
        <Route path="/features" element={<Features />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/about" element={<About />} />
        <Route path="/research" element={<Research />} />
        <Route path="/careers" element={<Careers />} />
        <Route path="/security" element={<Security />} />
        <Route path="/request-a-pilot" element={<RequestPilot />} />
        {/* Retired pages: sign-up is pilots-only for now, and the blog had no real posts. */}
        <Route path="/get-started" element={<Navigate to="/request-a-pilot" replace />} />
        <Route path="/blog" element={<Navigate to="/research" replace />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/cookies" element={<Cookies />} />

        {/* Auth + session states */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/logout" element={<Logout />} />
        <Route path="/session-expired" element={<SessionExpired />} />
        <Route path="/unauthorized" element={<Unauthorized />} />
        <Route path="/multiple-login" element={<MultipleLogin />} />
        {/* Old static-site URLs (bookmarks, old links) → their React pages. */}
        <Route path="/auth/signup.html" element={<Navigate to="/signup" replace />} />
        <Route path="/auth/login.html" element={<Navigate to="/login" replace />} />

        {/* Student */}
        <Route path="/student" element={<RequireRole roles={['student']} />}>
          <Route index element={<Navigate to="dashboard" replace />} />
          <Route path="dashboard" element={<StudentDashboard />} />
          <Route path="assignments" element={<AssignmentInbox />} />
          <Route path="progress" element={<Progress />} />
          <Route path="announcements" element={<Announcements />} />
          <Route path="assignments/:id/recheck" element={<Recheck />} />
          <Route path="upload" element={<StudentUpload />} />
          <Route path="grades" element={<StudentGrade />} />
          <Route path="answer-correction" element={<StudentAnswerCorrection />} />
          <Route path="profile" element={<Profile role="student" />} />
          <Route path="change-password" element={<ChangePassword role="student" />} />
        </Route>

        {/* Teacher */}
        <Route path="/teacher" element={<RequireRole roles={['teacher']} />}>
          <Route index element={<Navigate to="dashboard" replace />} />
          <Route path="dashboard" element={<TeacherDashboard />} />
          <Route path="assignments" element={<AssignmentList />} />
          <Route path="assignments/new" element={<CreateAssignment />} />
          <Route path="assignments/:id/setup" element={<CreateAssignment />} />
          <Route path="assignments/:id/grading" element={<GradingRun />} />
          <Route path="assignments/:id/submissions" element={<Submissions />} />
          <Route path="assignments/:id/analytics" element={<CohortAnalytics />} />
          <Route path="assistant" element={<Assistant />} />
          <Route path="queue" element={<GradingQueue />} />
          <Route path="students" element={<Students />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="settings" element={<TeacherSettings />} />
          <Route path="upload" element={<TeacherUpload />} />
          <Route path="grades" element={<TeacherViewGrade />} />
          <Route path="student-details" element={<StudentDetails />} />
          <Route path="correction" element={<TeacherCorrection />} />
          <Route path="question-correction" element={<TeacherQuestionCorrection />} />
          <Route path="profile" element={<Profile role="teacher" />} />
          <Route path="change-password" element={<ChangePassword role="teacher" />} />
        </Route>

        {/* Admin console */}
        <Route path="/admin" element={<RequireRole roles={['admin']} />}>
          <Route index element={<Navigate to="dashboard" replace />} />
          <Route path="dashboard" element={<AdminDashboard />} />
          <Route path="users" element={<AdminUsers />} />
          <Route path="reset-requests" element={<AdminResetRequests />} />
          <Route path="content" element={<AdminContent />} />
          <Route path="analytics" element={<AdminAnalytics />} />
          <Route path="tickets" element={<AdminTickets />} />
          <Route path="audit-logs" element={<AdminAuditLogs />} />
          <Route path="settings" element={<AdminSettings />} />
          <Route path="profile" element={<AdminProfile />} />
        </Route>
        {/* Reached from an unauthenticated invite link (?token=). */}
        <Route path={PAGES.firstLogin} element={<FirstLoginSetup />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  )
}
