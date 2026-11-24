import { useEffect, useRef, useState, type DragEvent, type FormEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { useAlerts } from '../hooks/useAlerts'
import { apiCache } from '../lib/apiCache'
import {
  formatFileSize,
  renameSolutionFile,
  requestPresignedUrls,
  uploadToPresignedUrl,
  UPLOAD_LIMITS,
  type UploadMetadata,
  type UploadType,
} from '../lib/upload'
import { AnimatePresence, motion } from 'motion/react'
import { AppHero, Panel, Stagger } from './app/AppKit'
import { Notice, ProgressBar } from './ui'
import WorkflowPage from './WorkflowPage'

export interface UploadPageCopy {
  title: string
  subtitle: string
  dropLabel: string
  submitLabel: string
  types: { value: UploadType; label: string }[]
  steps: { title: string; text: string }[]
  note: string
}

interface FileProgress {
  name: string
  percent: number
  error?: string
}

const HEADINGS: Record<UploadType, string> = {
  student: 'Upload Assignment',
  teacher: 'Upload Question Paper',
  answer: 'Upload Answer Sheet',
}

const PROGRESS_HIDE_DELAY_MS = 3000

/** Shared by the student and teacher upload pages (was javascript/upload_pdf.js). */
export default function UploadPage({ copy }: { copy: UploadPageCopy }) {
  const navigate = useNavigate()
  const { alerts, show, dismiss, clear } = useAlerts()
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Prefilled when opened from an assignment (e.g. the student inbox's "Start submission").
  const [params] = useSearchParams()
  const [uploadType, setUploadType] = useState<UploadType | ''>(() =>
    params.get('assignment_id') && copy.types.length === 1 ? copy.types[0].value : '',
  )
  const [assignmentId, setAssignmentId] = useState(() => params.get('assignment_id') ?? '')
  const [courseId, setCourseId] = useState(() => params.get('qp_id') ?? '')
  const { session } = useAuth()
  const [studentId, setStudentId] = useState(() => (session?.userType === 'student' ? session.userId : ''))
  const [files, setFiles] = useState<File[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [progress, setProgress] = useState<{ percent: number; text: string } | null>(null)
  const [fileProgress, setFileProgress] = useState<FileProgress[]>([])
  const timers = useRef<number[]>([])

  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  function addFiles(list: FileList | null) {
    if (!list) return
    const accepted: File[] = []
    for (const file of Array.from(list)) {
      if (!UPLOAD_LIMITS.allowedTypes.includes(file.type)) {
        show(`File ${file.name} is not a PDF`, 'danger')
      } else if (file.size > UPLOAD_LIMITS.maxFileSizeMb * 1024 * 1024) {
        show(`File ${file.name} exceeds maximum size of ${UPLOAD_LIMITS.maxFileSizeMb}MB`, 'danger')
      } else {
        accepted.push(file)
      }
    }
    setFiles((prev) => [...prev, ...accepted.filter((f) => !prev.some((p) => p.name === f.name))])
  }

  function onDrag(e: DragEvent, over: boolean) {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(over)
  }

  function resetForm() {
    setUploadType('')
    setAssignmentId('')
    setCourseId('')
    setStudentId(session?.userType === 'student' ? session.userId : '')
    setFiles([])
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  /** Validates the form and builds the presign request + the files to PUT. */
  function prepare(type: UploadType): { metadata: UploadMetadata; uploads: File[] } | null {
    const aId = assignmentId.trim()
    const qp = courseId.trim()
    if ((type === 'student' || type === 'answer') && !aId) {
      show('Please enter Assignment ID', 'danger')
      return null
    }
    if (type === 'answer' && !qp) {
      show('Please enter Question Paper ID', 'danger')
      return null
    }
    if (files.length === 0) {
      show(type === 'answer' ? 'Please select at least one answer sheet file' : 'Please select at least one file', 'danger')
      return null
    }

    const metadata: UploadMetadata = { uploadType: type, files: [] }
    if (aId) metadata.assignmentId = aId
    if (qp) metadata.courseId = qp
    if (type === 'student' && studentId.trim()) metadata.studentId = studentId.trim()

    if (type === 'answer') {
      metadata.qpId = qp
      const uploads = files.map((f) => {
        const renamed = renameSolutionFile(f, aId, qp)
        metadata.files.push({ name: renamed.name, originalName: f.name, type: renamed.type, size: renamed.size })
        return renamed
      })
      return { metadata, uploads }
    }

    metadata.files = files.map((f) => ({ name: f.name, type: f.type, size: f.size }))
    return { metadata, uploads: files }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!uploadType) {
      show('Please select upload type', 'danger')
      return
    }
    const prepared = prepare(uploadType)
    if (!prepared) return
    const { metadata, uploads } = prepared

    clear()
    setSubmitting(true)
    setProgress({ percent: 0, text: '0%' })
    setFileProgress(uploads.map((f) => ({ name: f.name, percent: 0 })))

    let done = 0
    let failed = 0
    const updateTotal = () => {
      const percent = Math.round(((done + failed) / uploads.length) * 100)
      setProgress({ percent, text: `${percent}% (${done}/${uploads.length} files)` })
    }

    try {
      const urls = await requestPresignedUrls(metadata)
      await Promise.all(
        uploads.map(async (file, i) => {
          try {
            await uploadToPresignedUrl(file, urls[i], (percent) =>
              setFileProgress((fp) => fp.map((p, j) => (j === i ? { ...p, percent } : p))),
            )
            done++
          } catch (err) {
            failed++
            setFileProgress((fp) => fp.map((p, j) => (j === i ? { ...p, error: (err as Error).message } : p)))
          }
          updateTotal()
        }),
      )

      if (failed === 0) {
        show(uploadType === 'answer' ? 'All answer sheets uploaded successfully!' : 'All files uploaded successfully!', 'success')
        // New submission: cached assignment lists/grades are stale now.
        apiCache.clearPrefix('student-data:')
        apiCache.clearPrefix('grades:')
        resetForm()
        redirectToCorrection(metadata)
      } else {
        show(`Upload completed with ${failed} failed file(s) out of ${uploads.length}`, 'warning')
      }
    } catch (err) {
      show(`Upload failed: ${(err as Error).message}`, 'danger')
    } finally {
      setSubmitting(false)
      timers.current.push(
        window.setTimeout(() => {
          setProgress(null)
          setFileProgress([])
        }, PROGRESS_HIDE_DELAY_MS),
      )
    }
  }

  /**
   * Sends the uploader to the post-upload correction workspace so they can
   * fix whatever OCR misread; the short delay lets the success alert show.
   */
  function redirectToCorrection(metadata: UploadMetadata) {
    if (!metadata.assignmentId) return
    const params = new URLSearchParams({ assignment_id: metadata.assignmentId })
    const qp = metadata.qpId || metadata.courseId
    if (qp) params.set('qp_id', qp)
    const target =
      metadata.uploadType === 'student' ? '/student/answer-correction' : '/teacher/question-correction'
    timers.current.push(window.setTimeout(() => navigate(`${target}?${params}`), 1500))
  }

  const typeLabel = uploadType ? copy.types.find((t) => t.value === uploadType)?.label : null

  return (
    <WorkflowPage title={copy.title}>
      <AppHero
        eyebrow={uploadType ? HEADINGS[uploadType] : 'Upload'}
        title={copy.title}
        text={copy.subtitle}
        aside={
          <motion.span
            className="app-drop-icon material-icons"
            style={{ width: 96, height: 96, borderRadius: 28, fontSize: 48 }}
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 3.5, repeat: Infinity, ease: 'easeInOut' }}
            aria-hidden
          >
            cloud_upload
          </motion.span>
        }
      />

      <Stagger className="app-upload">
        <Panel title={typeLabel ?? 'Upload details'}>
          <form onSubmit={handleSubmit} noValidate>
            {copy.types.length > 1 && (
              <div className="app-segment" role="radiogroup" aria-label="Upload type">
                {copy.types.map((t) => (
                  <button
                    key={t.value}
                    type="button"
                    role="radio"
                    aria-checked={uploadType === t.value}
                    className={uploadType === t.value ? 'is-active' : ''}
                    onClick={() => setUploadType(t.value)}
                  >
                    {uploadType === t.value && <motion.span layoutId="upload-type" className="app-segment-bg" />}
                    <span>{t.label}</span>
                  </button>
                ))}
              </div>
            )}

            <div className="ga-grid-2">
              <div className="ga-field">
                <label htmlFor="assignment-id">Assignment ID</label>
                <input
                  id="assignment-id"
                  className="ga-input"
                  placeholder="e.g. assignment456"
                  value={assignmentId}
                  onChange={(e) => setAssignmentId(e.target.value)}
                />
              </div>
              <div className="ga-field">
                <label htmlFor="course-id">Question Paper ID</label>
                <input
                  id="course-id"
                  className="ga-input"
                  placeholder="e.g. physics_101"
                  value={courseId}
                  onChange={(e) => setCourseId(e.target.value)}
                />
              </div>
            </div>
            {uploadType === 'student' && (
              <div className="ga-field">
                <label htmlFor="student-id">Student ID</label>
                <input
                  id="student-id"
                  className="ga-input"
                  placeholder="Enter your student ID"
                  value={studentId}
                  onChange={(e) => setStudentId(e.target.value)}
                />
              </div>
            )}

            <label
              className={`app-drop${dragOver ? ' is-over' : ''}`}
              onDragEnter={(e) => onDrag(e, true)}
              onDragOver={(e) => onDrag(e, true)}
              onDragLeave={(e) => onDrag(e, false)}
              onDrop={(e) => {
                onDrag(e, false)
                addFiles(e.dataTransfer.files)
              }}
            >
              <motion.span className="app-drop-icon material-icons" animate={dragOver ? { scale: 1.12, rotate: -6 } : { scale: 1, rotate: 0 }} aria-hidden>
                upload_file
              </motion.span>
              <strong>{copy.dropLabel}</strong>
              <small>or click to browse</small>
              <span className="app-chips">
                <span className="app-chip">PDF only</span>
                <span className="app-chip">up to {UPLOAD_LIMITS.maxFileSizeMb} MB</span>
                <span className="app-chip">multi-page</span>
              </span>
              <input
                ref={fileInputRef}
                type="file"
                id="file-input"
                accept=".pdf"
                multiple
                onChange={(e) => {
                  addFiles(e.target.files)
                  e.target.value = ''
                }}
              />
            </label>

            {files.length > 0 && (
              <div className="app-rows" style={{ marginTop: 16 }}>
                <AnimatePresence initial={false}>
                  {files.map((f) => {
                    return (
                      <motion.div key={f.name} className="app-file" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, x: 30 }}>
                        <span className="app-file-icon material-icons" aria-hidden>
                          picture_as_pdf
                        </span>
                        <span className="app-file-main">
                          <span className="app-file-name">{f.name}</span>
                          <span className="app-file-meta">{formatFileSize(f.size)}</span>
                        </span>
                        <button type="button" className="ga-btn ga-btn-ghost ga-btn-sm" onClick={() => setFiles((fs) => fs.filter((x) => x !== f))} aria-label={`Remove ${f.name}`}>
                          <span className="material-icons">close</span>
                        </button>
                      </motion.div>
                    )
                  })}
                </AnimatePresence>
              </div>
            )}

            {progress && (
              <motion.div style={{ marginTop: 18 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <ProgressBar percent={progress.percent} tone={progress.percent >= 100 ? 'success' : 'brand'} />
                <p className="ga-muted" style={{ margin: '8px 0 0' }}>
                  {progress.text}
                </p>
                {fileProgress.map((fp) => (
                  <div key={fp.name} className="app-file" style={{ marginTop: 10, flexWrap: 'wrap' }}>
                    <span className="app-file-main">
                      <span className="app-file-name">{fp.name}</span>
                      <span className="app-file-meta">{fp.error ? `Failed: ${fp.error}` : `${fp.percent}%`}</span>
                      <ProgressBar percent={fp.percent} tone={fp.error ? 'danger' : fp.percent >= 100 ? 'success' : 'brand'} />
                    </span>
                  </div>
                ))}
              </motion.div>
            )}

            <div style={{ marginTop: 18 }}>
              {alerts.map((a) => (
                <Notice
                  key={a.id}
                  tone={a.type === 'danger' ? 'danger' : a.type === 'warning' ? 'warning' : a.type === 'success' ? 'success' : 'info'}
                  action={
                    <button type="button" className="ga-btn ga-btn-ghost ga-btn-sm" onClick={() => dismiss(a.id)} aria-label="Dismiss">
                      ×
                    </button>
                  }
                >
                  {a.message}
                </Notice>
              ))}
            </div>

            <div className="ga-row" style={{ justifyContent: 'flex-end', marginTop: 8 }}>
              <motion.button type="submit" className="ga-btn ga-btn-primary" disabled={submitting} whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}>
                <span className="material-icons" aria-hidden>
                  {submitting ? 'hourglass_top' : 'upload'}
                </span>
                {submitting ? 'Uploading…' : copy.submitLabel}
              </motion.button>
            </div>
          </form>
        </Panel>

        <Panel title="How it works">
          <ol className="app-steps">
            {copy.steps.map((st) => (
              <li key={st.title}>
                <div>
                  <strong>{st.title}</strong>
                  <span>{st.text}</span>
                </div>
              </li>
            ))}
          </ol>
          <div className="app-tip">
            <span className="material-icons" aria-hidden>
              lightbulb
            </span>
            <span>{copy.note}</span>
          </div>
        </Panel>
      </Stagger>
    </WorkflowPage>
  )
}
