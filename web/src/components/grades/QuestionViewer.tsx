import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useMathJax } from '../../hooks/useMathJax'
import {
  formatTimestamp,
  getMarksStatus,
  renderContent,
  sortDetails,
  STATUS_BADGE,
  type MarksStatus,
  type QuestionDetail,
} from '../../lib/grading'

interface Props {
  details: QuestionDetail[]
  /** Custom card body (Student Details' editable card); defaults to the read-only card. */
  renderCard?: (q: QuestionDetail, index: number, status: MarksStatus) => ReactNode
  /** Open on this (sorted) index instead of the collapsed "pick a question" state. */
  initialIndex?: number
}

/**
 * One-question-at-a-time viewer: a pill "question path" (‹ Q1 Q2 … ›) above a
 * horizontal scroll-snapped row of question cards. The row stays collapsed
 * until a pill is picked. Shared by My Grades, Students Grades and Student
 * Details (was duplicated across three scripts).
 */
export default function QuestionViewer({ details, renderCard, initialIndex }: Props) {
  const questions = useMemo(() => sortDetails(details.filter((q) => q && typeof q === 'object')), [details])
  // Identity of the question *set* (not their marks), so editing a mark
  // doesn't reset the selection.
  const setKey = questions.map((q) => `${q.question_number}:${q.subpart ?? ''}`).join('|')
  const [current, setCurrent] = useState(initialIndex ?? -1)
  const containerRef = useRef<HTMLDivElement>(null)
  const cardRefs = useRef<(HTMLDivElement | null)[]>([])
  const [containerHeight, setContainerHeight] = useState<number | undefined>()

  // Set when the user picks a question (pill, ‹ ›, deep link) — not when
  // the row is swiped — so only those selections scroll the row.
  const scrollPending = useRef(initialIndex !== undefined)

  useEffect(() => {
    setCurrent(initialIndex ?? -1)
    scrollPending.current = initialIndex !== undefined
  }, [setKey, initialIndex])

  useMathJax(containerRef, [setKey])

  function goTo(index: number) {
    if (questions.length === 0) return
    scrollPending.current = true
    setCurrent(Math.max(0, Math.min(index, questions.length - 1)))
  }

  // Runs after the row has un-collapsed, so the card has its real position.
  useEffect(() => {
    if (!scrollPending.current || current < 0) return
    scrollPending.current = false
    const container = containerRef.current
    const card = cardRefs.current[current]
    if (container && card) {
      const left = card.getBoundingClientRect().left - container.getBoundingClientRect().left + container.scrollLeft
      container.scrollTo({ left, behavior: 'smooth' })
    }
  }, [current])

  // Keep the pills in sync when the user swipes the row directly.
  const isOpen = current >= 0
  useEffect(() => {
    const root = containerRef.current
    if (!root || !isOpen || !('IntersectionObserver' in window)) return
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio >= 0.6) {
            const idx = Number((entry.target as HTMLElement).dataset.qIndex)
            if (!Number.isNaN(idx)) setCurrent(idx)
          }
        })
      },
      { root, threshold: 0.6 },
    )
    cardRefs.current.forEach((el) => el && observer.observe(el))
    return () => observer.disconnect()
  }, [setKey, isOpen])

  // The row is as tall as its tallest card; pin it to the active card (and
  // follow it as panels inside open/close) so there's no dead space below.
  useLayoutEffect(() => {
    const card = current >= 0 ? cardRefs.current[current] : null
    if (!card) {
      setContainerHeight(undefined)
      return
    }
    const sync = () => setContainerHeight(card.scrollHeight)
    sync()
    const ro = new ResizeObserver(sync)
    ro.observe(card)
    return () => ro.disconnect()
  }, [current, setKey])

  return (
    <>
      <div className="q-path-nav">
        <button
          type="button"
          className="q-path-btn q-path-prev"
          aria-label="Previous question"
          disabled={!isOpen || current <= 0}
          onClick={() => goTo(current - 1)}
        >
          ‹
        </button>
        <div className="q-path-track">
          {questions.map((q, i) => (
            <button
              key={i}
              type="button"
              className={`q-path-pill status-${getMarksStatus(q.marks, q.max_marks)}${i === current ? ' active' : ''}`}
              onClick={() => goTo(i)}
            >
              Q{String(q.question_number ?? i + 1)}
              {q.subpart ?? ''}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="q-path-btn q-path-next"
          aria-label="Next question"
          disabled={!isOpen || current >= questions.length - 1}
          onClick={() => goTo(current + 1)}
        >
          ›
        </button>
      </div>
      <div className="q-path-hint" style={{ display: isOpen ? 'none' : undefined }}>
        Select a question above to view its details.
      </div>
      <div className="q-count">
        {questions.length > 0 && isOpen ? `Question ${current + 1} of ${questions.length}` : ''}
      </div>

      <div
        id="questions-container"
        ref={containerRef}
        className={isOpen ? undefined : 'is-collapsed'}
        style={{ height: containerHeight }}
      >
        {questions.map((q, i) => {
          const status = getMarksStatus(q.marks, q.max_marks)
          return (
            <div
              key={`${q.question_number}:${q.subpart ?? ''}:${i}`}
              ref={(el) => {
                cardRefs.current[i] = el
              }}
              className={`question-item status-${status}`}
              id={`q-card-${i}`}
              data-q-index={i}
            >
              {renderCard ? renderCard(q, i, status) : <ReadOnlyCard q={q} status={status} />}
            </div>
          )
        })}
      </div>
    </>
  )
}

/** Feedback/correction panel, collapsed by default. */
export function FeedbackToggle({
  feedback,
  status,
  defaultOpen = false,
}: {
  feedback?: string
  status: MarksStatus
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <>
      <button type="button" className="feedback-toggle-btn" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span>{status === 'correct' ? 'Feedback' : 'Correction'}</span>
        <span className="feedback-toggle-icon">▾</span>
      </button>
      <div
        className="feedback-container"
        hidden={!open}
        dangerouslySetInnerHTML={{ __html: renderContent(feedback || 'No additional reasoning provided') }}
      />
    </>
  )
}

function ReadOnlyCard({ q, status }: { q: QuestionDetail; status: MarksStatus }) {
  const ts = formatTimestamp(q.timestamp)
  return (
    <>
      <div className="gr-q-head">
        <div className="gr-q-title">
          <h3>
            Question {String(q.question_number ?? '')}
            {q.subpart ? ` (${q.subpart})` : ''}
          </h3>
          {STATUS_BADGE[status] && <span className={`status-badge status-badge-${status}`}>{STATUS_BADGE[status]}</span>}
        </div>
        <div className="gr-q-marks" aria-label={`${String(q.marks || 0)} out of ${String(q.max_marks || 0)} marks`}>
          {String(q.marks || 0)}
          <span> / {String(q.max_marks || 0)}</span>
        </div>
      </div>
      <p className="question-text" dangerouslySetInnerHTML={{ __html: renderContent(q.question || 'Question not available') }} />
      <div className="gr-q-block">
        <h4>Your answer</h4>
        <pre dangerouslySetInnerHTML={{ __html: renderContent(q.student_answer || 'No answer provided') }} />
      </div>
      <div className="reasoning-section rubric">
        {/* Corrections open by default: they are what the student came for. */}
        <FeedbackToggle feedback={q.feedback} status={status} defaultOpen={status !== 'correct'} />
      </div>
      {ts && <p className="gr-q-time">Answered on {ts}</p>}
    </>
  )
}
