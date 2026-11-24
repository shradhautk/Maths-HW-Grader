import type { CSSProperties } from 'react'
import { formatTimestamp, getGradeLetter, getMarksStatus, maxMarksOf, toNumber, type GradeResult } from '../../lib/grading'
import QuestionViewer from './QuestionViewer'
import '../../styles/ui.css'
import '../../styles/app.css'
import '../../styles/grades.css'

const RING_R = 46
const RING_C = 2 * Math.PI * RING_R

/**
 * One student's result for one assignment: score summary (ring, letter grade,
 * correct/partial/needs-review split) and the question-by-question breakdown.
 * `revealKey` remounts the container so the reveal animation replays on each
 * new search.
 */
export default function StudentResult({ result, revealKey }: { result: GradeResult; revealKey?: number }) {
  const summary = result.summary ?? []
  const first = summary[0]
  const details = Array.isArray(result.details) ? result.details : []

  const total = toNumber(first?.total_marks)
  const max = maxMarksOf(details)
  const percent = max > 0 ? Math.max(0, Math.min(100, Math.round((total / max) * 100))) : 0
  const grade = getGradeLetter(percent)

  const counts = { correct: 0, partial: 0, incorrect: 0 }
  details.forEach((q) => {
    const s = getMarksStatus(q.marks, q.max_marks)
    if (s !== 'unscored') counts[s]++
  })
  const scored = counts.correct + counts.partial + counts.incorrect
  const graded = formatTimestamp(first?.timestamp)

  return (
    <div key={revealKey} className="gr-result">
      {first && (
        <section className="app-panel" style={{ '--grade': `var(${grade.cssVar})` } as CSSProperties}>
          <div className="gr-summary">
            <div className="gr-ring" role="img" aria-label={`${percent}% of the maximum marks`}>
              <svg viewBox="0 0 108 108" aria-hidden>
                <circle cx="54" cy="54" r={RING_R} className="gr-ring-track" />
                <circle
                  cx="54"
                  cy="54"
                  r={RING_R}
                  className="gr-ring-fill"
                  strokeDasharray={RING_C}
                  strokeDashoffset={RING_C * (1 - percent / 100)}
                />
              </svg>
              <div className="gr-ring-label">
                <b>{percent}%</b>
                <small>of max marks</small>
              </div>
            </div>

            <div>
              <div className="gr-summary-top">
                <div className="gr-score">
                  {total}
                  <span> / {max} marks</span>
                </div>
                <span className="gr-letter">
                  Grade <b>{grade.letter}</b>
                </span>
              </div>
              <p className="gr-meta">
                {first.assignment_id && (
                  <>
                    Assignment <b>{first.assignment_id}</b>
                  </>
                )}
                {first.assignment_id && graded && ' · '}
                {graded && <>Graded {graded}</>}
              </p>

              {scored > 0 && (
                <>
                  <div className="gr-split" aria-hidden>
                    {counts.correct > 0 && <span className="gr-is-correct" style={{ flexGrow: counts.correct }} />}
                    {counts.partial > 0 && <span className="gr-is-partial" style={{ flexGrow: counts.partial }} />}
                    {counts.incorrect > 0 && <span className="gr-is-incorrect" style={{ flexGrow: counts.incorrect }} />}
                  </div>
                  <ul className="gr-counts">
                    <li className="gr-is-correct">
                      <b>{counts.correct}</b> correct
                    </li>
                    <li className="gr-is-partial">
                      <b>{counts.partial}</b> partially correct
                    </li>
                    <li className="gr-is-incorrect">
                      <b>{counts.incorrect}</b> need{counts.incorrect === 1 ? 's' : ''} review
                    </li>
                  </ul>
                </>
              )}
            </div>
          </div>

          <details className="gr-record gr-noprint">
            <summary>
              Record details
              <span className="material-icons" aria-hidden>
                expand_more
              </span>
            </summary>
            <div className="ga-table-wrap">
              <table className="ga-table">
                <thead>
                  <tr>
                    <th>Student ID</th>
                    <th>Assignment ID</th>
                    <th>Total marks</th>
                    <th>Evaluation ID</th>
                    <th>QP ID</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((g, i) => (
                    <tr key={i}>
                      <td>{g.student_id || 'N/A'}</td>
                      <td>{g.assignment_id || 'N/A'}</td>
                      <td>{g.total_marks ?? 'N/A'}</td>
                      <td className="gr-mono">{g.evaluation_id || 'N/A'}</td>
                      <td className="gr-mono">{g.qp_id || 'N/A'}</td>
                      <td>{formatTimestamp(g.timestamp) || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      )}

      {details.length > 0 && (
        <section className="app-panel">
          <header className="app-panel-head">
            <h2>Question breakdown</h2>
            <button type="button" className="ga-btn ga-btn-secondary ga-btn-sm gr-noprint" onClick={() => window.print()}>
              <span className="material-icons" aria-hidden style={{ fontSize: 16 }}>
                print
              </span>
              Print
            </button>
          </header>
          <QuestionViewer details={details} initialIndex={0} />
        </section>
      )}
    </div>
  )
}
