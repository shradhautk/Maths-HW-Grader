import { getGradeLetter, getMarksStatus, toNumber, type QuestionDetail } from '../../lib/grading'

interface Props {
  totalMarks: unknown
  /** Sum of every question's max_marks — the summary record carries no assignment-wide max. */
  maxMarks: number
  details?: QuestionDetail[]
}

/** Grade badge + fraction + progress bar + correct/partial/needs-review counts (was grade_display.js). */
export default function ScoreHero({ totalMarks, maxMarks, details }: Props) {
  const total = toNumber(totalMarks)
  const percent = maxMarks > 0 ? Math.round((total / maxMarks) * 100) : 0
  const grade = getGradeLetter(percent)
  const color = `var(${grade.cssVar})`

  const counts = { correct: 0, partial: 0, incorrect: 0 }
  details?.forEach((q) => {
    const s = getMarksStatus(q.marks, q.max_marks)
    if (s !== 'unscored') counts[s]++
  })

  return (
    <div className="score-hero">
      <span className="score-hero-badge" style={{ background: color }}>
        {grade.letter}
      </span>
      <div className="score-hero-info">
        <div className="score-hero-fraction">
          {total}
          <span className="score-hero-max">/{maxMarks}</span>
        </div>
        <div className="score-hero-label">Total score</div>
      </div>
      <div className="score-hero-progress">
        <div className="score-hero-progress-top">
          <span className="score-hero-percent-value">{percent}%</span>
          <span className="score-hero-percent-tag">of max marks</span>
        </div>
        <div className="score-hero-bar">
          <div className="score-hero-bar-fill" style={{ width: `${percent}%`, background: color }} />
        </div>
      </div>
      {details && details.length > 0 && (
        <div className="score-hero-stats">
          <div className="score-hero-stat">
            <div className="score-hero-stat-value correct">{counts.correct}</div>
            <div className="score-hero-stat-label">Correct</div>
          </div>
          <div className="score-hero-stat">
            <div className="score-hero-stat-value partial">{counts.partial}</div>
            <div className="score-hero-stat-label">Partial</div>
          </div>
          <div className="score-hero-stat">
            <div className="score-hero-stat-value incorrect">{counts.incorrect}</div>
            <div className="score-hero-stat-label">Needs review</div>
          </div>
        </div>
      )}
    </div>
  )
}
