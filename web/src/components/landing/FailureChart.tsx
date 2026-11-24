import { useState } from 'react'
import { ACCURACY, FAILURE_TYPES, SIZES, type MatrixSize } from './linalgBench'

// Grouped horizontal bars: one row per error type, one bar per matrix size.
// Matrix size is ordinal, so it takes a single-hue navy ramp (light → dark),
// validated ≥ 3:1 against the white chart surface. Hover or focus a bar for its
// value; the table below gives every number without needing colour.
const SIZE_COLOR: Record<MatrixSize, string> = { '3×3': '#7f90b6', '4×4': '#4b6194', '5×5': '#14213d' }
const MAX = Math.max(...FAILURE_TYPES.flatMap((f) => SIZES.map((s) => f.counts[s])))

export default function FailureChart() {
  const [active, setActive] = useState<{ type: string; size: MatrixSize } | null>(null)

  return (
    <figure className="lp-chart" aria-labelledby="fc-title">
      <figcaption id="fc-title" className="lp-chart-title">
        Failure types by matrix size
        <small>Classified failures across all 10 models, 2,200 responses per size</small>
      </figcaption>

      <ul className="lp-chart-legend" aria-label="Matrix size">
        {SIZES.map((s) => (
          <li key={s}>
            <i style={{ background: SIZE_COLOR[s] }} aria-hidden />
            {s} <span>({ACCURACY[s].fail} failures)</span>
          </li>
        ))}
      </ul>

      <div className="lp-chart-rows" onMouseLeave={() => setActive(null)}>
        {FAILURE_TYPES.map((f) => (
          <div key={f.type} className="lp-chart-row">
            <div className="lp-chart-label">
              <b>{f.type}</b>
              <small>{f.meaning}</small>
            </div>
            <div className="lp-chart-bars">
              {SIZES.map((s) => {
                const n = f.counts[s]
                const on = active?.type === f.type && active.size === s
                return (
                  <button
                    key={s}
                    type="button"
                    className={`lp-chart-hit${on ? ' is-on' : ''}`}
                    aria-label={`${f.type}, ${s}: ${n} of ${ACCURACY[s].fail} failures`}
                    onMouseEnter={() => setActive({ type: f.type, size: s })}
                    onFocus={() => setActive({ type: f.type, size: s })}
                    onBlur={() => setActive(null)}
                  >
                    <span className="lp-chart-bar" style={{ width: `${(n / MAX) * 100}%`, background: SIZE_COLOR[s] }} />
                    {on && (
                      <span className="lp-chart-tip" role="status">
                        <b>
                          {f.type} · {s}
                        </b>
                        {n} of {ACCURACY[s].fail} failures ({Math.round((n / ACCURACY[s].fail) * 100)}%)
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      <details className="lp-chart-table">
        <summary>Show the numbers as a table</summary>
        <table>
          <thead>
            <tr>
              <th scope="col">Error type</th>
              {SIZES.map((s) => (
                <th key={s} scope="col">
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {FAILURE_TYPES.map((f) => (
              <tr key={f.type}>
                <th scope="row">{f.type}</th>
                {SIZES.map((s) => (
                  <td key={s}>{f.counts[s]}</td>
                ))}
              </tr>
            ))}
            <tr>
              <th scope="row">All failures</th>
              {SIZES.map((s) => (
                <td key={s}>{ACCURACY[s].fail}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </details>
      <p className="lp-chart-source">
        Source: LinAlg-Bench dataset, results/error_tags (CC BY 4.0). “Other” groups the rarer error tags.
      </p>
    </figure>
  )
}
