import { useEffect, useRef } from 'react'
import { typeset } from '../hooks/useMathJax'

/**
 * Live preview of an answer. MathJax rewrites the DOM it typesets, so the
 * text is written imperatively into a node React never renders children into.
 */
export default function MathPreview({ text, latex, className }: { text: string; latex: boolean; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.textContent = text
    if (latex) {
      el.dataset.latex = 'true'
      // Debounce so typing doesn't re-typeset on every keystroke.
      const t = window.setTimeout(() => void typeset(el), 250)
      return () => window.clearTimeout(t)
    }
    delete el.dataset.latex
  }, [text, latex])

  return <div ref={ref} className={className} />
}
