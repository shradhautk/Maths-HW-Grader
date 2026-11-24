import { useEffect, type RefObject } from 'react'

declare global {
  interface Window {
    MathJax?: {
      typesetPromise?: (els?: Element[]) => Promise<void>
      typeset?: (els?: Element[]) => void
      typesetClear?: (els?: Element[]) => void
      [key: string]: unknown
    }
  }
}

const MATHJAX_SRC = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js'
let loading: Promise<void> | null = null

function loadMathJax(): Promise<void> {
  if (loading) return loading
  loading = new Promise((resolve, reject) => {
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
      },
      options: { enableMenu: false },
      startup: { typeset: false },
    }
    const script = document.createElement('script')
    script.src = MATHJAX_SRC
    script.async = true
    script.onload = () => resolve()
    script.onerror = () => {
      loading = null
      reject(new Error('MathJax failed to load'))
    }
    document.head.appendChild(script)
  })
  return loading
}

/** Typesets math inside `el`, loading MathJax on first use. Best-effort: failures leave raw TeX. */
export async function typeset(el: HTMLElement): Promise<void> {
  try {
    await loadMathJax()
    const mj = window.MathJax
    if (!mj || !el.isConnected) return
    mj.typesetClear?.([el])
    if (mj.typesetPromise) await mj.typesetPromise([el])
    else mj.typeset?.([el])
  } catch {
    /* math stays as raw TeX */
  }
}

/** Typesets math inside `ref` whenever `deps` change (skips content with no TeX). */
export function useMathJax(ref: RefObject<HTMLElement | null>, deps: unknown[]) {
  useEffect(() => {
    const el = ref.current
    if (!el || !el.textContent || !/[$\\]/.test(el.textContent)) return
    void typeset(el)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}
