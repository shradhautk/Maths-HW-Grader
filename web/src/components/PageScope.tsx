import { useEffect, type ReactNode, type Ref } from 'react'

/** Shared stylesheets in src/styles/themes/, scoped to `.theme-<name>`. */
export type Theme = 'aesthetic' | 'teacher-grade' | 'teacher-correction' | 'admin' | 'legal' | 'account'

interface Props {
  /** Page stylesheet in src/styles/pages/ (if any) — scopes its selectors (see vite.config.ts). */
  name: string
  /** Shared stylesheets the original page also loaded, in the order it loaded them. */
  theme?: Theme | Theme[]
  /** Classes the original page put on <body>, e.g. "auth-page". */
  className?: string
  /** Document title suffix: "GradeAnt | {title}". */
  title?: string
  /** Ref to the wrapper element (stands in for the original page's <body>). */
  rootRef?: Ref<HTMLDivElement>
  children: ReactNode
}

export default function PageScope({ name, theme, className, title, rootRef, children }: Props) {
  useEffect(() => {
    if (title) document.title = `GradeAnt | ${title}`
  }, [title])

  const themes = theme === undefined ? [] : Array.isArray(theme) ? theme : [theme]
  const classes = ['page-scope', ...themes.map((t) => `theme-${t}`), `page-${name}`, className].filter(Boolean).join(' ')
  return (
    <div ref={rootRef} className={classes}>
      {children}
    </div>
  )
}
