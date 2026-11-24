import { useEffect, useRef } from 'react'
import { animate, useInView } from 'motion/react'
import { EASE } from '../landing/motion'

/** Counts up from 0 when scrolled into view. */
export default function CountUp({ to, suffix = '', decimals = 0 }: { to: number; suffix?: string; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, amount: 0.6 })
  useEffect(() => {
    if (!inView || !ref.current) return
    const node = ref.current
    const controls = animate(0, to, {
      duration: 1.4,
      ease: EASE,
      onUpdate: (v) => (node.textContent = `${v.toFixed(decimals)}${suffix}`),
    })
    return () => controls.stop()
  }, [inView, to, suffix, decimals])
  return (
    <span ref={ref}>
      {(0).toFixed(decimals)}
      {suffix}
    </span>
  )
}
