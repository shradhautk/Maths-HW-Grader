/** "Fri, 24 Oct" — short date for notices and cards ('' if unparseable). */
export function formatWhen(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })
}
