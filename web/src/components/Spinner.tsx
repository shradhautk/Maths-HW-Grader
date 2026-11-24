/** Full-screen loading overlay; styled by each page's .loading-spinner/.spinner rules. */
export default function Spinner({ show }: { show: boolean }) {
  if (!show) return null
  return (
    <div className="loading-spinner" style={{ display: 'flex' }}>
      <div className="spinner" />
    </div>
  )
}
