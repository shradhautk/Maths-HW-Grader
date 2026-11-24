import type { AlertItem } from '../hooks/useAlerts'

export default function AlertStack({ alerts, onDismiss }: { alerts: AlertItem[]; onDismiss: (id: number) => void }) {
  return (
    <div className="alerts-container" aria-live="assertive">
      {alerts.map((a) => (
        <div key={a.id} className={`alert alert-${a.type}${a.fading ? ' fade-out' : ''}`}>
          {a.message}
          <button type="button" className="close-btn" onClick={() => onDismiss(a.id)}>
            &times;
          </button>
        </div>
      ))}
    </div>
  )
}
