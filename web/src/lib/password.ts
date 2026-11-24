export interface PasswordRule {
  id: 'length' | 'upper' | 'number' | 'special'
  label: string
  /** Required rules are enforced by the API; "special" only strengthens. */
  required: boolean
  test: (pw: string) => boolean
}

// Mirrors password_problem() in lambda_python_files/authetication.py.
export const PASSWORD_RULES: PasswordRule[] = [
  { id: 'length', label: '8+ characters', required: true, test: (pw) => pw.length >= 8 },
  { id: 'upper', label: 'Uppercase letter', required: true, test: (pw) => /[A-Z]/.test(pw) },
  { id: 'number', label: 'Number', required: true, test: (pw) => /\d/.test(pw) },
  { id: 'special', label: 'Special character', required: false, test: (pw) => /[^A-Za-z0-9]/.test(pw) },
]

export type StrengthLevel = 'empty' | 'weak' | 'fair' | 'good' | 'strong'

export function passwordStrength(pw: string): { level: StrengthLevel; label: string; score: number } {
  if (!pw) return { level: 'empty', label: '', score: 0 }
  const requiredMet = PASSWORD_RULES.filter((r) => r.required && r.test(pw)).length
  const bonus = PASSWORD_RULES.find((r) => r.id === 'special')!.test(pw) || pw.length >= 12
  if (requiredMet === 3 && bonus) return { level: 'strong', label: 'Strong', score: 4 }
  if (requiredMet === 3) return { level: 'good', label: 'Good', score: 3 }
  if (requiredMet === 2) return { level: 'fair', label: 'Fair', score: 2 }
  return { level: 'weak', label: 'Weak', score: 1 }
}

export function meetsRequiredRules(pw: string): boolean {
  return PASSWORD_RULES.filter((r) => r.required).every((r) => r.test(pw))
}
