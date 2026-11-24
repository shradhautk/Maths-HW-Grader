/** The server's password rules (backend/src/lib/passwords.js passwordProblem). */
export const PASSWORD_RULES = [
  { id: 'length', label: '8+ characters', test: (p: string) => [...p].length >= 8 },
  { id: 'upper', label: 'An uppercase letter', test: (p: string) => /\p{Lu}/u.test(p) },
  { id: 'number', label: 'A number', test: (p: string) => /\p{Nd}/u.test(p) },
]

/** The first rule a password doesn't meet, or undefined. */
export const unmetPasswordRule = (password: string) => PASSWORD_RULES.find((r) => !r.test(password))
