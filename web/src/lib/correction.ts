import { API } from '../config'
import { apiRequest } from './api'

/** POST to correct_questions_answer.py, which dispatches on `operation`. */
export function correctionRequest<T = unknown>(payload: Record<string, unknown>): Promise<T> {
  return apiRequest<T>(API.corrections,{ body: payload })
}
