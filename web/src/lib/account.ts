import { API } from '../config'
import { apiRequest } from './api'

export interface MeResponse {
  user_id: string
  name: string
  email: string
  userRole: string
  status: string
  must_change_password: boolean
  photoUrl: string | null
}

/** Files a reset request for an admin to approve (teachers/students can't self-reset). */
export function requestPasswordReset(email: string) {
  return apiRequest<{ message: string }>(API.auth.forgotPassword, { auth: false, body: { email } })
}

export function changePassword(currentPassword: string, newPassword: string) {
  return apiRequest<{ message: string }>(API.auth.changePassword, { body: { currentPassword, newPassword } })
}

export function fetchMe() {
  return apiRequest<MeResponse>(API.auth.me, { body: {} })
}

export const PHOTO_TYPES = ['image/jpeg', 'image/png', 'image/webp']
export const PHOTO_MAX_BYTES = 5 * 1024 * 1024

/** presigned PUT straight to S3, then tell the API to record it. Returns a fresh view URL. */
export async function uploadProfilePhoto(file: File): Promise<string> {
  if (!PHOTO_TYPES.includes(file.type)) throw new Error('Use a JPG, PNG or WebP image.')
  if (file.size > PHOTO_MAX_BYTES) throw new Error('Photo must be under 5 MB.')

  const { uploadUrl, key } = await apiRequest<{ uploadUrl: string; key: string }>(API.auth.profilePhoto, {
    body: { action: 'upload-url', contentType: file.type, size: file.size },
  })
  const put = await fetch(uploadUrl, { method: 'PUT', headers: { 'Content-Type': file.type }, body: file })
  if (!put.ok) throw new Error(`Photo upload failed (${put.status})`)

  const { photoUrl } = await apiRequest<{ photoUrl: string }>(API.auth.profilePhoto, {
    body: { action: 'confirm', key, contentType: file.type },
  })
  return photoUrl
}
