import { API, UPLOAD_BASE_URL, UPLOAD_SENDS_TOKEN } from '../config'
import { getToken } from './session'

export const UPLOAD_LIMITS = {
  maxFileSizeMb: 10,
  allowedTypes: ['application/pdf'],
  timeoutMs: 60_000,
}

export type UploadType = 'student' | 'teacher' | 'answer'

export interface UploadMetadata {
  uploadType: UploadType
  assignmentId?: string
  courseId?: string
  qpId?: string
  studentId?: string
  files: { name: string; type: string; size: number; originalName?: string }[]
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1048576).toFixed(1)} MB`
}

/** Asks the upload Lambda for one S3 presigned PUT URL per file, in order. */
export async function requestPresignedUrls(metadata: UploadMetadata): Promise<string[]> {
  let response: Response
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  // The Node backend requires the user's token; the legacy upload Lambda's CORS
  // config only allows Content-Type, so it is called without one (see config.ts).
  const token = UPLOAD_SENDS_TOKEN ? getToken() : null
  if (token) headers.Authorization = `Bearer ${token}`
  try {
    response = await fetch(`${UPLOAD_BASE_URL}${API.upload}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(metadata),
    })
  } catch (err) {
    throw new Error(`Failed to connect to server: ${(err as Error).message}`)
  }
  if (!response.ok) {
    throw new Error(`Server error (${response.status}): ${await response.text()}`)
  }
  const data = (await response.json()) as { urls?: string[] }
  if (!data.urls || data.urls.length < metadata.files.length) {
    throw new Error('Server did not return an upload URL for every file')
  }
  return data.urls
}

/** PUTs a file to S3 via XHR (fetch has no upload progress events). */
export function uploadToPresignedUrl(file: File, url: string, onProgress: (percent: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url, true)
    xhr.setRequestHeader('Content-Type', file.type)
    xhr.timeout = UPLOAD_LIMITS.timeoutMs
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
    })
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`HTTP error ${xhr.status}: ${xhr.statusText}`))
    xhr.onerror = () => reject(new Error('Network error occurred'))
    xhr.ontimeout = () => reject(new Error('Upload timed out'))
    xhr.send(file)
  })
}

/** The grading Lambda finds solution keys by name: {assignmentId}_{qpId}_answer.pdf */
export function renameSolutionFile(file: File, assignmentId: string, qpId: string): File {
  const clean = (s: string) => s.replace(/[^a-zA-Z0-9]/g, '_')
  const ext = file.name.split('.').pop()
  return new File([file], `${clean(assignmentId)}_${clean(qpId)}_answer.${ext}`, {
    type: file.type,
    lastModified: file.lastModified,
  })
}

/** Presigns and PUTs every file; resolves once all succeed, rejects on the first failure. */
export async function uploadFiles(
  metadata: Omit<UploadMetadata, 'files'>,
  files: File[],
  onProgress: (percent: number) => void,
): Promise<void> {
  const full: UploadMetadata = { ...metadata, files: files.map((f) => ({ name: f.name, type: f.type, size: f.size })) }
  const urls = await requestPresignedUrls(full)
  const progress = files.map(() => 0)
  await Promise.all(
    files.map((file, i) =>
      uploadToPresignedUrl(file, urls[i], (p) => {
        progress[i] = p
        onProgress(Math.round(progress.reduce((a, b) => a + b, 0) / files.length))
      }),
    ),
  )
}
