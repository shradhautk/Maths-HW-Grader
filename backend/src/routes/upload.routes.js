// POST /uploadpdf — presigned upload URLs (see services/upload.service.js).
import { Router } from 'express'
import { bodyOf, handle } from '../controllers/respond.js'
import { authenticate } from '../middleware/auth.js'
import { errorHandler, jsonBody } from '../middleware/errors.js'
import { presignUploads } from '../services/upload.service.js'

const router = Router()
router.use(jsonBody())
router.post(
  '/',
  handle(async (req) => presignUploads(bodyOf(req), await authenticate(req)), 'Could not prepare the upload.'),
)
router.use(errorHandler({ key: 'message' }))

export default router
