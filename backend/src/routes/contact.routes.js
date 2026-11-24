// POST /contact: the website's contact and request-a-pilot forms (public, no token).
// Errors are {"message": "..."}. Path comes from config/app.yaml (api.contact).
import { Router } from 'express'
import { bodyOf, handle } from '../controllers/respond.js'
import { errorHandler, jsonBody } from '../middleware/errors.js'
import * as contact from '../services/contact.service.js'

const router = Router()
router.use(jsonBody())
router.post(
  '/',
  handle((req) => contact.submit(bodyOf(req), req.get('x-forwarded-for')?.split(',')[0].trim() || req.ip), 'Could not send your message.'),
)
router.use((req, res) => res.status(404).json({ message: 'Not Found' }))
router.use(errorHandler({ key: 'message' }))

export default router
