// /auth/* (was authetication.py). Errors are {"message": "..."}.
// Paths come from config/app.yaml (api.auth).
import { Router } from 'express'
import config from '../config/index.js'
import * as c from '../controllers/auth.controller.js'
import { errorHandler, jsonBody } from '../middleware/errors.js'

const paths = config.app.paths.auth
const router = Router()
router.use(jsonBody())
router.post(paths.signup, c.signup)
router.post(paths.login, c.login)
router.post(paths.logout, c.logout)
router.post(paths.forgotPassword, c.forgotPassword)
router.post(paths.changePassword, c.changePassword)
router.post(paths.me, c.me)
router.post(paths.profilePhoto, c.profilePhoto)
router.use((req, res) => res.status(404).json({ message: 'Not Found' }))
router.use(errorHandler({ key: 'message' }))

export default router
