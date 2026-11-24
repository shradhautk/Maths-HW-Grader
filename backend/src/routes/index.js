// Route table. Every path comes from config/app.yaml (api section).
import { Router } from 'express'
import config from '../config/index.js'
import { health, ready } from '../controllers/health.controller.js'
import { LOCAL_STORAGE_ROUTE } from '../aws/s3.js'
import authRoutes from './auth.routes.js'
import contactRoutes from './contact.routes.js'
import localStorageRoutes from './localStorage.routes.js'
import operationRoutes from './operations.routes.js'
import uploadRoutes from './upload.routes.js'

const paths = config.app.paths
const router = Router()

router.get(paths.health, health)
router.get(paths.healthReady, ready)

router.use(paths.auth.base, authRoutes)
router.use(paths.contact, contactRoutes)
router.use(paths.upload, uploadRoutes)
router.use(operationRoutes)

if (config.drivers.storage === 'local') router.use(LOCAL_STORAGE_ROUTE, localStorageRoutes)

export default router
