// Single-resource "operation dispatch" APIs, each replacing one Python Lambda.
// Paths come from config/app.yaml (api section); the defaults are:
//   POST /getgrades      (get_grades.py)                errors: {"error": ...}
//   POST /correctanswer  (correct_questions_answer.py)  errors: {"error": ...}
//   POST /updatedb       (update_db.py)                 errors: {"error": ...}
//   POST /admin          (admin/lambda_admin.py)        errors: {"message": ...}
//   POST /workflow       (workflow.py)                  errors: {"error": ...}
import { Router } from 'express'
import config from '../config/index.js'
import { adminOperation } from '../controllers/admin.controller.js'
import { operationsController } from '../controllers/operations.controller.js'
import { errorHandler, jsonBody } from '../middleware/errors.js'
import { operations as correctionOps } from '../services/correction.service.js'
import { operations as gradesOps } from '../services/grades.service.js'
import { operations as marksOps } from '../services/marks.service.js'
import { operations as workflowOps } from '../services/workflow/index.js'

function resource(controller, errorShape) {
  const router = Router()
  router.use(jsonBody())
  router.post('/', controller)
  router.use(errorHandler(errorShape))
  return router
}

const paths = config.app.paths
const errors = { key: 'error', invalidJson: 'Invalid JSON format' }

const router = Router()
router.use(paths.grades, resource(operationsController(gradesOps, { area: 'grades' }), errors))
router.use(paths.corrections, resource(operationsController(correctionOps, { area: 'corrections' }), errors))
router.use(paths.marks, resource(operationsController(marksOps, { area: 'marks' }), errors))
router.use(paths.workflow, resource(operationsController(workflowOps, { area: 'workflow', internalMessage: 'Internal error' }), errors))
router.use(
  paths.admin,
  resource(adminOperation, { key: 'message', invalidJson: 'Request body must be valid JSON.', internal: 'Internal server error.' }),
)

export default router
