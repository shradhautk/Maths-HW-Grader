import * as admin from '../services/admin.service.js'
import { bodyOf, handle } from './respond.js'

export const adminOperation = handle((req) => admin.dispatch(bodyOf(req), req.get('authorization')), 'Internal server error.')
