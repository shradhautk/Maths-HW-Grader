import { bearerToken } from '../middleware/auth.js'
import * as auth from '../services/auth.service.js'
import { bodyOf, handle } from './respond.js'

export const signup = handle((req) => auth.signup(bodyOf(req)), 'Error creating user')
export const login = handle((req) => auth.login(bodyOf(req)), 'Error during login')
export const logout = handle((req) => auth.logout(req.get('authorization')), 'Error during logout')
export const forgotPassword = handle((req) => auth.forgotPassword(bodyOf(req)), 'Could not submit your request. Please try again.')
export const changePassword = handle(
  (req) => auth.changePassword(bearerToken(req), bodyOf(req)),
  'Could not update your password. Please try again.',
)
export const me = handle((req) => auth.me(bearerToken(req)), 'Could not load your account.')
export const profilePhoto = handle(
  (req) => auth.profilePhoto(bearerToken(req), bodyOf(req)),
  'Could not update your photo. Please try again.',
)
