// Teacher → class announcements; students read the ones for their classes.
import { randomUUID } from 'node:crypto'
import { badRequest, forbidden, notFound } from '../../lib/errors.js'
import { pyStr, sortBy, trunc, truthy } from '../../lib/py.js'
import * as wf from '../../repositories/workflow.repository.js'
import { myAssignments, now, required, requireRole, visibleAssignments } from './context.js'

export async function postAnnouncement(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [className, title, message] = required(body, 'class_name', 'title', 'message')
  const mine = new Set((await myAssignments(caller)).map((a) => a.class_name))
  if (caller.role !== 'admin' && !mine.has(className)) {
    throw badRequest('Create an assignment for that class first, then you can post to it.')
  }
  const item = {
    announcement_id: `N-${randomUUID().replace(/-/g, '').slice(0, 10)}`,
    class_name: trunc(pyStr(className), 100),
    title: trunc(pyStr(title).trim(), 150),
    message: trunc(pyStr(message).trim(), 2000),
    author_id: caller.user_id,
    author_name: caller.user.name || 'Teacher',
    created_at: now(),
  }
  if (!item.title || !item.message) throw badRequest('A title and a message are required.')
  await wf.putAnnouncement(item)
  return { announcement: item }
}

export async function listAnnouncements(body, caller) {
  let items
  if (caller.role === 'student') {
    let classes = caller.user.classes
    if (!truthy(classes)) {
      const visible = await visibleAssignments(caller)
      classes = sortBy([...new Set(visible.map((a) => a.class_name).filter(truthy))], (c) => c)
    }
    items = classes.length ? await wf.announcementsForClasses(classes) : []
  } else {
    requireRole(caller, 'teacher', 'admin')
    items = caller.role === 'admin' ? await wf.scanAnnouncements() : await wf.announcementsByAuthor(caller.user_id)
  }
  return { announcements: sortBy(items, (a) => a.created_at ?? '', { reverse: true }).slice(0, 50) }
}

export async function deleteAnnouncement(body, caller) {
  requireRole(caller, 'teacher', 'admin')
  const [announcementId] = required(body, 'announcement_id')
  const item = await wf.getAnnouncement(announcementId)
  if (!item) throw notFound('Announcement not found.')
  if (caller.role !== 'admin' && item.author_id !== caller.user_id) throw forbidden('You can only delete your own announcements.')
  await wf.deleteAnnouncement(announcementId)
  return { ok: true }
}
