import { useState } from 'react'
import { adminApiRequest } from '../../lib/api'
import AdminLayout, { capitalize, useAdminAlert } from './AdminLayout'
import { FilterPills, useAdminData } from './adminShared'

type Filter = 'all' | 'published' | 'pending' | 'draft'

interface Post {
  id: string
  title: string
  author: string
  dateLabel: string
  status: 'published' | 'pending' | 'draft'
}

const STATUS_PILL: Record<Post['status'], string> = { published: 'success', pending: 'warning', draft: 'info' }
const SMALL_BTN = { padding: '6px 11px', fontSize: 12 }

/** Blog/content moderation queue. */
function ContentList({ filter }: { filter: Filter }) {
  const showAlert = useAdminAlert()
  const { data, error, reload } = useAdminData<{ posts?: Post[] }>('listContent', { filter })

  async function setStatus(postId: string, status: Post['status']) {
    try {
      await adminApiRequest('updatePostStatus', { postId, status })
      showAlert(`Post ${status === 'published' ? 'published' : 'updated'}.`, 'success')
      void reload(true)
    } catch (err) {
      showAlert(`Couldn't update post: ${(err as Error).message}`, 'danger')
    }
  }

  const posts = data?.posts ?? []
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {error ? (
        <div className="admin-panel">Couldn't load content: {error}</div>
      ) : !data ? (
        <div className="admin-panel">Loading content…</div>
      ) : posts.length === 0 ? (
        <div className="admin-panel">No posts in this view.</div>
      ) : (
        posts.map((post) => (
          <div key={post.id} className="admin-panel" style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
            <div style={{ width: 64, height: 48, borderRadius: 8, background: 'var(--gradient-grow)', flexShrink: 0 }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--ink-900)' }}>{post.title}</div>
              <div style={{ fontSize: 11, color: 'var(--text-subtle)', marginTop: 3 }}>
                by {post.author} · {post.dateLabel}
              </div>
            </div>
            <span className={`status-pill ${STATUS_PILL[post.status] ?? 'info'}`}>{capitalize(post.status)}</span>
            <div className="admin-inline-actions">
              {post.status === 'pending' && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ ...SMALL_BTN, color: 'var(--accent)', borderColor: 'var(--teal-200)' }}
                  onClick={() => setStatus(post.id, 'published')}
                >
                  Approve
                </button>
              )}
              {post.status === 'published' ? (
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ ...SMALL_BTN, color: 'var(--color-danger)', borderColor: 'var(--color-danger-soft)' }}
                  onClick={() => setStatus(post.id, 'draft')}
                >
                  Unpublish
                </button>
              ) : (
                <button type="button" className="btn btn-secondary" style={SMALL_BTN} onClick={() => setStatus(post.id, 'published')}>
                  Publish
                </button>
              )}
              <button
                type="button"
                className="btn btn-secondary"
                style={SMALL_BTN}
                onClick={() => showAlert("Post editor isn't wired up yet — this button is ready for it.", 'warning')}
              >
                Edit
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  )
}

function NewPostButton() {
  const showAlert = useAdminAlert()
  return (
    <button
      type="button"
      className="btn btn-primary"
      onClick={() => showAlert("Post composer isn't wired up yet — this button is ready for it.", 'warning')}
    >
      <span className="material-icons" style={{ fontSize: 18 }}>add</span>New Post
    </button>
  )
}

export default function AdminContent() {
  const [filter, setFilter] = useState<Filter>('all')
  return (
    <AdminLayout docTitle="Content & Blog" title="Blog & Content" actions={<NewPostButton />}>
      <FilterPills<Filter>
        value={filter}
        onChange={setFilter}
        options={[
          { value: 'all', label: 'All' },
          { value: 'published', label: 'Published' },
          { value: 'pending', label: 'Pending' },
          { value: 'draft', label: 'Drafts' },
        ]}
      />
      <ContentList filter={filter} />
    </AdminLayout>
  )
}
