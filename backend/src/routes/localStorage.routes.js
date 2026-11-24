// Development-only stand-in for S3 presigned URLs (STORAGE_DRIVER=local):
// PUT stores the bytes under LOCAL_STORAGE_DIR, GET serves them back. Every URL
// is signed and expires, like a presigned S3 URL.
import express, { Router } from 'express'
import { readLocalObject, verifyLocalSignature, writeLocalObject } from '../aws/s3.js'
import config from '../config/index.js'

const router = Router()

const objectRef = (req) => {
  const [bucket, ...rest] = req.params.path
  return { bucket, key: rest.join('/') }
}

router.put('/*path', express.raw({ type: () => true, limit: config.storage.uploadMaxBytes }), async (req, res) => {
  const { bucket, key } = objectRef(req)
  const contentType = req.get('content-type') ?? ''
  const length = req.query.len ?? ''
  if (!verifyLocalSignature('put', bucket, key, req.query.exp, req.query.sig, contentType, length)) {
    return res.status(403).type('text/plain').send('SignatureDoesNotMatch')
  }
  // Like S3 with a signed Content-Length: the body must be exactly the signed size.
  if (length && (req.body?.length ?? 0) !== Number(length)) {
    return res.status(403).type('text/plain').send('SignatureDoesNotMatch')
  }
  await writeLocalObject(bucket, key, req.body ?? Buffer.alloc(0), contentType)
  res.status(200).end()
})

router.get('/*path', async (req, res) => {
  const { bucket, key } = objectRef(req)
  if (!verifyLocalSignature('get', bucket, key, req.query.exp, req.query.sig)) {
    return res.status(403).type('text/plain').send('SignatureDoesNotMatch')
  }
  try {
    const { body, contentType } = await readLocalObject(bucket, key)
    res.type(contentType).send(body)
  } catch {
    res.status(404).type('text/plain').send('NoSuchKey')
  }
})

export default router
