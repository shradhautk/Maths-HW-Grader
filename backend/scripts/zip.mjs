// Zip writing/reading for Lambda packages (package.mjs, package-lambdas.mjs).
// Written here instead of using tar/zip so output is identical on Windows,
// Linux and CI: every entry carries Unix permissions (executables 755, the
// rest 644), which Lambda needs, and timestamps are fixed so the same sources
// always give the same SHA-256. No dependencies.
import { closeSync, lstatSync, openSync, readdirSync, readFileSync, writeSync } from 'node:fs'
import path from 'node:path'
import { crc32, deflateRawSync, inflateRawSync } from 'node:zlib'

const DOS_DATE = (0 << 9) | (1 << 5) | 1 // 1980-01-01, fixed for reproducible builds
const DOS_TIME = 0
const UTF8 = 0x0800

/** Every regular file under `dir` as a "/"-separated relative path, sorted. Symlinks (node_modules/.bin) are skipped. */
export function listFiles(dir, rel = '') {
  const out = []
  for (const name of readdirSync(dir).sort()) {
    const abs = path.join(dir, name)
    const r = rel ? `${rel}/${name}` : name
    const st = lstatSync(abs)
    if (st.isSymbolicLink()) continue
    if (st.isDirectory()) out.push(...listFiles(abs, r))
    else if (st.isFile()) out.push(r)
  }
  return out
}

/**
 * Writes `files` (relative to `srcDir`) into `zipPath`, each stored under
 * `prefix + file` (e.g. "nodejs/" for a Lambda layer). Paths in `executables`
 * (relative to srcDir) get mode 755. Returns { files, bytes } (bytes = unzipped size).
 */
export function writeZip(zipPath, srcDir, files, { prefix = '', executables = new Set() } = {}) {
  if (files.length > 0xffff) throw new Error(`${files.length} files: more than a plain zip can hold (zip64 is not implemented)`)
  const fd = openSync(zipPath, 'w')
  const central = []
  let offset = 0
  let bytes = 0
  const write = (buf) => {
    writeSync(fd, buf)
    offset += buf.length
  }
  for (const rel of files) {
    const data = readFileSync(path.join(srcDir, rel))
    bytes += data.length
    const crc = crc32(data)
    const deflated = deflateRawSync(data, { level: 9 })
    const stored = deflated.length >= data.length
    const body = stored ? data : deflated
    const method = stored ? 0 : 8
    const name = Buffer.from(prefix + rel, 'utf8')
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4)
    local.writeUInt16LE(UTF8, 6)
    local.writeUInt16LE(method, 8)
    local.writeUInt16LE(DOS_TIME, 10)
    local.writeUInt16LE(DOS_DATE, 12)
    local.writeUInt32LE(crc, 14)
    local.writeUInt32LE(body.length, 18)
    local.writeUInt32LE(data.length, 22)
    local.writeUInt16LE(name.length, 26)
    local.writeUInt16LE(0, 28)
    central.push({ name, crc, method, csize: body.length, size: data.length, offset, mode: executables.has(rel) ? 0o100755 : 0o100644 })
    write(local)
    write(name)
    write(body)
    if (offset > 0xffffffff) throw new Error('package is over 4 GB (zip64 is not implemented)')
  }
  const cdStart = offset
  for (const c of central) {
    const h = Buffer.alloc(46)
    h.writeUInt32LE(0x02014b50, 0)
    h.writeUInt16LE((3 << 8) | 20, 4) // made by Unix, so the mode in the external attributes is honoured
    h.writeUInt16LE(20, 6)
    h.writeUInt16LE(UTF8, 8)
    h.writeUInt16LE(c.method, 10)
    h.writeUInt16LE(DOS_TIME, 12)
    h.writeUInt16LE(DOS_DATE, 14)
    h.writeUInt32LE(c.crc, 16)
    h.writeUInt32LE(c.csize, 20)
    h.writeUInt32LE(c.size, 24)
    h.writeUInt16LE(c.name.length, 28)
    h.writeUInt32LE((c.mode << 16) >>> 0, 38)
    h.writeUInt32LE(c.offset, 42)
    write(h)
    write(c.name)
  }
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(central.length, 8)
  end.writeUInt16LE(central.length, 10)
  end.writeUInt32LE(offset - cdStart, 12)
  end.writeUInt32LE(cdStart, 16)
  write(end)
  closeSync(fd)
  return { files: files.length, bytes }
}

/** Every entry in a zip buffer, read back through the central directory with its CRC checked. */
export function readEntries(zip) {
  const eocd = zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]))
  if (eocd < 0) throw new Error('not a zip file')
  const count = zip.readUInt16LE(eocd + 10)
  const out = new Map()
  let p = zip.readUInt32LE(eocd + 16)
  for (let i = 0; i < count; i++) {
    if (zip.readUInt32LE(p) !== 0x02014b50) throw new Error('corrupt central directory')
    const nameLen = zip.readUInt16LE(p + 28)
    const extraLen = zip.readUInt16LE(p + 30)
    const commentLen = zip.readUInt16LE(p + 32)
    const name = zip.toString('utf8', p + 46, p + 46 + nameLen)
    const local = zip.readUInt32LE(p + 42)
    const method = zip.readUInt16LE(local + 8)
    const csize = zip.readUInt32LE(local + 18)
    const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28)
    const raw = zip.subarray(start, start + csize)
    const data = method === 8 ? inflateRawSync(raw) : raw
    if (crc32(data) !== zip.readUInt32LE(p + 16)) throw new Error(`${name}: CRC mismatch`)
    out.set(name, { mode: zip.readUInt32LE(p + 38) >>> 16, data })
    p += 46 + nameLen + extraLen + commentLen
  }
  return out
}
