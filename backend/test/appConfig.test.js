// config/app.yaml: the shipped file is valid in every mode, and mistakes stop
// startup with a message naming the problem.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, test } from 'node:test'
import { fileURLToPath } from 'node:url'
import { parse } from 'yaml'
import { validateAppConfig } from '../src/config/appConfig.js'

const FILE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../config/app.yaml')
// The file as shipped, with the deployment switch pinned to aws: these checks are about the
// aws URLs, whatever target config/app.yaml currently selects. Hybrid tests set it themselves.
const shipped = () => ({ ...parse(readFileSync(FILE, 'utf8')), deployment: 'aws' })
const fails = (mutate, pattern, mode = 'development') => {
  const doc = shipped()
  mutate(doc)
  assert.throws(() => validateAppConfig(doc, mode), pattern)
}

describe('config/app.yaml', () => {
  test('the shipped file is valid for development, test and production', () => {
    for (const mode of ['development', 'test', 'production']) {
      const c = validateAppConfig(shipped(), mode)
      assert.equal(c.api.workflow, '/workflow')
      assert.deepEqual(c.roles.teacher.signup, ['self', 'invite']) // teachers sign up or join by invite
    }
    assert.deepEqual(validateAppConfig(shipped(), 'production').urls.corsOrigins, ['https://app.gradeant.io', 'https://grade-ant.com', 'https://www.grade-ant.com'])
  })

  test('a role cannot be given more than the code supports', () => {
    fails((d) => (d.permissions.marks.updateMarks = ['student', 'teacher']), /permissions\.marks\.updateMarks: student can't be given this/)
  })

  test('every operation must be listed, and names must be real', () => {
    fails((d) => delete d.permissions.workflow.publishGrades, /permissions\.workflow\.publishGrades is missing/)
    fails((d) => (d.permissions.workflow.publishGradez = ['teacher']), /publishGradez: unknown operation/)
    fails((d) => (d.permissions.grades.getAllStudentGrades = ['teachers']), /must be a list of roles/)
  })

  test('admins can never sign themselves up; roles are fixed', () => {
    fails((d) => (d.roles.admin.signup = ['self', 'invite']), /roles\.admin\.signup can't include "self"/)
    fails((d) => (d.roles.owner = { label: 'Owner', signup: ['invite'] }), /roles\.owner: unknown role/)
    fails((d) => (d.roles.teacher.signup = ['email']), /roles\.teacher\.signup must be a list of: self, invite/)
  })

  test('paths and URLs are checked', () => {
    fails((d) => (d.api.workflow = 'workflow'), /api\.workflow must be a path/)
    fails((d) => (d.api.marks = '/getgrades'), /"\/getgrades" is used for two different APIs/)
    fails((d) => (d.urls.production.api = 'not a url'), /urls\.production\.api must be an http\(s\) URL/, 'production')
    fails((d) => delete d.urls.staging, /urls\.staging is missing/, 'staging')
  })

  test('turning an operation off ([]) is allowed', () => {
    const doc = shipped()
    doc.permissions.workflow.askAssistant = []
    assert.deepEqual(validateAppConfig(doc, 'development').permissions.workflow.askAssistant, [])
  })
})

describe('config/app.yaml deployment switch', () => {
  const AWS = { database: 'dynamodb', ai: 'gemini', email: 'ses', secrets: 'aws', grading: 'lambda' }

  test('the aws target has the AWS providers', () => {
    const doc = shipped()
    doc.deployment = 'aws'
    const c = validateAppConfig(doc, 'production')
    assert.equal(c.deployment.name, 'aws')
    assert.deepEqual(c.deployment.providers, AWS)
  })

  test('a file without the section behaves as aws (older app.yaml files and fixtures)', () => {
    const doc = shipped()
    delete doc.deployment
    delete doc.deployments
    const c = validateAppConfig(doc, 'production')
    assert.equal(c.deployment.name, 'aws')
    assert.deepEqual(c.deployment.providers, AWS)
    assert.deepEqual(c.urls, validateAppConfig(shipped(), 'production').urls)
  })

  test('hybrid: providers switch and its urls are merged over urls.production only', () => {
    const doc = shipped()
    doc.deployment = 'hybrid'
    const prod = validateAppConfig(doc, 'production')
    assert.deepEqual(prod.deployment.providers, { database: 'supabase', ai: 'gemini', email: 'log', secrets: 'env', grading: 'sqs' })
    assert.equal(prod.urls.api, 'https://gradeant-api.onrender.com')
    assert.equal(prod.urls.web, 'https://grade-ant.com')
    assert.deepEqual(prod.urls.corsOrigins, ['https://grade-ant.com', 'https://www.grade-ant.com', 'https://grade-ant.netlify.app', 'https://gradeant.netlify.app'])
    // development and tests keep their own urls
    assert.equal(validateAppConfig(doc, 'development').urls.api, 'http://localhost:8787')
  })

  test('mistakes in the switch stop startup with a clear message', () => {
    fails((d) => (d.deployment = 'heroku'), /deployment: "heroku" is not defined under deployments/)
    fails((d) => (d.deployments.hybrid.providers.database = 'mongodb'), /providers\.database must be one of dynamodb, supabase/)
    fails((d) => (d.deployments.hybrid.providers.queue = 'kafka'), /providers\.queue: unknown provider/)
    fails((d) => (d.deployments.hybrid.urls.paths = '/x'), /urls\.paths: only web, api, upload, corsOrigins can be set here/)
    fails((d) => { d.deployment = 'hybrid'; d.deployments.hybrid.urls.api = 'not a url' }, /urls\.production\.api must be an http\(s\) URL/, 'production')
  })

  test('supabase with Secrets Manager secrets gets a warning', () => {
    const doc = shipped()
    doc.deployment = 'hybrid'
    doc.deployments.hybrid.providers.secrets = 'aws'
    assert.match(validateAppConfig(doc, 'production').deployment.warnings[0], /use secrets: env off AWS/)
  })
})
