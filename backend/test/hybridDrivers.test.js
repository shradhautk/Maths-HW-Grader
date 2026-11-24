// The free-tier hybrid's drivers: Gemini for the assistant and rubric drafting (same contract
// as Claude), and "Start grading" through the grader's SQS queue (same payload as the invoke).
import assert from 'node:assert/strict'
import { after, describe, test } from 'node:test'

process.env.APP_ENV = 'test'
const { default: config } = await import('../src/config/index.js')
const { askAI: askClaude, setGeminiClient } = await import('../src/ai/index.js')
const { triggerGrading, setGradingSender } = await import('../src/aws/lambda.js')

const saved = { drivers: { ...config.drivers }, grading: { ...config.grading }, gemini: { ...config.gemini } }
after(() => {
  Object.assign(config.drivers, saved.drivers)
  Object.assign(config.grading, saved.grading)
  Object.assign(config.gemini, saved.gemini)
  setGeminiClient(undefined)
  setGradingSender(undefined)
})

function fakeGemini(answer) {
  const calls = []
  return {
    calls,
    models: {
      async generateContent(params) {
        calls.push(params)
        return typeof answer === 'function' ? answer(params, calls.length) : answer
      },
    },
  }
}

describe('AI_DRIVER=gemini', () => {
  test('maps the chat to Gemini and returns its text', async () => {
    config.drivers.ai = 'gemini'
    config.gemini.model = 'gemini-test-flash'
    const fake = fakeGemini({ text: 'Q2 was the hardest.', candidates: [{ finishReason: 'STOP' }] })
    setGeminiClient(fake)
    const reply = await askClaude({
      system: 'You are GradeFlow.',
      messages: [
        { role: 'user', content: 'Hi' },
        { role: 'assistant', content: 'Hello!' },
        { role: 'user', content: 'Which question was hardest?' },
      ],
      effort: 'low',
      maxTokens: 1000,
    })
    assert.equal(reply, 'Q2 was the hardest.')
    const [params] = fake.calls
    assert.equal(params.model, 'gemini-test-flash')
    assert.deepEqual(params.contents.map((c) => [c.role, c.parts[0].text]), [['user', 'Hi'], ['model', 'Hello!'], ['user', 'Which question was hardest?']])
    assert.equal(params.config.systemInstruction, 'You are GradeFlow.')
    assert.equal(params.config.maxOutputTokens, 1000)
    assert.equal(params.config.responseJsonSchema, undefined)
  })

  test('rubric drafting asks for JSON with the schema', async () => {
    const schema = { type: 'object', properties: { items: { type: 'array' } } }
    const fake = fakeGemini({ text: '{"items":[]}', candidates: [{ finishReason: 'STOP' }] })
    setGeminiClient(fake)
    assert.equal(await askClaude({ system: 'S', messages: [{ role: 'user', content: 'Q' }], effort: 'medium', outputFormat: { type: 'json_schema', schema } }), '{"items":[]}')
    assert.equal(fake.calls[0].config.responseMimeType, 'application/json')
    assert.deepEqual(fake.calls[0].config.responseJsonSchema, schema)
  })

  test('a schema Gemini refuses (400) falls back to JSON in the prompt', async () => {
    const schema = { type: 'object' }
    const fake = fakeGemini((params, n) => {
      if (n === 1) throw Object.assign(new Error('Invalid JSON schema'), { status: 400 })
      return { text: '{}', candidates: [{ finishReason: 'STOP' }] }
    })
    setGeminiClient(fake)
    assert.equal(await askClaude({ system: 'S', messages: [{ role: 'user', content: 'Q' }], outputFormat: { type: 'json_schema', schema } }), '{}')
    assert.equal(fake.calls[1].config.responseJsonSchema, undefined)
    assert.match(fake.calls[1].config.systemInstruction, /Respond with only a JSON object matching this JSON Schema/)
  })

  test('a decline is null, like a Claude refusal; other errors are thrown', async () => {
    setGeminiClient(fakeGemini({ text: undefined, promptFeedback: { blockReason: 'SAFETY' } }))
    assert.equal(await askClaude({ system: 'S', messages: [{ role: 'user', content: 'Q' }] }), null)
    setGeminiClient(fakeGemini({ text: '', candidates: [{ finishReason: 'PROHIBITED_CONTENT' }] }))
    assert.equal(await askClaude({ system: 'S', messages: [{ role: 'user', content: 'Q' }] }), null)
    setGeminiClient(fakeGemini(() => { throw Object.assign(new Error('quota'), { status: 429 }) }))
    await assert.rejects(askClaude({ system: 'S', messages: [{ role: 'user', content: 'Q' }] }), /quota/)
  })
})

describe('GRADING_DRIVER=sqs', () => {
  const payload = { assignment_id: 'A-1', qp_id: 'QP-1', student_id: 'S1', method: 'ai', persona: 'balanced', partial_credit: true, rubric: [{ question_number: 1, max_marks: 5 }] }

  test('sends the invoke payload, byte for byte, to the grader queue', async () => {
    config.drivers.grading = 'sqs'
    config.grading.queueUrl = 'https://sqs.eu-north-1.amazonaws.com/123/gradeant-grader'
    const sent = []
    setGradingSender((msg) => sent.push(msg))
    await triggerGrading(payload)
    assert.deepEqual(sent, [{ QueueUrl: config.grading.queueUrl, MessageBody: JSON.stringify(payload) }])
  })

  test('nothing is sent while the queue is not configured', async () => {
    config.grading.queueUrl = ''
    const sent = []
    setGradingSender((msg) => sent.push(msg))
    await triggerGrading(payload)
    assert.deepEqual(sent, [])
  })
})
