// The AI behind the GradeFlow assistant and rubric drafting: Gemini (Google Gen AI SDK,
// @google/genai) with the one model used everywhere in GradeAnt (GEMINI_MODEL, default
// gemini-3.5-flash-preview). The prompts are the callers' own, unchanged from
// lambda_python_files/workflow.py (python-backend/gradeant_common/ai.py is the Python twin).
//
//   AI_DRIVER=gemini   (deployment providers.ai: gemini)  calls Gemini
//   AI_DRIVER=canned   (development, tests, providers.ai: off)  never calls a model
//
// askAI returns the reply text, or null when Gemini declines (blocked prompt / safety finish);
// callers then show a fixed message or keep the default rubric. `outputFormat` (a json_schema
// format) asks for JSON through responseJsonSchema; if Gemini refuses the schema (400) the
// schema goes into the prompt instead. `effort` has no Gemini equivalent and is ignored.
import { getGeminiApiKey } from '../aws/secrets.js'
import config from '../config/index.js'

let override
let gemini // GoogleGenAI client, created on first use

const DECLINED = new Set(['SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'RECITATION', 'IMAGE_SAFETY'])

const textOf = (content) =>
  typeof content === 'string' ? content : (Array.isArray(content) ? content : []).filter((b) => b?.type === 'text').map((b) => b.text).join('')

async function client() {
  if (!gemini) {
    const apiKey = await getGeminiApiKey()
    if (!apiKey) throw new Error('Gemini API key not configured (GEMINI_API_KEY or GEMINI_API_KEY_ARN)')
    const { GoogleGenAI } = await import('@google/genai')
    gemini = new GoogleGenAI({ apiKey })
  }
  return gemini
}

export async function askAI({ system, messages, effort, maxTokens = 16000, outputFormat }) {
  if (override) return override({ system, messages, effort, maxTokens, outputFormat })
  if (config.drivers.ai !== 'gemini') {
    // canned: keep the default rubric; give the assistant a clear "not connected" reply.
    return outputFormat ? null : '(Local dev server — the AI is not called. Set AI_DRIVER=gemini to use Gemini.)'
  }
  const ai = await client()
  const contents = messages.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: textOf(m.content) }] }))
  const generate = (systemInstruction, structured) =>
    ai.models.generateContent({
      model: config.gemini.model,
      contents,
      config: {
        systemInstruction,
        maxOutputTokens: maxTokens,
        ...(structured ? { responseMimeType: 'application/json', responseJsonSchema: outputFormat.schema } : {}),
      },
    })
  let response
  try {
    response = await generate(system, Boolean(outputFormat))
  } catch (err) {
    if (!outputFormat || err?.status !== 400) throw err
    response = await generate(
      `${system}\n\nRespond with only a JSON object matching this JSON Schema, and no other text:\n${JSON.stringify(outputFormat.schema)}`,
      false,
    )
  }
  if (response.promptFeedback?.blockReason) return null
  if (DECLINED.has(response.candidates?.[0]?.finishReason)) return null
  return response.text ?? null
}

/** Tests replace the model with a fake function; pass undefined to restore. */
export function setAIOverride(fn) {
  override = fn
}

/** Tests: a fake GoogleGenAI client ({ models: { generateContent } }); undefined restores the real one. */
export function setGeminiClient(fake) {
  gemini = fake
}
