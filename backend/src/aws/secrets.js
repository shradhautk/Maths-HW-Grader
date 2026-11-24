// Secrets come from AWS Secrets Manager in production (the Lambda Web
// Functions guide says not to keep secrets in revision environment variables)
// and from .env.<mode>.local in development. Each secret is fetched once per
// execution environment, lazily, so startup stays well under the 10 s limit.
import { GetSecretValueCommand, SecretsManagerClient } from '@aws-sdk/client-secrets-manager'
import config from '../config/index.js'

let client
const cache = new Map()

async function fetchSecret(arn, key) {
  client ??= new SecretsManagerClient({ region: config.aws.region })
  const res = await client.send(new GetSecretValueCommand({ SecretId: arn }))
  const raw = res.SecretString ?? ''
  // Accept a plain string secret or a JSON secret holding {"<key>": "..."}.
  try {
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === 'object' && typeof parsed[key] === 'string') return parsed[key]
  } catch {
    /* plain string */
  }
  return raw
}

function resolve(key, value, arn) {
  if (value) return Promise.resolve(value)
  // secrets: env (app.yaml deployment providers): hosts like Render have no Secrets Manager;
  // the value must be in the environment, and an ARN is never fetched.
  if (!arn || config.drivers.secrets === 'env') return Promise.resolve('')
  if (!cache.has(key)) {
    const pending = fetchSecret(arn, key).catch((err) => {
      cache.delete(key) // retry on the next request instead of caching the failure
      throw err
    })
    cache.set(key, pending)
  }
  return cache.get(key)
}

export const getJwtSecret = () => resolve('JWT_SECRET', config.auth.jwtSecret, config.auth.jwtSecretArn)

export const getGeminiApiKey = () => resolve('GEMINI_API_KEY', config.gemini.apiKey, config.gemini.apiKeyArn)

export const getSessionTokenSecret = () =>
  resolve('SESSION_TOKEN_SECRET', config.auth.sessionTokenSecret, config.auth.sessionTokenSecretArn)
