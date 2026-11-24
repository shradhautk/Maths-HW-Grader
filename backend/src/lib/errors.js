// HttpError carries a status and a message to the client. Each router decides
// whether the message goes out as {"error": ...} or {"message": ...}, matching
// the Python Lambda it replaces.
export class HttpError extends Error {
  constructor(status, message, { body } = {}) {
    super(message)
    this.status = status
    // When set, sent as the whole response body instead of {<key>: message}.
    this.body = body
  }
}

export const badRequest = (message) => new HttpError(400, message)
export const unauthorized = (message) => new HttpError(401, message)
export const forbidden = (message) => new HttpError(403, message)
export const notFound = (message) => new HttpError(404, message)
export const conflict = (message) => new HttpError(409, message)

/** A non-200 success-ish result (207 partial updates, 404 "nothing updated" with details). */
export class Reply {
  constructor(status, body) {
    this.status = status
    this.body = body
  }
}
export const reply = (status, body) => new Reply(status, body)

/** DynamoDB's conditional-check failure, regardless of client (SDK v3 or dynalite). */
export const isConditionalCheckFailed = (err) => err?.name === 'ConditionalCheckFailedException'
