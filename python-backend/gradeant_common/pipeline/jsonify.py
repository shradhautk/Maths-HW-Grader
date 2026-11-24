"""Answer-sheet PDF → transcript → segregated answers → JSON in the question paper's schema.

Port of lambda_python_files/excute_jsonifier.py (transcription) and jsonifier.py
(JsonifierProcessor.process_file). Prompts are VERBATIM. Differences, by design:
  - the PDF goes to the model as inline bytes (no Files API upload left behind)
  - a model failure in segregation / JSON conversion raises (the message is retried), instead of
    writing the error text into the pipeline as if it were an answer
  - the JSON is written ONCE (the original wrote it twice → the grader ran twice)
  - the root folder comes from the key, not a hard-coded 'Data_warehouse'
"""
import json
import logging

from . import PipelineInputError, pdf, s3io
from .llm import LLMError

log = logging.getLogger(__name__)

# excute_jsonifier.py PROMPT / GENERATION_CONFIG
TRANSCRIBE_PROMPT = "TASK - Transcribe this document. Output- Transcribed document "
TRANSCRIBE_CONFIG = {'max_output_tokens': 5120, 'temperature': 0.6}

# excute_jsonifier.py parse_pdf_with_gemini_text_method improvement_prompt. Written with explicit
# escapes: the original has trailing spaces that editors strip (test_prompts_match_original pins it).
IMPROVE_PROMPT = (
    '\n'
    '        Please analyze and improve the following text extracted from a PDF document. \n'
    '        Your task is to:\n'
    '        1. Clean up any formatting issues or OCR errors\n'
    '        2. Ensure proper sentence structure and paragraph breaks\n'
    '        3. Maintain the original meaning and content\n'
    '        4. Remove any duplicate content that might have occurred during extraction\n'
    '        5. Organize the content in a readable format\n'
    '        \n'
    '        Please provide the cleaned and improved text:\n'
    '        '
)

# jsonifier.py default_config
STRUCTURE_CONFIG = {'max_output_tokens': 8192, 'temperature': 0.6}
JSON_ATTEMPTS = 3


def segregation_prompt(qp_text):
    """jsonifier.py get_prompt_segregation (the transcript is appended after it)."""
    return (
        "You are a physics expert. You are given a Question Paper and a Text. Your task is to segregate the given Text into different answer numbers based on the Question Paper.\n"
        "If you are unsure about the answer number for a particular part of the text, reason based on the Question Paper to find the correct answer number.\n"
        "If you find any misnumbered answers in the provided Text based on the Question Paper, correct the answer number.\n"
        "If the text contains parts that do not have answer numbers, reason based on the Question Paper to assign the correct answer number to those parts.\n"
        "If the text for some subpart of a question is missing, it can be inside other subparts of questions (e.g., 2a contains the answer for 2b, 2c, etc.). Ensure that these parts are correctly identified and segregated.\n"
        "Do not create any subparts that are not in the Question Paper.\n"
        "Do not make any corrections to the text. Do not modify or remove the Text content; segregate it as it is.\n"
        "Return a JSON object mapping answer numbers to their corresponding answers.\n"
        f"#Question Paper:\n{qp_text}\n\n#Text: "
    )


def schema_prompt(schema_text):
    """jsonifier.py get_prompt_schema (the segregated answers are appended after it)."""
    return (
        f"Please fill the given Text in the following JSON Schema. \nStrict Rules:"
        f"1.Schema Compliance: The output must adhere exactly to the provided schema. Do not add or modify any fields beyond what is specified."
        f"2.Do not add/remove anything to 'Text' content. Ensure the full text, including any JSON content within the 'Text' section, is mapped correctly to the corresponding Questions in the JSON Schema reply."
        f"""3.Subparts should always be within a "subparts" array, even if there's only one."""
        f"4.Do not create subparts that are not explicitly mentioned in the input text.\n"
        f"#JSON Schema: {schema_text}\n\n# Text:\n"
    )


def output_to_json(response):
    """jsonifier.py output_to_json → (dict, None) or (None, error)."""
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end == -1 or start >= end:
            return None, 'No valid JSON object found in the response.'
        data = json.loads(response[start:end])
        if not isinstance(data, dict):
            return None, 'The JSON in the response is not an object.'
        return data, None
    except ValueError as exc:
        return None, f'JSONDecodeError: {exc}. Raw response length: {len(response)}'


def transcribe(pdf_bytes, client):
    """excute_jsonifier.process_pdf_combined (method 2 first): the PDF to the model; if that gives
    < 50 characters, PDF text cleaned by the model; if still < 10, the raw PDF text."""
    result = ''
    try:
        result = client.transcribe_pdf(pdf_bytes, TRANSCRIBE_PROMPT, **TRANSCRIBE_CONFIG)
    except LLMError as exc:
        log.warning('transcription from the PDF failed: %s', exc)
    if not result or len(result.strip()) < 50:
        raw = pdf.text_with_pages(pdf_bytes)
        result = ''
        if raw.strip():
            try:
                result = client.generate(f'{IMPROVE_PROMPT}\n\nContent to process:\n{raw}', **TRANSCRIBE_CONFIG)
            except LLMError as exc:
                log.warning('cleaning the PDF text failed: %s', exc)
                result = raw
    if not result or len(result.strip()) < 10:
        result = pdf.text_with_pages(pdf_bytes)
    if not result:
        raise PipelineInputError('No text could be read from the answer sheet')
    return result


def segregate(transcript, qp_text, client):
    return client.generate(segregation_prompt(qp_text) + transcript, **STRUCTURE_CONFIG)


def to_json(segregated, schema_text, client, attempts=JSON_ATTEMPTS):
    """Up to `attempts` model calls until the reply holds a JSON object → (data, last reply)."""
    error, response = 'No attempt made', ''
    for _ in range(attempts):
        response = client.generate(schema_prompt(schema_text) + segregated, **STRUCTURE_CONFIG)
        data, error = output_to_json(response)
        if error is None:
            return data, response
    raise PipelineInputError(f'Failed to generate valid JSON after {attempts} attempts: {error}')


def process(bucket, layout, stem, pdf_bytes, client):
    """Writes Parsed_pdfs/{text,JSON_Seg,JSON_Text}/<stem>.txt, then JSON/<stem>.json once (→ grader)."""
    # Inputs first, so a missing QP fails before any model call (the teacher may still be uploading).
    for key, what in ((layout.qp_pdf, 'Question-paper PDF'), (layout.qp_schema_answer_txt, 'Question-paper answer schema')):
        if not s3io.exists(bucket, key):
            raise PipelineInputError(f'{what} not found: {key}')
    qp_text = pdf.plain_text(s3io.read_bytes(bucket, layout.qp_pdf))
    schema_text = s3io.read_bytes(bucket, layout.qp_schema_answer_txt).decode('utf-8', errors='ignore')

    transcript = transcribe(pdf_bytes, client)
    s3io.write_text(bucket, layout.parsed_text(stem), transcript)

    segregated = segregate(transcript, qp_text, client)
    s3io.write_text(bucket, layout.parsed_seg(stem), segregated)

    data, reply = to_json(segregated, schema_text, client)
    s3io.write_text(bucket, layout.parsed_json_text(stem), reply)
    s3io.write_text(bucket, layout.parsed_json(stem), json.dumps(data, indent=2, ensure_ascii=False), 'application/json')
    return layout.parsed_json(stem)
