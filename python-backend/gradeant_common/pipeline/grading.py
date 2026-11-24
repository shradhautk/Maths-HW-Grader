"""Grading core: QA pairs → model grades → assignment_grading rows.

Port of grader_lambda.py GradingPipeline (extract_qa_pairs, prompt builders, evaluate_answer,
process_submission) with the plan's fixes:
  - a model failure or an unparseable reply never becomes 0 marks: GradingError, nothing written
  - marks are clamped to [0, max_marks]
  - row ids are deterministic (uuid5), so a re-run overwrites instead of duplicating
  - the questions of one submission are graded in parallel (threads; the old asyncio was sequential)
"""
import datetime
import difflib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .. import grading as grading_table
from ..config import env_float, env_int
from . import PipelineInputError, prompts
from .llm import LLMError


class GradingError(Exception):
    """A question could not be graded; the whole submission fails and nothing is written."""


def temperature():
    return env_float('GRADER_TEMPERATURE', 0.6)


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(n):
    return int(n) if float(n).is_integer() else n


# ---------------------------------------------------------------------------
# QA pairs (grader_lambda.py L2082-2172)
# ---------------------------------------------------------------------------

def _build_pair(*, question, answer, rubric, max_marks, question_number, subpart, answer_with):
    return {
        'question': question,
        'student_answer': answer,
        'rubric': rubric,
        'max_marks': max_marks,
        'question_number': question_number,
        'subpart': subpart,
        'answer_with': answer_with,
    }


def extract_qa_pairs(qp_json, sol_json, standard_sol_json=None, *, include_std_answer=False):
    """Same pairs as the original; a malformed QP raises PipelineInputError instead of returning []."""
    try:
        qp_questions = qp_json['questions']
        sol_answers = sol_json.get('questions', [])
        std_answers = standard_sol_json.get('questions', []) if standard_sol_json else []
        answer_with = 'standard_answer' if include_std_answer else 'without_standard_answer'

        qa_pairs = []
        for i, qp_q in enumerate(qp_questions):
            sol_a = sol_answers[i] if i < len(sol_answers) else {}
            std_a = std_answers[i] if i < len(std_answers) else {}

            if 'subparts' in qp_q:
                for j, qp_sub in enumerate(qp_q['subparts']):
                    sol_sub = sol_a.get('subparts', [])[j] if j < len(sol_a.get('subparts', [])) else {}
                    std_sub = std_a.get('subparts', [])[j] if j < len(std_a.get('subparts', [])) else {}
                    pair = _build_pair(
                        question=qp_sub['description'],
                        answer=sol_sub.get('reply', ''),
                        rubric=std_sub.get('rubric', '') if std_sub else None,
                        max_marks=qp_sub['max_marks'],
                        question_number=qp_q['question_number'],
                        subpart=qp_sub.get('subpart'),
                        answer_with=answer_with,
                    )
                    if include_std_answer:
                        pair['standard_answer'] = std_sub.get('standard_answer', '') if std_sub else None
                    qa_pairs.append(pair)
            else:
                pair = _build_pair(
                    question=qp_q.get('description', ''),
                    answer=sol_a.get('reply', ''),
                    rubric=std_a.get('rubric', '') if std_a else None,
                    max_marks=qp_q['max_marks'],
                    question_number=qp_q['question_number'],
                    subpart=None,
                    answer_with=answer_with,
                )
                if include_std_answer:
                    pair['standard_answer'] = std_a.get('standard_answer', '') if std_a else None
                qa_pairs.append(pair)
        return qa_pairs
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise PipelineInputError(f'Question paper / answer JSON has an unexpected shape: {type(exc).__name__}: {exc}') from exc


# ---------------------------------------------------------------------------
# Prompts (grader_lambda.py L2221-2259)
# ---------------------------------------------------------------------------

def build_eval_prompt(qa_item, persona='balanced', partial_credit=True):
    """Identical to the original for the default persona with partial credit on."""
    persona_text = prompts.PERSONAS.get(persona or 'balanced', prompts.PERSONAS['balanced'])
    has_std = bool(qa_item.get('standard_answer'))
    instructions = prompts.STD_ANSWER_INSTRUCTIONS if has_std else prompts.DEFAULT_INSTRUCTIONS
    if partial_credit is False:
        instructions += prompts.NO_PARTIAL_CREDIT

    rubric_section = f"'#Rubric':\n{qa_item['rubric']},\n" if qa_item.get('rubric') else ''
    std_section = f"#Standard answer:\n{qa_item['standard_answer']},\n" if has_std else ''

    return (
        f'You are -{persona_text}\n\n'
        f'#Instructions for evaluation:\n{instructions}\n'
        f'{{\n'
        f"'#Question':\n{qa_item['question']},\n"
        f'{std_section}'
        f"'#Student\\'s Answer':\n{qa_item['student_answer']},\n"
        f'{rubric_section}'
        f"'#Maximum Marks':\n{qa_item['max_marks']}\n"
        f'}}\n\n'
        '## Output Format:\n'
        'Return ONLY a valid JSON object. No markdown, no explanations.\n'
        '```json\n'
        '{{\n  "marks_awarded": <int>,\n  "reasoning": "<string>"\n}}\n'
        '```'
    )


def build_feedback_prompt(qa_item, marks, reasoning):
    has_std = bool(qa_item.get('standard_answer'))
    base = prompts.FEEDBACK_PROMPT_STD if has_std else prompts.FEEDBACK_PROMPT
    std_line = f"'#Standard Answer': {qa_item['standard_answer']}\n" if has_std else ''
    return (
        f'{base}\n'
        f"'#Question': {qa_item['question']}\n"
        f'{std_line}'
        f"'#Student\\'s Answer': {qa_item['student_answer']}\n"
        f"'#Marks Awarded': {marks}\n"
        f"'#Reasoning': {reasoning}\n"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def parse_marks(text, max_marks):
    """'{... "marks_awarded": n, "reasoning": "..."}' → (marks, reasoning). ValueError if unusable."""
    start, end = text.find('{'), text.rfind('}') + 1
    if start == -1 or end == 0 or end <= start:
        raise ValueError('No JSON object in the reply')
    blob = text[start:end]
    try:
        data = json.loads(blob)
    except ValueError:
        # The prompt's format block shows '{{ ... }}'; a model may echo the doubled braces.
        if blob.startswith('{{') and blob.endswith('}}'):
            data = json.loads(blob[1:-1])
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError('Reply JSON is not an object')
    marks = data.get('marks_awarded')
    if isinstance(marks, bool) or marks is None:
        raise ValueError('marks_awarded missing')
    if isinstance(marks, str):
        marks = float(marks.strip())
    if not isinstance(marks, (int, float)) or marks != marks:
        raise ValueError('marks_awarded is not a number')
    cap = _num(max_marks)
    marks = max(0.0, float(marks))
    if cap is not None:
        marks = min(marks, cap)
    reasoning = data.get('reasoning')
    return _clean(marks), '' if reasoning is None else str(reasoning)


def evaluate(qa_item, client, *, persona='balanced', partial_credit=True, attempts=3):
    """One graded copy of qa_item (+ marks, reasoning, feedback). Raises GradingError."""
    prompt = build_eval_prompt(qa_item, persona, partial_credit)
    marks = reasoning = None
    last = None
    for _ in range(attempts):
        try:
            raw = client.generate(prompt, temperature=temperature())
        except LLMError as exc:
            raise GradingError(f'Q{qa_item.get("question_number", "?")}: {exc}') from exc
        try:
            marks, reasoning = parse_marks(raw, qa_item.get('max_marks'))
            break
        except ValueError as exc:
            last = exc
    else:
        raise GradingError(f'Q{qa_item.get("question_number", "?")}: unusable model reply after {attempts} attempts ({last})')

    result = dict(qa_item)
    result['marks'] = marks
    result['reasoning'] = reasoning
    try:
        result['feedback'] = client.generate(build_feedback_prompt(qa_item, marks, reasoning), temperature=temperature())
    except LLMError:
        result['feedback'] = prompts.FEEDBACK_ERROR  # non-fatal, as in the original
    return result


def grade_pairs(pairs, client, *, persona='balanced', partial_credit=True):
    """Grades every pair in parallel; results in input order; the first failure propagates."""
    if not pairs:
        return []
    workers = max(1, min(len(pairs), env_int('GRADER_MAX_WORKERS', 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: evaluate(p, client, persona=persona, partial_credit=partial_credit), pairs))


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def id_part(value):
    if value is None or value == '':
        return '-'
    n = _num(value)
    if n is not None and not isinstance(value, bool):
        return str(_clean(n))  # 1, '1', Decimal('1') → '1'
    return str(value).strip()


def row_id(assignment_id, qp_id, student_id, question_number, subpart):
    key = '|'.join(id_part(x) for x in (assignment_id, qp_id, student_id, question_number, subpart))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'gradeant:' + key))


def summary_id(assignment_id, qp_id, student_id):
    key = '|'.join(id_part(x) for x in (assignment_id, qp_id, student_id)) + '|summary'
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'gradeant:' + key))


def detail_rows(results, assignment_id, qp_id, student_id, *, graded):
    """One row per pair (the original's fields). Ungraded rows carry no marks/reasoning/feedback.
    ocr_answer keeps the text as read from the paper; student corrections change only student_answer."""
    stamp, graded_at = int(time.time()), now_iso()
    rows = []
    for r in results:
        row = dict(r)
        if isinstance(row.get('student_answer'), str) and 'ocr_answer' not in row:
            row['ocr_answer'] = row['student_answer']
        if not graded:
            for k in ('marks', 'reasoning', 'feedback'):
                row.pop(k, None)
        else:
            row['graded_at'] = graded_at
        row.update(
            evaluation_id=row_id(assignment_id, qp_id, student_id, r.get('question_number'), r.get('subpart')),
            assignment_id=assignment_id,
            qp_id=qp_id,
            student_id=student_id,
            timestamp=stamp,
        )
        rows.append(row)
    return rows


CORRECTION_FLAG_REASON = 'answer_rewritten_after_ocr'
_COMPARE_CHARS = 4000  # cap the text compared (difflib is quadratic in the worst case)


def correction_similarity(ocr_answer, student_answer):
    """How much of the paper's text survives the student's correction: 1.0 = unchanged, 0.0 = rewritten.
    None when there is nothing to compare (no OCR text kept, e.g. a row from the legacy grader)."""
    if not isinstance(ocr_answer, str) or not isinstance(student_answer, str):
        return None
    a, b = ' '.join(ocr_answer.split())[:_COMPARE_CHARS], ' '.join(student_answer.split())[:_COMPARE_CHARS]
    if a == b:
        return 1.0
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def needs_review(ocr_answer, student_answer):
    """(flag, similarity): flag is True when a student's correction changed the answer too much to be
    an OCR fix (similarity below CORRECTION_SIMILARITY_MIN, default 0.6)."""
    similarity = correction_similarity(ocr_answer, student_answer)
    return similarity is not None and similarity < env_float('CORRECTION_SIMILARITY_MIN', 0.6), similarity


def summary_row(rows, assignment_id, qp_id, student_id, *, answer_with, method=None, persona=None):
    total = sum(float(_num(r.get('marks')) or 0.0) for r in rows)
    row = {
        'type': 'summary',
        'evaluation_id': summary_id(assignment_id, qp_id, student_id),
        'assignment_id': assignment_id,
        'qp_id': qp_id,
        'student_id': student_id,
        'total_marks': _clean(total),
        'timestamp': int(time.time()),
        'answer_with': answer_with,
        'graded_at': now_iso(),
    }
    if method:
        row['method'] = method
    if persona:
        row['persona'] = persona
    return row


def write_student(assignment_id, qp_id, student_id, rows, summary=None):
    """Replaces this student's rows for the assignment: stale rows (old uuid4 rows, an old summary)
    are deleted, details written, the summary written LAST (readers treat it as 'graded')."""
    existing = grading_table.student_rows(student_id, assignment_id=assignment_id, qp_id=qp_id)
    keep = {r['evaluation_id'] for r in rows} | ({summary['evaluation_id']} if summary else set())
    stale = [r['evaluation_id'] for r in existing if r.get('evaluation_id') not in keep]
    grading_table.replace_rows(rows, stale)  # DynamoDB batch, or one Supabase transaction
    if summary:
        grading_table.replace_rows([summary])
    return {'written': len(rows) + bool(summary), 'deleted': len(stale)}
