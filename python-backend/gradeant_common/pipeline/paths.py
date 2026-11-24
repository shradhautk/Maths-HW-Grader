"""The pipeline's S3 key layout, in one place.

Same keys as the originals (grader_lambda.py GradingPipeline._validate_paths and
jsonifier.py JsonifierProcessor.setup_s3_paths):

    <root>/students/assignments/<aid>__<qp>/original_pdf/<student>.pdf      uploaded answer sheet
    <root>/students/assignments/<aid>__<qp>/Parsed_pdfs/{text,JSON_Seg,JSON_Text}/<name>.txt
    <root>/students/assignments/<aid>__<qp>/Parsed_pdfs/JSON/<name>.json    → triggers the grader
    <root>/teachers/QP_Rubric/<aid>__<qp>/QP_pdfs/<qp>.pdf
    <root>/teachers/QP_Rubric/<aid>__<qp>/QP_schemas/QP_schemas_answers/<qp>_schema_answer.txt
    <root>/teachers/QPRubricWithAnswers/<aid>__<qp>/solution_pdfs/<aid>_<qp>_answer.pdf
    <base>/QP_schemas/QP_schemas_json/<qp>_schema.json     base = QPRubricWithAnswers/… if the
    <base>/rubric_jsons/<qp>_rubric.json                   solution PDF exists, else QP_Rubric/…
"""
import re
from dataclasses import dataclass

from ..config import env
from . import PipelineInputError

KINDS = {'original_pdf': 'original_pdf', 'Parsed_pdfs/JSON': 'parsed_json'}
_STEM = re.compile(r'^[A-Za-z0-9._-]{1,200}$')


def safe_stem(name):
    """excute_jsonifier.safe_file_name ('+' and ' ' → '_'), then any other character that can't be
    part of a student id → '_'. Used for uploaded PDFs, whose names the legacy uploader didn't choose."""
    stem = name.replace('+', '_').replace(' ', '_')
    return re.sub(r'[^A-Za-z0-9._-]', '_', stem)[:200]


def valid_id(value):
    """Assignment / question-paper / student ids that may appear in keys and rows."""
    return isinstance(value, str) and bool(_STEM.match(value))


def pipeline_bucket():
    """The one bucket the pipeline reads and writes. Required: no default."""
    bucket = env('PIPELINE_BUCKET')
    if not bucket:
        raise PipelineInputError('PIPELINE_BUCKET is not set.')
    return bucket


def pipeline_root():
    """Top-level folder in the bucket (a folder name, not a secret)."""
    return env('PIPELINE_ROOT', 'Data_warehouse')


@dataclass(frozen=True)
class Layout:
    root: str
    assignment_id: str
    qp_id: str

    @property
    def folder(self):
        return f'{self.assignment_id}__{self.qp_id}'

    # --- student side ------------------------------------------------------
    @property
    def student_base(self):
        return f'{self.root}/students/assignments/{self.folder}'

    @property
    def original_pdf_prefix(self):
        return f'{self.student_base}/original_pdf/'

    def original_pdf(self, student_id):
        return f'{self.original_pdf_prefix}{student_id}.pdf'

    def parsed_text(self, name):
        return f'{self.student_base}/Parsed_pdfs/text/{name}.txt'

    def parsed_seg(self, name):
        return f'{self.student_base}/Parsed_pdfs/JSON_Seg/{name}.txt'

    def parsed_json_text(self, name):
        return f'{self.student_base}/Parsed_pdfs/JSON_Text/{name}.txt'

    @property
    def parsed_json_prefix(self):
        return f'{self.student_base}/Parsed_pdfs/JSON/'

    def parsed_json(self, name):
        return f'{self.parsed_json_prefix}{name}.json'

    # --- teacher side ------------------------------------------------------
    @property
    def qp_rubric_base(self):
        return f'{self.root}/teachers/QP_Rubric/{self.folder}'

    @property
    def with_answers_base(self):
        return f'{self.root}/teachers/QPRubricWithAnswers/{self.folder}'

    @property
    def qp_pdf(self):
        return f'{self.qp_rubric_base}/QP_pdfs/{self.qp_id}.pdf'

    @property
    def qp_schema_answer_txt(self):
        return f'{self.qp_rubric_base}/QP_schemas/QP_schemas_answers/{self.qp_id}_schema_answer.txt'

    @property
    def solution_pdf(self):
        return f'{self.with_answers_base}/solution_pdfs/{self.assignment_id}_{self.qp_id}_answer.pdf'

    def rubric_base(self, has_solution):
        return self.with_answers_base if has_solution else self.qp_rubric_base

    def qp_schema_json(self, has_solution):
        return f'{self.rubric_base(has_solution)}/QP_schemas/QP_schemas_json/{self.qp_id}_schema.json'

    def rubric_json(self, has_solution):
        return f'{self.rubric_base(has_solution)}/rubric_jsons/{self.qp_id}_rubric.json'


def parse_student_key(key):
    """'<root>/students/assignments/<aid>__<qp>/<kind>/<file>' → (Layout, kind, stem).

    kind is 'original_pdf' or 'parsed_json'; anything else raises PipelineInputError. The stem
    (file name without extension) is the student id: for an uploaded PDF it is made safe with
    safe_stem() (as the original did); a parsed JSON name must already be a plain id.
    """
    parts = (key or '').split('/')
    if len(parts) < 6 or parts[1:3] != ['students', 'assignments'] or '__' not in parts[3]:
        raise PipelineInputError(f'Not a student pipeline key: {key!r}')
    sub = '/'.join(parts[4:-1])
    kind = KINDS.get(sub)
    if not kind:
        raise PipelineInputError(f'Unsupported pipeline folder {sub!r} in {key!r}')
    assignment_id, qp_id = parts[3].split('__', 1)
    assignment_id = assignment_id.split('__')[0]
    filename = parts[-1]
    stem, dot, ext = filename.rpartition('.')
    expected = 'pdf' if kind == 'original_pdf' else 'json'
    if not dot or ext.lower() != expected:
        raise PipelineInputError(f'Expected a .{expected} file: {key!r}')
    if kind == 'original_pdf':
        stem = safe_stem(stem)
    if not assignment_id or not qp_id or not _STEM.match(stem):
        raise PipelineInputError(f'Bad assignment, question paper or student id in {key!r}')
    return Layout(parts[0], assignment_id, qp_id), kind, stem
