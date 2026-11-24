"""POST /uploadpdf — Lambda gradeant-upload (handler upload.lambda_handler).

Presigned S3 PUT URLs for answer sheets and question papers; file bytes go straight
from the browser to S3. The endpoint stays off (503) until UPLOAD_BUCKET is set.
Errors are {"message": ...}. Response: {"urls": [...], "fileIds": [...], "uploadId": "upload-<uuid>"}.

UPLOAD_LAYOUT picks the object keys:
  template (default)  keys follow UPLOAD_KEY_TEMPLATE, as backend/src/services/upload.service.js
  pipeline            the grading pipeline's layout (gradeant_common.pipeline.paths), with the file
                      names the pipeline reads, chosen by the server (port of lambda_python_files/
                      upload_pdf.py, minus its gaps: it had no auth and kept user file names):
                        student → students/assignments/<aid>__<qp>/original_pdf/<studentId>.pdf
                        teacher → teachers/QP_Rubric/<aid>__<qp>/QP_pdfs/<qp>.pdf
                        answer  → teachers/QPRubricWithAnswers/<aid>__<qp>/solution_pdfs/<aid>_<qp>_answer.pdf
                      Exactly one file; the assignment must exist; UPLOAD_BUCKET must be PIPELINE_BUCKET.

Teacher/answer uploads (and any upload by a teacher in pipeline layout) need a teacher who owns
the assignment (admins pass). The body may be wrapped as {"metadata": {...}}, as the original accepted.
"""
import os
import re
import uuid

import boto3
from botocore.config import Config

from gradeant_common import config, dynamo
from gradeant_common.auth import authenticate
from gradeant_common.http import ApiError, create_response, handle, parse_body
from gradeant_common.ownership import can_manage
from gradeant_common.permissions import require_permission
from gradeant_common.pipeline import paths as pipeline_paths

ALLOWED_TYPES = {'application/pdf'}
UPLOAD_TYPES = {'student', 'teacher', 'answer'}
MAX_FILES = 20


def _safe(value):
    return re.sub(r'[^A-Za-z0-9._-]', '_', '' if value is None else str(value))[:200] or '_'


def object_key(meta, file_name):
    values = {
        'uploadType': _safe(meta.get('uploadType')),
        'assignmentId': _safe(meta.get('assignmentId')),
        'qpId': _safe(meta.get('qpId')),
        'studentId': _safe(meta.get('studentId')),
        'courseId': _safe(meta.get('courseId')),
        'fileName': _safe(file_name),
        'uuid': str(uuid.uuid4()),
    }
    template = config.env('UPLOAD_KEY_TEMPLATE', 'uploads/{uploadType}/{assignmentId}/{fileName}')
    return re.sub(r'\{(\w+)\}', lambda m: values.get(m.group(1), m.group(0)), template)


def _size(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _layout_mode():
    mode = config.env('UPLOAD_LAYOUT', 'template')
    if mode not in ('template', 'pipeline'):
        raise ApiError(503, 'File uploads are misconfigured on this server.')
    return mode


def _load_assignment(assignment_id):
    if not isinstance(assignment_id, str) or not assignment_id:
        return None
    return dynamo.table('assignments').get_item(Key={'assignment_id': assignment_id}).get('Item')


def _pipeline_keys(body, caller, upload_type, assignment, files):
    """Object keys in the pipeline layout (one file per request; the server picks the name)."""
    if config.env('PIPELINE_BUCKET') != config.env('UPLOAD_BUCKET'):
        raise ApiError(503, 'File uploads are misconfigured on this server.')
    if len(files) != 1:
        raise ApiError(400, 'Send exactly one PDF per upload.')
    assignment_id = body.get('assignmentId')
    if not assignment:
        raise ApiError(404, 'Assignment not found.')
    client_qp = body.get('qpId') or body.get('courseId')
    qp_id = assignment.get('qp_id') or client_qp
    if client_qp and assignment.get('qp_id') and client_qp != assignment['qp_id']:
        raise ApiError(400, "qpId doesn't match the assignment's question paper.")
    if not pipeline_paths.valid_id(assignment_id) or not pipeline_paths.valid_id(qp_id):
        raise ApiError(400, 'assignmentId and qpId may only contain letters, digits, ".", "_" and "-".')
    layout = pipeline_paths.Layout(pipeline_paths.pipeline_root(), assignment_id, qp_id)
    if upload_type == 'student':
        student_id = caller['user_id'] if caller['role'] == 'student' else body.get('studentId')
        if not pipeline_paths.valid_id(student_id):
            raise ApiError(400, 'studentId is required (letters, digits, ".", "_" and "-").')
        return [layout.original_pdf(student_id)]
    if upload_type == 'teacher':
        return [layout.qp_pdf]
    return [layout.solution_pdf]


def presign_uploads(body, caller):
    bucket = config.env('UPLOAD_BUCKET')
    if not bucket:
        raise ApiError(503, 'File uploads are not configured on this server.')
    if isinstance(body.get('metadata'), dict):  # the original upload Lambda's wrapped form
        body = body['metadata']
    mode = _layout_mode()
    upload_type = body.get('uploadType')
    if upload_type not in UPLOAD_TYPES:
        raise ApiError(400, 'uploadType must be student, teacher or answer.')
    if caller['role'] == 'student' and upload_type != 'student':
        raise ApiError(403, 'Students can only upload answer sheets.')
    # config/app.yaml permissions.upload.<uploadType>
    require_permission('upload', upload_type, caller['role'], "Your role can't upload this kind of file.")
    if caller['role'] == 'student' and body.get('studentId') and body.get('studentId') != caller['user_id']:
        raise ApiError(403, 'You can only upload your own answers.')
    assignment = _load_assignment(body.get('assignmentId'))
    if caller['role'] == 'student':
        # Only for a published assignment of one of the student's own classes.
        classes = (caller.get('user') or {}).get('classes') or []
        if not assignment or assignment.get('content_status') != 'published' or assignment.get('class_name') not in classes:
            raise ApiError(403, 'You can only upload answers for assignments in your own classes.')
    elif upload_type in ('teacher', 'answer') or mode == 'pipeline':
        # Question papers, solutions (and, in pipeline layout, sheets uploaded for a student) belong
        # to the assignment's teacher. Legacy flows may upload before the assignment exists: allowed
        # in template layout only.
        if assignment and not can_manage(caller, body.get('assignmentId')):
            raise ApiError(403, 'This assignment belongs to a different teacher.')

    max_bytes = config.env_int('UPLOAD_MAX_BYTES', 10 * 1024 * 1024)
    files = body.get('files') if isinstance(body.get('files'), list) else []
    if not files or len(files) > MAX_FILES:
        raise ApiError(400, f'Send between 1 and {MAX_FILES} files.')
    for f in files:
        if not isinstance(f, dict) or not isinstance(f.get('name'), str) or not f['name']:
            raise ApiError(400, 'Every file needs a name.')
        if f.get('type') not in ALLOWED_TYPES:
            raise ApiError(400, f"{f['name']}: only PDF files can be uploaded.")
        size = _size(f.get('size'))
        if not size > 0 or size > max_bytes:
            raise ApiError(400, f"{f['name']}: files must be under {round(max_bytes / 1048576)} MB.")

    if mode == 'pipeline':
        keys = _pipeline_keys(body, caller, upload_type, assignment, files)
    else:
        meta = dict(body)
        if meta.get('studentId') is None and caller['role'] == 'student':
            meta['studentId'] = caller['user_id']
        keys = [object_key(meta, f['name']) for f in files]
    ttl = config.env_int('UPLOAD_URL_TTL_SECONDS', 900)
    region = config.env('PIPELINE_BUCKET_REGION') if mode == 'pipeline' else ''
    s3 = boto3.client('s3', region_name=region or config.data_region(), config=Config(signature_version='s3v4'))
    # The validated size is part of the signature (ContentLength), so a client can't
    # declare a small file and then upload a huge one.
    urls = [
        s3.generate_presigned_url(
            'put_object',
            Params={'Bucket': bucket, 'Key': key, 'ContentType': f['type'], 'ContentLength': int(_size(f['size']))},
            ExpiresIn=ttl,
        )
        for key, f in zip(keys, files)
    ]
    return {'urls': urls, 'fileIds': [os.path.basename(k) for k in keys], 'uploadId': f'upload-{uuid.uuid4()}'}


def _route(event):
    if event.get('path') != config.api_paths().get('upload', '/uploadpdf') or (event.get('httpMethod') or 'POST') != 'POST':
        return create_response(404, {'message': 'Not Found'})
    body = parse_body(event)
    return create_response(200, presign_uploads(body, authenticate(event)))


def lambda_handler(event, context):
    return handle(event, context, _route, key='message', internal='Could not prepare the upload.')
