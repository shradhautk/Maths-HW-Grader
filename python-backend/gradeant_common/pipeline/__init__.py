"""The grading pipeline (PDF → JSON → graded rows), shared by the gradeant-jsonifier and
gradeant-grader workers. Port of lambda_python_files/{excute_jsonifier,jsonifier,grader_lambda}.py.

    paths    the one S3 key layout (students/…, teachers/QP_Rubric|QPRubricWithAnswers/…)
    s3io     S3 reads/writes for the pipeline bucket
    llm      LLM client interface (Gemini first, google-genai SDK)

Rule: `fitz` (PyMuPDF) and `google.genai` are imported only inside the functions that use
them. This layer is shared with the API functions, which don't ship those packages.
"""


class PipelineInputError(ValueError):
    """The input (key, QP, rubric, student JSON) can't be processed. The message fails and is retried, then dead-lettered."""
