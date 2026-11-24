"""PDF text with PyMuPDF (fitz). Imported lazily: only the worker layer ships PyMuPDF."""


def text_with_pages(data):
    """Student answer sheets: '--- Page n ---' separators, as excute_jsonifier.extract_text_from_pdf_fallback.
    '' when the PDF can't be read (scans have no text layer)."""
    import fitz
    try:
        doc = fitz.open(stream=data, filetype='pdf')
    except Exception:  # noqa: BLE001 — a broken PDF has no text; the caller decides
        return ''
    text = ''
    try:
        for page_num, page in enumerate(doc):
            try:
                text += f'\n--- Page {page_num + 1} ---\n'
                text += page.get_text('text')
            except Exception:  # noqa: BLE001 — skip an unreadable page, as the original
                continue
    finally:
        doc.close()
    return text.strip()


def plain_text(data):
    """Question papers: pages concatenated, as jsonifier.JsonifierProcessor.extract_text_from_pdf_s3."""
    import fitz
    try:
        doc = fitz.open(stream=data, filetype='pdf')
    except Exception:  # noqa: BLE001
        return ''
    try:
        return ''.join(doc[i].get_text() for i in range(len(doc)))
    finally:
        doc.close()
