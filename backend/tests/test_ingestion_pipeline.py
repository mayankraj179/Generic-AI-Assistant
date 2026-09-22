from __future__ import annotations

import pytest

from app.ingestion.pipeline import ParseError, capture_access, chunk, discover, parse, run_pipeline
from tests.fixtures.doc_builders import write_minimal_docx, write_minimal_pdf, write_minimal_xlsx


def test_pipeline_discovers_and_processes_files(tmp_path):
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    policy_file = docs_root / "policy.txt"
    policy_file.write_text(
        "Alpha paragraph.\n\nBeta paragraph with more content.\n\nGamma paragraph.",
        encoding="utf-8",
    )

    discovered = discover(str(docs_root))
    assert len(discovered) == 1
    assert discovered[0].uri == str(policy_file)
    assert discovered[0].source_type == "filesystem"

    labeled_source = discovered[0]
    labeled_source = type(labeled_source)(
        uri=f"{policy_file}?labels=role:employee,dept:hr",
        source_type=labeled_source.source_type,
    )
    with_access = capture_access(labeled_source)
    assert with_access.access_labels == frozenset({"role:employee", "dept:hr"})

    parsed = parse(with_access)
    assert parsed.title == "policy"
    assert "Alpha paragraph" in parsed.raw_text
    assert len(parsed.content_hash) == 64  # sha256 hex digest

    chunks = chunk(parsed)
    assert len(chunks) >= 1
    assert chunks[0].document_title == "policy"
    assert chunks[0].access_labels == with_access.access_labels
    assert chunks[0].display_text == parsed.raw_text or chunks[0].display_text in parsed.raw_text


def test_run_pipeline_combines_stage_outputs(tmp_path):
    docs_root = tmp_path / "documents"
    docs_root.mkdir()
    first = docs_root / "first.txt"
    first.write_text("First file text", encoding="utf-8")
    second = docs_root / "second.md"
    second.write_text("Second file text", encoding="utf-8")

    chunks = run_pipeline(str(docs_root))
    assert len(chunks) == 2
    titles = {chunk.document_title for chunk in chunks}
    assert {"first", "second"}.issubset(titles)


# ---------------------------------------------------------------------------
# Content hash: same text -> same hash, different text -> different hash
# ---------------------------------------------------------------------------


def test_content_hash_is_stable_for_identical_text(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("Identical policy content.", encoding="utf-8")
    file_b.write_text("Identical policy content.", encoding="utf-8")

    parsed_a = parse(capture_access(discover(str(file_a))[0]))
    parsed_b = parse(capture_access(discover(str(file_b))[0]))

    assert parsed_a.content_hash == parsed_b.content_hash


def test_content_hash_changes_with_text(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("Original content.", encoding="utf-8")
    original_hash = parse(capture_access(discover(str(file_path))[0])).content_hash

    file_path.write_text("Changed content.", encoding="utf-8")
    changed_hash = parse(capture_access(discover(str(file_path))[0])).content_hash

    assert original_hash != changed_hash


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_pdf_parses_correctly(tmp_path):
    pdf_path = tmp_path / "remote_work_policy.pdf"
    write_minimal_pdf(
        pdf_path,
        "Fictional Corp Remote Work Policy\n"
        "Employees may work remotely up to three days per week with manager approval.",
    )

    discovered = discover(str(pdf_path))
    assert len(discovered) == 1

    with_access = capture_access(discovered[0])
    parsed = parse(with_access)

    assert parsed.title == "remote_work_policy"
    assert "Fictional Corp Remote Work Policy" in parsed.raw_text
    assert "three days per week" in parsed.raw_text

    chunks = chunk(parsed)
    assert len(chunks) >= 1


def test_corrupt_pdf_raises_parse_error(tmp_path):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"this is not a real pdf file, just garbage bytes")

    with pytest.raises(ParseError):
        parse(capture_access(discover(str(corrupt))[0]))


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def test_docx_parses_correctly(tmp_path):
    docx_path = tmp_path / "employee_handbook.docx"
    write_minimal_docx(
        docx_path,
        heading="Employee Handbook",
        paragraphs=[
            "This handbook describes company policy.",
            "Employees receive 15 days of paid leave per year.",
        ],
    )

    with_access = capture_access(discover(str(docx_path))[0])
    parsed = parse(with_access)

    assert parsed.title == "employee_handbook"
    assert "Employee Handbook" in parsed.raw_text  # heading text preserved, as plain text
    assert "15 days of paid leave" in parsed.raw_text

    chunks = chunk(parsed)
    assert len(chunks) >= 1


def test_corrupt_docx_raises_parse_error(tmp_path):
    corrupt = tmp_path / "corrupt.docx"
    corrupt.write_bytes(b"this is not a real docx file, just garbage bytes")

    with pytest.raises(ParseError):
        parse(capture_access(discover(str(corrupt))[0]))


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def test_xlsx_parses_correctly(tmp_path):
    xlsx_path = tmp_path / "leave_balances.xlsx"
    write_minimal_xlsx(
        xlsx_path,
        sheet_title="Leave Balances",
        header=["Employee", "Balance"],
        rows=[["Alice", 12], ["Bob", 8]],
    )

    with_access = capture_access(discover(str(xlsx_path))[0])
    parsed = parse(with_access)

    assert parsed.title == "leave_balances"
    assert "Sheet: Leave Balances" in parsed.raw_text
    assert "Employee=Alice" in parsed.raw_text
    assert "Balance=12" in parsed.raw_text
    assert "Employee=Bob" in parsed.raw_text

    chunks = chunk(parsed)
    assert len(chunks) >= 1


def test_corrupt_xlsx_raises_parse_error(tmp_path):
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"this is not a real xlsx file, just garbage bytes")

    with pytest.raises(ParseError):
        parse(capture_access(discover(str(corrupt))[0]))


# ---------------------------------------------------------------------------
# Discovery finds the new binary formats
# ---------------------------------------------------------------------------


def test_discover_finds_pdf_docx_xlsx_alongside_text_files(tmp_path):
    docs_root = tmp_path / "mixed"
    docs_root.mkdir()
    write_minimal_pdf(docs_root / "a.pdf", "PDF content")
    write_minimal_docx(docs_root / "b.docx", heading="H", paragraphs=["DOCX content"])
    write_minimal_xlsx(docs_root / "c.xlsx", sheet_title="S", header=["X"], rows=[["y"]])
    (docs_root / "d.txt").write_text("text content", encoding="utf-8")

    discovered = discover(str(docs_root))
    names = {d.uri.split("\\")[-1].split("/")[-1] for d in discovered}
    assert names == {"a.pdf", "b.docx", "c.xlsx", "d.txt"}
