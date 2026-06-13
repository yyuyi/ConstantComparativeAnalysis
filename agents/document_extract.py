from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_text(text: Any) -> str:
    return str(text or "").replace("\x00", "").strip()


def _extract_txt(path: Path) -> Tuple[str, List[str]]:
    return path.read_text(encoding="utf-8", errors="replace"), []


def _extract_docx(path: Path) -> Tuple[str, List[str]]:
    try:
        import docx  # type: ignore
    except Exception as exc:
        return "", [f"python-docx unavailable: {exc}"]
    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs), []
    except Exception as exc:
        return "", [f"docx extraction failed: {exc}"]


def _extract_pdf_pypdf(path: Path) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        return "", [f"pypdf unavailable: {exc}"]

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return "", [f"pypdf could not open PDF: {exc}"]

    pages: List[str] = []
    failed_pages: List[int] = []
    blank_pages: List[int] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize_text(page.extract_text() or "")
        except Exception as exc:
            failed_pages.append(idx)
            warnings.append(f"pypdf page {idx} failed: {exc}")
            text = ""
        if not text:
            blank_pages.append(idx)
        pages.append(text)

    if failed_pages:
        warnings.append(f"pypdf extracted partial text; failed pages: {failed_pages}")
    if blank_pages and len(blank_pages) == len(pages):
        warnings.append("pypdf returned blank text for every page; PDF may be scanned or image-only.")
    elif blank_pages:
        warnings.append(f"pypdf returned blank text for pages: {blank_pages[:20]}")
    return "\n\n".join(page for page in pages if page), warnings


def _extract_pdf_pdftotext(path: Path) -> Tuple[str, List[str]]:
    exe = shutil.which("pdftotext")
    if not exe:
        return "", ["pdftotext unavailable"]
    try:
        proc = subprocess.run(
            [exe, "-layout", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as exc:
        return "", [f"pdftotext failed: {exc}"]
    warnings: List[str] = []
    if proc.returncode != 0:
        warnings.append(f"pdftotext exited with code {proc.returncode}: {proc.stderr.strip()[:400]}")
    return _normalize_text(proc.stdout), warnings


def _extract_pdf_pymupdf(path: Path) -> Tuple[str, List[str]]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return "", [f"PyMuPDF unavailable: {exc}"]
    warnings: List[str] = []
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return "", [f"PyMuPDF could not open PDF: {exc}"]
    pages: List[str] = []
    try:
        for idx, page in enumerate(doc, start=1):
            try:
                pages.append(_normalize_text(page.get_text("text")))
            except Exception as exc:
                warnings.append(f"PyMuPDF page {idx} failed: {exc}")
    finally:
        doc.close()
    return "\n\n".join(page for page in pages if page), warnings


def _extract_pdf_pdfplumber(path: Path) -> Tuple[str, List[str]]:
    try:
        import pdfplumber  # type: ignore
    except Exception as exc:
        return "", [f"pdfplumber unavailable: {exc}"]
    warnings: List[str] = []
    pages: List[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                try:
                    pages.append(_normalize_text(page.extract_text() or ""))
                except Exception as exc:
                    warnings.append(f"pdfplumber page {idx} failed: {exc}")
    except Exception as exc:
        return "", [f"pdfplumber could not open PDF: {exc}"]
    return "\n\n".join(page for page in pages if page), warnings


def _extract_pdf(path: Path) -> Tuple[str, str, List[str]]:
    candidates: List[Tuple[str, str, List[str]]] = []
    for method, fn in [
        ("pypdf", _extract_pdf_pypdf),
        ("pdftotext", _extract_pdf_pdftotext),
        ("pymupdf", _extract_pdf_pymupdf),
        ("pdfplumber", _extract_pdf_pdfplumber),
    ]:
        text, warnings = fn(path)
        candidates.append((method, _normalize_text(text), warnings))

    best_method, best_text, best_warnings = max(candidates, key=lambda item: len(item[1]))
    warnings: List[str] = []
    for method, text, method_warnings in candidates:
        if method == best_method:
            warnings.extend(method_warnings)
        elif method_warnings and not text:
            warnings.extend(method_warnings[:2])
    if not best_text:
        warnings.append(
            "No extractable PDF text found. If this is a scanned PDF, upload an OCR text version."
        )
    elif len(best_text) < 1000:
        warnings.append(
            f"Only {len(best_text)} characters extracted from PDF; transcript coverage may be incomplete."
        )
    return best_text, best_method, warnings


def extract_document_text(path: Path | str) -> Dict[str, Any]:
    """Extract text with diagnostics. Never raises for normal extraction failures."""
    doc_path = Path(path)
    suffix = doc_path.suffix.lower()
    warnings: List[str] = []
    method = suffix.lstrip(".") or "unknown"
    try:
        if suffix == ".txt":
            text, warnings = _extract_txt(doc_path)
            method = "txt"
        elif suffix == ".pdf":
            text, method, warnings = _extract_pdf(doc_path)
        elif suffix in {".docx", ".docs"}:
            text, warnings = _extract_docx(doc_path)
            method = "docx"
        elif suffix == ".doc":
            text = ""
            warnings = ["Legacy .doc files are not supported; upload .docx, .txt, or PDF."]
            method = "doc_unsupported"
        else:
            text, warnings = _extract_txt(doc_path)
            method = "text_fallback"
    except Exception as exc:
        text = ""
        warnings = [f"extraction failed: {exc}"]
    text = _normalize_text(text)
    return {
        "path": str(doc_path),
        "method": method,
        "chars": len(text),
        "words": len(text.split()),
        "text": text,
        "warnings": warnings,
    }
