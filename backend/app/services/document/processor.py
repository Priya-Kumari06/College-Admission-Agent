import os
import hashlib
from typing import List, Dict, Tuple
from pathlib import Path

import pdfplumber
import PyPDF2
from docx import Document as DocxDocument

from app.core.config import settings


def extract_text_from_pdf_pdfplumber(file_path: str) -> List[Dict]:
    """Extract text from PDF using pdfplumber (preferred - handles tables better)."""
    pages_content = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    for row in table:
                        if row:
                            table_text += " | ".join([cell or "" for cell in row]) + "\n"
                combined = f"{text}\n{table_text}".strip()
                if combined:
                    pages_content.append({
                        "page_number": page_num,
                        "content": combined
                    })
    except Exception as e:
        # Fallback to PyPDF2
        pages_content = extract_text_from_pdf_pypdf2(file_path)
    return pages_content


def extract_text_from_pdf_pypdf2(file_path: str) -> List[Dict]:
    """Fallback PDF extractor using PyPDF2."""
    pages_content = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_content.append({
                    "page_number": page_num,
                    "content": text.strip()
                })
    return pages_content


def extract_text_from_docx(file_path: str) -> List[Dict]:
    """Extract text from DOCX files."""
    doc = DocxDocument(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return [{"page_number": 1, "content": full_text}]


def extract_text_from_txt(file_path: str) -> List[Dict]:
    """Extract text from plain text files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return [{"page_number": 1, "content": content}]


def extract_text(file_path: str) -> List[Dict]:
    """Route to correct extractor based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf_pdfplumber(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    doc_name: str = ""
) -> List[Dict]:
    """
    Chunk extracted pages into smaller overlapping segments.
    Returns list of chunks with metadata.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        page_num = page["page_number"]
        text = page["content"]

        # Split by sentences for cleaner chunks
        sentences = text.replace("\n", " ").split(". ")
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append({
                        "chunk_id": f"{doc_name}_p{page_num}_c{chunk_id}",
                        "content": current_chunk.strip(),
                        "page_number": page_num,
                        "document_name": doc_name,
                        "char_count": len(current_chunk)
                    })
                    chunk_id += 1
                    # Overlap: carry last portion
                    overlap_words = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_words + sentence + ". "
                else:
                    current_chunk = sentence + ". "

        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"{doc_name}_p{page_num}_c{chunk_id}",
                "content": current_chunk.strip(),
                "page_number": page_num,
                "document_name": doc_name,
                "char_count": len(current_chunk)
            })
            chunk_id += 1

    return chunks


def get_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def validate_file(filename: str, file_size_bytes: int) -> Tuple[bool, str]:
    """Validate uploaded file."""
    allowed = settings.allowed_extensions.split(",")
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in allowed:
        return False, f"File type '.{ext}' not allowed. Allowed: {allowed}"
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, f"File too large. Maximum size: {settings.max_file_size_mb}MB"
    return True, "Valid"
