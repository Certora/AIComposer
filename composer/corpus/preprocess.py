"""
PDF preprocessing — base64 encode for LLM consumption and extract links.

The PDF is sent as a base64 document block so the Anthropic API can render
it visually. Link annotations are extracted separately since they are not
part of the visual rendering.
"""

import base64
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz


@dataclass
class PreprocessedPDF:
    """Result of preprocessing a PDF file.

    Contains a base64-encoded PDF for LLM document blocks, extracted URIs
    from link annotations, and a content hash for cache keying.
    """
    path: Path
    content_hash: str
    pdf_base64: str
    links: list[str] = field(default_factory=list)


def pdf_key(path: Path) -> str:
    pdf_bytes = path.read_bytes()
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
    return content_hash

def preprocess_pdf(path: Path) -> PreprocessedPDF:
    """Preprocess a PDF file: compute hash, base64 encode, extract links.

    Args:
        path: Path to the PDF file.

    Returns:
        PreprocessedPDF with content hash, base64 PDF, and extracted links.
    """
    pdf_bytes = path.read_bytes()
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
    pdf_base64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    doc : fitz.Document = fitz.open(path)
    links: list[str] = []
    for page in doc:
        for link in page.get_links():
            uri = link.get("uri")
            if uri:
                links.append(uri)

    return PreprocessedPDF(
        path=path,
        content_hash=content_hash,
        pdf_base64=pdf_base64,
        links=links,
    )
