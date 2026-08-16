from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pdfplumber
import pytesseract
from PIL import Image


class DocumentExtractionError(Exception):
    pass


def extract_text(file_path: str) -> Tuple[str, int, str]:
    """Extract text from PDF or image. Returns (text, pages, document_type)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = 0
        chunks: list[str] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        chunks.append(text)
        except Exception as exc:
            raise DocumentExtractionError(f"PDF extraction failed: {exc}") from exc
        return "\n\n".join(chunks), pages, "pdf"

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
        try:
            with Image.open(path) as image:
                text = pytesseract.image_to_string(image)
            return text, 1, "image_ocr"
        except Exception as exc:
            raise DocumentExtractionError(
                "Image OCR failed. Check that Tesseract is installed and available on PATH. "
                f"Details: {exc}"
            ) from exc

    raise DocumentExtractionError(
        f"Unsupported file type: {suffix}. Supported types: PDF, PNG, JPG, JPEG, WEBP, BMP, TIFF."
    )
