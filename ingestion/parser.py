from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def parse_pdf(path: str | Path) -> list[dict[str, Any]]:
    """Parse PDF into a list of page dicts with blocks and metadata."""
    doc = fitz.open(str(path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw = page.get_text("dict")
        pages.append(
            {
                "page_num": page_num + 1,
                "width": raw["width"],
                "height": raw["height"],
                "blocks": raw["blocks"],
            }
        )
    doc.close()
    return pages


def extract_text_with_positions(path: str | Path) -> list[dict[str, Any]]:
    """Extract text spans with bounding boxes for snippet extraction."""
    doc = fitz.open(str(path))
    result = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    spans.append(
                        {
                            "text": span["text"],
                            "bbox": span["bbox"],
                            "size": span["size"],
                            "flags": span["flags"],
                            "page": page_num + 1,
                        }
                    )
        result.append({"page_num": page_num + 1, "spans": spans})
    doc.close()
    return result


def get_page_text(path: str | Path, page_num: int) -> str:
    """Get plain text for a specific page (1-indexed)."""
    doc = fitz.open(str(path))
    page = doc[page_num - 1]
    text = page.get_text("text")
    doc.close()
    return text
