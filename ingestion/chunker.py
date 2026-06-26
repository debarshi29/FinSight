from __future__ import annotations

import hashlib
from collections import Counter

from core.models import Chunk
from ingestion.metadata import detect_company, detect_fiscal_year, detect_section_type


def _is_heading(span: dict, body_size: float) -> bool:
    return span["size"] > body_size * 1.2


def _block_text(block: dict) -> str:
    if "lines" not in block:
        return ""
    return "\n".join(
        "".join(span["text"] for span in line["spans"]) for line in block["lines"]
    )


def _body_size(blocks: list[dict]) -> float:
    sizes = [
        span["size"]
        for block in blocks
        if block["type"] == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    if not sizes:
        return 12.0
    counter = Counter(sizes)
    return counter.most_common(1)[0][0]


def _make_chunk_id(doc_id: str, page: int, index: int, text: str) -> str:
    h = hashlib.sha256(f"{doc_id}:{page}:{index}:{text[:100]}".encode()).hexdigest()[
        :12
    ]
    return f"{doc_id}_p{page}_c{index}_{h}"


def chunk_document(
    pages: list[dict],
    doc_id: str,
    source: str,
    target_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    company = detect_company(source)
    all_text = " ".join(
        _block_text(b) for p in pages for b in p.get("blocks", []) if b["type"] == 0
    )
    fiscal_year = detect_fiscal_year(all_text)

    chunks: list[Chunk] = []

    for page in pages:
        page_num = page["page_num"]
        blocks = page.get("blocks", [])
        if not blocks:
            continue

        body_size = _body_size(blocks)
        current_tokens: list[str] = []
        current_section = "unknown"
        chunk_index = 0

        for block in blocks:
            if block["type"] != 0:
                continue
            if "lines" not in block or not block["lines"]:
                continue

            text = _block_text(block)
            words = text.split()
            if not words:
                continue

            first_span = (
                block["lines"][0]["spans"][0] if block["lines"][0].get("spans") else {}
            )
            is_head = first_span and _is_heading(first_span, body_size)

            if is_head and len(current_tokens) >= overlap_tokens:
                chunk_text = " ".join(current_tokens)
                section_type = detect_section_type(chunk_text, current_section)
                chunk_id = _make_chunk_id(doc_id, page_num, chunk_index, chunk_text)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        source=source,
                        text=chunk_text,
                        page=page_num,
                        section=current_section,
                        section_type=section_type,
                        token_count=len(current_tokens),
                        fiscal_year=fiscal_year,
                        company=company,
                    )
                )
                chunk_index += 1
                current_tokens = current_tokens[-overlap_tokens:]
                current_section = text.strip()
            else:
                if is_head:
                    current_section = text.strip()
                current_tokens.extend(words)

                while len(current_tokens) >= target_tokens + overlap_tokens:
                    window = current_tokens[:target_tokens]
                    chunk_text = " ".join(window)
                    section_type = detect_section_type(chunk_text, current_section)
                    chunk_id = _make_chunk_id(doc_id, page_num, chunk_index, chunk_text)
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            source=source,
                            text=chunk_text,
                            page=page_num,
                            section=current_section,
                            section_type=section_type,
                            token_count=len(window),
                            fiscal_year=fiscal_year,
                            company=company,
                        )
                    )
                    chunk_index += 1
                    current_tokens = current_tokens[target_tokens - overlap_tokens :]

        if current_tokens:
            chunk_text = " ".join(current_tokens)
            section_type = detect_section_type(chunk_text, current_section)
            chunk_id = _make_chunk_id(doc_id, page_num, chunk_index, chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source=source,
                    text=chunk_text,
                    page=page_num,
                    section=current_section,
                    section_type=section_type,
                    token_count=len(current_tokens),
                    fiscal_year=fiscal_year,
                    company=company,
                )
            )

    return _deduplicate(chunks)


def _deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    result = []
    for chunk in chunks:
        key = hashlib.sha256(chunk.text[:200].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result
