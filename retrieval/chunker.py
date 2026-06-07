from collections import Counter
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    page: int
    section: str
    token_count: int


def is_heading(span: dict, body_size: float) -> bool:
    return span["size"] > body_size * 1.2


def extract_block_text(block: dict) -> str:
    if "lines" not in block:
        return ""
    return "\n".join(
        "".join(span["text"] for span in line["spans"]) for line in block["lines"]
    )


def get_body_size(blocks: list[dict]) -> float:
    sizes = [
        span["size"]
        for block in blocks
        if block["type"] == 0
        for line in block["lines"]
        for span in line["spans"]
    ]
    size_counts = Counter(sizes)
    most_common_size, _ = size_counts.most_common(1)[0]
    return most_common_size


def chunk_document(pages: list[dict], doc_id: str, source: str) -> list[Chunk]:
    chunks = []
    for page in pages:
        page_num = page["page_num"]
        current_page = page_num
        blocks = page["blocks"]
        body_size = get_body_size(blocks)
        current_text = []
        current_section = "unknown"
        chunk_index = 0
        for block in blocks:
            if block["type"] != 0:
                continue
            if "lines" not in block or not block["lines"]:
                continue
            block_text = extract_block_text(block)
            token_count = len(block_text.split())
            if token_count == 0:
                continue
            first_span = block["lines"][0]["spans"][0]
            is_head = is_heading(first_span, body_size)
            if is_head and current_text:
                chunk_id = f"{doc_id}_page{current_page}_chunk{chunk_index}"
                chunks.append(
                    Chunk(
                        chunk_id,
                        doc_id,
                        source,
                        "\n".join(current_text),
                        current_page,
                        current_section,
                        sum(len(t.split()) for t in current_text),
                    )
                )
                chunk_index += 1
                current_section = block_text.strip()
                current_text = []
            else:
                current_text.append(block_text)
                if is_head:
                    current_section = block_text.strip()
        if current_text:
            chunk_id = f"{doc_id}_page{current_page}_chunk{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id,
                    doc_id,
                    source,
                    "\n".join(current_text),
                    current_page,
                    current_section,
                    sum(len(t.split()) for t in current_text),
                )
            )
    return chunks


if __name__ == "__main__":
    import fitz

    doc = fitz.open("../Long_Term_Memory_Flow.pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pages.append(
            {"page_num": page_num + 1, "blocks": page.get_text("dict")["blocks"]}
        )

    chunks = chunk_document(pages, "test-doc", "test.pdf")
    for c in chunks[:5]:
        print(f"Section: {c.section}")
        print(f"Page: {c.page}")
        print(f"Tokens: {c.token_count}")
        print(f"Text: {c.text[:200]}")
        print("---")
