from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from demo.src.config import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_PATH, KNOWLEDGE_BASE_PATH


@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    doc_type: str
    title: str


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer splitting on paragraph/sentence boundaries.
            window = text[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
            if split_at > chunk_size // 3:
                end = start + split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_documents(knowledge_base: Path = KNOWLEDGE_BASE_PATH) -> list[Chunk]:
    if not knowledge_base.exists():
        raise FileNotFoundError(f"Knowledge base not found: {knowledge_base}")

    chunks: list[Chunk] = []
    for path in sorted(knowledge_base.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        doc_type = path.parent.name
        title = path.stem
        relative = str(path.relative_to(knowledge_base))
        for index, piece in enumerate(_split_text(content)):
            chunks.append(
                Chunk(
                    chunk_id=f"{relative}::{index}",
                    content=piece,
                    source=relative,
                    doc_type=doc_type,
                    title=title,
                )
            )
    return chunks


def build_index(force: bool = False) -> list[Chunk]:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists() and not force:
        return load_index()

    chunks = load_documents()
    payload = [asdict(chunk) for chunk in chunks]
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def load_index() -> list[Chunk]:
    if not INDEX_PATH.exists():
        return build_index(force=True)
    raw = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


if __name__ == "__main__":
    built = build_index(force=True)
    print(f"Indexed {len(built)} chunks from {KNOWLEDGE_BASE_PATH}")
