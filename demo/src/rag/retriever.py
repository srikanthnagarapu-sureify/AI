from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from demo.src.config import RETRIEVAL_K
from demo.src.rag.ingest import Chunk, build_index, load_index


TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    doc_type: str
    title: str
    score: float


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class Retriever:
    """Local TF-IDF retriever so demos work without an embedding API."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_index()
        if not self.chunks:
            self.chunks = build_index(force=True)
        self._doc_tokens = [_tokenize(chunk.content) for chunk in self.chunks]
        self._df: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] += 1
        self._n = len(self.chunks)

    def search(self, query: str, k: int = RETRIEVAL_K, doc_type: str | None = None) -> list[RetrievedChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        scored: list[tuple[float, Chunk]] = []

        for chunk, tokens in zip(self.chunks, self._doc_tokens):
            if doc_type and chunk.doc_type != doc_type:
                continue
            if not tokens:
                continue
            doc_tf = Counter(tokens)
            score = 0.0
            for term, q_count in query_tf.items():
                if term not in doc_tf:
                    continue
                idf = math.log((1 + self._n) / (1 + self._df[term])) + 1.0
                score += (q_count * idf) * (doc_tf[term] * idf)
            # Light boost when the filename/title matches the query.
            title_tokens = set(_tokenize(chunk.title))
            score += 1.5 * len(title_tokens.intersection(query_tokens))
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievedChunk] = []
        for score, chunk in scored[:k]:
            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    source=chunk.source,
                    doc_type=chunk.doc_type,
                    title=chunk.title,
                    score=round(score, 4),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    build_index(force=False)
    return Retriever()
