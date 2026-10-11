from __future__ import annotations

from dataclasses import dataclass

from demo.src.config import KNOWLEDGE_BASE_PATH
from demo.src.rag.retriever import RetrievedChunk, get_retriever


@dataclass
class ToolResult:
    name: str
    summary: str
    chunks: list[RetrievedChunk]


def search_knowledge_base(query: str, k: int = 5, doc_type: str | None = None) -> ToolResult:
    retriever = get_retriever()
    chunks = retriever.search(query, k=k, doc_type=doc_type)
    if not chunks:
        return ToolResult(
            name="search_knowledge_base",
            summary=f"No relevant documents found for: {query}",
            chunks=[],
        )
    sources = ", ".join(sorted({chunk.source for chunk in chunks}))
    return ToolResult(
        name="search_knowledge_base",
        summary=f"Retrieved {len(chunks)} chunk(s) from: {sources}",
        chunks=chunks,
    )


def list_products() -> ToolResult:
    products_dir = KNOWLEDGE_BASE_PATH / "products"
    product_names = sorted(path.stem for path in products_dir.glob("*.md"))
    content = "Insurellm products:\n" + "\n".join(f"- {name}" for name in product_names)
    chunk = RetrievedChunk(
        content=content,
        source="products/",
        doc_type="products",
        title="Product Catalog",
        score=1.0,
    )
    return ToolResult(
        name="list_products",
        summary=f"Found {len(product_names)} products",
        chunks=[chunk],
    )


def company_overview() -> ToolResult:
    overview = KNOWLEDGE_BASE_PATH / "company" / "overview.md"
    about = KNOWLEDGE_BASE_PATH / "company" / "about.md"
    chunks: list[RetrievedChunk] = []
    for path in (overview, about):
        if path.exists():
            chunks.append(
                RetrievedChunk(
                    content=path.read_text(encoding="utf-8")[:1800],
                    source=str(path.relative_to(KNOWLEDGE_BASE_PATH)),
                    doc_type="company",
                    title=path.stem,
                    score=1.0,
                )
            )
    return ToolResult(
        name="company_overview",
        summary=f"Loaded {len(chunks)} company document(s)",
        chunks=chunks,
    )


def list_available_tools() -> list[dict[str, str]]:
    return [
        {
            "name": "search_knowledge_base",
            "description": "Semantic/keyword search over company docs, products, employees, and contracts.",
        },
        {
            "name": "list_products",
            "description": "List all Insurellm product names from the knowledge base.",
        },
        {
            "name": "company_overview",
            "description": "Fetch high-level company overview and about information.",
        },
    ]
