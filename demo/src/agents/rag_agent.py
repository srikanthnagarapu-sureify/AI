from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from demo.src.agents.tools import (
    ToolResult,
    company_overview,
    list_products,
    search_knowledge_base,
)
from demo.src.config import OPENAI_CHAT_MODEL, has_openai
from demo.src.rag.retriever import RetrievedChunk


@dataclass
class AgentStep:
    thought: str
    tool: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "local"


class RagAgent:
    """
    Lightweight agent that plans tool use, retrieves grounded context,
    and synthesizes an answer (local or OpenAI-backed).
    """

    def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        history = history or []
        plan = self._plan(question)
        steps: list[AgentStep] = []
        gathered: list[RetrievedChunk] = []

        for thought, tool_name, tool_fn in plan:
            result: ToolResult = tool_fn()
            steps.append(
                AgentStep(
                    thought=thought,
                    tool=tool_name,
                    observation=result.summary,
                )
            )
            gathered.extend(result.chunks)

        # Deduplicate by source + content prefix.
        unique_sources: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in gathered:
            key = f"{chunk.source}:{chunk.content[:80]}"
            if key in seen:
                continue
            seen.add(key)
            unique_sources.append(chunk)

        if has_openai():
            answer = self._answer_with_openai(question, history, unique_sources)
            mode = "openai"
        else:
            answer = self._answer_locally(question, unique_sources)
            mode = "local"

        return AgentResult(answer=answer, steps=steps, sources=unique_sources[:8], mode=mode)

    def _plan(self, question: str) -> list[tuple[str, str, Callable[[], ToolResult]]]:
        q = question.lower()
        plan: list[tuple[str, str, Callable[[], ToolResult]]] = []

        def has_phrase(*phrases: str) -> bool:
            return any(re.search(rf"\b{re.escape(phrase)}\b", q) for phrase in phrases)

        wants_products = has_phrase(
            "product",
            "products",
            "offer",
            "pricing",
            "platform",
            "healthllm",
            "carllm",
            "homellm",
            "lifellm",
            "bizllm",
            "markellm",
            "claimllm",
            "rellm",
        )
        wants_company = has_phrase(
            "company", "insurellm", "culture", "career", "overview", "founded", "about insurellm"
        ) or q.strip() in {"what is insurellm?", "what is insurellm"}
        wants_people = has_phrase(
            "employee", "employees", "engineer", "engineering", "team", "staff", "people"
        ) or "who works" in q
        wants_contracts = has_phrase("contract", "contracts", "client", "customer", "deal")

        if wants_products and ("list" in q or "which" in q or "what products" in q):
            plan.append(
                (
                    "User is asking about product catalog coverage.",
                    "list_products",
                    list_products,
                )
            )

        if wants_company and not wants_people and not wants_products:
            plan.append(
                (
                    "Fetch company overview for grounded company context.",
                    "company_overview",
                    company_overview,
                )
            )

        doc_type = None
        if wants_people:
            doc_type = "employees"
        elif wants_contracts:
            doc_type = "contracts"
        elif wants_products and "list" not in q:
            doc_type = "products"
        elif wants_company:
            doc_type = "company"

        plan.append(
            (
                "Search the knowledge base for the most relevant evidence.",
                "search_knowledge_base",
                lambda q=question, dt=doc_type: search_knowledge_base(q, k=6, doc_type=dt),
            )
        )

        # Broaden search if we constrained by doc_type.
        if doc_type:
            plan.append(
                (
                    "Run a broader search in case relevant facts sit in other document types.",
                    "search_knowledge_base",
                    lambda q=question: search_knowledge_base(q, k=4, doc_type=None),
                )
            )

        return plan

    def _answer_locally(self, question: str, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return (
                "I could not find relevant information in the Insurellm knowledge base. "
                "Try asking about company overview, products, employees, or contracts."
            )

        q_tokens = {token.lower() for token in re.findall(r"[a-z0-9_]+", question.lower())}
        highlights: list[str] = []
        citations: list[str] = []

        for chunk in sources[:6]:
            citations.append(f"`{chunk.source}`")
            text = re.sub(r"[#*`]+", " ", chunk.content)
            text = re.sub(r"\s+", " ", text).strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            ranked: list[tuple[int, str]] = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 40:
                    continue
                overlap = sum(1 for token in q_tokens if token in sentence.lower())
                ranked.append((overlap, sentence))
            ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
            if ranked:
                best = ranked[0][1]
                if len(best) > 220:
                    best = best[:217] + "..."
                highlights.append(f"- {best} _(source: {chunk.source})_")

        if not highlights:
            for chunk in sources[:3]:
                snippet = re.sub(r"\s+", " ", chunk.content).strip()
                highlights.append(f"- {snippet[:220]}... _(source: {chunk.source})_")

        unique_highlights = []
        seen = set()
        for item in highlights:
            key = item[:120]
            if key in seen:
                continue
            seen.add(key)
            unique_highlights.append(item)

        summary = "\n".join(unique_highlights[:5])
        cited = ", ".join(dict.fromkeys(citations))
        return (
            f"**Answer (grounded from knowledge base)**\n\n"
            f"{summary}\n\n"
            f"**Sources used:** {cited}\n\n"
            "_Local demo mode: answers are synthesized from retrieved documents. "
            "Add `OPENAI_API_KEY` in `.env` for conversational LLM responses._"
        )

    def _answer_with_openai(
        self, question: str, history: list[dict], sources: list[RetrievedChunk]
    ) -> str:
        from openai import OpenAI

        client = OpenAI()
        context = "\n\n".join(
            f"Source: {chunk.source}\n{chunk.content}" for chunk in sources[:8]
        )
        system = (
            "You are Insurellm Expert Agent, a helpful internal assistant. "
            "Answer using only the provided context. If the context is insufficient, say so. "
            "Be concise, accurate, and cite source filenames inline when useful."
        )
        messages = [{"role": "system", "content": system}]
        for item in history[-6:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            }
        )
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or "No response generated."
