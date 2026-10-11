# Insurellm RAG + Agent Demo

Demo-ready application that combines **Retrieval-Augmented Generation (RAG)** with a **tool-using agent**.

Built for stakeholder review and live demos of applied AI learning from this repository.

## What it demonstrates

- Document ingestion from a company knowledge base
- Local RAG retrieval (works without an API key)
- Optional OpenAI-powered embeddings + LLM answers when `OPENAI_API_KEY` is set
- An agent that plans steps and calls tools (`search_knowledge_base`, `list_products`, `company_overview`)
- A Gradio UI showing the answer, agent steps, and retrieved sources

## Folder structure

```text
demo/
├── app.py                 # Launch the Gradio demo
├── README.md
├── src/
│   ├── config.py          # Paths and runtime settings
│   ├── rag/
│   │   ├── ingest.py      # Load + index knowledge base
│   │   └── retriever.py   # Retrieval interface
│   └── agents/
│       ├── tools.py       # Agent tools
│       └── rag_agent.py   # Planner + tool orchestration
└── data/                  # Generated index (gitignored)
```

Knowledge documents are reused from `week5/knowledge-base/` (company, products, employees, contracts).

## Quick start

From the repository root:

```bash
source .venv/bin/activate
cp .env.example .env          # optional: add OPENAI_API_KEY for full LLM answers
python demo/app.py
```

Then open the local Gradio URL shown in the terminal (default port `7865`).

Optional environment variables:

- `DEMO_PORT=7865` – Gradio port
- `DEMO_INBROWSER=1` – open browser automatically (default on)

## Suggested demo questions

- What is Insurellm?
- Which products does Insurellm offer?
- Tell me about Healthllm pricing.
- Who works at Insurellm in engineering?
- Summarize the company culture.

## Modes

| Mode | When | Behavior |
|------|------|----------|
| Local demo | No `OPENAI_API_KEY` | Keyword/TF-IDF retrieval + grounded synthesis |
| Full LLM | `OPENAI_API_KEY` set | OpenAI embeddings/Chat model + richer answers |

Both modes show agent steps and source documents so the demo stays review-friendly.
