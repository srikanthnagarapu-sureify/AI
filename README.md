# Mastering Generative AI and LLMs: An 8-Week Hands-On Journey

## Executive summary

This repository captures an end-to-end learning and implementation journey in Generative AI and LLM engineering. It is designed to demonstrate practical capability across the full AI lifecycle: experimentation, model integration, retrieval-augmented generation, fine-tuning, agent-based workflows, and deployment-ready user experiences.

The work is organized as a structured proof of learning and execution, suitable for both technical review and manager-facing presentation. Instead of presenting raw course materials alone, this repo frames the work as an applied AI capability program with tangible business outcomes and real project artifacts.

## Project objective

The goal is to build hands-on experience with modern Generative AI systems and to move from concept to working applications. The repository explores how LLMs can be applied to real-world business scenarios such as customer support, knowledge retrieval, software optimization, and personalized product intelligence.

## What this project demonstrates

- Building advanced Generative AI products using modern models and frameworks
- Working with over 20 model families across frontier and open-source ecosystems
- Developing robust workflows with LangChain, Hugging Face, Gradio, and related tooling
- Applying retrieval-augmented generation (RAG) to grounded knowledge tasks
- Exploring agentic workflows for automation and task execution
- Moving from inference to model adaptation and fine-tuning
- Creating polished user interfaces for AI-powered tools and experiences

## Business value

This repository reflects a practical AI capability roadmap that can be used to:

- automate knowledge-heavy tasks
- improve internal productivity and support quality
- build AI assistants grounded in company context and documentation
- accelerate technical workflows through intelligent coding assistance
- drive intelligent recommendations using structured and unstructured data

## Repository structure

The project is organized into a clear portfolio layout while preserving the course progression.

- `demo/` – **primary demo app**: RAG + tool-using agent with Gradio UI
- `guides/` – introductory technical guidance and setup material
- `extras/` – optional resources and supporting experiments
- `week2/` to `week8/` – progression of applied AI project work across the 8-week journey
- `scripts/` – local validation and smoke-test automation
- `.env.example` – environment configuration template

This structure keeps the learning path visible while presenting the repository in a professional, project-oriented format for stakeholder review.

## Weekly progression overview

The folders represent a progressive capability roadmap:

- `week2` – foundational LLM workflows and experimental practice
- `week3` – data, embeddings, and richer application flows
- `week4` – advanced AI engineering patterns and optimization work
- `week5` – practical application and interface-driven experiences
- `week6` – more advanced LLM engineering and systems thinking
- `week7` – deeper experimentation and production-like capabilities
- `week8` – agentic and end-to-end AI solution work

## Example use cases covered

This project includes work aligned to real-world Generative AI use cases such as:

- multi-modal customer support assistant
- enterprise knowledge worker over internal documents
- AI coding assistant for software optimization
- intelligent product pricing and recommendation workflows

## Quick start

```bash
cd /Users/srikanthnagarapu/product/AI
source $HOME/.local/bin/env
uv sync --frozen
cp .env.example .env
```

Or use the existing virtualenv:

```bash
source .venv/bin/activate
```

Then add your API keys to `.env` as needed (optional for the local demo).

## Validation commands

```bash
cd /Users/srikanthnagarapu/product/AI
source $HOME/.local/bin/env
uv run python scripts/smoke_test.py
```

This validates that the project environment and core dependencies are working.

## Run the RAG + Agent demo (recommended)

```bash
cd /Users/srikanthnagarapu/product/AI
source .venv/bin/activate
python demo/app.py
```

This launches the Insurellm assistant with:
- knowledge-base retrieval (RAG)
- an agent that plans and calls tools
- source citations and agent step visibility in the UI

Works in local demo mode without an API key. Add `OPENAI_API_KEY` to `.env` for fuller LLM answers.

See `demo/README.md` for structure and suggested demo questions.

## Run the original week5 sample app

```bash
cd /Users/srikanthnagarapu/product/AI
source $HOME/.local/bin/env
uv run python week5/app.py
```

## Recommended presentation angle for management

This project should be presented as an applied AI capability program, not as a raw course dump. Emphasize:

- practical learning outcomes
- hands-on implementation across multiple LLM workflows
- business-aligned use cases
- production-minded architecture and experimentation practices
- clear progression from foundation to agentic systems

## Notes

This repository is intended to showcase learning, experimentation, and project development in Generative AI and LLMs. It is suitable for portfolio, internal review, and certification-style presentation.
