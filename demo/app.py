from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Allow `uv run python demo/app.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.src.agents.rag_agent import RagAgent
from demo.src.config import has_openai
from demo.src.rag.ingest import build_index

load_dotenv(override=True)


def format_steps(steps) -> str:
    if not steps:
        return "_No agent steps recorded._"
    lines = ["### Agent steps"]
    for index, step in enumerate(steps, start=1):
        lines.append(
            f"**{index}. {step.tool}**\n"
            f"- Thought: {step.thought}\n"
            f"- Observation: {step.observation}"
        )
    return "\n\n".join(lines)


def format_sources(sources) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = ["### Retrieved sources"]
    for index, chunk in enumerate(sources, start=1):
        preview = chunk.content.replace("\n", " ").strip()
        if len(preview) > 320:
            preview = preview[:317] + "..."
        lines.append(
            f"**{index}. `{chunk.source}`** "
            f"(type: {chunk.doc_type}, score: {chunk.score})\n\n{preview}"
        )
    return "\n\n".join(lines)


def chat(history: list[dict]):
    question = history[-1]["content"]
    prior = history[:-1]
    agent = RagAgent()
    result = agent.run(question, prior)
    history.append({"role": "assistant", "content": result.answer})
    mode_note = f"**Mode:** `{result.mode}`"
    side_panel = f"{mode_note}\n\n{format_steps(result.steps)}\n\n{format_sources(result.sources)}"
    return history, side_panel


def main():
    build_index(force=False)
    mode = "OpenAI-enabled" if has_openai() else "Local demo (no API key)"

    def put_message(message: str, history: list[dict]):
        message = (message or "").strip()
        if not message:
            return "", history
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Source Sans Pro", "Helvetica", "sans-serif"])

    with gr.Blocks(title="Insurellm RAG Agent Demo", theme=theme) as ui:
        gr.Markdown(
            f"""
# Insurellm RAG + Agent Demo
Ask questions about Insurellm. The agent retrieves grounded context from the knowledge base, then answers.

**Runtime:** {mode}
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=560,
                    type="messages",
                    show_copy_button=True,
                )
                message = gr.Textbox(
                    placeholder="e.g. What products does Insurellm offer?",
                    show_label=False,
                    autofocus=True,
                )
                gr.Examples(
                    examples=[
                        "What is Insurellm?",
                        "Which products does Insurellm offer?",
                        "Tell me about Healthllm pricing.",
                        "Summarize Insurellm company culture.",
                        "Who works in engineering at Insurellm?",
                    ],
                    inputs=message,
                )
            with gr.Column(scale=1):
                details = gr.Markdown(
                    value="*Agent steps and retrieved sources will appear here.*",
                    container=True,
                    height=620,
                )

        message.submit(put_message, inputs=[message, chatbot], outputs=[message, chatbot]).then(
            chat, inputs=chatbot, outputs=[chatbot, details]
        )

    inbrowser = os.getenv("DEMO_INBROWSER", "1") == "1"
    port = int(os.getenv("DEMO_PORT", "7865"))
    ui.launch(inbrowser=inbrowser, server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    main()
