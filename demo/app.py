from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Allow `python demo/app.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.src.agents.rag_agent import RagAgent
from demo.src.config import has_openai
from demo.src.rag.ingest import build_index

load_dotenv(override=True)

CUSTOM_CSS = """
.gradio-container {
  max-width: 1200px !important;
  margin: auto;
}
#app-header {
  background: linear-gradient(135deg, #0f3d4c 0%, #1f6f8b 55%, #2a9d8f 100%);
  color: #fff;
  border-radius: 16px;
  padding: 22px 24px;
  margin-bottom: 14px;
}
#app-header h1 {
  color: #fff !important;
  margin: 0 0 8px 0;
  font-size: 1.7rem;
}
#app-header p {
  margin: 0;
  opacity: 0.95;
}
#side-panel {
  background: #f7fafc;
  border: 1px solid #d9e2ec;
  border-radius: 12px;
  padding: 14px;
  min-height: 560px;
}
"""

EXAMPLE_PROMPTS = [
    ["What is Insurellm?"],
    ["Which products does Insurellm offer?"],
    ["Tell me about Healthllm pricing."],
    ["What does Claimllm do?"],
    ["Summarize Insurellm company culture."],
    ["Where are Insurellm offices located?"],
    ["Who works in engineering at Insurellm?"],
    ["Tell me about Markellm."],
    ["What contracts exist for Healthllm?"],
    ["How does Lifellm help insurers?"],
]


def format_steps(steps) -> str:
    if not steps:
        return "_No agent steps recorded._"
    lines = ["### Agent reasoning"]
    for index, step in enumerate(steps, start=1):
        lines.append(
            f"**Step {index}: `{step.tool}`**\n"
            f"- Plan: {step.thought}\n"
            f"- Result: {step.observation}"
        )
    return "\n\n".join(lines)


def format_sources(sources) -> str:
    if not sources:
        return "_No sources retrieved._"
    lines = ["### Retrieved evidence"]
    for index, chunk in enumerate(sources, start=1):
        preview = chunk.content.replace("\n", " ").strip()
        if len(preview) > 280:
            preview = preview[:277] + "..."
        lines.append(
            f"**{index}. `{chunk.source}`** "
            f"<span style='color:#486581'>({chunk.doc_type} · score {chunk.score})</span>\n\n"
            f"{preview}"
        )
    return "\n\n".join(lines)


def chat(history: list[dict]):
    question = history[-1]["content"]
    prior = history[:-1]
    agent = RagAgent()
    result = agent.run(question, prior)
    history.append({"role": "assistant", "content": result.answer})
    mode_label = "OpenAI LLM" if result.mode == "openai" else "Local RAG (no API key)"
    side_panel = (
        f"**Runtime mode:** `{mode_label}`\n\n"
        f"{format_steps(result.steps)}\n\n"
        f"{format_sources(result.sources)}"
    )
    return history, side_panel


def main():
    build_index(force=False)
    mode = "OpenAI-enabled" if has_openai() else "Local RAG demo (no API key yet)"

    def put_message(message: str, history: list[dict]):
        message = (message or "").strip()
        if not message:
            return "", history
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Source Sans Pro", "Helvetica", "sans-serif"],
    )

    with gr.Blocks(title="Insurellm RAG Agent", theme=theme, css=CUSTOM_CSS) as ui:
        gr.HTML(
            f"""
            <div id="app-header">
              <h1>Insurellm Knowledge Agent</h1>
              <p>
                RAG + tool-using agent over company docs, products, employees, and contracts.
                <br/>Current runtime: <b>{mode}</b>
              </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat with the agent",
                    height=520,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=(None, None),
                )
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="Ask about Insurellm products, people, culture, or contracts...",
                        show_label=False,
                        scale=5,
                        autofocus=True,
                    )
                    send = gr.Button("Ask", variant="primary", scale=1)
                gr.Markdown("**Try these prompts** (click to fill, then press Ask):")
                gr.Examples(
                    examples=EXAMPLE_PROMPTS,
                    inputs=message,
                    label="Demo prompts",
                    examples_per_page=10,
                )
            with gr.Column(scale=2):
                details = gr.Markdown(
                    value=(
                        "### How this works\n"
                        "1. Agent plans which tools to call\n"
                        "2. RAG retrieves relevant knowledge-base chunks\n"
                        "3. Answer is grounded in those sources\n\n"
                        "*Ask a question to see live agent steps and evidence here.*"
                    ),
                    elem_id="side-panel",
                )

        gr.Markdown(
            "Tip for lead demo: add `OPENAI_API_KEY` to `.env` for richer conversational answers. "
            "Local mode still proves RAG retrieval + agent tool use."
        )

        submit_args = dict(
            fn=put_message,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        )
        message.submit(**submit_args).then(chat, inputs=chatbot, outputs=[chatbot, details])
        send.click(**submit_args).then(chat, inputs=chatbot, outputs=[chatbot, details])

    inbrowser = os.getenv("DEMO_INBROWSER", "1") == "1"
    port = int(os.getenv("DEMO_PORT", "7865"))
    ui.launch(inbrowser=inbrowser, server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    main()
