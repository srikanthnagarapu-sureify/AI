from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.src.agents.rag_agent import RagAgent
from week5.implementation.answer import answer_question

REQUIRED_MODULES = [
    "dotenv",
    "gradio",
    "openai",
    "numpy",
    "pandas",
    "requests",
    "plotly",
]

missing = []
for module_name in REQUIRED_MODULES:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - smoke validation only
        missing.append(f"{module_name}: {exc}")

required_dirs = [
    "demo",
    "guides",
    "week2",
    "week3",
    "week4",
    "week5",
    "week6",
    "week7",
    "week8",
]
missing_dirs = [name for name in required_dirs if not Path(name).exists()]

try:
    message, context = answer_question("What is Insurellm?", [])
    if not isinstance(message, str) or not context:
        raise RuntimeError("Fallback answer did not return a valid answer and context.")
except Exception as exc:  # pragma: no cover - smoke validation only
    missing.append(f"answer_question fallback: {exc}")

try:
    result = RagAgent().run("What products does Insurellm offer?")
    if not result.answer or not result.steps or not result.sources:
        raise RuntimeError("Demo agent did not return answer, steps, and sources.")
except Exception as exc:  # pragma: no cover - smoke validation only
    missing.append(f"demo RagAgent: {exc}")

if missing or missing_dirs:
    print("SMOKE TEST FAILED")
    if missing:
        print("Missing modules / runtime issues:")
        for item in missing:
            print(f" - {item}")
    if missing_dirs:
        print("Missing directories:")
        for item in missing_dirs:
            print(f" - {item}")
    raise SystemExit(1)

print("SMOKE TEST PASSED")
print("Core dependencies and course directories are available.")
print("Week5 fallback answer flow is working without requiring an API key.")
print("Demo RAG + Agent flow is working without requiring an API key.")
