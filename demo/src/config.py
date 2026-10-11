from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_ROOT.parent

KNOWLEDGE_BASE_PATH = REPO_ROOT / "week5" / "knowledge-base"
DATA_DIR = DEMO_ROOT / "data"
INDEX_PATH = DATA_DIR / "local_index.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

RETRIEVAL_K = 6
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def has_openai() -> bool:
    return bool(OPENAI_API_KEY)
