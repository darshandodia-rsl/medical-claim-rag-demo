"""Settings: where the files live and which models to use.

Everything here is read once, when the program starts. Nothing else in the project keeps global state.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# This file is src/claims_rag/config.py, so going up two folders gives the medical-claims-rag/ folder.
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Read GROQ_API_KEY, LLM_MODEL and EMBED_MODEL from the .env file (if there is one) into the environment.
load_dotenv(PROJECT_DIR / ".env")

# Input files.
POLICY_PATH = PROJECT_DIR / "data" / "policy" / "meridian_care_plus_policy.md"
CLAUSE_INDEX_PATH = PROJECT_DIR / "data" / "policy" / "clause_index.md"  # used only by tests
TEST_CLAIMS_PATH = PROJECT_DIR / "data" / "evaluation" / "test_claims.json"

# Where the vector database is stored on disk.
CHROMA_DIR = PROJECT_DIR / "chroma_db"

# Output logs (one JSON object per line).
DECISION_LOG_PATH = PROJECT_DIR / "logs" / "decisions.jsonl"
RETRIEVAL_EVAL_LOG_PATH = PROJECT_DIR / "logs" / "retrieval_eval.jsonl"

# The language model that decides claims, and the model that turns text into vectors.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
