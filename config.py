import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"

OUT_DIR.mkdir(exist_ok=True)

# data filenames and paths
RAW_LETTERS_FILE = OUT_DIR / os.environ.get("OUT_FILE", "letters.json")
ENRICHED_BASE_FILE = OUT_DIR / "intermediate_01.json"
SEMANTIC_ANALYSIS_FILE = OUT_DIR / "intermediate_02.json"
FINAL_PAYLOAD_FILE = OUT_DIR / "wrapped_payload.json"

# llm configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
ACTIVE_BACKEND: str = os.environ.get("ACTIVE_BACKEND", "local")

# credentials and api configuration
SLOWLY_TOKEN = os.environ.get("SLOWLY_TOKEN", "")
SLOWLY_POST_ID = os.environ.get("SLOWLY_POST_ID", "")
_raw_user_id = os.environ.get("SLOWLY_USER_ID", "")

if not SLOWLY_TOKEN or not SLOWLY_POST_ID or not _raw_user_id:
    raise EnvironmentError(
        "missing critical environment variables!\n"
        "ensure SLOWLY_TOKEN, SLOWLY_USER_ID, and SLOWLY_POST_ID are defined in your .env file.\n"
        "example:\n"
        "SLOWLY_TOKEN='your_jwt_here'\n"
        "SLOWLY_POST_ID='your_post_id'\n"
        "SLOWLY_USER_ID='your_user_id'"
    )

SLOWLY_USER_ID = int(_raw_user_id)
del _raw_user_id
