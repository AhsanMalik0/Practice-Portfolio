import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
DOCS_DIR = os.getenv("DOCS_DIR", "policy_docs")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "store_bot")
USE_ATLAS = os.getenv("ATLAS_VECTOR_SEARCH", "0") == "1"
VECTOR_INDEX = "policy_vector_index"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))
PORT = int(os.getenv("PORT", "8000"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "store_bot")