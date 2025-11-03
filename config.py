import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# API KEYS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

# Paths
DATA_DIR = "data"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "docs.index")
META_FILE = os.path.join(INDEX_DIR, "metadata.npy")
DOCS_FILE = os.path.join(INDEX_DIR, "documents.npy")

# Models
EMBED_MODEL = "all-MiniLM-L6-v2"
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEMINI_MODEL = "models/gemini-2.5-flash"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
