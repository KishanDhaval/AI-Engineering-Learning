import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "IT_Company_HR_Policy.pdf")
FAISS_CLOUD_PATH = os.path.join(BASE_DIR, "faiss_index_cloud")
FAISS_LOCAL_PATH = os.path.join(BASE_DIR, "faiss_index_local")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_EMBED = "mistral-embed"
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_EMBED = "nomic-embed-text"