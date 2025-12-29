import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [RAW_DIR, PROCESSED_DIR, CHUNKS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# LlamaParser
LLAMA_PARSER_API_KEY = os.getenv("LLAMA_PARSER_API_KEY")

# Azure Document Intelligence (for PDF parsing)
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

# Mistral Document AI (fallback for failed documents)
MISTRAL_DOC_AI_ENDPOINT = os.getenv("MISTRAL_DOC_AI_ENDPOINT")
MISTRAL_DOC_AI_KEY = os.getenv("MISTRAL_DOC_AI_KEY")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "invest-up-docs")

# Scraper settings
BASE_URL = "https://invest.up.gov.in"
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", 5))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 1.0))

# Document extensions to capture
DOCUMENT_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".doc", ".docx", ".ppt", ".pptx"}

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM settings
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2000
