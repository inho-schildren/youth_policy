import os
from dotenv import load_dotenv

load_dotenv()

# ── API ─────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
LLM_MODEL        = "gpt-4o-mini"
EMBEDDING_MODEL  = "text-embedding-3-small"
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ── 기존 경로 ────────────────────────────────────────────────
# PDF_FOLDER  = "./realdata"       # ← PDF 파일 위치
# META_PATH   = "metadata.json"    # ← 메타데이터 저장 위치
# DOCS_PATH   = "output.json"      # ← documents 저장 위치
# CHROMA_DIR  = "./db/chroma_db"   # ← 벡터DB 저장 위치

# ── 고도화 경로 ─────────────────────────────────────────────────
PDF_FOLDER  = "./data/raw"
META_PATH   = "./data/metadata.json"
DOCS_PATH   = "./data/output_v2.json"
CHUNKS_PATH = "./data/chunks_v2.json"
CHROMA_DIR  = "./db/chroma_db_v2"

# ── 파라미터 ─────────────────────────────────────────────
MAX_PAGES        = 30    # 메타데이터 추출 시 읽을 최대 페이지 수
MAX_TEXT_LENGTH  = 5000  # LLM에 넘길 최대 텍스트 길이

# ── Retriever ─────────────────────────────────────────────
RETRIEVER_K  = 10
BM25_WEIGHT  = 0.85
DENSE_WEIGHT = 0.2

# ── MMR ───────────────────────────────────────────────────
MMR_FETCH_K  = 15
MMR_LAMBDA   = 0.9

# ── Reranker ──────────────────────────────────────────────
RERANKER_TOP_N = 7

# ── SemanticChunker ───────────────────────────────────────
# CHUNK_THRESHOLD = 85