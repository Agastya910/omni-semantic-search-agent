
# models used for embedding and LLM
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2:7b-instruct-q4_0"

# Qdrant database configuration
DATA_FOLDER = "data_source"
VECTOR_DB_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "rag_collection"

SPARSE_VECTOR_NAME = "sparse-text"

TOP_K_RETRIEVAL, TOP_K_RERANK = 20, 5
TOP_K = 5  # Number of top similar chunks to retrieve during reranking






