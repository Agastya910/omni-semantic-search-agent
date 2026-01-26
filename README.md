# 🤖 Personal Semantic Agent: Production-Grade RAG

A local, high-performance Retrieval-Augmented Generation (RAG) system built for speed, accuracy, and efficiency on standard hardware.

This project was built from scratch to demonstrate how to tackle real-world industry challenges: handling thousands of diverse files, ensuring low-latency retrieval without a GPU, and implementing multi-stage ranking for precision.

## 🚀 Key Features

- **Multi-Format Ingestion**: Seamlessly parses PDF, DOCX, and PPTX using Docling.
- **Hybrid Search**: Combines semantic (Dense) vectors with keyword (Sparse/BM25) search to ensure proper nouns and specific terms are never missed.
- **CPU Optimized**: Uses Rust-based Qdrant and ONNX-powered FlashRank to deliver sub-second latency without an expensive GPU.
- **Smart Indexing**: Automatic deduplication logic ensures you don't re-process files that are already indexed.
- **Refined Reranking**: Implements a two-stage retrieval process (Retrieve -> Rerank) to maximize context relevance for the LLM.

## 🛠️ The Tech Stack

| Component    | Technology     | Why? |
|--------------|----------------|------|
| Ingestion    | Docling        | Exceptional layout analysis and smart chunking for complex documents. |
| Vector DB    | Qdrant         | Written in Rust; extremely fast HNSW indexing and native Hybrid Search support. |
| Embeddings   | nomic-embed-text | High-performing open-source local embeddings via Ollama. |
| Reranking    | FlashRank      | Ultra-lightweight ONNX models for high-precision ranking on CPU. |
| LLM Engine   | Ollama         | Provides a robust, local API for state-of-the-art models like Llama 3.

## 🏗️ Architecture Workflow

1. **Ingestion**: Files are monitored in a target folder, parsed, and split into meaningful segments.
2. **Hybrid Indexing**: Chunks are converted into both semantic and sparse vectors and stored in Qdrant.
3. **Retrieval**: The system fetches the top 25 matches using Reciprocal Rank Fusion (RRF).
4. **Reranking**: FlashRank acts as a "judge" to pick the top 5 most relevant chunks from the initial results.
5. **Generation**: The LLM synthesizes the final answer using only the most relevant context.

## 🚦 Getting Started

### Prerequisites

- Docker Desktop (For running the Qdrant container)
- Ollama (Ensure llama3 and nomic-embed-text are pulled)
- Python 3.10+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/personal-semantic-agent.git
   cd personal-semantic-agent
   ```

2. Start the Vector Database:
   ```bash
   docker-compose up -d
   ```

3. Install Dependencies:
   ```bash
   #  using uv (recommended)
   uv sync
   ```

4. Run the Agent:
   ```bash
   python main.py
   ```

## 📺 Watch the Build

I built this entire system live on YouTube, explaining every engineering decision from library selection to pipeline optimization.


## ⚖️ License

MIT License - feel free to use this for your own projects!