""" goes over all the documents in the DATA_FOLDER, gives them to ingestion.py
gets the chunks, 
embeds the chunks using database.py
stores the vectors in qdrant vector db 

takes the user query, embeds it using ollama_client.py

retrieves similar chunks from qdrant db using database.py search function

reranks the retrieved chunks using reranking.py

finally gives the chunks to llm to generate answer. 

"""


import os
from core.ingestion import IngestionPipeline
from core.database import VectorDB
from core.reranking import Reranker
from api.ollama_client import OllamaClient
from config import DATA_FOLDER, TOP_K_RETRIEVAL, TOP_K_RERANK

class Pipeline:
    def __init__(self):
        self.ingestion = IngestionPipeline()
        self.db= VectorDB()
        self.reranker = Reranker()
        self.llm = OllamaClient()
        # initialize the vector db 
        self.db.ensure_collection()

    # Scan the data folder and ingest documents using ingestion pipeline, then embed and store in vector db
    def index_documents(self):
            """Scans folder and indexes documents ONLY if they are new."""
            if not os.path.exists(DATA_FOLDER):
                os.makedirs(DATA_FOLDER)
                print(f"📁 Data source folder '{DATA_FOLDER}' created. You can add documents to it.")
                return

            files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(('.pdf', '.docx', '.txt'))]
            
            if not files:
                print("⚠️ No files found to index.")
                return

            print(f"📂 Found {len(files)} files. Checking for new content...")
            
            for file in files:
                file_path = os.path.join(DATA_FOLDER, file)
                if self.db.file_exists(file_path):
                    #print(f"⏩ Skipping {file} (Already indexed)")
                    continue
                chunks = self.ingestion.process_file(file_path)
                if chunks:
                    self.db.upsert_chunks(chunks)
                    print(f"✅ Indexed {len(chunks)} chunks from {file}")

    def query(self, user_query: str):
        print(f"\n🔎 Searching for: {user_query}")
        
        # 1. Retrieval
        raw_results = self.db.search(user_query, limit=TOP_K_RETRIEVAL)
        
        # 2. Reranking
        ranked_results = self.reranker.rerank(user_query, raw_results, top_k=TOP_K_RERANK)

        # 3. Context Construction
        context_text = "\n\n".join([f"Source ({r['meta']['source']}): {r['text']}" for r in ranked_results])
        
        # 4. LLM Generation
        system_prompt = """
You are a professional assistant. Answer the user's question ONLY using the provided context.
Every time you mention a fact, you MUST cite the source file name in brackets, that is the 'source' field in the context. and meta as well.
If the information is not in the context, say you don't know.
"""
        final_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}"
        
        print("🤖 Generating answer...")
        # Return the generator directly
        return self.llm.generate_response(system_prompt, final_prompt)