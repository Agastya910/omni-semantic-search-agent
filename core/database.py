""" Creating the qdrant database if not already present, adding new information to it. """
""" does the embedding using ollama nomic-embed-text model which we run using ollama client locally on our machine """
""" Embedding is stored in qdrant vector db along with sparse vectors for better retrieval """

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams, Modifier
import uuid
import time
from api.ollama_client import OllamaClient

from config import VECTOR_DB_URL, COLLECTION_NAME, SPARSE_VECTOR_NAME

class VectorDB:
    def __init__(self):
        self.client= QdrantClient(url=VECTOR_DB_URL, timeout=60)
        self.ollama_client = OllamaClient()
        self.ollama= OllamaClient()
    def ensure_collection(self):
        """ in case a collection is not present create it """
        max_retries = 5
        for i in range(max_retries):
            try: 
                # Try to get the collection; if it doesn't exist, create it
                try:
                    self.client.get_collection(collection_name=COLLECTION_NAME)
                except Exception:
                    print(f"Creating collection: {COLLECTION_NAME}")
                    self.client.create_collection(
                        collection_name=COLLECTION_NAME,
                        vectors_config={
                            "dense": VectorParams(size=768, distance=Distance.COSINE)
                        },
                        sparse_vectors_config={
                            SPARSE_VECTOR_NAME: SparseVectorParams(
                                index=models.SparseIndexParams(on_disk=False),
                                modifier=Modifier.IDF
                            )
                        }
                    )
                return
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Retrying to ensure collection due to error: {e}")
                    time.sleep(2 ** i)  # Exponential backoff
                else:
                    raise e
                
    def upsert_chunks(self, chunks):
        points = []
        for chunk in chunks:
            dense_vector = self.ollama.get_embeddings(chunk["text"])
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    SPARSE_VECTOR_NAME: models.Document(text=chunk["text"], model="Qdrant/bm25")
                },
                payload={"text": chunk["text"], **chunk["metadata"]}
            ))
        
        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query: str, limit:  int = 50):
        query_dense= self.ollama_client.get_embeddings(query)
        response= self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=limit),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using=SPARSE_VECTOR_NAME, 
                    limit=limit
                ),
            ], 
            query = models.FusionQuery(fusion= models.Fusion.RRF),
            limit=limit, 
            with_payload=True
        )
        return response.points
    
    def file_exists(self, file_path: str) -> bool:
        """Checks if any chunks from this file already exist in the DB."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        result = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=file_path))
                ]
            ),
            limit=1,
        )
        return len(result[0]) > 0


