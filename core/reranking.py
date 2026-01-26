from flashrank import Ranker, RerankRequest
from config import TOP_K
class Reranker:
    def __init__(self):
        # Uses a tiny model (~30MB) that runs fast on CPU/Low VRAM
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="opt")

    def rerank(self, query: str, search_results, top_k: int = TOP_K):
        """Re-orders Qdrant results by relevance."""
        
        # Convert Qdrant objects to FlashRank dictionaries
        passages = [
            {"id": str(res.id), "text": res.payload["text"], "meta": res.payload}
            for res in search_results
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)
        
        return results[:top_k]