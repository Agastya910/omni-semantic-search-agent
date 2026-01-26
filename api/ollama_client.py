import ollama
from config import EMBEDDING_MODEL, LLM_MODEL

class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(timeout=120.0)

    def get_embeddings(self, text: str)-> list[float]:
        """ Generate vector embedding for a text string."""
        response = self.client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    
    def generate_response(self, system_prompt: str, user_prompt: str):
        """Generates text response from the LLM."""
        response = self.client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], stream=True
        )
        for chunk in response:
            yield chunk["message"]["content"]

