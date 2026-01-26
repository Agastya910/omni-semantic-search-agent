from pipeline import Pipeline
import logging
# Add this at the very top of main.py
logging.getLogger("httpx").setLevel(logging.WARNING)
from config import DATA_FOLDER

if __name__ == "__main__":
    print("🚀 Starting RAG Agent...")
    agent = Pipeline()
    
    agent.index_documents()

    print("\n✅ System Ready. Type 'exit' to quit.")
    
    while True:
        query = input(f"\n Hi I am here to help ask me questions from the files in the folder >> {DATA_FOLDER} ")
        if query.lower() in ["exit", "quit"]:
            break
        
        response_generator = agent.query(query)
        
        print(f"\n💡 Answer: ", end="", flush=True)
        for chunk in response_generator:
            print(chunk, end="", flush=True)
        print("\n")