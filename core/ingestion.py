""" Read a file , convert it using DocumentConverter and create chunks using HybridChunker """
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from pathlib import Path

class IngestionPipeline:
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(tokenizer= 'sentence-transformers/all-MiniLM-L6-v2')

    def process_file(self, file_path:str):
        
        """Reads a file, converts to Markdown, and chunks it with a fallback for .txt files."""
        path = Path(file_path)
        print(f"📄 Processing: {file_path}...")
        
        try:
            if path.suffix.lower() == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                doc = DoclingDocument(name=path.stem)
                doc.add_text(label="text", text=content)
            else:
                result = self.converter.convert(file_path)
                doc = result.document

            chunks = list(self.chunker.chunk(doc))

            processed_chunks = []

            for i, chunk in enumerate(chunks):
                heading = "General"
                if chunk.meta.headings:
                    heading = chunk.meta.headings[0]
                processed_chunks.append({
                    "text": chunk.text,
                    "metadata": {
                        "source": str(file_path),
                        "page": heading,
                        "chunk_id": i
                    }
                })
            return processed_chunks
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []
        
        