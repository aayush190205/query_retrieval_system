import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ingestion.parser import read_document
from app.ingestion.chunker import chunk_text
from app.embeddings.embedder import upsert_chunks


file_path = r"C:\Users\user\OneDrive\Desktop\query_retrieval_system\data\BAJHLIP23020V012223.pdf"
text = read_document(file_path)

print("\n--- Preview of full document ---\n")
print(text[:500])

chunks = chunk_text(text, max_words=200, overlap=20)

upsert_chunks(chunks, doc_id="bajaj-policy")
print(f"\nTotal Chunks Created: {len(chunks)}")
print("\n--- Preview of First Chunk ---\n")
print(chunks[0])    
