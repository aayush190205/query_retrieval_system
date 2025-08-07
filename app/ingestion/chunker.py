# app/ingestion/chunker.py

def chunk_text(text: str, max_words: int = 200, overlap: int = 20) -> list:
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(' '.join(chunk))
        i += max_words - overlap  # step forward with overlap

    return chunks
