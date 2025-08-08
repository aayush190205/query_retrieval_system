import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

def search_similar_chunks(query: str, top_k: int = 6):
    query_vector = model.encode(query).tolist()

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = result.get("matches", [])
    results = []
    seen_texts = set()

    for match in matches:
        chunk_text = match["metadata"].get("text", "").strip()
        if chunk_text and chunk_text not in seen_texts:
            results.append({
                "score": match["score"],
                "text": chunk_text
            })
            seen_texts.add(chunk_text)

    if len(results) < 2:
        results.append({
            "score": 0.0,
            "text": "Not mentioned in the policy document."
        })

    return results
