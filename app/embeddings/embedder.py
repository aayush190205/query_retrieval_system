import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
load_dotenv()

# Load keys
# openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
pinecone_region = os.getenv("PINECONE_REGION")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Create index if not exists
if pinecone_index not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_region
        )
    )

index = pc.Index(pinecone_index)





model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> list:
    return model.encode(text).tolist()



def upsert_chunks(chunks: list, doc_id: str):
    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embed_text(chunk)
        vectors.append({
            "id": f"{doc_id}-{i}",
            "values": vec,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} chunks to Pinecone.")
