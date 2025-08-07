from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.ingestion.parser import read_document
from app.embeddings.embedder import upsert_chunks
from app.semantic_search import search_similar_chunks

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class HackRXRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

@app.post("/api/v1/hackrx/run", response_model=HackRXResponse)
def hackrx_run(data: HackRXRequest):
    pdf_url = data.documents
    questions = data.questions

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_data = requests.get(pdf_url)
            if pdf_data.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download document")
            tmp_file.write(pdf_data.content)
            tmp_path = tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    chunks = read_document(tmp_path)
    upsert_chunks(chunks, doc_id="submission-doc")

    answers = []
    for q in questions:
        top_chunks = search_similar_chunks(q, top_k=4)
        context = "\n---\n".join([c["text"] for c in top_chunks])

        messages = [
            {
                "role": "system",
                "content": "You are an expert insurance policy reader. Answer the question ONLY based on the context below. If not found, reply: 'Not mentioned in the policy document.' First say YES/NO, then % if applicable, and mention source lines."
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
        ]

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "HackRX Retrieval System"
            },
            json={
                "model": "openrouter/horizon-beta",
                "messages": messages,
                "temperature": 0.0
            }
        )

        if response.status_code == 200:
            try:
                result = response.json()
                answer = result["choices"][0]["message"]["content"] if "choices" in result else "⚠️ Unexpected model response"
            except:
                answer = "⚠️ Could not parse model response"
        else:
            answer = "⚠️ Model request failed"

        answers.append(answer)
        answers.append(pdf_url)

    return {"answers": answers}
