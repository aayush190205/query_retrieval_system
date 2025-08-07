import os
import requests
from app.semantic_search import search_similar_chunks

# Get your OpenRouter API key
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Step 1: Define your query
query = "Does this policy cover maternity expenses, and what are the conditions?"

# Step 2: Retrieve top matching chunks
results = search_similar_chunks(query, top_k=3)
print(results)
print('\n')
context = "\n---\n".join([r["text"] for r in results])

# Step 3: Define the prompt
messages = [
    {
        "role": "system",
        "content": "You are a insurance expert. Answer the user's question based on if the insurance is cover in this plan or not given in the context using only the context provided. Be accurate and concise."
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    }
]

# Step 4: Send request to OpenRouter
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer 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",
        "HTTP-Referer": "http://localhost",  # or your website
        "X-Title": "Query Retrieval System"
    },
    json={
        "model": "openrouter/horizon-beta",
        "messages": messages,
        "temperature": 0.0
    }
)

# Step 5: Extract and print the answer
reply = response.json()
print("Raw reply:", reply)

print("\n💡 Answer:\n", reply['choices'][0]['message']['content'])
