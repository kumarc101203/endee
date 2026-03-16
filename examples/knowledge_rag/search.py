import os
from endee import Endee
from sentence_transformers import SentenceTransformer
from openai import OpenAI

INDEX = "knowledge_base"

# -----------------------
# Connect to Endee
# -----------------------
client = Endee()
index = client.get_index(INDEX)

# -----------------------
# Load embedding model
# -----------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Initialize LLM (Groq API)
# -----------------------
llm = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------
# User Query
# -----------------------
query = input("\nAsk a question about Endee: ")

# -----------------------
# Generate embedding
# -----------------------
query_embedding = model.encode(query).tolist()

# -----------------------
# Vector search in Endee
# -----------------------
results = index.query(
    vector=query_embedding,
    top_k=3
)

print("\nRetrieved Context:\n")

context = []

for r in results:
    text = r["meta"]["text"]
    context.append(text)
    print("-", text)

# -----------------------
# Combine retrieved context
# -----------------------
context_text = "\n".join(context)

# -----------------------
# Create RAG prompt
# -----------------------
prompt = f"""
You are an AI assistant that answers questions about the Endee vector database.

Use the following documentation context to answer the question.

Context:
{context_text}

Question:
{query}

Answer clearly and concisely based only on the provided context.
"""

# -----------------------
# LLM Response
# -----------------------
response = llm.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# -----------------------
# Print AI answer
# -----------------------
print("\nAI Answer:\n")
print(response.choices[0].message.content)