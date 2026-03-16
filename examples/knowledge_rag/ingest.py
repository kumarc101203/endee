from endee import Endee, Precision
from sentence_transformers import SentenceTransformer

INDEX = "knowledge_base"

client = Endee()

try:
    client.create_index(
        name=INDEX,
        dimension=384,
        space_type="cosine",
        precision=Precision.INT8
    )
    print("Index created")
except:
    print("Index already exists")

index = client.get_index(INDEX)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents from file
with open("data/endee_docs.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

print("Generating embeddings...")
embeddings = model.encode(documents)

print("Uploading vectors...")

items = []

for i, emb in enumerate(embeddings):
    items.append({
        "id": f"doc{i}",
        "vector": emb.tolist(),
        "meta": {"text": documents[i]}
    })

index.upsert(items)

print("Vectors inserted successfully.")