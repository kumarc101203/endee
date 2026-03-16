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

docs = [
    "Endee is a high performance vector database for AI search.",
    "Vector databases enable semantic search systems.",
    "Retrieval augmented generation improves LLM answers.",
    "Embeddings convert text into numerical vectors."
]

print("Generating embeddings...")
embeddings = model.encode(docs)

print("Uploading vectors...")

items = []

for i, emb in enumerate(embeddings):
    items.append({
        "id": f"doc{i}",
        "vector": emb.tolist(),
        "meta": {"text": docs[i]}
    })

index.upsert(items)

print("Vectors inserted successfully.")