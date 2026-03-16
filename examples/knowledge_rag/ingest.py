import os
import re
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer

INDEX = "knowledge_base"
DOCS_PATH = "data/docs"

client = Endee()

# ---------- Create index if not exists ----------
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


# ---------- Chunking function ----------
def chunk_text(text, size=600):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size]
        chunks.append(chunk)
        start += size

    return chunks


documents = []
sources = []

print("Reading documentation files...")

for file in os.listdir(DOCS_PATH):

    if file.endswith(".md"):

        path = os.path.join(DOCS_PATH, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # ---------- Clean markdown ----------
        text = re.sub(r'`+', '', text)
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        chunks = chunk_text(text)

        for chunk in chunks:

            chunk = chunk.strip()

            # ignore very small fragments
            if len(chunk) > 120:
                documents.append(chunk)
                sources.append(file)

print("Total chunks:", len(documents))


# ---------- Generate embeddings ----------
print("Generating embeddings...")
embeddings = model.encode(documents)


# ---------- Prepare items ----------
items = []

for i, emb in enumerate(embeddings):

    items.append({
        "id": f"doc{i}",
        "vector": emb.tolist(),
        "meta": {
            "text": documents[i],
            "source": sources[i]
        }
    })


# ---------- Upload to Endee ----------
print("Uploading vectors...")

index.upsert(items)

print("Documentation vectors inserted successfully.")