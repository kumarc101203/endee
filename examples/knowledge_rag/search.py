from endee import Endee
from sentence_transformers import SentenceTransformer

INDEX = "knowledge_base"

client = Endee()
index = client.get_index(INDEX)

model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("Enter your query: ")

query_embedding = model.encode(query).tolist()

results = index.query(
    vector=query_embedding,
    top_k=3
)

print("\nTop Results:\n")

for r in results:
    print(r["meta"]["text"])