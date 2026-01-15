import json
import faiss
from sentence_transformers import SentenceTransformer
import time

encoding_model = "sentence-transformers/all-MiniLM-L6-v2"

start_time = time.time()
model = SentenceTransformer(encoding_model)
model.half().eval()
with open("values.json") as f:
    vals = json.load(f)
end_time = time.time()
print(f"Loading vals time: {end_time - start_time:.2f} seconds")

with open("entity_relations.json") as f:
    entity_relations = json.load(f)
end_time = time.time()
print(f"Loading ent, rel time: {end_time - start_time:.2f} seconds")


start_time = time.time()
index = faiss.read_index("/home/rtn27/LMLM_develop/large-index-experiment/full_database_index_v1.faiss")
end_time = time.time()
print(f"Read index time: {end_time - start_time:.2f} seconds")


start_time = time.time()
prompt = "Arthur's magazine publication date"
prompt_embedding = model.encode([prompt])
distances, indices = index.search(prompt_embedding, k=5)
end_time = time.time()
print(f"Top k search time: {end_time - start_time:.2f} seconds")

for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}: {vals[idx]}")
    print(f"Entity relation {entity_relations[idx]}")
    print("\n\n")