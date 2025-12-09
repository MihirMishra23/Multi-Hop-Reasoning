import faiss
from sentence_transformers import SentenceTransformer
import time
import json 

def _normalize_text(text: str) -> str:
    return text.lower().replace("_", " ").strip()
encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(encoding_model, device = "cuda")
model.half().eval()

with open('/home/rtn27/LMLM_develop/large-index-experiment/first_100k_texts.json', 'r') as f:
    entity_relations = json.load(f)

# Load the saved index
loaded_index = faiss.read_index('/home/rtn27/LMLM_develop/large-index-experiment/saved_index.faiss')
print(f"Loaded index with {loaded_index.ntotal} vectors")

# Example search query
query_text = "george gregory cromer previous alec position"
normalized_query = _normalize_text(query_text)

# Encode the query
start_time = time.time()
query_embedding = model.encode([normalized_query], convert_to_numpy=True)
end_time = time.time()
print(f"Time to encode query: {end_time - start_time:.4f} seconds")

# Search the index
k = 5  # Number of nearest neighbors to retrieve
start_time = time.time()
distances, indices = loaded_index.search(query_embedding, k)
end_time = time.time()
print(f"Time to search index: {end_time - start_time:.4f} seconds")

# Display results
print(f"\nTop {k} results for query: '{query_text}'")
for i in range(k):
    if indices[0][i] != -1:  # Valid result
        result_idx = indices[0][i]
        distance = distances[0][i]
        result_text = entity_relations[result_idx] if result_idx < len(entity_relations) else "Index out of range"
        print(f"Rank {i+1}: Distance={distance:.4f}, Text='{result_text}'")
