import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import time

def _normalize_text(text: str) -> str:
    return text.lower().replace("_", " ").strip()

np.random.seed(1234)
xb = np.random.random((1000000, 96)).astype('float32')
xq = np.random.random((10000, 96)).astype('float32')
xt = np.random.random((100000, 96)).astype('float32')
 
res = faiss.StandardGpuResources()
# Disable the default temporary memory allocation since an RMM pool resource has already been set.
res.noTempMemory()



encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(encoding_model, device = "cuda")
model.half().eval()


start_time = time.time()
with open('/home/rtn27/LMLM_develop/large-index-experiment/first_100k_texts.json', 'r') as f:
    entity_relations = json.load(f)
end_time = time.time()
print(f"Time to load JSON: {end_time - start_time:.4f} seconds")

start_time = time.time()
encodings = model.encode(entity_relations, convert_to_numpy = True, batch_size =1)
end_time = time.time()
print(f"Time to encode stuff: {end_time - start_time:.4f} seconds")

index = faiss.GpuIndexFlatL2(res, 384)
start_time = time.time()
index.add(encodings)
end_time = time.time()
print(f"Time to add encodings to index: {end_time - start_time:.4f} seconds")
print(index.ntotal)

faiss.write_index(faiss.index_gpu_to_cpu(index), '/home/rtn27/LMLM_develop/large-index-experiment/saved_index.faiss')
print("Index saved successfully")
