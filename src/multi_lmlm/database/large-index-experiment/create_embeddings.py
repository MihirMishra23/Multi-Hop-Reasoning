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


def main():
    encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(encoding_model)
    model.half().eval()
    start_time = time.time()
    with open('/home/rtn27/LMLM_develop/large-index-experiment/entity_relations.json', 'r') as f:
        entity_relations = json.load(f)

    end_time = time.time()
    print(f"Time to lod stuff: {end_time - start_time:.4f} seconds")

    pool = model.start_multi_process_pool()

    BATCH_SIZE = 512
    CHUNK_SIZE = 10600000 
    start_time = time.time()
    # encodings = model.encode(entity_relations[:100000], convert_to_numpy = True, batch_size =BATCH_SIZE, show_progress_bar=True, pool = pool)
    
    chunk_nb = 1
    for start in range(0, len(entity_relations), CHUNK_SIZE):
        print(f"On chunk number {chunk_nb}")
        batch_texts = entity_relations[start : start + CHUNK_SIZE]

        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
            pool = pool
        )

        np.save(f"/home/rtn27/LMLM_develop/large-index-experiment/embeddings_{start//CHUNK_SIZE:05d}.npy", embeddings.astype("float32"))
        chunk_nb += 1

    model.stop_multi_process_pool(pool)
    end_time = time.time()
    print("finished all chunks!")
    print(f"Time to encode stuff: {end_time - start_time:.4f} seconds")
    index = faiss.GpuIndexFlatL2(res, 384)
    start_time = time.time()
    index.add(encodings)
    end_time = time.time()
    print(f"Time to add encodings to index: {end_time - start_time:.4f} seconds")
    print(index.ntotal)

    faiss.write_index(faiss.index_gpu_to_cpu(index), '/home/rtn27/LMLM_develop/large-index-experiment/full_database_index_v1.faiss')
    print("Index saved successfully")

if __name__ == "__main__":
    main()