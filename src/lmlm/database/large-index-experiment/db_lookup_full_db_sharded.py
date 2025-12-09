import json
import faiss
from multiprocessing import Pool
import time

from sentence_transformers import SentenceTransformer

NUM_DEVICES = 6
K = 5

paths = [f"/home/rtn27/LMLM_develop/large-index-experiment/index/shards/database_index_shard_{i}.faiss" for i in range(6)]

_index = None


def load_index(path):
    global _index
    if _index is None:
        print("loading index from memory")
        _index = faiss.read_index(path)
    

def search_index(query_embedding):
    global _index
    if _index is None:
        raise Exception("Cannot search, load the index first")
    else:
        print("Index already in memory, using it")
    return _index.search(query_embedding, K)


if __name__ == "__main__" :
    start_time = time.time()
    with open("values.json") as f:
        vals = json.load(f)
    end_time = time.time()
    print(f"Loading vals time: {end_time - start_time:.2f} seconds")


    now=time.time()
    encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(encoding_model)
    end = time.time()
    print(f"Total to load model took: ", end - now)
    prompt = "Arthur's magazine publication date"
    now=time.time()
    prompt_embedding = model.encode([prompt])
    end = time.time()
    print(f"Total to create embedding took: ", end - now)
    

    now = time.time()
    print(f"Total time took: ", end - now)
    print("finished loading indexies")


    now = time.time()
    with Pool(NUM_DEVICES) as p:
        indexes = p.map(load_index, paths)
        distances_and_indexes_arr = p.map(search_index, [prompt_embedding] * NUM_DEVICES)
        # Combine all distances and indices from all shards
        all_distances = []
        all_indices = []
        
        for distances, indices in distances_and_indexes_arr:
            all_distances.extend(distances[0])  # distances[0] because it's a 2D array
            all_indices.extend(indices[0])      # indices[0] because it's a 2D array
        
        # Find indices of top k minimum distances
        sorted_idx = sorted(range(len(all_distances)), key=lambda i: all_distances[i])
        top_k_indices = [all_indices[i] for i in sorted_idx[:K]]
        top_k_distances = [all_distances[i] for i in sorted_idx[:K]]
        
        print(f"Top {K} indices: {top_k_indices}")
        print(f"Top {K} distances: {top_k_distances}")
        for i, idx in enumerate(top_k_indices):
            print(f"Result {i+1}: Index {idx}, Distance {top_k_distances[i]:.4f}, Value: {vals[idx]}")

    end = time.time()
    print(f"Total time took: ", end - now)


    # now = time.time()
    # index = faiss.read_index("/home/rtn27/LMLM_develop/large-index-experiment/index/full_database_index_v1.faiss")
    # end = time.time()
    
