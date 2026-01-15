import faiss
import numpy as np
import time

NUM_SHARDS = 6
curr_idx_start = 0
for i in range(NUM_SHARDS):     # around 5
    start_time = time.time()
    index = faiss.IndexIDMap2(faiss.IndexFlatL2(384))
    print(f"processing chunk {i}")
    arr = np.load(f"/home/rtn27/LMLM_develop/large-index-experiment/embeddings-unnormalized/embeddings_{i:05d}.npy", mmap_mode="r")
    print("finished loading chunk")
    arr_len = arr.shape[0]
    ids = np.arange(curr_idx_start, curr_idx_start + arr_len, dtype = "int64")
    index.add_with_ids(np.asarray(arr, dtype="float32"), ids)   # FAISS requires normal ndarray
    print("Finished adding this chunk to index")
    faiss.write_index(index, f'/home/rtn27/LMLM_develop/large-index-experiment/index/shards/database_index_shard_{i}.faiss')
    curr_idx_start += arr_len
    end_time = time.time()
    print(f"Time to add encodings to index: {end_time - start_time:.4f} seconds")
print("finished building all indexes")

