import json
import random

with open("/home/rtn27/.cache/huggingface/hub/datasets--kilian-group--LMLM-database/snapshots/a0a9141268a614c6e050e1a5a3a7ebae3e4ee43d/dwiki6.1M-annotator_database.json") as f:
    db = json.load(f)

num_triplets_original = len(db["triplets"])
print("original triplets:", num_triplets_original)

print("original values :",len(db["return_values"]) )
print("original relationships :",len(db["relationships"]) )
print("original entities :",len(db["entities"]) )

with open("entity_relations.json") as f:
    ers = json.load(f)

with open("values.json") as f:
    vals = json.load(f)

print("entity_relations count:", len(ers))
print("values count:", len(vals))

N = len(db["triplets"])
i = random.randrange(N)

ent_orig, rel_orig, val_orig = db["triplets"][i]  # adjust to real structure
print("ORIG:", ent_orig, rel_orig, val_orig)
print("NEW ER:", ers[i])
print("NEW VAL:", vals[i])