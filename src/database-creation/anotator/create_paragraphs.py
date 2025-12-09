import json
from datasets import load_dataset, Dataset
    
input_file = "/home/rtn27/LMLM/build-database/data/hotpot_dev_distractor_v1.json"
with open(input_file, "r") as f:
    qa_data = json.load(f)

data = [] #list of paragraphs of data to annotate
for e in qa_data:

    context = e["context"]
    sentences = []
    for c in context:
        p =(c[0]) + "\n"
        p +=  "".join([sentence for sentence in c[1]])
        sentences.append(p)

    data.append({"paragraphs": sentences})
    

dataset = Dataset.from_list(data)
dataset = dataset.shuffle(seed = 42)

# Save to JSON file
output_file = "/home/rtn27/LMLM/build-database/data/atomic_sentences_hotpotqa_1k_seed_42.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset[:1000], f, indent=2, ensure_ascii=False)
