from data import get_dataset

ds = get_dataset(name = "hotpotqa", setting = "distractor", split = "train", sub_split = "easy", seed = 42, limit = 1000)

count = 0
for e in ds:
    print("question: ", e["question"])
    print("answer: ", e["answers"])
    print("golden context", e["golden_contexts"])
    count += 1
    if count > 10:
        break

print("\n\n")
from datasets import load_dataset
ds = load_dataset("hotpot_qa", "distractor", split="train")
for e in ds:
    if "Spanish breed of dog typical of the region of Las Enca" in e["question"]:
        print(e)