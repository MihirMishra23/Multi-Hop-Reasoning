from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

dataset2 = load_dataset("trl-lib/DeepMath-103K", split="train")

for d in dataset:
    print(d)
    break


print("\n\n\n-----\n\n\n")
for d2 in dataset2:
    print(d2)
    break

def process_ex(example):
    example = {"prompt": [{"content": example["question"], "role": "user"}], "solution": example["answer"]}
    return example


processed_dataset =dataset.map(process_ex)
print("\n\n\n-----\n\n\n")
for d in processed_dataset:
    print(d)
    break

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=processed_dataset,
)
trainer.train()