from datasets import load_dataset
from lmlm_grpo import LMLMGRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.rewards import accuracy_reward
from eval.metrics import exact_match_score
from trl.trainer.grpo_config import GRPOConfig
from lmlm.constants import ANSWER_START_TOKEN, ANSWER_END_TOKEN

model_path = "/share/j_sun/rtn27/checkpoints/qwen3-1.7B_sft_v1.3_5743/sft_ep5_bsz32_new_qa_26-01-06_04-44/"

dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
dataset = dataset.shuffle(seed = 42)

def process_ex(example):
    example = {"prompt": "Question:\n" + example["question"] + "\nAnswer:\n", "solution": example["answer"]}
    return example

processed_dataset = dataset.map(process_ex)

train_set = processed_dataset.select(range(len(processed_dataset)-8100,len(processed_dataset)-100, 1))
eval_set = processed_dataset.select(range(len(processed_dataset)-100,len(processed_dataset), 1))

print(train_set[0])
print(eval_set[0])

tok = AutoTokenizer.from_pretrained(model_path)

def extract_answer_from_tags(text : str):
    try:
        return text.split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0]
    except Exception as e:
        return ""

def em_accuracy(completions, solution, **kwargs):
    return [1 if exact_match_score(extract_answer_from_tags(c), s) else 0 for (c,s) in zip(completions, solution)]

print(model_path.split('checkpoints/')[1].split("sft_ep")[0])

args = GRPOConfig(f"{model_path}-GRPO")
args.beta = 0.001 #same setting as search-r1 [https://arxiv.org/pdf/2503.09516] and Deepseek-r1
args.num_generations = 8
args.num_generations_eval = 1 # faster eval
args.per_device_train_batch_size = 16
args.gradient_accumulation_steps = 32
args.use_vllm = True
args.log_completions=True
args.vllm_mode = 'colocate'
args.max_completion_length = 512
args.do_eval = True
args.eval_strategy = "steps"
args.eval_steps = 4
args.logging_steps = 1
args.run_name = model_path.split('checkpoints/')[1].split("sft")[0]

trainer = LMLMGRPOTrainer(
    model=model_path,
    reward_funcs=em_accuracy,
    lmlm_database_path = "/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json",
    adaptive_k = True,
    processing_class=tok,
    tools = True,
    train_dataset=train_set,
    eval_dataset=eval_set,
    args=args,
)

print(trainer.args.output_dir)

from lmlm.constants import DB_END_TOKEN
result = "China" + DB_END_TOKEN
print(trainer.processing_class(result, add_special_tokens = False)["input_ids"])
print(trainer.processing_class(result, add_special_tokens = True)["input_ids"])

trainer.train()