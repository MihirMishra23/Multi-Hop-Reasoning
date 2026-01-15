from dataclasses import dataclass, field
from datasets import load_dataset
from trainer.lmlm_basetrainer import LMLMGRPOTrainer
# from trainer.lmlm_grpotrainer import LMLMGRPOTrainer
from transformers import AutoTokenizer, HfArgumentParser
from eval.metrics import exact_match_score
from trl.trainer.grpo_config import GRPOConfig
from multi_lmlm.constants import ANSWER_START_TOKEN, ANSWER_END_TOKEN
import wandb
import os

@dataclass
class ScriptArguments:
    """Arguments for dataset and paths."""
    model_path: str = field(metadata={"help": "Path to the pretrained model"})
    database_path: str = field(metadata={"help": "Path to the LMLM database JSON file"})

    # Dataset parameters
    dataset_name: str = field(
        default="hotpotqa/hotpot_qa",
        metadata={"help": "HuggingFace dataset name"}
    )
    dataset_config: str = field(
        default="distractor",
        metadata={"help": "Dataset configuration"}
    )
    train_size: int = field(default=8000, metadata={"help": "Number of training examples"})
    eval_size: int = field(default=100, metadata={"help": "Number of evaluation examples"})


@dataclass
class LMLMArguments:
    """LMLM specific arguments."""
    adaptive_k: bool = field(
        default=False,
        metadata={"help": "Use adaptive k for database retrieval"}
    )
    tools: bool = field(
        default=False,
        metadata={"help": "Enable tool calling"}
    )


def extract_answer_from_tags(text: str):
    """Extract answer from between answer tags."""
    try:
        return text.split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0]
    except Exception:
        return ""


def em_accuracy(completions, solution, **kwargs):
    """Calculate exact match accuracy for completions."""
    return [1 if exact_match_score(extract_answer_from_tags(c), s) else 0 
            for (c, s) in zip(completions, solution)]


def process_example(example):
    """Process HotpotQA example into prompt-solution format."""
    return {
        "prompt": f"Question:\n{example['question']}\nAnswer:\n",
        "solution": example["answer"]
    }


def main():
    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((ScriptArguments, LMLMArguments, GRPOConfig))
    script_args, lmlm_args, grpo_config = parser.parse_args_into_dataclasses()

    # wandb name
    grpo_config.run_name = script_args.model_path.split('/')[-1]+'-'+str(grpo_config.loss_type)+'-g'+str(grpo_config.num_generations)+'-bs'+str(grpo_config.per_device_train_batch_size)+'-s'+str(grpo_config.gradient_accumulation_steps)+'-b'+str(grpo_config.beta)+'-ep'+str(grpo_config.num_train_epochs)+'-n'+str(script_args.train_size)
    # grpo_config.output_dir = os.path.join(script_args.save_dir, grpo_config.run_name)
    os.makedirs(grpo_config.output_dir, exist_ok=True)

    if wandb.run is not None:
        wandb.run.name = grpo_config.run_name
    else:
        print(f"Wandb run is not initialized, skipping wandb name setting")

    # Load and process dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split="train")
    dataset = dataset.shuffle(seed=42)
    
    processed_dataset = dataset.map(process_example)
    
    # Create train/eval splits
    total_size = len(processed_dataset)
    train_start = max(0, total_size - script_args.train_size - script_args.eval_size)
    train_end = min(total_size, train_start + script_args.train_size)
    
    train_set = processed_dataset.select(range(train_start, train_end))
    eval_set = processed_dataset.select(range(train_end, total_size))
    
    print(f"Train set size: {len(train_set)}")
    print(f"Eval set size: {len(eval_set)}")
    print(f"Train example: {train_set[0]}")
    print(f"Eval example: {eval_set[0]}")
    
    # Load tokenizer
    print(f"Loading tokenizer from: {script_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    
    
    print(f"GRPO Config:")
    print(f"  Output dir: {grpo_config.output_dir}")
    print(f"  Run name: {grpo_config.run_name}")
    print(f"  Num generations: {grpo_config.num_generations}")
    print(f"  Batch size: {grpo_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {grpo_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {grpo_config.learning_rate}")
    print(f"  Max grad norm: {grpo_config.max_grad_norm}")
    print(f"  Use vLLM: {grpo_config.use_vllm}")
    
    # Initialize trainer
    print("Initializing LMLMGRPOTrainer...")
    trainer = LMLMGRPOTrainer(
        model=script_args.model_path,
        reward_funcs=em_accuracy,
        lmlm_database_path=script_args.database_path,
        adaptive_k=lmlm_args.adaptive_k,
        processing_class=tokenizer,
        tools=lmlm_args.tools,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=grpo_config,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()