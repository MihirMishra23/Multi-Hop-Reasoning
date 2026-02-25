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
from data import get_dataset
import random

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
    return_triples: bool = field(
        default=False,
        metadata={"help": "Return triples for tool calling"}
    )
    use_inverses : bool = field(
        default = False,
        metadata={"help" : "When building the db, augment (e,r,v) -> (v,r,e) for each triplet"}
    )
    retrieval_threshold : float = field(
        default = 0.6,
        metadata = {"help" : "cosing similarity threshold"}
    )
    two_phase: bool = field(
        default=False,
        metadata={"help": "Enable two-phase generation (Phase 1: DB creation from contexts, Phase 2: QA with per-example DB)"}
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


def db_size_threshold(completions, **kwargs):
    """Reward function that checks if triplet to context character ratio is above 0.017."""
    rewards = []
    contexts = kwargs.get("contexts", [])
    for i, comp in enumerate(completions):
        try:
            triplets = comp.split("\n")
            # Sanity check to ensure this is a db generation (Phase 1)
            if "\t" in triplets[0] and "\t" in triplets[1]:
                # Count number of triplets (non-empty lines with tabs)
                num_triplets = sum(1 for t in triplets if "\t" in t)

                # Calculate context character count
                if i < len(contexts):
                    context = contexts[i]
                    # If context is a list of strings, join them
                    if isinstance(context, list):
                        context_str = "\n\n".join(context)
                    else:
                        context_str = str(context)
                    context_chars = len(context_str)

                    # Calculate ratio and assign reward
                    if context_chars > 0:
                        ratio = num_triplets / context_chars
                        print(f"DEBUG: Example {i} - Num Triplets: {num_triplets}, Context Chars: {context_chars}, Ratio: {ratio:.4f}")
                        reward = 1 if ratio > 0.01 else 0
                    else:
                        reward = 0
                else:
                    reward = 0  # No context available
            else:
                reward = 0  # Phase 2 completion or malformed
        except Exception:
            reward = 0  # If parsing fails
        rewards.append(reward)
    return rewards


def process_example(example):
    """Process HotpotQA example into prompt-solution format."""
    return {
        "prompt": f"Question:\n{example['question']}\nAnswer:\n",
        "contexts": example.get("contexts", []),
        "solution": example["answers"][0]
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

    train_dataset = get_dataset(name = script_args.dataset_name, setting = script_args.dataset_config, split = "train", sub_split = "train", limit = script_args.train_size, seed = 42)
    test_dataset = get_dataset(name = script_args.dataset_name, setting = script_args.dataset_config, split = "train", sub_split = "eval", limit = script_args.eval_size, seed = 42)
    
    train_set = train_dataset.map(process_example)
    eval_set = test_dataset.map(process_example)
    
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
    print(f"  Adaptive k: {lmlm_args.adaptive_k}")
    print(f"  Return triples: {lmlm_args.return_triples}")

    # Initialize trainer
    print("Initializing LMLMGRPOTrainer...")
    trainer = LMLMGRPOTrainer(
        retrieval_threshold = lmlm_args.retrieval_threshold,
        use_inverses = lmlm_args.use_inverses,
        model=script_args.model_path,
        reward_funcs=[em_accuracy, db_size_threshold],
        lmlm_database_path=script_args.database_path,
        adaptive_k=lmlm_args.adaptive_k,
        return_triples=lmlm_args.return_triples,
        processing_class=tokenizer,
        tools=lmlm_args.tools,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=grpo_config,
        two_phase=lmlm_args.two_phase,
    )
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    print("Training completed!")


if __name__ == "__main__":
    main()
