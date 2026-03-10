import math
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from trainer.lmlm_basetrainer import LMLMGRPOTrainer, parse_triplets
# from trainer.lmlm_grpotrainer import LMLMGRPOTrainer
from transformers import AutoTokenizer, HfArgumentParser
from eval.metrics import exact_match_score
from trl.trainer.grpo_config import GRPOConfig
from reward_func import em_accuracy, f1_reward, db_coverage_reward, db_size_threshold
import wandb
import os
from data import get_dataset, get_dataset_from_path
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
    train_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to custom train JSON (list of QA objects). If set, overrides dataset_name for training."},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to custom eval JSON. If set, overrides dataset_name for eval. If only train_data_path set, eval uses HotpotQA."},
    )


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
    reward_func: str = field(
        default="em_coverage",
        metadata={"help": "Reward function to use"}
    )


def process_example(example):
    """Process HotpotQA example into prompt-solution format."""
    return {
        "prompt": f"Question:\n{example['question']}\nAnswer:\n",
        "question": example["question"],
        "contexts": example.get("golden_contexts", []),
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
    if script_args.train_data_path:
        print(f"Loading custom train data from: {script_args.train_data_path}")
        train_dataset = get_dataset_from_path(
            script_args.train_data_path, limit=script_args.train_size, seed=42
        )
        if script_args.eval_data_path:
            print(f"Loading custom eval data from: {script_args.eval_data_path}")
            test_dataset = get_dataset_from_path(
                script_args.eval_data_path, limit=script_args.eval_size, seed=42
            )
        else:
            print(f"Loading eval from HotpotQA (eval_data_path not set)")
            test_dataset = get_dataset(
                name=script_args.dataset_name,
                setting=script_args.dataset_config,
                split="train",
                sub_split="eval",
                limit=script_args.eval_size,
                seed=42,
            )
    else:
        print(f"Loading dataset: {script_args.dataset_name}")
        train_dataset = get_dataset(
            name=script_args.dataset_name,
            setting=script_args.dataset_config,
            split="train",
            sub_split="train",
            limit=script_args.train_size,
            seed=42,
        )
        test_dataset = get_dataset(
            name=script_args.dataset_name,
            setting=script_args.dataset_config,
            split="train",
            sub_split="eval",
            limit=script_args.eval_size,
            seed=42,
        )

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

    # In two_phase (TRR++) mode:
    #   - em_accuracy scores Phase-2 QA completions (returns None for Phase-1 triplets)
    #   - db_coverage_reward scores Phase-1 DB completions via graph reachability (returns None for Phase-2)
    # db_size_threshold is excluded; db_coverage_reward replaces it with a semantically richer signal.
    if lmlm_args.two_phase:
        reward_funcs = []
        if "em" in lmlm_args.reward_func:
            reward_funcs.append(em_accuracy)
        if "f1" in lmlm_args.reward_func:
            reward_funcs.append(f1_reward)
        if "coverage" in lmlm_args.reward_func:
            reward_funcs.append(db_coverage_reward)
        if "size" in lmlm_args.reward_func:
            reward_funcs.append(db_size_threshold)
    else:
        reward_funcs = [em_accuracy]

    # Initialize trainer
    print("Initializing LMLMGRPOTrainer...")
    trainer = LMLMGRPOTrainer(
        retrieval_threshold = lmlm_args.retrieval_threshold,
        use_inverses = lmlm_args.use_inverses,
        model=script_args.model_path,
        reward_funcs=reward_funcs,
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
