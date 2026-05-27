"""
Eval-only script: initialize GRPO trainer with SFT checkpoint, run evaluate() only.
Walks the exact same code path as GRPO training (FSDP, vLLM colocate, _generate_two_phase,
_tool_call_loop, reward computation) but without any weight updates.

Usage:
    Same args as grpo_train.py. The script calls trainer.evaluate() instead of trainer.train().
"""

from dataclasses import dataclass, field
from typing import Optional
from trainer.lmlm_basetrainer import LMLMGRPOTrainer, parse_triplets
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from trl.trainer.grpo_config import GRPOConfig
from reward_func import em_accuracy, f1_reward, db_coverage_reward, db_size_threshold, format_reward_zero_rl
import json
import wandb
import os
from data import get_dataset


# ---------------------------------------------------------------------------
# Reuse the same argument dataclasses from grpo_train.py
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Path to the pretrained model"})
    database_path: str = field(metadata={"help": "Path to the LMLM database JSON file"})
    dataset_name: str = field(default="hotpotqa/hotpot_qa", metadata={"help": "HuggingFace dataset name"})
    dataset_config: str = field(default="distractor", metadata={"help": "Dataset configuration"})
    train_size: int = field(default=8000, metadata={"help": "Number of training examples"})
    eval_size: int = field(default=100, metadata={"help": "Number of evaluation examples"})


@dataclass
class LMLMArguments:
    retrieval_threshold: float = field(default=0.6)
    retrieval_top_k: int = field(default=1)
    use_chat_template: bool = field(default=False)
    two_phase: bool = field(default=False)
    reward_func: str = field(default="em_coverage")
    phase1_reward_type: str = field(default="binary")
    phase1_prompt_type: str = field(default="sft")
    phase1_db_weight_mode: str = field(default="count_dynamic")
    num_db_rollouts: int = field(default=1)


@dataclass
class AblationArguments:
    tier_path: Optional[str] = field(default=None)
    tier_min_score: int = field(default=1)
    tier_max_score: int = field(default=7)
    curriculum: bool = field(default=False)
    curriculum_phases: str = field(default="5-7,3-7,1-7")
    curriculum_steps: str = field(default="0.33,0.67")
    adaptive_k: bool = field(default=False)
    use_inverses: bool = field(default=False)
    vanilla_grpo: bool = field(default=False)
    tools: bool = field(default=False)
    return_triples: bool = field(default=False)


def process_example(example):
    return {
        "prompt": f"Question:\n{example['question']}\nAnswer:\n",
        "question": example["question"],
        "contexts": example.get("golden_contexts", []),
        "solution": example["answers"][0]
    }


def main():
    parser = HfArgumentParser((ScriptArguments, LMLMArguments, AblationArguments, GRPOConfig))
    script_args, lmlm_args, ablation_args, grpo_config = parser.parse_args_into_dataclasses()

    # Force do_eval on. We use train(max_steps=1, eval_on_start=True) instead
    # of plain evaluate() to avoid the FSDP lazy-init `_is_root` error that
    # happens when summon_full_params runs before any forward pass. train()
    # does the full FSDP setup, then eval_on_start triggers an eval BEFORE
    # any optimizer step -- so the eval uses pure SFT weights.
    grpo_config.do_eval = True
    grpo_config.eval_on_start = True
    grpo_config.max_steps = 1
    grpo_config.eval_strategy = "no"  # we only want the eval_on_start; no periodic eval
    grpo_config.save_strategy = "no"  # don't save anything

    grpo_config.run_name = "eval_only_" + os.path.basename(grpo_config.output_dir)
    os.makedirs(grpo_config.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    train_dataset = get_dataset(name=script_args.dataset_name, setting=script_args.dataset_config, split="train", sub_split="train", limit=script_args.train_size, seed=42)
    test_dataset  = get_dataset(name=script_args.dataset_name, setting=script_args.dataset_config, split="train", sub_split="eval",  limit=script_args.eval_size,  seed=42)

    train_set = train_dataset.map(process_example)
    eval_set  = test_dataset.map(process_example)
    print(f"Train set size: {len(train_set)}, Eval set size: {len(eval_set)}")

    # Load tokenizer
    print(f"Loading tokenizer from: {script_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)

    # Build reward functions (same as grpo_train.py)
    reward_funcs = []
    if "em"       in lmlm_args.reward_func: reward_funcs.append(em_accuracy)
    if "f1"       in lmlm_args.reward_func: reward_funcs.append(f1_reward)
    if "coverage" in lmlm_args.reward_func: reward_funcs.append(db_coverage_reward)
    if "size"     in lmlm_args.reward_func: reward_funcs.append(db_size_threshold)
    if "format"   in lmlm_args.reward_func: reward_funcs.append(format_reward_zero_rl)
    if not reward_funcs:
        reward_funcs = [em_accuracy]

    print("Initializing LMLMGRPOTrainer (eval-only mode)...")
    trainer = LMLMGRPOTrainer(
        model=script_args.model_path,
        reward_funcs=reward_funcs,
        lmlm_database_path=script_args.database_path,
        processing_class=tokenizer,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=grpo_config,
        retrieval_threshold=lmlm_args.retrieval_threshold,
        retrieval_top_k=lmlm_args.retrieval_top_k,
        use_chat_template=lmlm_args.use_chat_template,
        two_phase=lmlm_args.two_phase,
        phase1_reward_type=lmlm_args.phase1_reward_type,
        phase1_prompt_type=lmlm_args.phase1_prompt_type,
        phase1_db_weight_mode=lmlm_args.phase1_db_weight_mode,
        num_db_rollouts=lmlm_args.num_db_rollouts,
        adaptive_k=ablation_args.adaptive_k,
        tools=ablation_args.tools,
        return_triples=ablation_args.return_triples,
        use_inverses=ablation_args.use_inverses,
        vanilla_grpo=ablation_args.vanilla_grpo,
    )

    print("=" * 60)
    print("EVAL-ONLY MODE: calling trainer.train(max_steps=1, eval_on_start=True)")
    print("This walks the exact same code path as GRPO training")
    print("(FSDP wrap, vLLM colocate, _generate_two_phase, etc.)")
    print("eval_on_start triggers eval BEFORE step 1 -- weights stay at SFT.")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("EVAL-ONLY RESULTS (from trainer.state.log_history):")
    eval_logs = [h for h in trainer.state.log_history if any(k.startswith("eval_") for k in h)]
    for entry in eval_logs:
        for k, v in sorted(entry.items()):
            print(f"  {k}: {v}")
        print("-" * 60)
    print("=" * 60)

    # Save results
    results_path = os.path.join(grpo_config.output_dir, "eval_only_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_logs, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
