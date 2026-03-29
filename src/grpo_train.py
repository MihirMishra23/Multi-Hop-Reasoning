from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from trainer.lmlm_basetrainer import LMLMGRPOTrainer, parse_triplets
# from trainer.lmlm_grpotrainer import LMLMGRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from eval.metrics import exact_match_score
from trl.trainer.grpo_config import GRPOConfig
from reward_func import em_accuracy, f1_reward, db_coverage_reward, db_size_threshold, format_reward_zero_rl
import json
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

    # Curriculum / tier filtering
    tier_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tier JSON produced by generate_tier.py. When set, training examples are filtered to those whose rollout score falls in [tier_min_score, tier_max_score]."}
    )
    tier_min_score: int = field(
        default=1,
        metadata={"help": "Minimum rollout success count (inclusive) to keep an example."}
    )
    tier_max_score: int = field(
        default=7,
        metadata={"help": "Maximum rollout success count (inclusive) to keep an example."}
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
    phase1_reward_type: str = field(
        default="binary",
        metadata={"help": "Phase 1 reward type in two_phase mode: 'binary' (db_size_threshold/db_coverage_reward) or 'utilization' (used_triplets/total_triplets ratio)"}
    )
    phase1_prompt_type: str = field(
        default="sft",
        metadata={"help": "Prompt type key into database_creation.json / lmlm_agent.json (e.g. 'sft', 'sft_with_question', 'zero_rl', 'formatted_zero_rl')"}
    )
    num_db_rollouts: int = field(
        default=1,
        metadata={"help": "Number of DB rollouts per question in two_phase mode (K). N must be divisible by K. Each DB gets N//K QA rollouts."}
    )
    phase1_db_weight_mode: str = field(
        default="count_dynamic",
        metadata={"help": "How to weight A_db before combining with A_qa. Options: 'none' (no weight), 'fixed' (phase1_advantage_weight), 'dynamic' (scale_ratio=r_qa_mean/r_db_mean), 'count' (M=N/K), 'count_dynamic' (M*scale_ratio, current default)."}
    )
    retrieval_top_k: int = field(
        default=1,
        metadata={"help": "Nb of examples retrieved from db"}
    )
    use_chat_template: bool = field(
        default=False,
        metadata={"help": "Wrap prompts in the model's chat template before generation (needed for instruct/base models in zero-RL)"}
    )
    vanilla_grpo: bool = field(
        default=False,
        metadata={"help": "Vanilla GRPO: treat (db_g, qa_g) as one trajectory with r_db=r_qa. "
                  "Requires two_phase=True. Auto-sets num_db_rollouts=num_generations (K=G, M=1). "
                  "All tokens in trajectory g share a single advantage A_g normalized within the question group."}
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

    # wandb run name = save dir basename (single source of truth)
    grpo_config.run_name = os.path.basename(grpo_config.output_dir)
    os.makedirs(grpo_config.output_dir, exist_ok=True)

    if wandb.run is not None:
        wandb.run.name = grpo_config.run_name
    else:
        print(f"Wandb run is not initialized, skipping wandb name setting")

    # Load and process dataset
    print(f"Loading dataset: {script_args.dataset_name}")

    train_dataset = get_dataset(name = script_args.dataset_name, setting = script_args.dataset_config, split = "train", sub_split = "train", limit = script_args.train_size, seed = 42)
    test_dataset = get_dataset(name = script_args.dataset_name, setting = script_args.dataset_config, split = "train", sub_split = "eval", limit = script_args.eval_size, seed = 42)

    if script_args.tier_path is not None:
        print(f"Applying tier filter from {script_args.tier_path} (score {script_args.tier_min_score}..{script_args.tier_max_score})")
        with open(script_args.tier_path, "r", encoding="utf-8") as f:
            tier_data = json.load(f)
        valid_ids = {
            qid for qid, item in tier_data["results"].items()
            if script_args.tier_min_score <= int(item["score"]) <= script_args.tier_max_score
        }
        before = len(train_dataset)
        train_dataset = train_dataset.filter(lambda ex: ex["id"] in valid_ids)
        print(f"Tier filter: {before} -> {len(train_dataset)} examples")

    train_set = train_dataset.map(process_example)
    eval_set = test_dataset.map(process_example)
    
    print(f"Train set size: {len(train_set)}")
    print(f"Eval set size: {len(eval_set)}")
    # Load tokenizer
    print(f"Loading tokenizer from: {script_args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)

    # formatted_zero_rl_v6 uses DB special tokens as single vocabulary entries.
    # Starting from a base model these tokens are absent, so we must register them,
    # resize the embedding table, and save the result to disk BEFORE the trainer
    # initialises colocated vLLM.  vLLM fixes org_vocab_size from the on-disk
    # config at startup; if the training model has a larger embedding the
    # _move_model_to_vllm weight-sync will fail with an assertion error.
    if lmlm_args.phase1_prompt_type == "formatted_zero_rl_v6":
        from multi_lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN
        _db_tokens = [DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN]
        _to_add = [t for t in _db_tokens if t not in tokenizer.get_vocab()]
        if _to_add:
            print(f"formatted_zero_rl_v6: registering {len(_to_add)} DB special tokens "
                  f"and pre-saving resized model so vLLM sees the correct vocab size...")
            tokenizer.add_special_tokens({"additional_special_tokens": _to_add})
            _presave_path = os.path.join(grpo_config.output_dir, "init_with_db_tokens")
            os.makedirs(_presave_path, exist_ok=True)
            _tmp_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_path, torch_dtype="auto"
            )
            _tmp_model.resize_token_embeddings(len(tokenizer))
            _tmp_model.save_pretrained(_presave_path)
            tokenizer.save_pretrained(_presave_path)
            del _tmp_model
            script_args.model_path = _presave_path
            print(f"  Saved to {_presave_path} (new vocab size: {len(tokenizer)})")
    
    
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
        if "format" in lmlm_args.reward_func:
            reward_funcs.append(format_reward_zero_rl)
    else:
        reward_funcs = []
        if "em" in lmlm_args.reward_func:
            reward_funcs.append(em_accuracy)
        if "f1" in lmlm_args.reward_func:
            reward_funcs.append(f1_reward)
        if "format" in lmlm_args.reward_func:
            reward_funcs.append(format_reward_zero_rl)
        if not reward_funcs:
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
        phase1_reward_type=lmlm_args.phase1_reward_type,
        phase1_prompt_type=lmlm_args.phase1_prompt_type,
        num_db_rollouts=lmlm_args.num_db_rollouts,
        phase1_db_weight_mode=lmlm_args.phase1_db_weight_mode,
        retrieval_top_k = lmlm_args.retrieval_top_k,
        use_chat_template=lmlm_args.use_chat_template,
        vanilla_grpo=lmlm_args.vanilla_grpo,
    )
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=grpo_config.resume_from_checkpoint)
    print("Training completed!")


if __name__ == "__main__":
    main()
