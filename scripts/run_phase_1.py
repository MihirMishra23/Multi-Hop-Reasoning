from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from accelerate import PartialState
from data import get_dataset
from trainer.lmlm_basetrainer import LMLMGRPOTrainer
from trl.trainer import GRPOConfig
import json
import os
from datetime import datetime

@dataclass
class ScriptArguments:
    """Arguments for Phase 1 generation script."""
    model_path: str = field(metadata={"help": "Path to the pretrained model"})
    dataset: str = field(
        default="hotpotqa",
        metadata={"help": "Dataset name"}
    )
    split: str = field(
        default="dev",
        metadata={"help": "Dataset split to use"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    nb_examples: int = field(
        default=1000,
        metadata={"help": "Number of examples to process"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for generation"}
    )
    output_file: str = field(
        default="",
        metadata={"help": "Output file path (default: auto-generated with timestamp)"}
    )
    limit : int = field(
        default=1000,
        metadata={"help": "how many examples to run over"}
    )
    sub_split : str = field(default = None, metadata={"help": "Which (custom) subsplit from trainset to load from. Ask Ryan or Linxi for details, or read code in hoptotqa.py"})
    questions_file: str = field(
        default=None,
        metadata={"help": "Path to JSON file containing questions (bypasses normal dataset loading)"}
    )

def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Initialize accelerate state for logging
    PartialState()

    if script_args.questions_file:
        print(f"Loading questions from file: {script_args.questions_file}")
        from datasets import Dataset as HFDataset
        with open(script_args.questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        ds = HFDataset.from_list(questions_data)
        print(f"Loaded {len(ds)} questions from {script_args.questions_file}")
    else:
        print(f"Loading dataset: {script_args.dataset}")
        print(f"  Split: {script_args.split}")
        print(f"  Limit: {script_args.nb_examples}")
        print(f"  Seed: {script_args.seed}")
        print(f" sub split :", script_args.sub_split)

        ds = get_dataset(
            name=script_args.dataset,
            setting="distractor",
            split=script_args.split,
            seed=script_args.seed,
            limit=script_args.limit,
            sub_split=script_args.sub_split,
        )

    print("First element is : ", ds[0])
    print("elemnt at last index is is :", ds[-1])

    print(f"Loading model from: {script_args.model_path}")
    tok = AutoTokenizer.from_pretrained(script_args.model_path)
    args = GRPOConfig(
        vllm_mode = 'colocate',
        vllm_gpu_memory_utilization=0.8,
        temperature=1.0,  # Neutral temperature for consistent generation
        top_p=1.0,        # No nucleus filtering
        top_k=0,          # No top-k filtering
        num_generations=script_args.batch_size,
        steps_per_generation=1,
        per_device_train_batch_size = script_args.batch_size,
        
    )

    print("Initializing trainer...")
    trainer = LMLMGRPOTrainer(
        model=script_args.model_path,
        processing_class=tok,
        reward_funcs=[],
        two_phase=True,
        lmlm_database_path=None,
        args=args
,
    )

    # Collect all results
    all_triplets = []

    print(f"Starting generation with batch size {script_args.batch_size}...")
    num_batches = (len(ds) + script_args.batch_size - 1) // script_args.batch_size

    for i in range(0, len(ds), script_args.batch_size):
        batch = ds[i:i+script_args.batch_size]
        qa_prompts = batch["question"]
        contexts = batch["golden_contexts"]

        print(f"Processing batch {i//script_args.batch_size + 1}/{num_batches}")

        results = trainer._generate_two_phase(
            qa_prompts=qa_prompts,
            contexts=contexts,
            fast_build_db=True
        )

        all_triplets.extend(results)

    entities = set()
    relationships = set()
    return_values = set()
    triplets = []

    for triplet in all_triplets:
        if len(triplet) != 3:
            print(f"\n[ERROR]: malformed triplet: {triplet}")
            continue
        entities.add(triplet[0])
        relationships.add(triplet[1])
        return_values.add(triplet[2])
        triplets.append([triplet[0], triplet[1], triplet[2]])

    # Generate output filename if not provided
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not script_args.output_file:
        script_args.output_file = f"phase1_database_{script_args.model_path.split('/')[-1]}_{script_args.dataset}_{script_args.split}_{timestamp}.json"

    # Build database with metadata first
    lmlm_database = {
        "metadata": {
            "model_path": script_args.model_path,
            "dataset": script_args.dataset,
            "split": script_args.split,
            "seed": script_args.seed,
            "nb_examples": script_args.nb_examples,
            "batch_size": script_args.batch_size,
            "timestamp": timestamp,
            "total_triplets": len(triplets),
            "unique_entities": len(entities),
            "unique_relationships": len(relationships),
            "unique_return_values": len(return_values)
        },
        "triplets": triplets,
        "entities": list(entities),
        "relationships": list(relationships),
        "return_values": list(return_values)
    }

    # Save combined results
    print(f"Saving {len(all_triplets)} results to {script_args.output_file}")
    os.makedirs(os.path.dirname(script_args.output_file), exist_ok=True)
    with open(script_args.output_file, 'w') as f:
        json.dump(lmlm_database, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
    