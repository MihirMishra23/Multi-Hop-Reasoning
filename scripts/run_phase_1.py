from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from accelerate import PartialState
from data import get_dataset
from trainer.lmlm_basetrainer import LMLMGRPOTrainer
import json
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

def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Initialize accelerate state for logging
    PartialState()

    print(f"Loading dataset: {script_args.dataset}")
    print(f"  Split: {script_args.split}")
    print(f"  Limit: {script_args.nb_examples}")
    print(f"  Seed: {script_args.seed}")

    ds = get_dataset(
        name=script_args.dataset,
        setting="distractor",
        split=script_args.split,
        seed=script_args.seed,
        limit=script_args.nb_examples
    )

    print(f"Loading model from: {script_args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(script_args.model_path).to("cuda")
    tok = AutoTokenizer.from_pretrained(script_args.model_path)

    print("Initializing trainer...")
    trainer = LMLMGRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[],
        two_phase=True,
        lmlm_database_path=None
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
    lmlm_database = {"triplets": []}

    for triplet in all_triplets:
        if len(triplet) != 3:
            print(f"\n[ERROR]: malformed triplet: {triplet}")
            continue
        entities.add(triplet[0])
        relationships.add(triplet[1])
        return_values.add(triplet[2])
        lmlm_database["triplets"].append([triplet[0], triplet[1], triplet[2]])

    lmlm_database["entities"]= list(entities)
    lmlm_database["relationships"]= list(relationships)
    lmlm_database["return_values"] = list(return_values)

    # Generate output filename if not provided
    if not script_args.output_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_args.output_file = f"phase1_database_{script_args.model_path.split('/')[-1]}_{script_args.dataset}_{script_args.split}_{timestamp}.json"

    # Save combined results
    print(f"Saving {len(all_triplets)} results to {script_args.output_file}")
    with open(script_args.output_file, 'w') as f:
        json.dump(lmlm_database, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()
    