import asyncio
import json
import os
import argparse

from google import genai
from pydantic import BaseModel
from datasets import load_dataset
from datetime import datetime
from constants import REPO_ROOT
from data import get_dataset


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract knowledge triplets from multi-hop QA datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "musique", "2wiki"],
                        help="Dataset to use: hotpotqa or musique")
    parser.add_argument("--hotpot-setting", type=str, required=False, choices=["distractor", "fullwiki"],
                        help="HotpotQA dataset setting (only for hotpotqa): distractor or fullwiki")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use")
    parser.add_argument("--model", type=str, required=True, help="Gemini model to use")
    parser.add_argument("--nb-examples", type=int, required=True, help="Number of examples to process")
    parser.add_argument("--sample-from", type=str, required=True, choices=["start", "end"],
                        help="Sample from start or end of dataset")
    parser.add_argument("--use-context", type=str, required=True, help="Use only golden or all contexts", choices=["golden", "all"])
    parser.add_argument("--prompt-name", type=str, required=True, help="Prompt name from prompts.json")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for shuffling")
    parser.add_argument("--max-concurrent", type=int, required=True, help="Maximum concurrent API requests")
    parser.add_argument("--nb-parts-per-prompt", type = int, required = False, help = "How many parts to split the context into. If none provided, defaults to 1. Only for all context mode", default = 1)
    parser.add_argument("--database-path", type = str, required = False, help = "Output path which overrides the default", default = None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help = "Store additional mapping from context to tripelts, useful for analyzing triplets quality and debugging.")
    
    args = parser.parse_args(argv)

    # Validate hotpot-setting is provided for hotpotqa
    if args.dataset == "hotpotqa" and args.hotpot_setting is None:
        parser.error("--hotpot-setting is required when using --dataset hotpotqa")

    # Load prompts from JSON
    prompts_path = os.path.join(REPO_ROOT, "data/prompts", "database_creation.json")
    with open(prompts_path, "r") as f:
        prompts_data = json.load(f)

    if args.prompt_name not in prompts_data:
        raise ValueError(f"Prompt '{args.prompt_name}' not found in prompts.json. Available prompts: {list(prompts_data.keys())}")

    prompt_config = prompts_data[args.prompt_name]
    args.prompt = prompt_config["prompt"]

    # Initialize client
    args.client = genai.Client()

    return args



class KnowledgeTriplets(BaseModel):
    triplets: list[tuple[str,str,str]]


class ProcessedQuestion(BaseModel):
    index : int
    question : str
    golden_contexts : str
    knowledge_triplets: KnowledgeTriplets
    context_triplet_mappings: list[dict]


def split_into_parts(items: list, num_parts: int) -> list[list]:
    """Split a list into num_parts parts where max and min lengths differ by at most 1."""
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if num_parts >= len(items):
        # Return each item as its own part (or empty parts if num_parts > len)
        return [[item] for item in items] + [[] for _ in range(num_parts - len(items))]

    base_size = len(items) // num_parts
    remainder = len(items) % num_parts

    parts = []
    start = 0
    for i in range(num_parts):
        part_size = base_size + (1 if i < remainder else 0)
        parts.append(items[start:start + part_size])
        start += part_size

    return parts



# Async function to process a single context
async def process_context(example, idx: int, semaphore: asyncio.Semaphore, args: argparse.Namespace, dataset_len: int, failed_indexes: list) -> ProcessedQuestion:
    question = example["question"]
    all_contexts = example.get("contexts", [])
    golden_contexts = example.get("golden_contexts", [])

    if args.use_context == "golden":
        contexts = golden_contexts
    else:
        contexts = all_contexts
    num_parts = 1 if args.use_context == "golden" else args.nb_parts_per_prompt
    context_parts = split_into_parts(contexts, num_parts)


    all_triplets = []
    context_triplet_mappings = []

    async with semaphore:
        max_retries = 5
        retry_delay = 60  # seconds

        for part_idx, context_part in enumerate(context_parts):
            if not context_part:  # Skip empty parts
                continue

            part_context_str = "\n\n".join(context_part)
            formatted_prompt = args.prompt.format(context=part_context_str)

            part_label = f"{idx + 1}/{dataset_len}" if num_parts == 1 else f"{idx + 1}/{dataset_len} (part {part_idx + 1}/{num_parts})"
            print(f"Sending request {part_label}...")

            for attempt in range(max_retries):
                try:
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: args.client.models.generate_content(
                            model=args.model,
                            contents=formatted_prompt,
                            config={
                                "response_mime_type": "application/json",
                                "response_json_schema": KnowledgeTriplets.model_json_schema(),
                            },
                        )
                    )
                    kt = KnowledgeTriplets.model_validate_json(response.text)
                    all_triplets.extend(kt.triplets)

                    context_triplet_mappings.append({
                        "index": idx,
                        "context": part_context_str,
                        "triplets": kt.triplets
                    })

                    print(f"Completed request {part_label}")
                    break  

                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if it's a rate limit error
                    if "rate" in error_msg or "quota" in error_msg or "429" in error_msg or "validation" in error_msg:
                        if attempt < max_retries - 1:
                            if "validation" in error_msg:
                                print(f"Validation error on request {part_label}, the error is: {error_msg}. ", end="")
                                print(f"The response output was :", response)
                            else:
                                print(f"Rate limited on request {part_label}. ", end="")
                            print(f"Waiting {retry_delay} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            print(f"Failed request {part_label} after {max_retries} attempts due to rate limiting: {e}")
                            failed_indexes.append((idx + 1, part_idx + 1))
                            break  # Move to next part
                    else:
                        # For other errors, retry once then fail
                        if attempt < 1:
                            print(f"Error on request {part_label}: {e}. Retrying...")
                            await asyncio.sleep(2)
                            continue
                        else:
                            print(f"Failed request {part_label}: {e}")
                            raise

    golden_contexts_str = "\n\n".join(golden_contexts)
    return ProcessedQuestion(
        index=idx,
        golden_contexts=golden_contexts_str,
        knowledge_triplets=KnowledgeTriplets(triplets=all_triplets),
        question=question,
        context_triplet_mappings=context_triplet_mappings
    )


async def main(args: argparse.Namespace):
    # Load dataset
    dataset = get_dataset(args.dataset, args.hotpot_setting, split=args.split, seed = args.seed)
    if args.sample_from == "end":
        start_idx = len(dataset) - args.nb_examples
        dataset = dataset.select(range(start_idx, len(dataset)))
        print(f"Sampling {args.nb_examples} examples from END (start_idx={start_idx})")
    else:  # start
        start_idx = 0
        dataset = dataset.select(range(0, args.nb_examples))
        print(f"Sampling {args.nb_examples} examples from START")

    print(f"Loaded {len(dataset)} examples")

    output_path = f"{args.dataset}_output_{args.split}_seed_{args.seed}_sample_from_{args.sample_from}_nb_{args.nb_examples}_date_{datetime.today().strftime('%m-%d')}"

    failed_indexes = []

    # Main async function to process all contexts
    async def process_all_contexts():
        # Limit concurrent requests to avoid rate limiting (adjust as needed)
        semaphore = asyncio.Semaphore(args.max_concurrent)
        tasks = []
        for idx, example in enumerate(dataset):
            tasks.append(process_context(example, idx, semaphore, args, len(dataset), failed_indexes))
        return await asyncio.gather(*tasks)

    # Run async processing
    print("Starting async processing...")
    results = await process_all_contexts()
    print(f"Completed all {len(results)} requests")
    print(f"Failed indexes: ", failed_indexes)

    #create a json of the desired format
    lmlm_database = {"entities": [], "relationships" : [], "return_values" : [], "triplets" : []}
    entities = set()
    relationships = set()
    return_values = set()
    for processed_question in results:
        for triplet in processed_question.knowledge_triplets.triplets:
            entities.add(triplet[0])
            relationships.add(triplet[1])
            return_values.add(triplet[2])
            lmlm_database["triplets"].append([triplet[0], triplet[1], triplet[2]])

    lmlm_database["entities"]= list(entities)
    lmlm_database["relationships"]= list(relationships)
    lmlm_database["return_values"] = list(return_values)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(REPO_ROOT, "src/database_creation/gemini", output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save database
    if args.database_path is not None:
        database_path = args.database_path
    else:
        database_path = os.path.join(output_dir, "database.json")
    with open(database_path, "w") as f:
        json.dump(lmlm_database, f, indent=4)
    print("saved database to:", database_path)

    # Save metadata
    metadata_json = [data.model_dump(mode = 'json')["knowledge_triplets"] for data in results]
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_json, f, indent=4)
    print("saved metadata to:", metadata_path)

    if args.debug:
        context_triplet_mappings = []
        for processed_question in results:
            context_triplet_mappings.extend(processed_question.context_triplet_mappings)
        mapping_path = os.path.join(output_dir, "context_to_triplets.json")
        with open(mapping_path, 'w') as f:
            json.dump(context_triplet_mappings, f, indent=4)
        print("saved context to triplets mapping to:", mapping_path)


if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))