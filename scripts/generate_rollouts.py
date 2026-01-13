
import os
import time
import asyncio
import argparse
from google import genai
from lmlm.database.database_manager import DatabaseManager
from synthetic_data.utils import  RolloutMetadata, assert_valid_rollout
from lmlm.constants import DB_END_TOKEN, DB_RETRIEVE_TOKEN, DB_SEP_TOKEN, DB_START_TOKEN,ANSWER_START_TOKEN, ANSWER_END_TOKEN
from openai import OpenAI
from datetime import datetime
from constants import REPO_ROOT
from data import get_dataset

from eval.metrics import f1_score
import json


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rollouts using LMLM agent with database lookups")
    parser.add_argument("--database-path", type=str, required=True, help="Path to database JSON file")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to metadata JSON file")
    parser.add_argument("--model", type=str, required=True, help="Model to use for generation")
    parser.add_argument("--max-generations", type=int, required=True, help="Maximum number of generation iterations")
    parser.add_argument("--start-idx", type=int, required=True, help="Starting index for dataset")
    parser.add_argument("--nb-examples", type=int, required=True, help="Number of examples to process")
    parser.add_argument("--max-concurrent", type=int, required=True, help="Maximum concurrent requests")
    parser.add_argument("--hotpot-setting", type=str, required=True, choices=["distractor", "fullwiki"],
                        help="HotpotQA dataset setting: distractor or fullwiki")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for shuffling")
    parser.add_argument("--prompt-name", type=str, required=True, help="Prompt name from lmlm_agent.json")
    parser.add_argument("--db-top-k", type=int, required=True, help="Number of top results to retrieve from database")
    parser.add_argument("--db-threshold", type=float, required=True, help="Threshold for database retrieval")
    parser.add_argument("--adaptive-k", action=argparse.BooleanOptionalAction, required=True,
                        help="Whether to use adaptive k for database retrieval")
    parser.add_argument("--return-triplets", action=argparse.BooleanOptionalAction, required=True,
                        help="whether to return triplets or just values")

    args = parser.parse_args(argv)

    # Load prompts from JSON and add to args
    prompts_path = os.path.join(REPO_ROOT, "data/prompts", "lmlm_agent.json")
    with open(prompts_path, "r") as f:
        prompts_data = json.load(f)

    if args.prompt_name not in prompts_data:
        raise ValueError(f"Prompt '{args.prompt_name}' not found in lmlm_agent.json. Available prompts: {list(prompts_data.keys())}")

    prompt_config = prompts_data[args.prompt_name]
    system_prompt_template = prompt_config["prompt"]

    # Format the system prompt with DB tokens and add to args
    args.system_prompt = system_prompt_template.format(
        DB_START_TOKEN=DB_START_TOKEN,
        DB_SEP_TOKEN=DB_SEP_TOKEN,
        DB_RETRIEVE_TOKEN=DB_RETRIEVE_TOKEN,
        DB_END_TOKEN=DB_END_TOKEN
    )

    # Load database and add to args
    args.db = DatabaseManager()
    args.db.load_database(args.database_path, args.db_top_k, args.db_threshold, args.adaptive_k)

    # Load metadata and add to args
    with open(args.metadata_path, "r") as f:
        args.metadata = json.load(f)

    # Create OpenAI client and add to args
    args.client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai"
    )

    return args


def gemini_w_db_lookup(prompt: str, args: argparse.Namespace) -> str:
    current_gen = 1
    res = ""
    input = [{"role" : "system", "content" : args.system_prompt},{"role" : "user" , "content" : prompt}, {"role" : "assistant", "content" : ""}]
    while (current_gen <= args.max_generations):
        print("On generation number: ", current_gen)
        print("The prompt is :", input)
        stream = args.client.chat.completions.create(
            model=args.model,
            messages=input,
            stream=True,
            max_completion_tokens=2048,
        )

        for char in "".join([chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content is not None]):
            if char is None:
                break
            res+=char
            if not res.endswith(DB_RETRIEVE_TOKEN):
                continue
            _, query = res.rsplit(DB_START_TOKEN, 1)
            try:
                return_values = args.db.retrieve_from_database(DB_START_TOKEN + query, args.db_threshold, return_triplets = args.return_triplets)[:4] #limit to 4 return values
                return_value = ", ".join(return_values)
            except Exception as e:
                print(f"Database lookup failed: {e}")
                return_value = "unknown"
                # Handle DB lookup failure with fallback policy

            #### Step 4: Append retrieved value and db_end token
            res += return_value + DB_END_TOKEN
            input = [{"role" : "system", "content" : args.system_prompt},{"role" : "user" , "content" : prompt}, {"role" : "assistant", "content" : res}]
            break

        current_gen += 1
        if ANSWER_END_TOKEN in res:
            return res
    return res

async def process_example(semaphore, example, idx, args: argparse.Namespace):
    async with semaphore:
        max_retries = 10
        retry_delay = 60
        for attempt in range(max_retries):
            try:
                print(f"\nProcessing example {idx}...")
                question = example["question"]
                print("here 1")
                triplets = args.metadata[idx]["triplets"]
                print("here 2")
                triplets_formatted_str = "\n".join([f"({triplet[0]}, {triplet[1]}, {triplet[2]})" for triplet in triplets])
                print("here 3")
                prompt = f"Here is the question: {question}"

                # Run the synchronous gemini_w_db_lookup() in a thread pool
                result = await asyncio.to_thread(gemini_w_db_lookup, prompt, args)

                if result is None:
                    return None

                assert_valid_rollout(result)
                print("here 4")

                lmlm_answer = result.split(ANSWER_START_TOKEN)[-1].split(ANSWER_END_TOKEN)[0]
                answers = example["answers"]
                print("here 5")

                score = max([f1_score(lmlm_answer, a)[0] for a in answers])
                print(f"completed exaample {idx}!")
                return RolloutMetadata(full_response = "", annotated_text = result, triplets = triplets_formatted_str, golden_answer = answers, f1_score = score, lmlm_answer = lmlm_answer, question = question)
            except Exception as e:
                print(f"lookup failed with error {e} Retrying, currently at {attempt} retries... ")
                await asyncio.sleep(retry_delay)

        print("Failed all retries for example : ", example)

async def main(args: argparse.Namespace):
    # Load HotpotQA dataset
    print(f"Loading HotpotQA dataset from HuggingFace...")
    dataset = get_dataset("hotpotqa", args.hotpot_setting, split=args.split, seed = args.seed)
    print(f"Using HotpotQA setting: {args.hotpot_setting}")
    print(f"Length of the dataset is: {len(dataset)}")

    # Select the specific range
    end_idx = min(args.start_idx + args.nb_examples, len(dataset))
    dataset = dataset.select(range(args.start_idx, end_idx))
    print(f"Selected examples from index {args.start_idx} to {end_idx} ({len(dataset)} examples)")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Create tasks for all examples
    tasks = []
    for idx, example in enumerate(dataset):
        tasks.append(process_example(semaphore, example, idx + 1, args))

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Filter out None values (only keep f1_score > 0.5)
    rollouts = [r for r in results if r is not None]

    formatted_rollouts = {"examples":  [r.model_dump(mode = 'json') for r in rollouts]}

    output_path = f"{os.path.dirname(os.path.abspath(__file__))}/hotpot_qa_rollouts_{datetime.today().strftime('%m-%d')}_count_{len(rollouts)}_start_idx_{args.start_idx}_total_{args.nb_examples}.json"
    print(f"\n\ndone! Collected {len(rollouts)} successful rollouts out of {len(tasks)} examples. ")
    with open(output_path, 'w') as f:
        json.dump(formatted_rollouts, f, indent=2)
    print("saved results to ", output_path)

if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))
