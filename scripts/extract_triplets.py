import asyncio
import json
import os
import argparse

from google import genai
from pydantic import BaseModel
from datasets import load_dataset
from datetime import datetime
from constants import REPO_ROOT

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract knowledge triplets from multi-hop QA datasets")
parser.add_argument("--dataset", type=str, required=True, choices=["hotpot_qa", "musique"],
                    help="Dataset to use: hotpot_qa or musique")
parser.add_argument("--hotpot-setting", type=str, required=False, choices=["distractor", "fullwiki"],
                    help="HotpotQA dataset setting (only for hotpot_qa): distractor or fullwiki")
parser.add_argument("--split", type=str, required=True, help="Dataset split to use")
parser.add_argument("--model", type=str, required=True, help="Gemini model to use")
parser.add_argument("--nb-examples", type=int, required=True, help="Number of examples to process")
parser.add_argument("--sample-from", type=str, required=True, choices=["start", "end"],
                    help="Sample from start or end of dataset")
parser.add_argument("--use-only-golden", action=argparse.BooleanOptionalAction, required=True, help="Use only golden/supporting contexts")
parser.add_argument("--prompt-name", type=str, required=True, help="Prompt name from prompts.json")
parser.add_argument("--seed", type=int, required=True, help="Random seed for shuffling")
parser.add_argument("--max-concurrent", type=int, required=True, help="Maximum concurrent API requests")

args = parser.parse_args()

# Validate hotpot-setting is provided for hotpot_qa
if args.dataset == "hotpot_qa" and args.hotpot_setting is None:
    parser.error("--hotpot-setting is required when using --dataset hotpot_qa")

# Load prompts from JSON
prompts_path = os.path.join(REPO_ROOT + "/data/prompts", "database_creation.json")
with open(prompts_path, "r") as f:
    prompts_data = json.load(f)

if args.prompt_name not in prompts_data:
    raise ValueError(f"Prompt '{args.prompt_name}' not found in prompts.json. Available prompts: {list(prompts_data.keys())}")

prompt_config = prompts_data[args.prompt_name]
PROMPT = prompt_config["prompt"]
EXAMPLES = prompt_config.get("examples", "")

# Set configuration from arguments
DATASET = args.dataset
SPLIT = args.split
MODEL = args.model
NB_EXAMPLES = args.nb_examples
SAMPLE_FROM = args.sample_from
USE_ONLY_GOLDEN_CONTEXT = args.use_only_golden
SEED = args.seed
MAX_CONCURRENT = args.max_concurrent

client = genai.Client()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load dataset
print(f"Loading {DATASET} dataset from HuggingFace...")
if DATASET == "hotpot_qa":
    raw_dataset = load_dataset("hotpot_qa", args.hotpot_setting, split=SPLIT)
    print(f"Using HotpotQA setting: {args.hotpot_setting}")
elif DATASET == "musique":
    raw_dataset = load_dataset("dgslibisey/MuSiQue", split=SPLIT)
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

# Shuffle dataset
raw_dataset = raw_dataset.shuffle(seed=SEED)

print(f"Length of the dataset is: {len(raw_dataset)}")

# Select examples based on sample-from parameter
if SAMPLE_FROM == "end":
    START_IDX = len(raw_dataset) - NB_EXAMPLES
    raw_dataset = raw_dataset.select(range(START_IDX, len(raw_dataset)))
    print(f"Sampling {NB_EXAMPLES} examples from END (start_idx={START_IDX})")
else:  # start
    START_IDX = 0
    raw_dataset = raw_dataset.select(range(0, NB_EXAMPLES))
    print(f"Sampling {NB_EXAMPLES} examples from START")

print(f"Loaded {len(raw_dataset)} examples")

# Preprocess dataset to normalize format
def preprocess_dataset(dataset, dataset_name):
    """Normalize dataset format to match HotpotQA structure"""
    if dataset_name == "hotpot_qa":
        # Already in correct format
        return dataset
    elif dataset_name == "musique":
        # Convert MuSiQue to HotpotQA format
        def convert_musique_to_hotpot(example):
            paragraphs = example.get('paragraphs', [])
            supporting_para_ids = set(example.get('paragraph_support_idx', []))

            # Extract titles and sentences
            titles = []
            sentences = []
            supporting_titles = []

            for i, para in enumerate(paragraphs):
                title = para.get('title', '')
                text = para.get('paragraph_text', '')
                # Split paragraph into sentences (simple split by period)
                para_sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

                titles.append(title)
                sentences.append(para_sentences)

                # Track supporting titles
                if i in supporting_para_ids:
                    supporting_titles.append(title)

            return {
                'question': example['question'],
                'context': {
                    'title': titles,
                    'sentences': sentences
                },
                'supporting_facts': {
                    'title': supporting_titles,
                    'sent_id': []  # MuSiQue doesn't have sentence-level support
                }
            }

        return dataset.map(convert_musique_to_hotpot, remove_columns=dataset.column_names)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

print("Preprocessing dataset to normalize format...")
raw_dataset = preprocess_dataset(raw_dataset, DATASET)
print("Dataset preprocessing complete")

OUTPUT_PATH = f"{DATASET}_output_{SPLIT}_seed_{SEED}_sample_from_{SAMPLE_FROM}_nb_{NB_EXAMPLES}_date_{datetime.today().strftime('%m-%d')}"

failed_indexes = []
class KnowledgeTriplets(BaseModel):
    triplets: list[tuple[str,str,str]]


class ProcessedQuestion(BaseModel):
    index : int
    question : str
    golden_contexts : str
    knowledge_triplets: KnowledgeTriplets

# Async function to process a single context
async def process_context(example, idx: int, semaphore: asyncio.Semaphore) -> KnowledgeTriplets:
    question = example["question"]
    supporting_facts = example["supporting_facts"]

    supporting_titles = set(supporting_facts['title'])

    # Get contexts
    # Format: {'title': ['Title1', 'Title2', ...], 'sentences': [['sent1', 'sent2'], ['sent3'], ...]}
    context_data = example.get('context', {})
    # if not context_data or 'title' not in context_data or 'sentences' not in context_data:
    #     continue

    titles = context_data.get('title', [])
    sentences = context_data.get('sentences', [])

    # Filter for supporting contexts
    supporting_contexts = []
    for i, title in enumerate(titles):
        if not USE_ONLY_GOLDEN_CONTEXT or title in supporting_titles:
            # Get sentences for this title
            sents = sentences[i] if i < len(sentences) else []
            # Concatenate sentences
            paragraph = f"{title}: " + " ".join(sents).strip()
            supporting_contexts.append(paragraph)

    golden_contexts = "\n\n".join(supporting_contexts)

    async with semaphore:
        formatted_prompt = PROMPT.format(context=golden_contexts)
        print(f"Sending request {idx + 1}/{len(raw_dataset)}...")
        max_retries = 5
        retry_delay = 60  # seconds

        for attempt in range(max_retries):
            try:
                # Using synchronous client in async context with run_in_executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=MODEL,
                        contents=formatted_prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_json_schema": KnowledgeTriplets.model_json_schema(),
                        },
                    )
                )
                kt = KnowledgeTriplets.model_validate_json(response.text)
                print(f"Completed request {idx + 1}/{len(raw_dataset)}")
                return ProcessedQuestion(index = idx, golden_contexts = golden_contexts, knowledge_triplets = kt, question = question)

            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a rate limit error
                if "rate" in error_msg or "quota" in error_msg or "429" in error_msg or "validation" in error_msg:
                    if attempt < max_retries - 1:
                        if "validation" in error_msg:
                            print(f"Validation error on request {idx + 1}, the error is: {error_msg}. ", end = "")
                            print(f"The response output was :", response)
                        else:
                            print(f"Rate limited on request {idx + 1}. ", end="")
                        print(f"Waiting {retry_delay} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"Failed request {idx + 1} after {max_retries} attempts due to rate limiting: {e}")
                        failed_indexes.append(idx + 1)
                        return ProcessedQuestion(index = idx, golden_contexts = golden_contexts, knowledge_triplets = KnowledgeTriplets(triplets = []), question = question)
                else:
                    # For other errors, retry once then fail
                    if attempt < 1:
                        print(f"Error on request {idx + 1}: {e}. Retrying...")
                        await asyncio.sleep(2)
                        continue
                    else:
                        print(f"Failed request {idx + 1}: {e}")
                        raise

# Main async function to process all contexts
async def process_all_contexts():
    # Limit concurrent requests to avoid rate limiting (adjust as needed)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for idx, example in enumerate(raw_dataset):
        tasks.append(process_context(example, idx, semaphore))
    return await asyncio.gather(*tasks)

# Run async processing
print("Starting async processing...")
results = asyncio.run(process_all_contexts())
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
output_dir = os.path.join(REPO_ROOT, "src/database_creation/gemini", OUTPUT_PATH)
os.makedirs(output_dir, exist_ok=True)

# Save database
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