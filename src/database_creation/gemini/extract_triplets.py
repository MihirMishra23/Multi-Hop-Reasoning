import asyncio
import json
import os

from google import genai
from pydantic import BaseModel
from datasets import load_dataset
from datetime import datetime

client = genai.Client()

SPLIT="train"
EXAMPLES = "For example, for the sentence: 'James Clark is an Australian soccer player with a Austrian wife named Alissa Jordan' would have the triplets: (James Clark, nationality, Australian), (James Clark, occupation, soccer player), (James Clark, wife name, Alissa Jordan), (Alissa Jordan, husband, James Clark), (Alissa Jordan, nationality, Austrian)."

PROMPT = "Extract triplets from the following context. Each triplet must be of the form (entity, relationship, value). " \
"Rules: " \
"- Entities is the subject of a sentence. Do not use pronouns (e.g he, or it) to describe an entity, use the full descriptive name. " \
"- Relationships refer  to any characteristic of an entity. Even if a relationship is not explicity mentioned, you must included it. Relationships can be things that entity was involved in, or any descriptive characterisitc of that entity." \
"If an entity is described with multiple characteristics in one go, create seperate triplet entries for each of those characteristics." \
"Example: from the text 'Albert Einstein was a German theoretical physicist best known for developing the theory of relativity.' the triplets are (Albert Einstein, nationality, Germany), (Albert Einstein, occupation, theoretical physicist), (Albert Einstein, best known for, developping the theory of relativity)" \
" Given these examples, generate explicit and implicit triplets from the following context. \n\n {context}"
MODEL = "gemini-3-pro-preview"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NB_EXAMPLES = 12000
START_IDX = 5999
OUTPUT_PATH = f"output_{SPLIT}_42_start_idx_{START_IDX}_{NB_EXAMPLES}_date_{datetime.today().strftime('%m-%d')}"


# Load HotpotQA dataset directly from HuggingFace
print("Loading HotpotQA dataset from HuggingFace...")
raw_dataset = load_dataset("hotpot_qa", "distractor", split=SPLIT)

# Shuffle and limit
raw_dataset = raw_dataset.shuffle(seed=42)

raw_dataset = raw_dataset.select(range(min(NB_EXAMPLES, len(raw_dataset))))

print(f"Loaded {len(raw_dataset)} examples")


class KnowledgeTriplets(BaseModel):
    triplets: list[tuple[str,str,str]]


class ProcessedQuestion(BaseModel):
    index : int
    question : str
    golden_contexts : str
    knowledge_triplets: KnowledgeTriplets

# Async function to process a single context
async def process_context(example, idx: int, semaphore: asyncio.Semaphore) -> KnowledgeTriplets:
    print(example)
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
        if title in supporting_titles:
            print("title: ", title, "\n")
            # Get sentences for this title
            sents = sentences[i] if i < len(sentences) else []
            # Concatenate sentences
            paragraph = f"{title}: " + " ".join(sents).strip()
            supporting_contexts.append(paragraph)

    golden_contexts = "\n\n".join(supporting_contexts)

    async with semaphore:
        formatted_prompt = PROMPT.format(context=golden_contexts)
        print(f"Sending request {idx + 1}/{len(supporting_facts)}...")

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
                print(f"Completed request {idx + 1}/{len(supporting_facts)}")
                return ProcessedQuestion(index = idx, golden_contexts = golden_contexts, knowledge_triplets = kt, question = question)

            except Exception as e:
                raise(e)
                error_msg = str(e).lower()
                # Check if it's a rate limit error
                if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"Rate limited on request {idx + 1}. Waiting {retry_delay} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        print(f"Failed request {idx + 1} after {max_retries} attempts due to rate limiting: {e}")
                        raise
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
    semaphore = asyncio.Semaphore(251)

    tasks = []
    for idx, example in enumerate(raw_dataset):
        if idx > START_IDX:
            tasks.append(process_context(example, idx, semaphore))
    return await asyncio.gather(*tasks)

# Run async processing
print("Starting async processing...")
results = asyncio.run(process_all_contexts())
print(f"Completed all {len(results)} requests")

#create a json of the desired format
lmlm_database = {"entities": [], "relationships" : [], "return_values" : [], "triplets" : []}
entities = set()
relationships = set()
return_values = set()
for processed_question in results:
    for triplet in processed_question.knowledge_triplets.triplets:
        print(triplet)
        entities.add(triplet[0])
        relationships.add(triplet[1])
        return_values.add(triplet[2])
        lmlm_database["triplets"].append([triplet[0], triplet[1], triplet[2]])

lmlm_database["entities"]= list(entities)
lmlm_database["relationships"]= list(relationships)
lmlm_database["return_values"] = list(return_values)

# Create output directory if it doesn't exist
output_dir = os.path.join("/home/rtn27/Multi-Hop-Reasoning/src/database-creation/gemini", OUTPUT_PATH)
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