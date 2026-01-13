
import os
from data.hotpotqa import load_hotpotqa
import time
import asyncio
from google import genai
from multi_lmlm.database.database_manager import DatabaseManager
from synthetic_data.utils import  RolloutMetadata, assert_valid_rollout
from multi_lmlm.constants import DB_END_TOKEN, DB_RETRIEVE_TOKEN, DB_SEP_TOKEN, DB_START_TOKEN,ANSWER_START_TOKEN, ANSWER_END_TOKEN
from openai import OpenAI
from datetime import datetime

from eval.metrics import f1_score
import json

database_path = "/home/rtn27/Multi-Hop-Reasoning/src/database-creation/gemini/output_train_42_start_idx_5999_12000_date_12-18/database.json"
metadata_path = "/home/rtn27/Multi-Hop-Reasoning/src/database-creation/gemini/output_train_42_start_idx_5999_12000_date_12-18/metadata.json"


db = DatabaseManager()
db.load_database(database_path, 4, 0.6, True)

with open(metadata_path, "r") as f:
    metadata = json.load(f)
    
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

MODEL="gemini-2.5-pro"

SYSTEM_PROMPT = f"""You are a database lookup expert. Your goal is to answer questions only by using database lookups.
Here is an example to guide you. For the question:
'What is the nationality of James henry Miller’s wife?' you would output:
'<thinking> James Henry Miller was married to {DB_START_TOKEN} James Henry Miller {DB_SEP_TOKEN} spouse{DB_RETRIEVE_TOKEN} June Miller, A Shoulder to Cry On{DB_END_TOKEN} June Miller. June Miller's nationality was {DB_START_TOKEN} June Miller {DB_SEP_TOKEN} nationality{DB_RETRIEVE_TOKEN} American, The war of 1878{DB_END_TOKEN} American. </thinking> <answer> American </answer>'.

For the lookup part, if the result is unknown you should try a different lookup. For example, to answer:
'What is the birthday of the director of the movie Interstellar?', you would output:
'<thinking> The director of Interstellar is {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} directed by {DB_RETRIEVE_TOKEN} unknown {DB_END_TOKEN} unknown, lets try again. {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} director {DB_RETRIEVE_TOKEN} Christopher Nolan, October 26 2014 {DB_END_TOKEN} Christopher Nolan. The birthday of Christopher Nolan is {DB_START_TOKEN} Christopher Nolan {DB_RETRIEVE_TOKEN} July 30, 1970 {DB_END_TOKEN} July 30, 1970 </thinking> <answer> July 30, 1970 </answer>'.

Rules:
- You may ONLY use information in the prompt or that you already generated. To gain information, you MUST output {DB_START_TOKEN} ENTITY {DB_SEP_TOKEN} RELATIONSHIP {DB_RETRIEVE_TOKEN}.
- Do not write any justification inside the answer tags, only the answer. Your reasoning must be concise and direct.
- All information used must be between the {DB_RETRIEVE_TOKEN} and {DB_END_TOKEN} tags.
- If one lookup does not give you the information you need or outputs 'unknown', you must issue another lookup or try a different strategy to gain information.
- If the lookup does not give you the information you need, you must issue another lookup.
- Follow the style and formatting of the provided examples. Be concise.
- You may retrieve multiple values which are not directly what was asked for. If possible, use the retrieved information to guide your answer or next lookup.
- The ONLY facts you are allowed to use are those that appear between {DB_RETRIEVE_TOKEN} and {DB_END_TOKEN} in this conversation.
- If unknown or some non-sensical text is inside the {DB_RETRIEVE_TOKEN} ... {DB_END_TOKEN} span, you MUST issue more lookups while the answer is unknown.
- When you output your final answer, only include the answer, nothing else.
- Notice that when there is 'unknown', you must absolutely not put the answer to the question afterwards. You must absolutely continue issuing database lookups until you find the answer.
- You must only use information that is given to you in the prompt or that you have already generated. You may not use any external information that you have not already generated.
- The information following the {DB_END_TOKEN} token must be taken from what is inside the {DB_RETRIEVE_TOKEN} ... {DB_END_TOKEN} span. Do not mention any other information not declared in this span first.
"""

MAX_GENERATIONS=6


def gemini_w_db_lookup(prompt):
    current_gen = 1
    res = ""
    input = [{"role" : "system", "content" : SYSTEM_PROMPT},{"role" : "user" , "content" : prompt}, {"role" : "assistant", "content" : ""}]
    while (current_gen <= MAX_GENERATIONS):
        print("On generation number: ", current_gen)
        stream = client.chat.completions.create(
            model=MODEL,
            messages=input,
            stream=True,
            max_completion_tokens=2048,
        )
        
        for char in "".join([chunk.choices[0].delta.content for chunk in stream if chunk.choices[0].delta.content is not None]):
            #time.sleep(0.01)
            if char is None:
                break
            res+=char
            if not res.endswith(DB_RETRIEVE_TOKEN):
                continue
            _, query = res.rsplit(DB_START_TOKEN, 1)
            try:
                return_values = db.retrieve_from_database(DB_START_TOKEN + query, 0.6)[:4] #limit to 4 return values
                return_value = ", ".join(return_values)
            except Exception as e:
                print(f"Database lookup failed: {e}")
                return_value = "unknown"
                # Handle DB lookup failure with fallback policy

            #### Step 4: Append retrieved value and db_end token
            res += return_value + DB_END_TOKEN
            input = [{"role" : "system", "content" : SYSTEM_PROMPT},{"role" : "user" , "content" : prompt}, {"role" : "assistant", "content" : res}]   
            break
             
        current_gen += 1
        if ANSWER_END_TOKEN in res:
            return res

async def process_example(semaphore, example, idx):
    async with semaphore:
        max_retries = 10
        retry_delay = 60
        for attempt in range(max_retries):
            try:
                print(f"\nProcessing example {idx}...")
                question = example["question"]
                print("here 1")
                triplets = metadata[idx]["triplets"]
                print("here 2")
                triplets_formatted_str = "\n".join([f"({triplet[0]}, {triplet[1]}, {triplet[2]})" for triplet in triplets])
                print("here 3")
                prompt = f"Here is the question: {question}"

                # Run the synchronous gemini_w_db_lookup() in a thread pool
                result = await asyncio.to_thread(gemini_w_db_lookup, prompt)

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

async def main():
    # Load questions from HotpotQA train dataset
    START_IDX = 0
    NB_EXAMPLES =  251
    MAX_CONCURRENT = 10
    dataset = load_hotpotqa(
        setting="distractor",
        split="train",
        source="auto",
        limit=NB_EXAMPLES,
        seed=42
    )

    dataset = dataset.select(range(START_IDX, NB_EXAMPLES))

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Create tasks for all examples
    tasks = []
    for idx, example in enumerate(dataset):
        tasks.append(process_example(semaphore, example, idx + 1))

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Filter out None values (only keep f1_score > 0.5)
    rollouts = [r for r in results if r is not None]

    formatted_rollouts = {"examples":  [r.model_dump(mode = 'json') for r in rollouts]}

    output_path = f"{os.path.dirname(os.path.abspath(__file__))}/{datetime.today().strftime('%m-%d')}_rollouts_{len(rollouts)}_start_idx_{START_IDX}_examples_{NB_EXAMPLES}.json"
    print(f"\n\ndone! Collected {len(rollouts)} successful rollouts out of {len(tasks)} examples. ")
    with open(output_path, 'w') as f:
        json.dump(formatted_rollouts, f, indent=2)  
    print("saved results to ", output_path)
if __name__ == '__main__':
    asyncio.run(main())
