
import os
from data.hotpotqa import load_hotpotqa
import json
import time
import asyncio
from google import genai
from lmlm.database.database_manager import DatabaseManager
from pydantic import BaseModel
from lmlm.constants import DB_END_TOKEN, DB_RETRIEVE_TOKEN, DB_SEP_TOKEN, DB_START_TOKEN,ANSWER_START_TOKEN, ANSWER_END_TOKEN, THINKING_START_TOKEN, THINKING_END_TOKEN
from openai import OpenAI
from datetime import datetime
from eval.metrics import f1_score
import json
from utils import RolloutMetadata, assert_valid_rollout

metadata_path = "/share/j_sun/rtn27/lmlm_database/output_train_42_6000_date_12-10/metadata.json"
database_path = "//share/j_sun/rtn27/lmlm_database/output_train_42_6000_date_12-10/database.json"

db = DatabaseManager()
db.load_database(database_path, 4, 0.6, True)

with open(metadata_path, "r") as f:
    metadata = json.load(f)


client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

MODEL="gemini-3-pro-preview"

OLD_EX = f"""
For the question: 'What is the nationality of James henry Miller’s wife?' you would output:
I need to use the '(James Henry Miller, spouse)' lookup, and potentially the (June Miller, nationality) lookup. 
'{THINKING_START_TOKEN} James Henry Miller was married to {DB_START_TOKEN} James Henry Miller {DB_SEP_TOKEN} spouse{DB_RETRIEVE_TOKEN} June Miller{DB_END_TOKEN} June Miller. June Miller's nationality was {DB_START_TOKEN} June Miller {DB_SEP_TOKEN} nationality{DB_RETRIEVE_TOKEN} American{DB_END_TOKEN} American. {THINKING_END_TOKEN} <answer> American </answer>'.
"""

ANOTHER_OLD_EX="""Here are some examples, format your response like these below:

For the question: 'What is the birthday of the director of the movie Interstellar?', you would output:
'I need to use the (Interstallar, director) lookup, and potentially the (Christopher Nolan, birthday) lookup. I am not allowed to mention Christopher Nolan until I retrieve it from the database. {THINKING_START_TOKEN} The director of Interstellar is {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} directed by {DB_RETRIEVE_TOKEN} unknown {DB_END_TOKEN} unknown, lets try again. {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} director {DB_RETRIEVE_TOKEN} Christopher Nolan {DB_END_TOKEN} Christopher Nolan. The birthday of Christopher Nolan is {DB_START_TOKEN} Christopher Nolan {DB_SEP_TOKEN} birthday {DB_RETRIEVE_TOKEN} July 30, 1970 {DB_END_TOKEN} July 30, 1970 {THINKING_END_TOKEN} <answer> July 30, 1970 </answer>'.

For the question: 'What country is the Louvre located in?', you would output:
'I need to use the (Louvre, located in) lookup. {THINKING_START_TOKEN} The Louvre is located in {DB_START_TOKEN} The Louvre {DB_SEP_TOKEN} located in {DB_RETRIEVE_TOKEN} 10 August 1793, Paris, France {DB_END_TOKEN} France. <answer> France </answer>'.

For the question what basketball player on the Los Angeles Lakers is born in 2004?", you would output:
'I need to use the (Los Angeles Lakers, team members) lookup, and the (Bronny James, date of birth) lookup. Since 'Bronny James' does not appear in the question, I am not allowed to mention "Bronny James" before retrieving it from the database.
'{THINKING_START_TOKEN} The Los Angeles Lakers' team members are {DB_START_TOKEN} Los Angeles Lakers {DB_SEP_TOKEN} team members {DB_RETRIEVE_TOKEN} Lebron James, Luka Doncic, Austin Reaves, Deandre Ayton{DB_END_TOKEN} players who are born after 2004. Lets try a different lookup: {DB_START_TOKEN} Los Angeles Lakers {DB_SEP_TOKEN} youngest team member {DB_RETRIEVE_TOKEN} Austin Reaves {DB_END_TOKEN}. Austin Reaves is born on {DB_START_TOKEN} Austin Reaves {DB_SEP_TOKEN} birthday {DB_RETRIEVE_TOKEN} May 29, 1998{DB_END_TOKEN} 1998, so this is not the right player. Lets try looking up the recent draft pick. The Los Angeles Lakers recently drafted {DB_START_TOKEN} Los Angeles Lakers {DB_SEP_TOKEN} recently drafted {DB_RETRIEVE_TOKEN} Bronny James {DB_END_TOKEN}. Bronny James is born in {DB_START_TOKEN} Bronny James {DB_SEP_TOKEN} birthday {DB_RETRIEVE_TOKEN} October 6th, 2004 {DB_END_TOKEN}{THINKING_END_TOKEN} <answer> Bronny James </answer>'.
"""


SYSTEM_PROMPT = f"""You are being used to train a database lookup expert. Your goal is to train this model answer questions only by using database lookups, without using any external knowledge. You are given a list of (entity , relationship) pairs to guide which lookups you should issue.
You may NOT use the information in the (entity,relationship) pairs without using a lookup. You must lookup the first and second value in the triplet to issue a lookup. 

Rules:
- You may ONLY use information in the prompt or that you already generated. To gain information, you MUST output {DB_START_TOKEN} ENTITY {DB_SEP_TOKEN} RELATIONSHIP {DB_RETRIEVE_TOKEN} VALUE {DB_END_TOKEN} VALUE.
- The (entity, relation) pairs you are allowed to lookup are the ones that figure in the question. You may NOT lookup an entity if this entity does not figure in the question, or is very common knowledge (like a Country, or president).
- You may NOT look up an entity that does not appear in the question. To lookup an entity from the provided (entity, relationship) pairs, you MUST issue other lookups first.
- Do not write any justification inside the answer tags, only the answer.
- If one lookup does not give you the information you need or outputs 'unknown', you must issue another lookup or try a different strategy to gain information.
- You may ONLY make basic logical inferences. You may not make any inferences using external knowledge that is not declared inside {DB_RETRIEVE_TOKEN} ... {DB_END_TOKEN}.
- You may retrieve multiple values which are not directly what was asked for. If possible, use the retrieved information to guide your answer or next lookup.
- The information following the {DB_END_TOKEN} token must be taken from what is inside the {DB_RETRIEVE_TOKEN} ... {DB_END_TOKEN} span. Do not mention any other information not declared in this span first.
- You may NOT lookup an entity that does not appear in the question, unless it was retrieved explicitly.
- You may NOT mention anything about what you plan to lookup, or any (entity, relationship) pairs that are mentioned in the question.
- Strictly follow the below formatting in order to answer the question.

Here are some examples, format your response like these below:

For the question: 'What is the birthday of the director of the movie Interstellar?' and the provided triplets (Interstellar, director), (Christopher Nolan, birthday), you would output:
'<plan> I need to use the (Interstallar, director) lookup, and potentially the (Christopher Nolan, birthday) lookup. I am not allowed to mention Christopher Nolan until I retrieve it from the database. </plan> {THINKING_START_TOKEN} The director of Interstellar is {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} directed by {DB_RETRIEVE_TOKEN} unknown {DB_END_TOKEN} unknown, lets try again. {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} director {DB_RETRIEVE_TOKEN} Christopher Nolan {DB_END_TOKEN} Christopher Nolan. The birthday of Christopher Nolan is {DB_START_TOKEN} Christopher Nolan {DB_SEP_TOKEN} birthday {DB_RETRIEVE_TOKEN} July 30, 1970 {DB_END_TOKEN} July 30, 1970 {THINKING_END_TOKEN} <answer> July 30, 1970 </answer>'.

For the question: 'What country is the Louvre located in?' and the provided triplets (Louvre, located in) you would output:
'<plan> I need to use the (Louvre, located in) lookup. I have no restrictions with what I am allowed to lookup. </plan> {THINKING_START_TOKEN} The Louvre is located in {DB_START_TOKEN} The Louvre {DB_SEP_TOKEN} located in {DB_RETRIEVE_TOKEN} 10 August 1793, Paris, France {DB_END_TOKEN} France. <answer> France </answer>'.

For the question what basketball player on the Los Angeles Lakers is born in 2004?", and the provided triplets (Los Angeles Lakers, team members), (Bronny James, date of birth) you would output:
'<plan> I need to use the (Los Angeles Lakers, team members) lookup, and the (Bronny James, date of birth) lookup. Since 'Bronny James' does not appear in the question, I am not allowed to mention "Bronny James" before retrieving it from the database. </plan>
'{THINKING_START_TOKEN} The Los Angeles Lakers' team members are {DB_START_TOKEN} Los Angeles Lakers {DB_SEP_TOKEN} team members {DB_RETRIEVE_TOKEN} Lebron James, Luka Doncic, Austin Reaves, Deandre Ayton{DB_END_TOKEN} players who are born after 2004. Lets try a different lookup: {DB_START_TOKEN} Los Angeles Lakers {DB_SEP_TOKEN} youngest team member {DB_RETRIEVE_TOKEN} Austin Reaves {DB_END_TOKEN}. Austin Reaves is born on {DB_START_TOKEN} Austin Reaves {DB_SEP_TOKEN} birthday {DB_RETRIEVE_TOKEN} May 29, 1998{DB_END_TOKEN} 1998, so this is not the right player. Lets try looking up the recent draft pick. The Los Angeles Lakers recently drafted {DB_START_TOKEN} Los Angeles Lakers {DB_SEP_TOKEN} recently drafted {DB_RETRIEVE_TOKEN} Bronny James {DB_END_TOKEN}. Bronny James is born in {DB_START_TOKEN} Bronny James {DB_SEP_TOKEN} birthday {DB_RETRIEVE_TOKEN} October 6th, 2004 {DB_END_TOKEN}{THINKING_END_TOKEN} <answer> Bronny James </answer>'.
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

            if ANSWER_END_TOKEN in res:
                return res

            if not res.endswith(DB_RETRIEVE_TOKEN):
                continue
            _, query = res.rsplit(DB_START_TOKEN, 1)
            try:
                return_values = db.retrieve_from_database(DB_START_TOKEN + query, 0.6, top_k = 4)
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
        

    return res


async def process_example(semaphore, example, idx):
    async with semaphore:
        max_retries = 10
        retry_delay = 30
        for attempt in range(max_retries):
            try:
                print(f"\nProcessing example {idx}...")
                question = example["question"]
                triplets = metadata[idx]["triplets"]
                allowed_lookups = "\n".join([f"{DB_START_TOKEN} {triplet[0]} {DB_SEP_TOKEN} {triplet[1]} {DB_RETRIEVE_TOKEN}" for triplet in triplets])
                prompt = f"Here is the question: {question}\n And here are some lookups you may issue: {allowed_lookups}. Make sure you do not make any explicit references to 'entities' or 'relationships' or the pairs directly"
                # Run the synchronous gemini_w_db_lookup() in a thread pool
                result = await asyncio.to_thread(gemini_w_db_lookup, prompt)

                if result is None:
                    return None

                lmlm_answer = result.split(ANSWER_START_TOKEN)[-1].split(ANSWER_END_TOKEN)[0]
                answers = example["answers"]

                score = float(max([f1_score(lmlm_answer, a)[0] for a in answers]))
                triplets_formatted_str = "\n".join([f"({triplet[0]}, {triplet[1]}, {triplet[2]})" for triplet in triplets])
                try: 
                    assert_valid_rollout(result)
                    parsed_rollout = THINKING_START_TOKEN + result.split(THINKING_START_TOKEN)[1]
                except Exception as e:
                    print("invalid rollouts, adding anyways")
                    return RolloutMetadata(full_response = result, annotated_text = "", triplets = triplets_formatted_str, golden_answer = answers, f1_score = -1.0, lmlm_answer = lmlm_answer, question = question)

                print(f"Completed example {idx}!")
                return RolloutMetadata(full_response = result, annotated_text = parsed_rollout, triplets = triplets_formatted_str, golden_answer = answers, f1_score = score, lmlm_answer = lmlm_answer, question = question)
            except AssertionError as e:
                print(f"Ill formatted response, error {e} retrying, currently at {attempt} retries")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                print(f"lookup failed with error {e} Retrying, currently at {attempt} retries... ")
                await asyncio.sleep(retry_delay)

        print("Failed all retries for example : ", example)

async def main():
    # Load questions from HotpotQA train dataset
    dataset = load_hotpotqa(
        setting="distractor",
        split="train",
        source="auto",
        limit=6000,
        seed=42
    )
    NB_EXAMPLES = 6000
    MAX_CONCURRENT = min(NB_EXAMPLES, 150)  # Control concurrency with semaphore

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Create tasks for all examples
    tasks = []
    for idx, example in enumerate(dataset):
        if idx >= NB_EXAMPLES:
            break
        tasks.append(process_example(semaphore, example, idx))

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Filter out None values (only keep f1_score > 0.5)
    rollouts = [r for r in results if r is not None]

    formatted_rollouts = {"examples":  [r.model_dump(mode = 'json') for r in rollouts]}

    output_path = f"{os.path.dirname(os.path.abspath(__file__))}/{datetime.today().strftime('%m-%d')}_rollouts_{len(rollouts)}_examples_{NB_EXAMPLES}.json"
    print(f"\n\ndone! Collected {len(rollouts)} successful rollouts out of {len(tasks)} examples.")
    with open(output_path, 'w') as f:
        json.dump(formatted_rollouts, f, indent=2)
    print("Saved results to :", output_path)

if __name__ == '__main__':
    asyncio.run(main())
