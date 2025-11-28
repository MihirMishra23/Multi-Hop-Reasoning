
import time
from google import genai
from src.lmlm.database.database_manager import DatabaseManager
database_path = "/home/rtn27/Multi-Hop-Reasoning/src/database-creation/build-database-gemini/generated_database_train_42_2k.json"
from src.lmlm.constants import DB_END_TOKEN, DB_RETRIEVE_TOKEN, DB_SEP_TOKEN, DB_START_TOKEN,ANSWER_START_TOKEN, ANSWER_END_TOKEN
from openai import OpenAI
from src.data.hotpotqa import load_hotpotqa
import os
from src.eval.metrics import f1_score
import json
db = DatabaseManager()
db.load_database(database_path)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

MODEL="gemini-2.5-pro"

SYSTEM_PROMPT = f"""You are a database lookup expert. Your goal is to answer questions by using database lookups. 
Here is an example to guide you, for the question: 
'What is the nationality of James henry Miller’s wife?' You would output: '<thinking> James Henry Miller was married to {DB_START_TOKEN} James Henry Miller {DB_SEP_TOKEN} spouse{DB_RETRIEVE_TOKEN} June Miller{DB_END_TOKEN} June Miller. June Miller's nationality was {DB_START_TOKEN} June Miller {DB_SEP_TOKEN} nationality{DB_RETRIEVE_TOKEN} American{DB_END_TOKEN} American. </thinking> <answer> American </answer>. For just the lookup part, if the result is unknown you should try a different lookup. For example, to answer the question what is the birthday of the director of the movie Interstellar, you would output: <thinking> The director of Interstellar is {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} directed by <|db_result|> unknown {DB_END_TOKEN} unknown, lets try again.  {DB_START_TOKEN} Interstellar {DB_SEP_TOKEN} director <|db_result|> Christopher Nolan {DB_END_TOKEN} Christopher Nolan. The birthday of Christopher Nolan is {DB_START_TOKEN} Christopher Nolan <|db_result|> July 30, 1970 {DB_END_TOKEN} July 30, 1970 </thinking> <answer> July 30, 1970 </answer>'

Rules:
- You may ONLY use infromation in the prompt or that you already generated. To gain information, you MUST output {DB_START_TOKEN} ENTITY {DB_SEP_TOKEN} RELATIONSHIP {DB_RETRIEVE_TOKEN}. Do not write any justification inside the answer tags, only the answer. Your reasoning must be concise and direct. 
- All information used must be between the {DB_RETRIEVE_TOKEN} and {DB_END_TOKEN} tag. If one lookup does not give you the information you need or outputs 'unknown', you must issue another lookup or try a different strategy to gain information. If the lookup does not give you the information you need, you must issue another lookup. 
- Follow the style and formatting of the provided examples. Be concise.  
- The ONLY facts you are allowed to use are those that appear between
   {DB_RETRIEVE_TOKEN} and {DB_END_TOKEN} in this conversation.
- Ignore your own training data and world knowledge completely.
- If unknown or some non-sensical text is in inside the {DB_RETRIEVE_TOKEN} ... {DB_END_TOKEN} span,
     you MUST issue more lookups. While the answer is unknown.
- When you output your final answer, only include the answer, nothing else.
- Notice that when there is 'unknown', you must absolutely not put the answer to the question afterwards. You must absolutely continue issuing database lookups until you find the answer. You must only use information that is given to you in the prompt or that you have already generated. You may not use any external information that you have not already generated.
- The information following the <|db_end|> token must be taken from what is inside the {DB_RETRIEVE_TOKEN} ... {DB_END_TOKEN} span. Do not mention any other information not declared in this span first.
"""

MAX_GENERATIONS=10




def func(prompt):
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
            print(char, end = "", flush=True)
            #time.sleep(0.01)
            if char is None:
                break
            res+=char
            if not res.endswith(DB_RETRIEVE_TOKEN):
                continue
            _, query = res.rsplit(DB_START_TOKEN, 1)
            try:
                return_value = db.retrieve_from_database(DB_START_TOKEN + query, 0.6)
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

rollouts = []
if __name__ == '__main__':

    # Load questions from HotpotQA train dataset
    dataset = load_hotpotqa(
        setting="distractor",
        split="train",
        source="auto",
        limit=2000,
        seed=42
    )
    NB_EXAMPLES = 10
    count = 0
    for example in dataset:
        count += 1
        if count > NB_EXAMPLES:
            break
        question = example["question"]
        prompt = f"Here is the question: {question}"
        result = func(prompt)
        if result is None:
            continue
        lmlm_answer = result.split(ANSWER_START_TOKEN)[-1].split(ANSWER_END_TOKEN)[0]
        print(f"\nquestion : {question} \n")
        answers = example["answers"]
        print(f"\nAnswers was : {answers} \n")
        print(f"lmlm answer: ", lmlm_answer, "\n\n")
        score = max([f1_score(lmlm_answer, a)[0] for a in answers])
        if score > 0.5:
            print("Successful answer, adding trajectory: ", result)
            rollouts.append(result)

    print("done!")
    with open('rollouts.json', 'w') as f:
        json.dump(rollouts, f, indent=2)
