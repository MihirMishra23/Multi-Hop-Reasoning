import json
import os
import time
from openai import OpenAI

# Configuration constants
START_IDX = 41
LAST_IDX = 41  #inclusive
MODEL = "gpt-4.1"

indices = [0,1,2,3,4,5,6, 7 , 41,58]



#OTHER EXAMPLES I WAS USING
# EXAMPLES = """
# Example 1:

# Question: Which tennis player has won the most Grand Slam titles?
# Answer: Novak Djokovic
# Context: Novak Djokovic has won 24 grand slam titles, and Rafael Nadal has won 22 grand slam titles, and Roger Federer has won 20 Grand Slam Titles, so Novak has won the most.

# Novak Djokovic has won <|db_entity|> Novak Djokovic <|db_relationship|> Grand Slam titles<|db_return|> 24<|db_end|> 24 Grand Slam titles. Rafael Nadal has won <|db_entity|> Rafael Nadal <|db_relationship|> Grand Slam titles<|db_return|> 22<|db_end|> 22 Grand Slam titles. Roger Federer has won <|db_entity|> Roger Federer <|db_relationship|> Grand Slam titles<|db_return|> 20<|db_end|> 20 Grand Slam titles. Therefore, Novak Djokovic has won the most Grand Slam titles. <ANSWER> Novak Djokovic

# Example 2:

# Question: What year did the British politician born in 1967 began to represent the Daventry in the UK House of Commons?
# Answer: Chris Heaton‑Harris
# Context: Chris Heaton‑Harris was born 28 of November, 1967. He began representing Daventry in the UK House of Commons on 6 May 2010.

# Keir Starmer was born on <|db_entity|> Keir Starmer <|db_relationship|> birthday<|db_return|>September 2, 1962<|db_end|> September 2, 1962. Nigel Farage was born on <|db_entity|> Nigel Farage <|db_relationship|> birthday<|db_return|> April 3, 1964<|db_end|> April 3, 1964. Nick Clegg was born on <|db_entity|> Nick Clegg <|db_relationship|> birthday<|db_return|>Janurary 7th, 1967<|db_end|>. Nick Clegg <|db_relationship|> represented constituency <|db_return|> Sheffield Hallam <|db_end|>. Chris Heaton‑Harris was born on <|db_entity|> Chris Heaton‑Harris <|db_relationship|> birthday<|db_return|>November 28, 1967<|db_end|>. Chris Heaton‑Harris <|db_relationship|> represented constituency <|db_return|> Daventry <|db_end|>. Therefore, the answer is Chris Heaton‑Harris. <ANSWER> Chris Heaton‑Harris.



# Example: In what year did the English actor from the film The System die?
# <thinking> The System<|db_relationship|> starred<|db_return|> Oliver Reed<|db_end|>. The nationality of Oliver Reed is<|db_entity|> Oliver Reed<|db_relationship|> nationality<|db_return|> English<|db_end|> English. Oliver Reed<|db_entity|> Oliver Reed<|db_relationship|> died in<|db_return|> 1999<|db_end|>. Therefore the english actor from the film The System died in 1999.<thinking/> <answer> 1999. <answer/>

# """



#RULES TO PUT:
# Rules:
# - Each db lookup should be inferable from previous context. You may not use the result of a db lookup call before the <|db_return|> tag. 
# - Each lookup must follow the (entity, relation) -> value pattern.
# - Each relationship must be specific, atomic, and capture a single verifiable fact.
# - After doing your database lookup calls, please output an answer tag before your final answer.
# - The values returned by the database lookup should be specific, simple values.
# - Any composite information must be split into multiple database lookup calls.
# - You MUST use at least one database lookup before answering the question. You may NOT answer the question directly, without using a database lookup.
# - If you are looking for an entity that satisfies a property, you MUST perform at least one other db lookup before performing one on this entity.

PROMPT_TEMPLATE = """I want to generate examples of db look up generation for answering questions. Create either direct look ups of up entities, or when the entity is not known generate a 'guess and check' strategy.
 A db lookup performs an (entity, relationship) lookup and produces a value, using the <|db_entity|> entity<|db_relationship|> relationship<|db_return|>  value<|db_end|>

Here are some examples:

- If you are looking for a property of an entity, you can perform a <|db_entity|> check as follows:
Example 1:
Question: What is the nationality of James henry Miller’s wife? 
Answer: American
Context: James Henry Miller's wife is June Miller, and she is american.
<thinking> James Henry Miller was married to <|db_entity|> James Henry Miller <|db_relationship|> spouse<|db_return|> June Miller<|db_end|> June Miller. June Miller's nationality was <|db_entity|> June Miller <|db_relationship|> nationality<|db_return|> American<|db_end|> American. <answer> American <american/>"


- If you are looking for an entity that satisfies a specific property, you MUST perform a dblookup on a DIFFERENT entity first. Here is an example:
Example 2:
Question: Which basktball player has the most total career points?
Answer: Lebron James
Context: Lebron James has scored the most career points, with 42,184 total points.
<thinking> Kareem Abdul-Jabbar has <|db_entity|> Kareem Abdul-Jabbar<|db_relationship|> career points<|db_return|> 38,387<|db_return|> 38,387 total career points. Lebron James has <|db_entity|> Lebron James<|db_relationship|> career points<|db_return|> 42,184<|db_return|> 42,184 total career points. Therefore Lebron has the most total career points. <thinking/> <answer> Lebron<answer/>"

- Here is an example rollout where you must guess and check different values before finding the correct answer.
Example 3:
Question: Which genus of moth in the world's seventh-largest country contains only one species?
Answer: Rhangena
Context: The world's seventh largest country is India, the Rhangena moth is located in India, and contains a single species.
<thinking> To answer this question, we first need to determine the world's seventh largest country. Australia is <|db_entity|> Australia<|db_relationship|> area ranking<|db_return|> 6th<|db_end|> the 6th largest country in the world by area. Argentina is<|db_entity|> Argentina<|db_relationship|> area ranking<|db_return|> 8th<|db_end|> the 8th largest country by area. India is <|db_entity|> India<|db_relationship|> area ranking<|db_return|> 7th<|db_end|> the 7th largest country, therefore the genus of moth is found in India. The genus Goniocraspedon is found in <|db_entity|> Goniocraspedon<|db_relationship|> found in<|db_return|> India<|db_end|> is found in India. The number of species of Goniocraspedon is
 <|db_entity|> Goniocraspedon<|db_relationship|> number of species<|db_return|> 3<|db_end|> 3, so Goniocraspedon does not contain only one species. The genus Rhangena is found in <|db_entity|> Rhangena<|db_relationship|> found in<|db_return|> India<|db_end|> India, and the number of species of Rhangena is<|db_entity|> Rhangena<|db_relationship|> number of species<|db_return|> 1<|db_end|> 1, Therefore the correct answer is Rhangena.
 <thinking/> <answer> Rhangena<answer/>

Here is your question:
Question: {question}
Answer: {answer}
Context: {context}

Now output a sequence of <thinking> enclosed database lookups and eventually answer the question using these. Follow the above examples and instructions and do not output anything else."""

def create_finetuning_data(input_file, output_file):
    """
    Process question-answer pairs through OpenAI and save results.

    Args:
        input_file (str): Path to the input JSON file with QA pairs
        output_file (str): Path to save the processed results
    """

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Load the QA pairs
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    results = []

    for i, qa_pair in enumerate(qa_data):
        if i not in indices:
            continue

        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        context = qa_pair.get("context", "")

        # Format the prompt using the template
        prompt = PROMPT_TEMPLATE.format(question=question, answer=answer, context=context)

        try:
            # Send request to OpenAI
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            # Extract the response
            generated_text = response.choices[0].message.content

            # Store the result
            result = {
                "prompt": prompt,
                "original_question": question,
                "original_answer": answer,
                "generated_response": generated_text
            }

            results.append(result)

            print(f"Processed {i+1}/{len(qa_data)}: {question[:50]}...")

            # Add a small delay to respect rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            # Store error information
            error_result = {
                "original_question": question,
                "original_answer": answer,
                "error": str(e)
            }
            print(error_result)

    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete!")
    print(f"Processed {len(results)} items")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Configuration
    # input_file = "context_examples/495_w_context_examples.json"
    input_file = "data/495_w_context_examples.json"
    output_file = "lmlm_generated_rollouts.json"

    # Use the constant prompt template

    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
    else:
        create_finetuning_data(input_file, output_file)