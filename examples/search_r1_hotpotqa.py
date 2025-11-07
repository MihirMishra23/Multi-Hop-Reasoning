#!/usr/bin/env python3
"""
Script to pull random questions from HotpotQA and generate responses using GPT-4.
"""

import random
import os
import re
import argparse
from datasets import load_dataset
from openai import OpenAI


def has_search_tag(text):
    """Check if the text contains a <search> tag."""
    return '<search>' in text and '</search>' in text


def has_answer_tag(text):
    """Check if the text contains an <answer> tag."""
    return '<answer>' in text and '</answer>' in text


def extract_search_query(text):
    """Extract the search query from <search> tags."""
    match = re.search(r'<search>(.*?)</search>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_gpt4_response_multiturn(question):
    """Get a response from GPT-4 with multi-turn interaction for searches."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    initial_prompt = f"Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> (\"entity\", \"relationship\") </search> and it will return the top searched results between <information> and </information>. IMPORTANT: Output ONLY ONE <search> tag at a time, then STOP and wait for the search results. Do not output multiple searches in the same response. After receiving the search results, you can conduct more reasoning and perform another search if needed, or provide the final answer. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>. The answer should be concise and direct - just provide the essential information without repeating the question or adding extra explanation. IMPORTANT: Do not include any blank lines or line breaks between tags - keep everything flowing on the same line with just spaces between tags.\n\nQuestion: {question}"
    
    # Initialize conversation history
    messages = [{
        "content": initial_prompt,
        "role": "user"
    }]
    
    turn_count = 0
    max_turns = 10  # Prevent infinite loops
    
    while turn_count < max_turns:
        turn_count += 1
        
        # Get GPT-4 response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant's response to conversation history
        messages.append({
            "content": assistant_message,
            "role": "assistant"
        })
        
        # Print the response
        print("\nGPT-4 Response:")
        print(assistant_message)
        
        # Check if the assistant provided a final answer
        if has_answer_tag(assistant_message):
            print("\n✓ Final answer received!")
            break
        
        # Check if the assistant is requesting a search
        if has_search_tag(assistant_message):
            search_query = extract_search_query(assistant_message)
            user_input = input("> ")
            
            if user_input.lower() == '':
                search_results = "No results found."
            else:
                search_results = user_input
            
            # Add the search results as a user message
            user_message = f"<information>{search_results}</information>"
            messages.append({
                "content": user_message,
                "role": "user"
            })
            
            print("\nContinuing with GPT-4...")
        else:
            # No search tag and no answer tag - something unexpected
            print("\n⚠ GPT-4 response doesn't contain <search> or <answer> tags. Stopping.")
            break
    
    if turn_count >= max_turns:
        print(f"\n⚠ Reached maximum number of turns ({max_turns})")
    
    return messages


def print_reasoning_trace(messages, question):
    """Print the final reasoning trace with all interactions."""
    print("\n" + "=" * 80)
    print("FINAL REASONING TRACE")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")
    
    # Skip the first message (initial prompt) and process the rest
    for i in range(1, len(messages)):
        msg = messages[i]
        if msg['role'] == 'assistant':
            print(msg['content'], end=' ')
        elif msg['role'] == 'user' and i > 0:  # Skip the initial prompt
            # This is a search result provided by the user
            print(msg['content'], end=' ')
    print()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run HotpotQA questions with GPT-4')
    parser.add_argument('--qid', '--question-id', type=str, help='Specific question ID to run')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    # seed = 2  # You can change this seed value
    # random.seed(seed)
    # print(f"Using random seed: {seed}")
    
    print("Loading HotpotQA dataset...")
    
    # Load the HotpotQA dataset (distractor setting)
    # You can change 'distractor' to 'fullwiki' if needed
    dataset = load_dataset('hotpot_qa', 'distractor', split='train')
    
    print(f"Dataset loaded! Total examples: {len(dataset)}\n")
    
    # Determine which question(s) to process
    if args.qid:
        # Find the specific question by ID
        print(f"Looking for question ID: {args.qid}")
        example = None
        for item in dataset:
            if item['id'] == args.qid:
                example = item
                break
        
        if example is None:
            print(f"Error: Question ID '{args.qid}' not found in dataset.")
            return
        
        examples_to_process = [example]
        print("=" * 80)
        print("SPECIFIC HOTPOTQA QUESTION")
        print("=" * 80)
    else:
        # Get 1 random example
        random_indices = random.sample(range(len(dataset)), 1)
        examples_to_process = [dataset[idx] for idx in random_indices]
        print("=" * 80)
        print("RANDOM HOTPOTQA QUESTION")
        print("=" * 80)
    
    for i, example in enumerate(examples_to_process, 1):
        question = example['question']
        print(f"\nQuestion ID: {example['id']}")
        print(f"Question: {question}")
        
        print("\nStarting multi-turn conversation with GPT-4...")
        
        # Run the multi-turn conversation
        conversation = get_gpt4_response_multiturn(question)
        
        print("\n" + "=" * 80)
        print("CONVERSATION COMPLETE")
        print("=" * 80)
        
        # Print the final reasoning trace
        print_reasoning_trace(conversation, question)
        
        print("Gold answer: ", example['answer'])


if __name__ == "__main__":
    main()
