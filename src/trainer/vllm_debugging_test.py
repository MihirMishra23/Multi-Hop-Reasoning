#!/usr/bin/env python3
"""
vLLM debugging test script that replicates exact initialization and sampling parameters
from grpo_train.sh and lmlm_basetrainer.py to isolate generation bugs.

Usage:
    python src/trainer/vllm_debugging_test.py
"""

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================================
# Configuration matching grpo_train.sh
# ============================================================================
MODEL_PATH = "/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_two_phase_hotpotqa_ep5_bsz48"
VLLM_GPU_MEMORY_UTILIZATION = 0.15
VLLM_MAX_MODEL_LENGTH = 4096
VLLM_TENSOR_PARALLEL_SIZE = 1
MAX_COMPLETION_LENGTH = 1024
TEMPERATURE = 1.3
TOP_P = 0.95
TOP_K = 0
NUM_ROLLOUTS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 1
STEPS_PER_GENERATION = 8

# ============================================================================
# Load tokenizer with fix_mistral_regex flag
# ============================================================================
print(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# Get stop tokens
DB_RETRIEVE_TOKEN = "<|db_return|>"
stop_token_ids = [
    tokenizer.eos_token_id,
    tokenizer.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0]
]

print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"DB_RETRIEVE_TOKEN: {DB_RETRIEVE_TOKEN} (ID: {stop_token_ids[1]})")
print(f"Stop token IDs: {stop_token_ids}")
print(f"Token ID 0 decodes to: '{tokenizer.decode([0])}'")

# ============================================================================
# Initialize vLLM with exact parameters from lmlm_basetrainer.py line 707-731
# ============================================================================
print("\nInitializing vLLM...")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    max_num_seqs=PER_DEVICE_TRAIN_BATCH_SIZE * VLLM_TENSOR_PARALLEL_SIZE * STEPS_PER_GENERATION,
    max_model_len=VLLM_MAX_MODEL_LENGTH,
    distributed_executor_backend=None,  # Single GPU, no TP
    seed=42,
    max_num_batched_tokens=4096,
    # Important so temperature scaling/logit tweaking affects the TIS log probs
    logprobs_mode="processed_logprobs",
    # Disabled in original: enforce_eager=True, enable_prefix_caching=False
)

# ============================================================================
# Test prompts (Phase 1: Triplet extraction from contexts)
# ============================================================================
test_prompts = [
    # Example 1: This one worked in step 1 (generated 69 triplets)
    """Please extract knowledge triplets from the context.
Context:

Grigori Aleksandrov: Grigori Vasilyevich Aleksandrov or Alexandrov (Russian: Григо́рий Васи́льевич Алекса́ндров ; original family name was Мормоненко or Mormonenko; 23 January 1903 – 16 December 1983) was a prominent Soviet film director who was named a People's Artist of the USSR in 1947 and a Hero of Socialist Labor in 1973.  He was awarded the Stalin Prizes for 1941 and 1950.

October: Ten Days That Shook the World: October: Ten Days That Shook the World (Russian: Октябрь (Десять дней, которые потрясли мир) ; translit.  "Oktyabr': Desyat' dney kotorye potryasli mir") is a 1928 Soviet silent historical film by Sergei Eisenstein and Grigori Aleksandrov.  It is a celebratory dramatization of the 1917 October Revolution commissioned for the tenth anniversary of the event.  Originally released as October in the Soviet Union, the film was re-edited and released internationally as Ten Days That Shook The World, after John Reed's popular book on the Revolution.  In U.S. released by Amkino Corporation and First National (later was a subsidiary of Warner Bros.).

Triplets:
""",
    # Example 2: This one failed in step 2 (generated all "!!!")
    """Please extract knowledge triplets from the context.
Context:

Jo Beth Taylor: Joanne Rebecca Guilfoyle (born 29 May, 1971 in Perth, Western Australia), known professionally as Jo Beth Taylor, is an Australian television presenter, actor and singer most well known for hosting three weekly programs at the same time in the 1990s on the Nine Network: "Australia's Funniest Home Video Show" (1993–1997), "Hey Hey It's Saturday" (1995–1997) and "What's Up Doc? " (1996–1997), before taking a hiatus from television for more than two years.

Hey Hey It's Saturday: Hey Hey It's Saturday was a long-running variety television program on Australian television.  It initially ran for 27 years on the Nine Network from on 9 October 1971 to 20 November 1999 (there was a recess in 1978).  Its host throughout its entire run was Daryl Somers, who later also became executive producer of the program.  The original producer, Gavin Disney, left the program in the 1980s and Somers then jointly formed his own production company, "Somers Carroll Productions", with on-screen partner Ernie Carroll, the performer of Somers' puppet sidekick Ossie Ostrich.

Triplets:
""",
]

# ============================================================================
# Setup sampling parameters matching lmlm_basetrainer.py line 1330-1344
# ============================================================================
generation_kwargs = {
    "n": NUM_ROLLOUTS,
    "repetition_penalty": 1.0,  # Default (no penalty set in grpo_train.sh)
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "min_p": 0.0,
    "max_tokens": MAX_COMPLETION_LENGTH,
    "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
    "stop_token_ids": stop_token_ids,
}

sampling_params = SamplingParams(**generation_kwargs)

print(f"\nSampling parameters:")
for key, value in generation_kwargs.items():
    print(f"  {key}: {value}")

# ============================================================================
# Test generation for both prompts
# ============================================================================
print("\n" + "="*80)
print("TESTING GENERATION")
print("="*80)

for i, prompt in enumerate(test_prompts):
    print(f"\n{'='*80}")
    print(f"TEST {i+1}: {'EXAMPLE THAT WORKED' if i == 0 else 'EXAMPLE THAT FAILED'}")
    print(f"{'='*80}")
    print(f"Prompt preview: {prompt[:100]}...")

    print(f"\nGenerating {NUM_ROLLOUTS} completions...")
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)

    for j, output in enumerate(outputs[0].outputs):
        completion_text = output.text
        token_ids = output.token_ids

        print(f"\n--- Completion {j+1}/{NUM_ROLLOUTS} ---")
        print(f"Length: {len(token_ids)} tokens")
        print(f"First 10 token IDs: {token_ids[:10]}")
        print(f"First 100 chars: {completion_text[:100]}")

        # Check for degenerate output
        if len(set(token_ids[:50])) == 1:
            print(f"⚠️  WARNING: Repetitive output detected! Token {token_ids[0]} repeated.")

        # Check if it's all exclamation marks
        if completion_text.strip() == "!" * len(completion_text.strip()):
            print(f"❌ ERROR: Output is all exclamation marks!")

        # Try to count triplets
        triplet_lines = [line for line in completion_text.split('\n') if '\t' in line]
        print(f"Triplet lines detected: {len(triplet_lines)}")

        if triplet_lines:
            print(f"Sample triplet: {triplet_lines[0][:100]}")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
