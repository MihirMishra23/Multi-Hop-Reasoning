#!/usr/bin/env python3
"""Test Qwen3 with proper implementation"""

import sys
sys.path.insert(0, '/home/rtn27/Multi-Hop-Reasoning/rebuttal-search-r1/Search-R1')

from verl.third_party.vllm.vllm_v_0_6_3.llm import LLM
from vllm import SamplingParams

print("Loading Qwen3-1.7B with proper Qwen3 implementation...")
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    trust_remote_code=True,
    gpu_memory_utilization=0.3,
)

prompts = [
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=50,
)

print("\nGenerating...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")

    # Check if it's not just exclamation marks
    if generated_text.strip() and not all(c == '!' for c in generated_text.strip()):
        print("\n✓ SUCCESS! Model is generating proper text, not just '!!!'")
    else:
        print("\n✗ FAIL: Model still generating gibberish")
