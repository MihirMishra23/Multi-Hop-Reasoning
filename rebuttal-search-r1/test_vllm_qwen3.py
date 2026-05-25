#!/usr/bin/env python3
"""Quick test to see if vLLM can generate from Qwen3 at all"""

from vllm import LLM, SamplingParams

# Load the model
print("Loading Qwen3-1.7B...")
llm = LLM(model="Qwen/Qwen3-1.7B", trust_remote_code=True)

# Simple test prompt
prompts = [
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]

# Try generation with different parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

print("\nGenerating...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}")
    print()
