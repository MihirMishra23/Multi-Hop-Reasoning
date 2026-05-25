#!/usr/bin/env python3
"""Test if Qwen3 generates proper Search-R1 rollouts"""

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Search-R1'))

from vllm import LLM, SamplingParams

# Test prompt from HotpotQA training
prompt = """<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: In what year did Ricardo Izecson dos Santos Leite win the Ballon d'Or?
<|im_end|>
<|im_start|>assistant
"""

print("=" * 70)
print("Testing Qwen3-1.7B Search-R1 Rollout Generation")
print("=" * 70)

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    trust_remote_code=True,
    gpu_memory_utilization=0.4,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=500,
)

print("\nGenerating rollout...")
outputs = llm.generate([prompt], sampling_params)

generated = outputs[0].outputs[0].text
print(f"\nGenerated ({len(generated)} chars):")
print("-" * 70)
print(generated[:500])
print("-" * 70)

# Check if output is valid
if all(c == '!' for c in generated.strip()[:100]):
    print("\n❌ FAIL: Still generating only '!!!'")
    sys.exit(1)
elif len(generated.strip()) < 10:
    print("\n❌ FAIL: Output too short")
    sys.exit(1)
else:
    print("\n✅ SUCCESS: Model generating proper text (not just '!!!')")

    # Check if it's using Search-R1 format
    if '<think>' in generated or '<search>' in generated or '<answer>' in generated:
        print("✅ Model is using Search-R1 format tags!")
    else:
        print("⚠️  Model generated text but not using Search-R1 format yet (expected for untrained model)")
