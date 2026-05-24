#!/usr/bin/env python3
"""Test Search-R1's vLLM wrapper with Qwen3"""

import sys
sys.path.insert(0, '/home/rtn27/Multi-Hop-Reasoning/rebuttal-search-r1/Search-R1')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from verl.third_party.vllm.vllm_v_0_6_3.llm import LLM
from vllm import SamplingParams

prompt = """<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of France?
<|im_end|>
<|im_start|>assistant
"""

print("Testing Search-R1's vLLM wrapper with Qwen3...")

try:
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
    )
    print("✓ Model loaded via Search-R1 wrapper\n")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=300,
    )

    outputs = llm.generate([prompt], sampling_params)
    generated = outputs[0].outputs[0].text

    print(f"Generated ({len(generated)} chars):")
    print("-" * 60)
    print(generated[:400])
    print("-" * 60)

    if all(c == '!' for c in generated.strip()[:50]):
        print("\n❌ FAIL: Still generating '!!!'")
        sys.exit(1)
    else:
        print("\n✅ SUCCESS: Search-R1 pipeline generates proper text!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
