#!/usr/bin/env python3
"""Direct test of Qwen3 with vLLM"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm import LLM, SamplingParams

print("=" * 60)
print("Testing Qwen3-1.7B with proper Qwen3 implementation")
print("=" * 60)

try:
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

prompts = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=50,
    stop=["<|im_end|>"],
)

print("Generating response...")
try:
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated = output.outputs[0].text
        print(f"\nPrompt: {output.prompt[:80]}...")
        print(f"Generated: {generated!r}\n")

        # Check if output is valid
        if not generated.strip():
            print("✗ FAIL: Empty output")
        elif all(c == '!' for c in generated.strip()):
            print("✗ FAIL: Still generating only exclamation marks")
        elif len(generated.strip()) < 3:
            print("✗ FAIL: Output too short")
        else:
            print("✓ SUCCESS! Model generating proper text")

except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
