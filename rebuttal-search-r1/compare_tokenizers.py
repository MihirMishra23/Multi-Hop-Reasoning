#!/usr/bin/env python3
"""Compare Qwen2 and Qwen3 tokenizers"""

from transformers import AutoTokenizer

print("=== Qwen2-1.5B ===")
tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
print(f"Vocab size: {tok2.vocab_size}")
print(f"Token 0: {tok2.decode([0])!r}")
print(f"EOS: {tok2.eos_token!r} (id={tok2.eos_token_id})")
print(f"PAD: {tok2.pad_token!r} (id={tok2.pad_token_id})")

print("\n=== Qwen3-1.7B ===")
tok3 = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
print(f"Vocab size: {tok3.vocab_size}")
print(f"Token 0: {tok3.decode([0])!r}")
print(f"EOS: {tok3.eos_token!r} (id={tok3.eos_token_id})")
print(f"PAD: {tok3.pad_token!r} (id={tok3.pad_token_id})")

# Check if vocabs match
print(f"\nVocab sizes match: {tok2.vocab_size == tok3.vocab_size}")
print(f"EOS IDs match: {tok2.eos_token_id == tok3.eos_token_id}")
