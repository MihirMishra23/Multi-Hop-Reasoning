#!/usr/bin/env python3
"""Test Qwen3 tokenizer"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

# Check what "!" tokenizes to
text = "!"
tokens = tokenizer.encode(text)
print(f"Text: {text!r}")
print(f"Tokens: {tokens}")
print(f"Token IDs: {tokens}")

# Check repeated exclamation marks
text2 = "!!!!!!!!"
tokens2 = tokenizer.encode(text2)
print(f"\nText: {text2!r}")
print(f"Tokens: {tokens2}")

# Try decoding
decoded = tokenizer.decode([tokens[0]] * 500)
print(f"\nDecoding 500x token {tokens[0]}: {decoded[:200]!r}...")

# Check vocab size and special tokens
print(f"\nVocab size: {tokenizer.vocab_size}")
print(f"EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
