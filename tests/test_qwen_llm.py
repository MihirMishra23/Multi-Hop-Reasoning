from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


def load_qwen_module():
    root = Path(__file__).parents[1] / "src" / "llm"
    package = types.ModuleType("qwen_test_package")
    package.__path__ = [str(root)]

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.no_grad = nullcontext
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object

    with patch.dict(
        sys.modules,
        {
            "qwen_test_package": package,
            "torch": torch,
            "transformers": transformers,
        },
    ):
        base_spec = importlib.util.spec_from_file_location(
            "qwen_test_package.base", root / "base.py"
        )
        base = importlib.util.module_from_spec(base_spec)
        sys.modules[base_spec.name] = base
        base_spec.loader.exec_module(base)

        qwen_spec = importlib.util.spec_from_file_location(
            "qwen_test_package.qwen", root / "qwen.py"
        )
        qwen = importlib.util.module_from_spec(qwen_spec)
        sys.modules[qwen_spec.name] = qwen
        qwen_spec.loader.exec_module(qwen)
    return qwen


qwen = load_qwen_module()


class FakeTensor:
    shape = (1, 1)


class FakeTokenizer:
    eos_token_id = 0
    chat_template = None

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"input_ids": FakeTensor()}

    @staticmethod
    def decode(tokens, skip_special_tokens):
        return "answer"


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [[10, 11]]


class QwenLLMTests(unittest.TestCase):
    def test_explicit_input_budget_overrides_tokenizer_metadata(self):
        llm = qwen.QwenLLM.__new__(qwen.QwenLLM)
        llm.model_name = "qwen3-1.7b"
        llm.timeout_s = 60
        llm.max_retries = 0
        llm.tokenizer = FakeTokenizer()
        llm.model = FakeModel()
        llm.use_device_map = True

        response = llm.run(
            "raw prompt",
            temperature=0,
            max_tokens=300,
            extra={"raw_prompt": True, "max_input_tokens": 8000},
        )

        self.assertEqual(response.text, "answer")
        self.assertEqual(
            llm.tokenizer.calls,
            [("raw prompt", {"return_tensors": "pt", "truncation": True, "max_length": 8000})],
        )
        self.assertNotIn("max_input_tokens", llm.model.calls[0])
        self.assertNotIn("raw_prompt", llm.model.calls[0])


if __name__ == "__main__":
    unittest.main()
