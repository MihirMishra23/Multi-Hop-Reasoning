import importlib.util
import sys
import types
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "rebuttal-search-r1" / "eval_search_r1_qwen25.py"
SPEC = importlib.util.spec_from_file_location("eval_search_r1_qwen25", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_search_r1_tag_parsing():
    text = "<think>first</think><search> alpha beta </search>"
    assert MODULE.extract_search(text) == "alpha beta"
    assert MODULE.extract_answer(text) == ""
    assert MODULE.extract_answer("<answer> first </answer><answer> final </answer>") == "final"


def test_search_r1_passage_format_matches_upstream_infer():
    result = [
        {"document": {"contents": '"Title"\nBody text'}, "score": 0.9},
    ]
    assert MODULE.format_passages(result) == 'Doc 1(Title: "Title") Body text\n'


def test_corrected_tokenizer_is_serialized_for_vllm(monkeypatch, tmp_path):
    calls = {}

    class FakeTokenizer:
        def save_pretrained(self, path):
            calls["saved"] = Path(path)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            calls["model_path"] = model_path
            calls["kwargs"] = kwargs
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    tokenizer, vllm_path = MODULE.load_tokenizer("checkpoint", "revision", True)

    assert isinstance(tokenizer, FakeTokenizer)
    assert calls["model_path"] == "checkpoint"
    assert calls["kwargs"] == {
        "revision": "revision",
        "fix_mistral_regex": True,
    }
    assert Path(vllm_path) == calls["saved"]
    assert Path(vllm_path).parent == tmp_path
