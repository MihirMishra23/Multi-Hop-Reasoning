import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_hermes_eval_module():
    module_path = (
        Path(__file__).parents[1] / "rebuttal-search-r1" / "eval_search_r1_hermes.py"
    )
    module_name = "eval_search_r1_hermes_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    fake_data = types.ModuleType("data")
    fake_data.get_dataset = lambda **kwargs: None
    fake_eval = types.ModuleType("eval")
    fake_metrics = types.ModuleType("eval.metrics")
    fake_metrics.exact_match_score = lambda prediction, gold: prediction == gold
    fake_metrics.f1_score = lambda prediction, gold: 0.0
    fake_reward = types.ModuleType("hotpotqa_f1")
    fake_reward.compute_score = lambda **kwargs: 0.0
    replacements = {
        "data": fake_data,
        "eval": fake_eval,
        "eval.metrics": fake_metrics,
        "hotpotqa_f1": fake_reward,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


hermes = load_hermes_eval_module()


def test_hermes_tool_call_and_answer_parsing():
    text = (
        '<thinking>find it</thinking><tool_call>{"name":"search",'
        '"arguments":{"query_list":["alpha", "beta"]}}</tool_call>'
    )
    assert hermes.parse_tool_call(text) == ["alpha", "beta"]
    assert hermes.extract_answer("prefix <answer>  Paris </answer>") == "Paris"


def test_confiqa_manifest_requires_exact_query_prefix(tmp_path):
    store_ids = ["12", "7", "44"]
    digest = hashlib.sha256("\n".join(store_ids).encode()).hexdigest()
    manifest = {
        "dataset_provenance": {
            "selection": {
                "setting": "cf_100_conflict_free",
                "seed": 42,
                "count": 3,
                "ordered_ids_sha256": digest,
                "ordered_ids": store_ids,
            }
        }
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        corpus_manifest=str(path),
        confiqa_setting="cf_100_conflict_free",
        seed=42,
        expected_store_samples=3,
    )

    assert hermes.validate_confiqa_manifest(args, store_ids[:2]) == manifest
    with pytest.raises(ValueError, match="query IDs are not the ordered prefix"):
        hermes.validate_confiqa_manifest(args, ["7", "12"])
