#!/usr/bin/env python3
"""Run-scoped semantic-equivalence checks for cross-example retrieval batching."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

EVALUATOR = Path(__file__).resolve().parents[1] / "rebuttal-search-r1" / "eval_search_r1_hermes.py"
spec = importlib.util.spec_from_file_location("eval_search_r1_hermes_test", EVALUATOR)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeResponse:
    def __init__(self, queries):
        self.queries = queries

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "result": [
                [
                    {
                        "document": {
                            "id": f"{query}-{rank}",
                            "contents": f"Title {query} {rank}\nBody {query} {rank}",
                        },
                        "score": 1.0 / rank,
                    }
                    for rank in range(1, 4)
                ]
                for query in self.queries
            ]
        }


def fake_post(url, json, timeout):
    del url, timeout
    return FakeResponse(json["queries"])


class BatchedRetrievalTests(unittest.TestCase):
    def test_batched_payloads_are_byte_identical_to_legacy_requests(self):
        groups = [["alpha"], ["beta", "gamma"], [], ["delta", "epsilon", "zeta"]]
        with patch.object(module.requests, "post", fake_post):
            for side in ("left", "right", "middle"):
                legacy = [
                    module.call_retrieval(
                        "http://retriever/retrieve",
                        group,
                        3,
                        30,
                        max_tool_response_length=91,
                        tool_response_truncate_side=side,
                    )
                    for group in groups
                ]
                batched = module.call_retrieval_batched(
                    "http://retriever/retrieve",
                    groups,
                    3,
                    30,
                    chunk_size=2,
                    max_tool_response_length=91,
                    tool_response_truncate_side=side,
                )
                self.assertEqual(batched, legacy)


if __name__ == "__main__":
    unittest.main()
