from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_merge_module():
    module_path = Path(__file__).parents[1] / "scripts" / "merge_eval_shards.py"
    spec = importlib.util.spec_from_file_location("merge_eval_shards_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merge_eval_shards = load_merge_module()


def artifact(qids, expected_examples):
    return {
        "metadata": {
            "dataset": "hotpotqa",
            "setting": "distractor",
            "dataset_source": "hf",
            "split": "dev",
            "type": "ircot",
            "model": "qwen3-1.7b",
            "model-path": None,
            "resolved_model_id": "Qwen/Qwen3-1.7B",
            "model_revision": "revision",
            "code_commit": "commit",
            "retrieval": {"backend": "official_elasticsearch_bm25", "k": 8},
            "ircot": {"max_steps": 8},
            "expected_examples": expected_examples,
            "dataset_provenance": {
                "source": {"kind": "test"},
                "selection": {
                    "seed": 0,
                    "setting": "distractor",
                    "ordered_ids": qids,
                },
            },
        },
        "inference_params": {"temperature": 0},
        "results": {qid: {"pred": qid, "gold_answer": [qid]} for qid in qids},
    }


class MergeEvalShardsTests(unittest.TestCase):
    def write_artifact(self, directory, start_index, payload):
        path = Path(directory) / f"eval_n{len(payload['results'])}_i{start_index}_shard.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_merges_complete_contiguous_shards_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as directory:
            second = self.write_artifact(directory, 2, artifact(["q2", "q3"], 2))
            first = self.write_artifact(directory, 0, artifact(["q0", "q1"], 2))

            merged, shard_records = merge_eval_shards.merge_shards(
                [second, first], expected_count=4, expected_shards=2
            )

        self.assertEqual(list(merged["results"]), ["q0", "q1", "q2", "q3"])
        self.assertEqual(merged["metadata"]["total_examples"], 4)
        self.assertEqual(merged["metadata"]["dataset_provenance"]["selection"]["count"], 4)
        self.assertEqual([record["start_index"] for record in shard_records], [0, 2])

    def test_rejects_a_gap_between_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.write_artifact(directory, 0, artifact(["q0"], 1))
            second = self.write_artifact(directory, 2, artifact(["q2"], 1))

            with self.assertRaisesRegex(ValueError, "not contiguous"):
                merge_eval_shards.merge_shards(
                    [first, second], expected_count=2, expected_shards=2
                )

    def test_rejects_incomplete_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_artifact(directory, 0, artifact(["q0"], 2))

            with self.assertRaisesRegex(ValueError, "incomplete"):
                merge_eval_shards.merge_shards(
                    [path], expected_count=2, expected_shards=1
                )


if __name__ == "__main__":
    unittest.main()
