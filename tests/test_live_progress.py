import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.live_progress import (
    append_completion_records,
    completion_record,
    load_completion_ids,
    score_result,
    score_results,
    write_progress,
)


def _result(pred="The Eiffel Tower", answers=None, raw="FINAL_ANSWER: The Eiffel Tower"):
    return {
        "pred": pred,
        "gold_answer": answers or ["Eiffel Tower", "the eiffel tower"],
        "trace": [{"answer": pred, "raw_response": raw}],
    }


class LiveProgressTests(unittest.TestCase):
    def test_completion_record_keeps_raw_completion_and_best_alias_score(self):
        result = _result()
        record = completion_record("q1", result, "direct")

        self.assertEqual(record["qid"], "q1")
        self.assertEqual(record["completion"], "FINAL_ANSWER: The Eiffel Tower")
        self.assertEqual(record["em"], 1.0)
        self.assertEqual(record["f1"], 1.0)
        self.assertEqual(record["metric"], "answer_em_f1")

    def test_completion_jsonl_is_deduplicated_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.completions.jsonl"
            logged_ids = set()

            first = append_completion_records(str(path), {"q1": _result()}, "direct", logged_ids)
            second = append_completion_records(str(path), {"q1": _result()}, "direct", logged_ids)

            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            self.assertEqual(load_completion_ids(str(path)), {"q1"})
            self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_atomic_progress_snapshot_has_rolling_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            progress_path = Path(temporary_directory) / "run.progress.json"
            results = {"q1": _result(), "q2": _result(pred="Paris", answers=["London"])}
            payload = write_progress(
                str(progress_path),
                completed=2,
                expected=4,
                scores=score_results(results, "rag"),
                checkpoint_path="preds/run.json",
                completions_path="preds/run.completions.jsonl",
                status="running",
                last_qids=["q2"],
            )

            on_disk = json.loads(progress_path.read_text())
            self.assertEqual(on_disk, payload)
            self.assertEqual(payload["fraction"], 0.5)
            self.assertEqual(payload["rolling"]["count"], 2)
            self.assertAlmostEqual(payload["rolling"]["em"], 0.5)
            self.assertAlmostEqual(payload["rolling"]["f1"], 0.5)

    def test_ircot_live_score_uses_official_drop_metric(self):
        ftfy = types.ModuleType("ftfy")
        ftfy.fix_text = lambda value: value
        with patch.dict(sys.modules, {"ftfy": ftfy}):
            score = score_result(_result(pred="12", answers=["12"]), "ircot")

        self.assertEqual(score["metric"], "official_ircot_drop")
        self.assertEqual(score["em"], 1.0)
        self.assertEqual(score["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
