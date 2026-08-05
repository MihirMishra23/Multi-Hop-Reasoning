from __future__ import annotations

import json
import tempfile
import unittest

try:
    import datasets  # noqa: F401
    import ftfy  # noqa: F401
    import numpy  # noqa: F401
    import scipy  # noqa: F401

    from src.data.ircot_official import load_ircot_official_evaluation
    from src.eval.ircot_official_metrics import drop_metrics, evaluate_predictions

    HAS_EVAL_DEPENDENCIES = True
except ImportError:
    HAS_EVAL_DEPENDENCIES = False


@unittest.skipUnless(HAS_EVAL_DEPENDENCIES, "IRCoT evaluation dependencies are not installed")
class IRCoTOfficialEvaluationTests(unittest.TestCase):
    def test_released_jsonl_schema_is_normalized_without_reordering(self):
        records = [
            {
                "question_id": "q2",
                "question_text": "Second?",
                "answers_objects": [{"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["Two"]}],
                "contexts": [{"title": "T2", "paragraph_text": "P2", "is_supporting": True}],
            },
            {
                "question_id": "q1",
                "question_text": "First?",
                "answers_objects": [{"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["One"]}],
                "contexts": [{"title": "T1", "paragraph_text": "P1", "is_supporting": False}],
            },
        ]
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.close()

        dataset = load_ircot_official_evaluation(handle.name)

        self.assertEqual(dataset["id"], ["q2", "q1"])
        self.assertEqual(dataset[0]["answers"], ["Two"])
        self.assertEqual(dataset[0]["golden_contexts"], ["T2: P2"])
        self.assertEqual(dataset[0]["supporting_facts"], [{"title": "T2", "sentence_id": 0}])

    def test_drop_metric_matches_upstream_number_and_span_rules(self):
        self.assertEqual(drop_metrics(["the 2"], ["2"]), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(drop_metrics(["Paris France"], ["Paris"]), (0.0, 0.67, 0.5, 1.0))
        metrics = evaluate_predictions(
            [
                {
                    "pred": "Paris",
                    "gold_answer_objects": [
                        {"number": "", "date": {"day": "", "month": "", "year": ""}, "spans": ["Paris"]}
                    ],
                }
            ]
        )
        self.assertEqual(metrics, {"em": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "count": 1})

    def test_drop_date_answer_uses_upstream_reader_format(self):
        metrics = evaluate_predictions(
            [
                {
                    "pred": "5-May-2020",
                    "gold_answer_objects": [
                        {"number": "", "date": {"day": "5", "month": "May", "year": "2020"}, "spans": []}
                    ],
                }
            ]
        )
        self.assertEqual(metrics, {"em": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0, "count": 1})


if __name__ == "__main__":
    unittest.main()
