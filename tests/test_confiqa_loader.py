import json

import pytest

from data import confiqa
from data.provenance import dataset_source, selected_rows_provenance


def _fake_rows(count):
    return [
        {
            "question": f"question {index}",
            "orig_context": f"original context {index}",
            "orig_answer": f"original {index}",
            "orig_alias": [],
            "orig_path_labeled": f"[('h{index}', 'r', 'o{index}')]",
            "cf_context": f"counterfactual context {index}",
            "cf_answer": f"counterfactual {index}",
            "cf_alias": [],
            "cf_path_labeled": f"[('h{index}', 'r', 'c{index}')]",
        }
        for index in range(count)
    ]


@pytest.fixture()
def fake_confiqa(tmp_path, monkeypatch):
    path = tmp_path / "ConFiQA-MR.json"
    path.write_text(json.dumps(_fake_rows(600)), encoding="utf-8")
    monkeypatch.setattr(confiqa, "_resolve_confiqa_path", lambda *args, **kwargs: str(path))
    return path


def test_variants_share_order_and_apply_edits_after_shuffle(fake_confiqa):
    datasets = {
        setting: confiqa.load_confiqa(setting=setting, seed=42, limit=550)
        for setting in ("orig", "cf_100", "cf_500")
    }
    assert datasets["orig"]["id"] == datasets["cf_100"]["id"] == datasets["cf_500"]["id"]
    assert sum(datasets["orig"]["is_counterfactual"]) == 0
    assert sum(datasets["cf_100"]["is_counterfactual"]) == 100
    assert sum(datasets["cf_500"]["is_counterfactual"]) == 500
    assert datasets["cf_100"][99]["answers"][0].startswith("counterfactual")
    assert datasets["cf_100"][100]["answers"][0].startswith("original")


def test_fifty_row_smoke_has_expected_counterfactual_counts(fake_confiqa):
    expected = {"orig": 0, "cf_100": 50, "cf_500": 50, "cf": 50}
    for setting, count in expected.items():
        dataset = confiqa.load_confiqa(setting=setting, seed=42, limit=50)
        assert sum(dataset["is_counterfactual"]) == count


def test_legacy_edit_before_shuffle_bug_is_reproduced_by_the_old_ordering():
    selected_ids = confiqa._ordered_source_indices(6000, 42)[:1000]
    assert sum(source_id < 100 for source_id in selected_ids) == 14
    assert sum(source_id < 500 for source_id in selected_ids) == 79


def test_confiqa_source_is_immutable_and_exact():
    source = dataset_source("confiqa")
    assert len(source["revision"]) == 40
    assert source["sha256"] == "dbb76f361831b754b87219344d80a386eaf83078ffe5888272dbe9d8e6c0eede"


def test_ordered_ids_are_recorded_not_only_hashed():
    provenance = selected_rows_provenance(
        "confiqa", ["7", "2", "9"], seed=42, setting="cf_100", counterfactual_count=3
    )
    assert provenance["selection"]["ordered_ids"] == ["7", "2", "9"]
    assert provenance["selection"]["counterfactual_count"] == 3
