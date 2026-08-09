import gzip
import hashlib
import json
from pathlib import Path

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


def test_conflict_free_manifest_records_exact_universe_and_inverse_audit():
    manifest = confiqa.load_conflict_free_manifest()
    universe = manifest["selection_universe"]
    assert universe["seed"] == 42
    assert universe["retained_count"] == 1000
    assert len(universe["ordered_source_ids"]) == 1000
    assert (
        universe["ordered_source_ids_sha256"]
        == "2a47114948532dec8c406a26b6d7dcfc8cec39f1ad8b7e6e4803cf728cd341bd"
    )
    assert manifest["selection_algorithm"]["maximum_counterfactual_count"] == 356
    assert manifest["smoke_query_selection"]["count"] == 50
    assert (
        manifest["smoke_query_selection"]["ordered_source_ids_sha256"]
        == "4b1ccfa4a255e99559d159fa82c27a78da531819d30a7fdfc8abe041010df316"
    )

    expected = {"cf_100_conflict_free": 100, "cf_356_conflict_free": 356}
    for setting, count in expected.items():
        condition = manifest["conditions"][setting]
        assert condition["actual_counterfactual_count"] == count
        assert len(condition["selected_cf_source_ids"]) == count
        assert condition["ambiguity"]["forward_direct"]["ambiguous_key_count"] == 0
        assert (
            condition["ambiguity"]["actual_database_with_inverses"][
                "ambiguous_key_count"
            ]
            > 0
        )


def test_conflict_free_selection_is_by_source_id(monkeypatch):
    rows = _fake_rows(4)
    manifest = {
        "conditions": {
            "cf_100_conflict_free": {
                "label": "CF-100-conflict-free",
                "selected_cf_source_ids": [3, 1],
            }
        }
    }
    monkeypatch.setattr(confiqa, "load_conflict_free_manifest", lambda: manifest)
    dataset = confiqa._normalize_confiqa(
        rows, [2, 3, 0, 1], "cf_100_conflict_free"
    )
    assert dataset["id"] == ["2", "3", "0", "1"]
    assert dataset["is_counterfactual"] == [False, True, False, True]
    assert dataset[1]["answers"][0] == "counterfactual 3"
    assert dataset[3]["answers"][0] == "counterfactual 1"


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


def test_bundled_popqa_artifact_preserves_historical_bytes():
    artifact = (
        Path(__file__).parents[1] / "data" / "artifacts" / "popqa_corpus_1000_ex_seed_42.json.gz"
    )
    digest = hashlib.sha256()
    with gzip.open(artifact, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == "00c9c266728f63ba0fc259d727ccab488a9ad54bc594eabd10ffe131ca0875fc"
