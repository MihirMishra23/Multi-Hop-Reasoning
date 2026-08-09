import gzip
import json
from pathlib import Path

import pytest
from datasets import Dataset

from data import popqa, provenance
from data.popqa_corpus import (
    WikipediaClient,
    build_corpus,
    extract_paragraphs,
    records_from_response,
)


def test_extract_paragraphs_keeps_nonempty_lines():
    extract = "Lead paragraph.\n\nHistory\nFirst history paragraph.\n  \nReferences\n"
    assert extract_paragraphs(extract) == [
        "Lead paragraph.",
        "History",
        "First history paragraph.",
        "References",
    ]


def test_records_from_response_preserves_requested_order_and_redirects():
    payload = {
        "query": {
            "normalized": [{"from": "alpha", "to": "Alpha"}],
            "redirects": [{"from": "Alpha", "to": "Alpha Prime"}],
            "pages": [
                {
                    "pageid": 2,
                    "title": "Missing page",
                    "missing": True,
                },
                {
                    "pageid": 1,
                    "title": "Alpha Prime",
                    "fullurl": "https://en.wikipedia.org/wiki/Alpha_Prime",
                    "extract": "Lead.\n\nHistory\nDetails.",
                    "revisions": [{"revid": 123, "timestamp": "2026-01-02T03:04:05Z"}],
                },
            ],
        }
    }

    records = records_from_response(["alpha", "Missing page"], payload)
    assert [record["requested_title"] for record in records] == ["alpha", "Missing page"]
    assert records[0]["title"] == "Alpha Prime"
    assert records[0]["paragraphs"] == ["Lead.", "History", "Details."]
    assert records[0]["revision_id"] == 123
    assert records[1]["status"] == "missing"


def test_client_follows_extract_continuation_and_merges_pages(monkeypatch):
    responses = iter(
        [
            {
                "continue": {"excontinue": 1, "continue": "||"},
                "query": {
                    "pages": [
                        {"pageid": 1, "title": "Alpha", "extract": "Alpha context."},
                        {"pageid": 2, "title": "Beta"},
                    ]
                },
            },
            {
                "query": {
                    "pages": [
                        {"pageid": 1, "title": "Alpha"},
                        {"pageid": 2, "title": "Beta", "extract": "Beta context."},
                    ]
                }
            },
        ]
    )
    continuations = []
    delays = []
    client = WikipediaClient(request_delay=0.25, sleeper=delays.append)

    def fake_request(titles, continuation=None):
        continuations.append(continuation)
        return next(responses)

    monkeypatch.setattr(client, "_request", fake_request)
    records = client.fetch(["Alpha", "Beta"])

    assert continuations == [None, {"excontinue": 1, "continue": "||"}]
    assert delays == [0.25]
    assert [record["paragraphs"] for record in records] == [
        ["Alpha context."],
        ["Beta context."],
    ]


class _StubClient:
    api_url = "https://example.invalid/w/api.php"
    user_agent = "PopQATestBot/1.0 (test@example.invalid)"

    def __init__(self):
        self.calls = []

    def fetch(self, titles):
        self.calls.append(list(titles))
        return [
            {
                "requested_title": title,
                "title": title,
                "status": "ok",
                "page_id": index,
                "revision_id": 100 + index,
                "revision_timestamp": "2026-01-02T03:04:05Z",
                "url": f"https://example.invalid/wiki/{title}",
                "paragraphs": [f"Context for {title}"],
            }
            for index, title in enumerate(titles)
        ]


def test_build_corpus_is_resumable_and_writes_deterministic_gzip(tmp_path):
    selection = {
        "titles": ["Alpha", "Beta", "Gamma"],
        "rows": 4,
        "provenance": {"source": {"revision": "abc"}, "selection": {"count": 4}},
    }
    client = _StubClient()
    first = build_corpus(
        output_dir=tmp_path,
        selection=selection,
        client=client,
        batch_size=2,
        request_delay=0,
        create_gzip=True,
    )
    assert client.calls == [["Alpha", "Beta"], ["Gamma"]]
    assert first["corpus"]["status_counts"] == {"ok": 3}

    gzip_path = tmp_path / "popqa_contexts.jsonl.gz"
    first_bytes = gzip_path.read_bytes()
    second_client = _StubClient()
    second = build_corpus(
        output_dir=tmp_path,
        selection=selection,
        client=second_client,
        batch_size=2,
        request_delay=0,
        create_gzip=True,
    )
    assert second_client.calls == []
    assert second["corpus"]["sha256"] == first["corpus"]["sha256"]
    assert gzip_path.read_bytes() == first_bytes


def test_loader_joins_corpus_by_subject_title_not_position(tmp_path, monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "id": 1,
                "question": "Question A",
                "possible_answers": ["answer a"],
                "subj": "A",
                "s_wiki_title": "Article A",
            },
            {
                "id": 2,
                "question": "Question B",
                "possible_answers": ["answer b"],
                "subj": "B",
                "s_wiki_title": "Article B",
            },
            {
                "id": 3,
                "question": "Question A again",
                "possible_answers": ["answer a"],
                "subj": "A",
                "s_wiki_title": "Article A",
            },
            {
                "id": 4,
                "question": "Missing",
                "possible_answers": ["missing"],
                "subj": "Missing",
                "s_wiki_title": "Missing article",
            },
        ]
    )
    monkeypatch.setattr(popqa, "load_dataset", lambda *args, **kwargs: raw)
    corpus_path = tmp_path / "corpus.jsonl.gz"
    records = [
        {
            "requested_title": "Article B",
            "title": "Article B",
            "status": "ok",
            "paragraphs": ["B context"],
        },
        {
            "requested_title": "Article A",
            "title": "Canonical A",
            "status": "ok",
            "paragraphs": ["A context"],
        },
    ]
    with gzip.open(corpus_path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    with pytest.warns(RuntimeWarning, match="covers 2/3"):
        dataset = popqa.load_popqa(split="test", corpus_path=str(corpus_path))

    assert dataset[0]["contexts"] == ["A context"]
    assert dataset[0]["context_titles"] == ["Canonical A"]
    assert dataset[1]["contexts"] == ["B context"]
    assert dataset[2]["contexts"] == ["A context"]
    assert dataset[3]["contexts"] == []


def test_json_array_corpus_is_also_title_keyed(tmp_path, monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "id": 7,
                "question": "Question",
                "possible_answers": ["answer"],
                "subj": "Subject",
                "s_wiki_title": "Correct title",
            }
        ]
    )
    monkeypatch.setattr(popqa, "load_dataset", lambda *args, **kwargs: raw)
    corpus_path = Path(tmp_path) / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"title": "Wrong title", "paragraphs": ["wrong"]},
                {"title": "Correct title", "paragraphs": ["correct"]},
            ]
        ),
        encoding="utf-8",
    )

    dataset = popqa.load_popqa(split="test", corpus_path=str(corpus_path))
    assert dataset[0]["contexts"] == ["correct"]


def test_corpus_resolution_defaults_to_full_and_honors_overrides(monkeypatch):
    monkeypatch.delenv("POPQA_CORPUS_PATH", raising=False)
    downloads = []
    monkeypatch.setattr(
        popqa,
        "hf_dataset_file",
        lambda name: downloads.append(name) or "/cache/popqa_contexts.jsonl.gz",
    )

    full = popqa._resolve_popqa_corpus_path(None)
    monkeypatch.setenv("POPQA_CORPUS_PATH", "/data/environment.jsonl.gz")
    environment = popqa._resolve_popqa_corpus_path(None)
    explicit = popqa._resolve_popqa_corpus_path("/data/explicit.jsonl.gz")

    assert full == "/cache/popqa_contexts.jsonl.gz"
    assert environment == "/data/environment.jsonl.gz"
    assert explicit == "/data/explicit.jsonl.gz"
    assert downloads == ["popqa_contexts"]


def test_full_popqa_corpus_source_is_immutable_and_checksummed():
    source = provenance.dataset_source("popqa_contexts")

    assert source["hf_repo"] == "ryannoonan/popqa-wikipedia-contexts"
    assert source["revision"] == "1b91217d540cd4ab34e3efee1d7abe07c46f0209"
    assert source["file"] == "popqa_contexts.jsonl.gz"
    assert source["sha256"] == (
        "afcc52bd4ab5ebe4f63a249fb305b21de288e8a4eafbf8c73cbd183ec3320482"
    )


def test_hf_dataset_file_verifies_a_registered_checksum(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.jsonl.gz"
    artifact.write_bytes(b"verified artifact")
    source = {
        "hf_repo": "owner/repo",
        "revision": "immutable-revision",
        "file": artifact.name,
        "sha256": provenance.sha256_file(artifact),
    }
    monkeypatch.setattr(provenance, "dataset_source", lambda name: source)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(artifact))

    assert provenance.hf_dataset_file("popqa_contexts") == str(artifact)

    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="expected 000000"):
        provenance.hf_dataset_file("popqa_contexts")
