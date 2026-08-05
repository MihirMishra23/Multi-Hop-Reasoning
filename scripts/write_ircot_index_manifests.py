#!/usr/bin/env python3
"""Write versioned manifests for the exact indexes built by upstream IRCoT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests


UPSTREAM_COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
PROCESSED_ARCHIVE = "processed_data.zip"
PROCESSED_ARCHIVE_URL = "https://drive.google.com/file/d/1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC/view"
PROCESSED_ARCHIVE_SHA256 = "271fff07efb120a71739c89ab69ab10f4c00059e74b2f7b451a607158c364906"
DATASETS = {
    "hotpotqa": {
        "documents": 5233329,
        "archive": "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        "url": "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
        "evaluation": "hotpotqa/dev_subsampled.jsonl",
        "source": "HotpotQA Wikipedia snapshot",
    },
    "2wikimultihopqa": {
        "documents": 430225,
        "archive": "2wikimultihopqa.zip",
        "url": "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1",
        "evaluation": "2wikimultihopqa/dev_subsampled.jsonl",
        "source": "2WikiMultiHopQA release",
    },
    "musique": {
        "documents": 139416,
        "archive": "musique_v1.0.zip",
        "url": "https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view",
        "evaluation": "musique/dev_subsampled.jsonl",
        "source": "MuSiQue v1.0 release",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_ids_sha256(path: Path) -> str:
    ids = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                ids.append(str(json.loads(line)["question_id"]))
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--elasticsearch-url", default="http://127.0.0.1:9200")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_archive_path = root / "downloads" / PROCESSED_ARCHIVE
    processed_archive_sha256 = sha256_file(processed_archive_path)
    if processed_archive_sha256 != PROCESSED_ARCHIVE_SHA256:
        raise ValueError(
            f"Processed-data archive checksum is {processed_archive_sha256}; "
            f"expected {PROCESSED_ARCHIVE_SHA256}"
        )

    version = requests.get(args.elasticsearch_url, timeout=30).json()["version"]
    for dataset, specification in DATASETS.items():
        count_response = requests.get(f"{args.elasticsearch_url}/{dataset}/_count", timeout=30)
        count_response.raise_for_status()
        count = int(count_response.json()["count"])
        if count != specification["documents"]:
            raise ValueError(f"{dataset} count is {count}; expected {specification['documents']}")
        settings = requests.get(f"{args.elasticsearch_url}/{dataset}/_settings", timeout=30).json()
        mappings = requests.get(f"{args.elasticsearch_url}/{dataset}/_mapping", timeout=30).json()
        archive_path = root / "downloads" / specification["archive"]
        evaluation_path = root / "data" / "processed_data" / specification["evaluation"]
        manifest = {
            "dataset": dataset,
            "source": specification["source"],
            "source_url": specification["url"],
            "license": "See the upstream dataset release",
            "source_archive": str(archive_path),
            "source_archive_sha256": sha256_file(archive_path),
            "evaluation_archive_url": PROCESSED_ARCHIVE_URL,
            "evaluation_archive": str(processed_archive_path),
            "evaluation_archive_sha256": processed_archive_sha256,
            "evaluation_file": str(evaluation_path),
            "evaluation_file_sha256": sha256_file(evaluation_path),
            "evaluation_ids_sha256": evaluation_ids_sha256(evaluation_path),
            "evaluation_count": sum(1 for line in evaluation_path.open() if line.strip()),
            "upstream_repository": "https://github.com/StonyBrookNLP/ircot",
            "upstream_commit": UPSTREAM_COMMIT,
            "index_command": f"python retriever_server/build_index.py {dataset}",
            "elasticsearch_version": version,
            "index_name": dataset,
            "index_document_count": count,
            "index_settings": settings,
            "index_mappings": mappings,
        }
        logical_index = json.dumps(
            {
                "archive_sha256": manifest["source_archive_sha256"],
                "upstream_commit": UPSTREAM_COMMIT,
                "version": version,
                "count": count,
                "settings": settings,
                "mappings": mappings,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest["logical_index_sha256"] = hashlib.sha256(logical_index.encode("utf-8")).hexdigest()
        output_path = output_dir / f"{dataset}_index.json"
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{output_path} {sha256_file(output_path)}")


if __name__ == "__main__":
    main()
