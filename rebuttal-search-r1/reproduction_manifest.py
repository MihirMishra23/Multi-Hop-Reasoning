#!/usr/bin/env python3
"""Immutable artifact provenance for the supported Search-R1 reconstruction."""

from __future__ import annotations

import argparse
import json
import re


SOURCE_RECIPE_COMMIT = "c30a5d919c8eec291ed5e3e6fefca18eb8138113"
CLASSIFICATION = (
    "API-compatible reconstruction; W&B recorded the package snapshot, but the "
    "exact editable verl source commit was not recorded"
)
RUNTIME = {
    "python": "3.11",
    "torch": "2.8.0+cu128",
    "verl": "0.7.0",
    "sglang": "0.5.2",
    "transformers": "4.56.1",
    "huggingface_hub": "0.34.4",
    "ray": "2.53.0",
    "uvicorn": "0.40.0",
}

ARTIFACTS = {
    "qwen3_1_7b": {
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    },
    "qwen3_4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    },
    "e5_base_v2": {
        "repo_id": "intfloat/e5-base-v2",
        "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
    },
    "hotpotqa": {
        "repo_id": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
    },
}


def artifact(name: str) -> dict[str, str]:
    try:
        selected = ARTIFACTS[name].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown reproduction artifact: {name}") from exc
    _validate_revision(selected["revision"])
    return selected


def _validate_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Artifact revisions must be full 40-character commit hashes")


def resolve_snapshot(repo_id: str, revision: str) -> str:
    _validate_revision(revision)
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=revision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", choices=sorted(ARTIFACTS))
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--field", choices=("repo_id", "revision"))
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()

    if args.describe:
        print(
            json.dumps(
                {
                    "source_recipe_commit": SOURCE_RECIPE_COMMIT,
                    "classification": CLASSIFICATION,
                    "runtime": RUNTIME,
                    "artifacts": ARTIFACTS,
                },
                sort_keys=True,
            )
        )
        return

    if args.field:
        if not args.artifact:
            parser.error("--field requires --artifact")
        print(artifact(args.artifact)[args.field])
        return

    if args.artifact:
        selected = artifact(args.artifact)
        repo_id = args.repo_id or selected["repo_id"]
        revision = args.revision or selected["revision"]
    else:
        if not args.repo_id or not args.revision:
            parser.error("use --artifact, or provide both --repo-id and --revision")
        repo_id = args.repo_id
        revision = args.revision
    print(resolve_snapshot(repo_id, revision))


if __name__ == "__main__":
    main()
