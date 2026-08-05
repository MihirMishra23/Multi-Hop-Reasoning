#!/usr/bin/env python3
"""Fetch the released IRCoT prompt assets from the pinned upstream commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
BASE_URL = f"https://raw.githubusercontent.com/StonyBrookNLP/ircot/{COMMIT}/prompts"
ASSETS = {
    "hotpotqa/gold_with_1_distractors_context_cot_qa_codex.txt": "00fd7b411360004f55c5d295d71d257d45f00c8125033faa88232935607993c3",
    "hotpotqa/gold_with_2_distractors_context_cot_qa_codex.txt": "af56b51582a5fdbebf5a5d50ab95cefcb95d92986cf820774ccbaa897a71552f",
    "hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt": "299fbb3b9e3b642cbe4c09ec035e64a2627a5d118aa2aaede8c8829e4ab7dbe8",
    "2wikimultihopqa/gold_with_1_distractors_context_cot_qa_codex.txt": "890050a6dd6c396b2af3cb2c1f88f01049e7bee49e64f11cb5fd08eaae27015a",
    "2wikimultihopqa/gold_with_2_distractors_context_cot_qa_codex.txt": "6cd153370f5f35a0e3db53d8def28abfe341ae67cddd2ccc5365eeb61769e235",
    "2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt": "f4ffc3fdcf1616b60b5317fc44da61ba1306c8413d86c59d01621b0433daad95",
    "musique/gold_with_1_distractors_context_cot_qa_codex.txt": "4f6b4b337de812ff94dbc713e3eabdbeb97f93fd5549ac2e3b92c5e0813cf5d0",
    "musique/gold_with_2_distractors_context_cot_qa_codex.txt": "454bac330eb8800e629a2ca9938b7358da8131a75c563b51ff1836cf4501377f",
    "musique/gold_with_3_distractors_context_cot_qa_codex.txt": "406602227e58733de4396230008ea9ccc8a2c167be2006f24e4fe77e3446b5d5",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("provenance/ircot/prompts"),
        help="Destination root for the pinned prompt files",
    )
    args = parser.parse_args()

    fetched = []
    for relative_path, expected_sha256 in ASSETS.items():
        destination = args.output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{BASE_URL}/{relative_path}") as response:
            content = response.read()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {relative_path}: {actual_sha256} != {expected_sha256}"
            )
        destination.write_bytes(content)
        fetched.append(
            {
                "path": str(destination),
                "sha256": actual_sha256,
                "bytes": len(content),
            }
        )
    print(json.dumps({"commit": COMMIT, "assets": fetched}, indent=2))


if __name__ == "__main__":
    main()
