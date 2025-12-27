import argparse
import json
import os
from typing import List

from database_creation.annotator import (
    iter_annotate_batches,
    load_prompt_template,
    load_qa_json,
    parse_db_lookups,
    prepare_paragraphs,
    save_annotations,
    save_database,
    save_paragraphs,
)


def load_paragraphs(paragraphs_path: str) -> List[str]:
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "paragraphs" in data:
        return data["paragraphs"]
    return data


def load_annotations(annotations_path: str) -> List[str]:
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the annotator database pipeline.")
    parser.add_argument(
        "--qa-input",
        default="/share/j_sun/lmlm_multihop/dataset/hotpot_dev_distractor_v1.json",
        help="Path to raw HotpotQA JSON.",
    )
    parser.add_argument(
        "--paragraphs-in",
        default="/home/rtn27/LMLM/build-database/data/hotpotqa_dev_distractor_1k_seed_42_paragraphs.json",
        help="Path to prebuilt paragraphs JSON.",
    )
    parser.add_argument(
        "--paragraphs-out",
        default="/home/rtn27/LMLM/build-database/data/atomic_sentences_hotpotqa_1k_seed_42.json",
        help="Output path for paragraphs JSON.",
    )
    parser.add_argument("--annotations-in", help="Path to existing annotations JSON.")
    parser.add_argument(
        "--annotations-out",
        default="/home/rtn27/LMLM/build-database/annotation/annotated_results.json",
        help="Output path for annotations JSON.",
    )
    parser.add_argument(
        "--annotations-batch-dir",
        default="/home/rtn27/LMLM/build-database/annotation",
        help="Directory for per-batch annotations.",
    )
    parser.add_argument(
        "--database-out",
        default="/home/rtn27/LMLM/build-database/triplets/hotpotqa_1k_42_dev_triplets.json",
        help="Output path for parsed database JSON.",
    )
    parser.add_argument(
        "--prompt-path",
        default="/home/rtn27/LMLM/prompts/llama-v6.1.json",
        help="Prompt template JSON for the annotator.",
    )
    parser.add_argument(
        "--model-path",
        default="kilian-group/LMLM-Annotator",
        help="Annotator model path or HF id.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-start", type=int, default=328)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--no-flatten", action="store_true")
    parser.add_argument("--resume-annotate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.paragraphs_in:
        paragraphs = load_paragraphs(args.paragraphs_in)
    else:
        if not args.qa_input:
            raise ValueError("Provide --qa-input or --paragraphs-in to build paragraphs.")
        qa_data = load_qa_json(args.qa_input)
        paragraphs_data = prepare_paragraphs(
            qa_data,
            seed=args.seed,
            limit=args.limit,
            flatten=not args.no_flatten,
        )
        paragraphs = paragraphs_data["paragraphs"]
        if args.paragraphs_out:
            save_paragraphs(paragraphs_data, args.paragraphs_out)

    annotations_path = None
    initial_annotations = None
    start_index = args.batch_start if args.batch_start is not None else 0
    if args.annotations_in and not args.resume_annotate:
        annotations_path = args.annotations_in
    else:
        if not args.prompt_path or not args.model_path:
            raise ValueError("Provide --prompt-path and --model-path to run annotation.")
        prompt_template = load_prompt_template(args.prompt_path)
        if args.annotations_in:
            initial_annotations = load_annotations(args.annotations_in)
            if args.batch_start is None:
                start_index = len(initial_annotations)
        if not args.annotations_out and args.annotations_in:
            args.annotations_out = args.annotations_in
        if not args.annotations_out:
            raise ValueError("Provide --annotations-out to save annotations.")

        if args.annotations_batch_dir:
            os.makedirs(args.annotations_batch_dir, exist_ok=True)

        annotated = []
        for batch_index, _, annotated_data in iter_annotate_batches(
            paragraphs=paragraphs,
            prompt_template=prompt_template,
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            start_index=start_index,
            max_new_tokens=args.max_new_tokens,
            initial_annotations=initial_annotations,
        ):
            annotated = annotated_data
            if args.annotations_batch_dir:
                batch_path = os.path.join(
                    args.annotations_batch_dir,
                    f"annotated_results_batch_{batch_index}.json",
                )
                save_annotations(annotated, batch_path)

        save_annotations(annotated, args.annotations_out)
        annotations_path = args.annotations_out

    if args.database_out:
        if not annotations_path:
            raise ValueError("Provide --annotations-in or run annotation to parse a database.")
        parsed = parse_db_lookups(annotations_path, verbose=args.verbose)
        save_database(parsed, args.database_out)


if __name__ == "__main__":
    main()
