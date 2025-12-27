import argparse
import json
import math
import os
from typing import List, Optional

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
from tqdm import tqdm


def load_paragraphs(paragraphs_path: str) -> List[str]:
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "paragraphs" in data:
        return data["paragraphs"]
    return data


def load_annotations(annotations_path: str) -> List[str]:
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the annotator database pipeline.")
    parser.add_argument(
        "--mode",
        choices=["prepare", "annotate", "parse", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--qa-input",
        default="/share/j_sun/lmlm_multihop/datasets/hotpot_dev_distractor_v1.json",
        help="Path to raw HotpotQA JSON.",
    )
    parser.add_argument("--paragraphs-in", default="", help="Path to prebuilt paragraphs JSON.")
    parser.add_argument(
        "--paragraphs-out",
        default="/share/j_sun/lmlm_multihop/outputs/paragraphs/hotpotqa_dev_distractor_1k_seed_42.json",
        help="Output path for paragraphs JSON.",
    )
    parser.add_argument("--annotations-in", help="Path to existing annotations JSON.")
    parser.add_argument(
        "--annotations-out",
        default="/share/j_sun/lmlm_multihop/outputs/annotations/annotated_results.json",
        help="Output path for annotations JSON.",
    )
    parser.add_argument(
        "--annotations-batch-dir",
        default="/share/j_sun/lmlm_multihop/outputs/annotations/batches",
        help="Directory for per-batch annotations.",
    )
    parser.add_argument(
        "--database-out",
        default="/share/j_sun/lmlm_multihop/outputs/databases/hotpotqa_1k_42_dev_triplets.json",
        help="Output path for parsed database JSON.",
    )
    parser.add_argument(
        "--prompt-path",
        default="/share/j_sun/lmlm_multihop/prompts/llama-v6.1.json",
        help="Prompt template JSON for the annotator.",
    )
    parser.add_argument(
        "--model-path",
        default="kilian-group/LMLM-Annotator",
        help="Annotator model path or HF id.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-start", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--resume-annotate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.paragraphs_in and args.qa_input:
        raise ValueError("Provide only one of --qa-input or --paragraphs-in.")

    if args.mode in {"prepare", "annotate", "all"}:
        if not args.paragraphs_in and not args.qa_input:
            raise ValueError("Provide --qa-input or --paragraphs-in for prepare/annotate.")

    if args.mode in {"annotate", "all"}:
        if not args.prompt_path or not args.model_path:
            raise ValueError("Provide --prompt-path and --model-path to run annotation.")

    if args.mode == "parse" and not args.annotations_in:
        raise ValueError("Provide --annotations-in to parse a database.")


def should_prepare(args: argparse.Namespace) -> bool:
    return args.mode in {"prepare", "annotate", "all"} and not args.paragraphs_in


def should_annotate(args: argparse.Namespace) -> bool:
    if args.mode not in {"annotate", "all"}:
        return False
    if args.annotations_in and not args.resume_annotate:
        return False
    return True


def should_parse(args: argparse.Namespace) -> bool:
    return args.mode in {"parse", "all"} and bool(args.database_out)


def run_prepare(args: argparse.Namespace) -> List[str]:
    if args.paragraphs_in:
        return load_paragraphs(args.paragraphs_in)

    qa_data = load_qa_json(args.qa_input)
    paragraphs_data = prepare_paragraphs(
        qa_data,
        seed=args.seed,
        limit=args.limit,
        show_progress=True,
    )
    if args.paragraphs_out:
        save_paragraphs(paragraphs_data, args.paragraphs_out)
    return paragraphs_data["paragraphs"]


def run_annotate(args: argparse.Namespace, paragraphs: List[str]) -> str:
    prompt_template = load_prompt_template(args.prompt_path)
    initial_annotations: Optional[List[str]] = None
    start_index = args.batch_start if args.batch_start is not None else 0

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

    remaining = max(0, len(paragraphs) - start_index)
    total_batches = math.ceil(remaining / args.batch_size) if args.batch_size else 0

    annotated = []
    batch_iter = iter_annotate_batches(
        paragraphs=paragraphs,
        prompt_template=prompt_template,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        start_index=start_index,
        max_new_tokens=args.max_new_tokens,
        initial_annotations=initial_annotations,
    )
    for batch_index, _, annotated_data in tqdm(batch_iter, total=total_batches, desc="Annotating"):
        annotated = annotated_data
        if args.annotations_batch_dir:
            batch_path = os.path.join(
                args.annotations_batch_dir,
                f"annotated_results_batch_{batch_index}.json",
            )
            save_annotations(annotated, batch_path)

    save_annotations(annotated, args.annotations_out)
    return args.annotations_out


def run_parse(args: argparse.Namespace, annotations_path: str) -> None:
    parsed = parse_db_lookups(annotations_path, verbose=args.verbose, show_progress=True)
    save_database(parsed, args.database_out)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    print(f"Mode: {args.mode}")
    paragraphs = None
    annotations_path = args.annotations_in

    if should_prepare(args):
        print("Stage: prepare paragraphs")
        paragraphs = run_prepare(args)

    if should_annotate(args):
        print("Stage: annotate paragraphs")
        if paragraphs is None:
            paragraphs = load_paragraphs(args.paragraphs_in)
        annotations_path = run_annotate(args, paragraphs)

    if should_parse(args):
        print("Stage: parse annotations")
        if not annotations_path:
            raise ValueError("Provide --annotations-in or run annotation to parse a database.")
        run_parse(args, annotations_path)


if __name__ == "__main__":
    main()
