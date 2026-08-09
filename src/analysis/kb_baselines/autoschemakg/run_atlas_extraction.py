"""Run atlas-rag KnowledgeGraphExtractor against a local vLLM OpenAI-compatible
server. Expects the input JSON to already exist (from build_atlas_input.py).

This produces atlas-rag's native output under
    {output_dir}/kg_extraction/{model_name}_{filename}_output_{timestamp}_*.json
which we later convert to KBevo unified-KB schema via atlas_to_kbevo_kb.py.

Usage:
    PYTHONPATH=src python kb_baselines/autoschemakg/run_atlas_extraction.py \
        --input kb_baselines/autoschemakg/output/inputs/hotpotqa_dev_n5_seed42.json \
        --output-dir kb_baselines/autoschemakg/output/extractions/hotpotqa_dev_n5_seed42 \
        --vllm-endpoint http://localhost:8000/v1 \
        --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --batch-size 16
"""

import argparse
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="atlas-rag input JSON (list of {id,text})")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY", help="OpenAI client api_key (vLLM ignores it)")
    p.add_argument("--model-name", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-workers", type=int, default=8,
                   help="atlas-rag concurrent API workers (per stage)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--include-concept", action="store_true",
                   help="Also generate concept layer (slower; we don't need it for KBevo schema)")
    p.add_argument("--debug-mode", action="store_true",
                   help="atlas-rag debug mode: process only 20 samples")
    p.add_argument("--chunk-size", type=int, default=8192,
                   help="atlas-rag's text chunker char limit per chunk")
    args = p.parse_args()

    # atlas-rag's load_dataset expects all input files in `data_directory` matching
    # `filename_pattern`. Easiest path: stage the input file in its own dir.
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    staging_dir = Path(args.output_dir).resolve() / "atlas_input_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_file = staging_dir / input_path.name
    if staged_file.resolve() != input_path.resolve():
        shutil.copy(input_path, staged_file)
    filename_pattern = input_path.stem.split("_")[0]  # any unique prefix

    # Build OpenAI client + LLMGenerator pointing at vLLM
    from openai import OpenAI
    from atlas_rag.llm_generator import LLMGenerator
    from atlas_rag.kg_construction.triple_extraction import (
        KnowledgeGraphExtractor,
        ProcessingConfig,
    )

    client = OpenAI(base_url=args.vllm_endpoint, api_key=args.api_key)
    llm = LLMGenerator(client=client, model_name=args.model_name, backend="vllm",
                       max_workers=args.max_workers)

    config = ProcessingConfig(
        model_path=args.model_name,
        data_directory=str(staging_dir),
        filename_pattern=filename_pattern,
        batch_size_triple=args.batch_size,
        batch_size_concept=args.batch_size,
        output_directory=str(Path(args.output_dir).resolve()),
        include_concept=args.include_concept,
        debug_mode=args.debug_mode,
        max_workers=args.max_workers,
        max_new_tokens=args.max_new_tokens,
        chunk_size=args.chunk_size,
        chunk_overlap=0,
    )

    extractor = KnowledgeGraphExtractor(model=llm, config=config)
    print(f"[atlas-rag] Running extraction:")
    print(f"  input:          {input_path}")
    print(f"  filename_patt:  {filename_pattern}")
    print(f"  model:          {args.model_name}")
    print(f"  vllm endpoint:  {args.vllm_endpoint}")
    print(f"  output_dir:     {args.output_dir}")

    extractor.run_extraction()

    out_dir = Path(args.output_dir) / "kg_extraction"
    print(f"[atlas-rag] Done. Raw KG output JSON(s):")
    for f in sorted(out_dir.glob("*.json")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
