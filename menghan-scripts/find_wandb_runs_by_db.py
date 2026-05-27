#!/usr/bin/env python3
"""Find wandb runs that referenced one of the target DATABASE_PATHs.

This searches the wandb project in three ways (in order):
    1. Server-side filter on config.database_path  (fast, exact match)
    2. Client-side scan of run.config dict          (catches alt key names)
    3. Download config.yaml file from each run and grep   (catches CLI args
       captured by wandb-metadata.json that aren't in structured config)

The 3 default targets are the Gemini KBs referenced in the W2 rebuttal table:
    - hotpotqa_validation_42_1000_all_context_database.json
    - musique_validation_42_1000_all_context_database.json
    - 2wiki_db.json (as2637's prebuilt 2wiki KB)

Usage:
    python menghan-scripts/find_wandb_runs_by_db.py                # all 3
    python menghan-scripts/find_wandb_runs_by_db.py --deep         # also grep files
    python menghan-scripts/find_wandb_runs_by_db.py --since 2025-12-01
"""

import argparse
import os
import sys
import tempfile

DEFAULT_TARGETS = {
    "hotpotqa": "/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json",
    "musique":  "/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json",
    "2wiki":    "/share/j_sun/as2637/database/2wiki_db.json",
}

# Substring tokens that uniquely identify each target DB (cheaper to grep)
TARGET_TOKENS = {
    "hotpotqa": "hotpotqa_validation_42_1000_all_context_database.json",
    "musique":  "musique_validation_42_1000_all_context_database.json",
    "2wiki":    "as2637/database/2wiki_db.json",
}

INTERESTING_METRICS = (
    "eval/em", "eval/f1", "eval_em", "eval_f1",
    "test/em", "test/f1",
    "eval/exact_match", "eval/f1_score",
    "train/em", "train/f1",
)


def print_run(r, hit_via, hit_label, target):
    print()
    print(f"  [{hit_label}]  hit via: {hit_via}")
    print(f"  name:       {r.name}")
    print(f"  url:        {r.url}")
    print(f"  state:      {r.state}")
    print(f"  created:    {r.created_at}")
    for k in ("model_path", "dataset_name", "two_phase",
              "phase1_prompt_type", "use_inverses",
              "retrieval_threshold", "retrieval_top_k",
              "learning_rate", "num_train_epochs"):
        v = r.config.get(k)
        if v is not None:
            print(f"  cfg.{k:22s}={v}")
    summary = dict(r.summary) if r.summary else {}
    shown = [f"{k}={summary[k]}" for k in INTERESTING_METRICS if k in summary]
    if shown:
        print(f"  metrics:    {'  '.join(shown)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity",  default=os.environ.get("WANDB_ENTITY",  "ryan-noonan-cornell-university"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "LMLM-Multihop"))
    p.add_argument("--deep", action="store_true",
                   help="also DOWNLOAD config.yaml + wandb-metadata.json from each run and grep "
                        "(slow but catches CLI args not in run.config dict)")
    p.add_argument("--since", default=None,
                   help="only scan runs created >= this ISO date (e.g. 2025-12-01)")
    p.add_argument("--limit", type=int, default=2000,
                   help="max runs to scan in deep mode")
    p.add_argument("--per-page", type=int, default=200)
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb not installed. pip install wandb", file=sys.stderr)
        sys.exit(1)

    print(f"Querying wandb: {args.entity}/{args.project}")
    api = wandb.Api()

    # ── Strategy 1: server-side filter on config.database_path ──
    print("\n[1/3] Server-side filter on config.database_path...")
    target_paths = list(DEFAULT_TARGETS.values())
    filters = {"$or": [{"config.database_path": t} for t in target_paths]}
    if args.since:
        filters = {"$and": [filters, {"created_at": {"$gte": args.since}}]}
    runs1 = list(api.runs(f"{args.entity}/{args.project}", filters=filters, per_page=args.per_page))
    print(f"  matched {len(runs1)} runs")

    # ── Strategy 2: client-side scan run.config (catches alt key names) ──
    print("\n[2/3] Client-side scan of run.config dict...")
    list_filter = {"created_at": {"$gte": args.since}} if args.since else {}
    all_runs = list(api.runs(f"{args.entity}/{args.project}", filters=list_filter, per_page=args.per_page))
    print(f"  scanning {len(all_runs)} runs total")
    runs2 = {}  # (label, run) -> hit_via
    for r in all_runs:
        cfg = r.config
        # check any string-valued config field
        for k, v in cfg.items():
            if not isinstance(v, str):
                continue
            for label, tok in TARGET_TOKENS.items():
                if tok in v:
                    runs2[(label, r.id)] = (r, f"config.{k}")
                    break
    print(f"  matched {len(runs2)} (run, label) pairs via dict scan")

    # ── Strategy 3: download config.yaml + wandb-metadata.json files, grep ──
    runs3 = {}
    if args.deep:
        print(f"\n[3/3] Deep mode — downloading config.yaml from up to {args.limit} runs...")
        scanned = 0
        for r in all_runs[: args.limit]:
            scanned += 1
            if scanned % 50 == 0:
                print(f"  scanned {scanned}/{len(all_runs[:args.limit])}...")
            try:
                # try config.yaml first
                file = r.file("config.yaml")
                if file is None:
                    continue
                with tempfile.TemporaryDirectory() as td:
                    file.download(root=td, replace=True)
                    fp = os.path.join(td, "config.yaml")
                    if not os.path.exists(fp):
                        continue
                    with open(fp) as f:
                        content = f.read()
                for label, tok in TARGET_TOKENS.items():
                    if tok in content:
                        runs3[(label, r.id)] = (r, "file:config.yaml")
            except Exception as e:
                # Some runs lack the file; skip silently
                continue
        print(f"  matched {len(runs3)} (run, label) pairs via file grep")
    else:
        print("\n[3/3] Deep mode skipped. Use --deep to enable file-grep search.")

    # ── Merge + present ──
    by_target = {label: [] for label in DEFAULT_TARGETS}
    for r in runs1:
        db = r.config.get("database_path", "")
        for label, target in DEFAULT_TARGETS.items():
            if db == target:
                by_target[label].append((r, "server-filter"))
    for (label, _rid), (r, via) in runs2.items():
        by_target[label].append((r, via))
    for (label, _rid), (r, via) in runs3.items():
        by_target[label].append((r, via))

    # Dedup by run.id, prefer earliest strategy
    for label in by_target:
        seen = set()
        dedup = []
        for r, via in by_target[label]:
            if r.id in seen:
                continue
            seen.add(r.id)
            dedup.append((r, via))
        dedup.sort(key=lambda rv: rv[0].created_at or "", reverse=True)
        by_target[label] = dedup

    for label, target in DEFAULT_TARGETS.items():
        print()
        print("=" * 80)
        print(f"[{label}]  {target}")
        print("=" * 80)
        matched = by_target[label]
        if not matched:
            print("  (no matches in wandb — try --deep, or these runs may not be wandb-logged)")
            continue
        for r, via in matched:
            print_run(r, via, label, target)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
