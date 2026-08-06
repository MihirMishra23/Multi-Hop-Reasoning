# Dataset provenance

Every public loader is pinned to an immutable upstream revision in
`src/data/provenance.py`; evaluation outputs also store the ordered example IDs
and their SHA-256 digest. `MULTIHOP_DATA_DIR` may override the download cache,
but no Cornell filesystem path is required.

| Dataset | Pinned source | Notes |
|---|---|---|
| ConFiQA-MR | Context-DPO commit `557dadee` | Canonical JSON is SHA-256 `dbb76f...e6c0eede`; this exactly matches the former Unicorn copy. |
| HotpotQA | `hotpotqa/hotpot_qa@1908d6a` | `distractor` / `fullwiki` configs. |
| MuSiQue | `dgslibisey/MuSiQue@c8f4f8c` | Hugging Face mirror of the Stony Brook release. |
| MQuAKE | `henryzhongsc/MQuAKE-Remastered@b54712d` | Uses the CF6334 parquet. |
| 2WikiMultiHopQA | `kamelliao/2wikimultihopqa@f4f0d7e` | Uses the pinned `data/dev.json` mirror of the official release. |
| SynthWorlds | `kenqgu/SynthWorlds@d0f02ed` | `qa-sm` / `qa-rm` configs. |
| TriviaQA | `mandarjoshi/trivia_qa@0f7faf3` | Uses `rc.wikipedia`. |
| PopQA | `akariasai/PopQA@098765c` | The exact team-generated 1,000-example Wikipedia context artifact is bundled under `data/artifacts/`; it is not part of upstream PopQA. |

## PopQA Wikipedia corpus

PopQA supplies questions and Wikipedia subject titles, but not article text. The
repository therefore contains a resumable corpus builder that fetches plain-text
article extracts from the official MediaWiki API and records each returned page
ID, revision ID, revision timestamp, canonical URL, and redirect target.

Run a small, seed-42 smoke build:

```bash
python scripts/build_popqa_corpus.py \
  --output-dir data/generated/popqa-smoke \
  --seed 42 \
  --limit 5 \
  --gzip
```

Run the full 14,267-question build by omitting `--seed` and `--limit`:

```bash
python scripts/build_popqa_corpus.py \
  --output-dir data/generated/popqa-full \
  --gzip
```

The full set currently contains 12,244 unique requested subject titles. The
builder batches requests, honors `Retry-After`, retries transient failures, and
appends progress to `popqa_contexts.jsonl`, so rerunning the same command resumes
completed pages. It also writes a deterministic `popqa_contexts.jsonl.gz` and a
`manifest.json` containing the PopQA source revision, ordered row IDs, corpus
checksum, and scrape status counts. Use `--refresh` only when intentionally
creating a new Wikipedia snapshot.

To evaluate with a generated corpus, point the loader at it without changing
code:

```bash
export POPQA_CORPUS_PATH=/path/to/popqa_contexts.jsonl.gz
```

Contexts are joined through each row's `s_wiki_title`, not list position. This
allows repeated titles to reuse one article and prevents a missing page from
shifting every later question onto the wrong context.

### Distribution

Keep the builder, tests, and manifest format in Git. Publish the completed
`popqa_contexts.jsonl.gz` plus `manifest.json` in a Hugging Face dataset
repository, then pin that dataset repository's commit and artifact SHA-256 in
`src/data/provenance.py`. Do not commit successive full corpus snapshots to this
Git repository: they are derived Wikipedia data, are expected to be tens of
megabytes compressed, and would permanently inflate Git history. The bundled
legacy artifact remains only for reproducing the historical 1,000-row runs.

## ConFiQA Table 2 protocol

Use seed 42 and evaluate the first 1,000 rows of the shuffled ConFiQA-MR file.
The loader fixes the row order before applying edits, so every setting contains
the same ordered questions:

- `orig`: 0 counterfactual rows.
- `cf_100`: positions 0-99 are counterfactual.
- `cf_500`: positions 0-499 are counterfactual.
- `cf`: every selected row is counterfactual.

This order-of-operations matters. The old loader edited raw rows 0-99/0-499
and shuffled afterward, leaving only 14/79 edited rows in the prior 1,000-row
seed-42 evaluation.

For a 50-example smoke test, `cf_100` and `cf_500` are intentionally identical
(all 50 rows are counterfactual). A full 1,000-example run is required to
distinguish their reported scores.

Search-R1 Table 2 evaluation uses an index built only from the selected ConFiQA
contexts for that setting. The Wiki-18 corpus is not part of this ConFiQA
protocol.
