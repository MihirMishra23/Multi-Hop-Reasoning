# IRCoT with repository models

This evaluator ports the IRCoT inference method from
`StonyBrookNLP/ircot@3c1820f698eea5eeddb4fba3c56b64c961e063e4` and substitutes the
repository's configured model for the retired `code-davinci-002` model. It is
therefore a method reproduction, not a claim that Qwen predictions reproduce
the original paper's model outputs.

The faithful path supports the three datasets shared with the current Table 3
matrix: HotpotQA, 2WikiMultiHopQA, and MuSiQue. Upstream IRCoT did not release
a PopQA prompt or report a PopQA result.

## One-time setup

Install the sentence segmenter used by the official loop and fetch the released
few-shot prompt files at the pinned revision:

```bash
pip install -e .
python -m spacy download en_core_web_sm
python scripts/fetch_ircot_assets.py
```

Set up the official Elasticsearch 7.10 retrieval service using the pinned
upstream repository. Follow its `README.md` to download the raw corpora, build
the indexes, and launch `retriever_server/serve.py`. Before evaluation, verify
the document counts from the upstream release:

| Corpus | Required documents |
| --- | ---: |
| HotpotQA | 5,233,329 |
| 2WikiMultiHopQA | 430,225 |
| MuSiQue | 139,416 |

Do not substitute the per-question distractor contexts or FlashRAG index for
these dataset-level indexes and label the result as faithful IRCoT.

The released paper development protocol uses each dataset's exact 100-row
`processed_data/{dataset}/dev_subsampled.jsonl`, prompt set 1, and the complete
Cartesian product of retrieval depth `2, 4, 6, 8` with prompt distractor count
`1, 2, 3`. It selects the best setting by the released DROP F1 evaluator. The
paper then applies that setting to prompt sets 1, 2, and 3 on the released test
subsample and reports their mean and sample standard deviation. Do not call an
arbitrary first-100 slice or a single untuned setting the paper protocol.
The cluster jobs pin Qwen3-1.7B to `70d244cc...` and Qwen3-4B to
`1cfa9a72...`; the resolved full revision is stored with every prediction.

## Smoke evaluation

Run a small subset first:

```bash
python src/eval_multihop.py \
  --method ircot \
  --dataset hotpotqa \
  --setting distractor \
  --split dev \
  --model qwen3-1.7b \
  --batch-size 1 \
  --total-count 2 \
  --ircot-retriever-url http://RETRIEVER_HOST:8000 \
  --ircot-index-manifest provenance/ircot/hotpotqa_index.json \
  --ircot-evaluation-file /path/to/processed_data/hotpotqa/dev_subsampled.jsonl \
  --ircot-prompt-set 1 \
  --ircot-retrieval-k 6 \
  --ircot-distractor-count 2 \
  --ircot-max-evidence 15 \
  --ircot-max-steps 10 \
  --ircot-generator-max-tokens 300 \
  --debug-evidence
```

IRCoT tuned retrieval depth over `2, 4, 6, 8` and prompt distractor count over
`1, 2, 3`. A new Qwen comparison should tune those choices on a development
set, then record the selected values for the held-out evaluation. The saved
JSON records the prompt hash, upstream commit, model revision, retrieval
service URL, corpus name, raw prompts/responses, and cumulative retrieval
traces.

`--ircot-index-manifest` is optional for a control-flow smoke test, but required
for any reportable run. The manifest should record the raw corpus URL/version,
license, file SHA-256 values, preprocessing/index command, Elasticsearch
version and settings, final index document count, and a content hash or hashes
for the archived index. Its own SHA-256 is stored in every prediction file.

For the exact development sweep, start the real index service, run the
two-example real-corpus smoke, and only then submit the grid:

```bash
sbatch scripts/ircot_index_service.slurm
IRCOT_RETRIEVER_URL=http://RETRIEVER_HOST:18000 \
  sbatch scripts/ircot_official_smoke.slurm
IRCOT_RETRIEVER_URL=http://RETRIEVER_HOST:18000 \
  bash scripts/launch_ircot_dev_grid.sh
python scripts/summarize_ircot_dev_grid.py /path/to/predictions/dev_grid \
  --output /path/to/dev_grid_summary.json
```

## Faithfulness boundary

The evaluator preserves the released prompt reader and 8,000-token fitting
rules, literal completion-prompt transport, newline stop, 300-token generation
budget, spaCy sentence boundaries, factual-sentence query selection, official
WH-word removal, Elasticsearch request schema, 90/100 fuzzy evidence
deduplication, 600-word retrieval filter, 350-word prompt truncation, 15-item
evidence cap, answer/max-sentence exits, and the separate final QA reader.

The intentional difference is the configured repository model and its
tokenizer/generation implementation. Prompt assets and attribution are tracked
under `provenance/ircot/` and `third_party/ircot/`.
