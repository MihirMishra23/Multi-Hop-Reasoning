# PopQA provenance

The PopQA questions are pinned to `akariasai/PopQA@098765c`. The generated
Wikipedia context corpus is pinned independently to
`ryannoonan/popqa-wikipedia-contexts@1b91217` and verified with SHA-256
`afcc52bd4ab5ebe4f63a249fb305b21de288e8a4eafbf8c73cbd183ec3320482`.

PopQA supplies questions and subject titles, but not article text. The corpus
builder fetches article extracts from the MediaWiki API and records page and
revision provenance. Contexts are joined to questions using `s_wiki_title`, not
list position.

Build a fresh corpus with:

```bash
python scripts/build_popqa_corpus.py \
  --output-dir data/generated/popqa-full \
  --gzip
```

The published pinned corpus is used automatically. Set `POPQA_CORPUS_PATH` only
when intentionally evaluating with another corpus artifact.

## Long-tail evaluation subset

The paper-style long-tail condition selects every row with subject popularity
`s_pop < 100` before shuffling. At the pinned PopQA revision this produces
exactly 1,399 questions. Seed 42 fixes their evaluation order.

Prepare the exact IDs, selected corpus, hashes, and metric manifest and launch
the Direct, RAG, and KBEVO evaluation matrix with:

```bash
bash scripts/launch_popqa_longtail.sh
```

The preparation step records all 1,399 ordered IDs and their SHA-256 digest.
Every method uses normalized exact match against any answer alias; F1 is also
reported by the shared evaluator.
