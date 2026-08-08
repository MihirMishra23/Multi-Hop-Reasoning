# Search-R1 on ConFiQA

This is a minimal batch wrapper around the public Search-R1 inference recipe;
it does not reimplement training. It uses the public Qwen2.5 checkpoints at
their pinned Hugging Face revisions:

- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo@bd4f5b0`
- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo@713cbe3`

For each ConFiQA setting, build an E5 index from exactly the selected contexts,
then evaluate:

```bash
python rebuttal-search-r1/prepare_confiqa_corpus.py \
  --setting cf_100 --num-samples 1000 --seed 42 --output-dir /tmp/confiqa_cf100

python rebuttal-search-r1/build_retrieval_index.py \
  --corpus /tmp/confiqa_cf100/corpus.jsonl \
  --output /tmp/confiqa_cf100/e5_Flat.index

python rebuttal-search-r1/eval_search_r1_qwen25.py \
  --model-path PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo \
  --confiqa-setting cf_100 --num-samples 1000 --seed 42 \
  --retrieval-corpus /tmp/confiqa_cf100/corpus.jsonl \
  --retrieval-index /tmp/confiqa_cf100/e5_Flat.index \
  --output /tmp/confiqa_cf100/searchr1_3b.json
```

The prompt, search loop, temperature 0.7, E5-base-v2 retriever and retrieval
top-k 3 follow `PeterGriffinJin/Search-R1@598e61b`. Wiki-18 is not used: Table 2
is a closed-corpus ConFiQA experiment, so the corpus is derived from the same
selected ConFiQA rows and variant as the questions.
