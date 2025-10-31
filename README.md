# Limited Memory Language Models with Multi-Hop Reasoning

## HotpotQA Evaluation Pipeline

The `evaluation` package mirrors the HotpotQA scorer available in
`LLMAgents/memento`. It computes exact-match, F1, supporting-fact, and joint
metrics and can stream gold annotations directly from Hugging Face.

### Quick start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the evaluator (module path: `eval.hotpotqa.evaluate`):

   ```bash
   python -m eval.hotpotqa.evaluate path/to/predictions.json \
     --split validation \
     --setting distractor
   ```

   Use `--data-path /path/to/hotpot-jsons` for offline evaluation. Metrics are
   printed in plain text by default; add `--pretty` for a JSON summary or
   `--metrics-output results.json` to export them.

### Prediction format

The default HotpotQA JSON structure is supported out of the box:

```json
{
  "answer": {
    "5a8b57f25542995d1e6f1371": "yes"
  },
  "sp": {
    "5a8b57f25542995d1e6f1371": [["Scott Derrickson", 0], ["Ed Wood", 0]]
  }
}
```

Lists of records with `id`/`pred` keys or raw `{"id": "answer"}` mappings are
also accepted and automatically converted to this layout.