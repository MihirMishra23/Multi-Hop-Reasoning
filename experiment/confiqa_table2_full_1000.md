# ConFiQA Table 2 full rerun (n=1,000)

Numerical runs used commit `66732cb`, seed 42, and A100-SXM4-80GB GPUs. Final
Search-R1 runs used CUDA for both index construction and query retrieval.

| Model | Original EM / F1 | CF100 EM / F1 | CF500 EM / F1 |
| --- | ---: | ---: | ---: |
| KBEVO Qwen3-1.7B | 71.3 / 75.31 | 37.1 / 42.21 | 25.5 / 29.87 |
| KBEVO Qwen3-4B | 78.6 / 82.96 | 42.7 / 48.21 | 29.9 / 33.70 |
| Search-R1 Qwen2.5-3B | 53.3 / 60.55 | 50.3 / 57.37 | 42.9 / 49.84 |
| Search-R1 Qwen2.5-7B | 57.8 / 65.16 | 53.3 / 60.03 | 37.1 / 42.48 |

The paper reports EM only. Comparison below is `paper -> rerun (delta)`.

| Model | Original | CF100 | CF500 |
| --- | ---: | ---: | ---: |
| KBEVO Qwen3-1.7B | 74.0 -> 71.3 (-2.7) | 68.5 -> 37.1 (-31.4) | 64.2 -> 25.5 (-38.7) |
| KBEVO Qwen3-4B | 76.8 -> 78.6 (+1.8) | 75.6 -> 42.7 (-32.9) | 71.4 -> 29.9 (-41.5) |
| Search-R1 Qwen2.5-3B | 45.9 -> 53.3 (+7.4) | 43.4 -> 50.3 (+6.9) | 37.3 -> 42.9 (+5.6) |
| Search-R1 Qwen2.5-7B | 55.2 -> 57.8 (+2.6) | 52.2 -> 53.3 (+1.1) | 37.6 -> 37.1 (-0.5) |

Validation: every output contains 1,000 rows; counterfactual counts are exactly
0/100/500; all settings use the same ordered example IDs (SHA-256
`2a47114948532dec8c406a26b6d7dcfc8cec39f1ad8b7e6e4803cf728cd341bd`); and all
12 final jobs exited successfully. Artifacts are on Unicorn at
`/share/j_sun/lmlm_multihop/results/confiqa_table2_full`.

Search-R1 CPU-retriever originals are retained as `results_cpu_retriever.json`.
The final `results.json` files use CUDA consistently across all three settings.
The KBEVO counterfactual discrepancy is the main remaining reproduction issue.
