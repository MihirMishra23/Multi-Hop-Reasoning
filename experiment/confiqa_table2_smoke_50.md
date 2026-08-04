# ConFiQA Table 2 smoke (50 examples)

Run on Unicorn from code commit `6147881`, seed 42. All completed jobs exited 0;
artifacts are under
`/share/j_sun/lmlm_multihop/results/confiqa_table2_smoke`.

| Model | `orig` EM / F1 | `cf_100` EM / F1 | `cf_500` EM / F1 |
|---|---:|---:|---:|
| KBEVO Qwen3-1.7B | 80.0 / 83.6 | 50.0 / 52.0 | 48.0 / 53.9 |
| KBEVO Qwen3-4B | 80.0 / 86.9 | 46.0 / 48.5 | 48.0 / 51.6 |
| Search-R1 Qwen2.5-3B | 58.0 / 65.0 | 46.0 / 49.8 | 46.0 / 49.8 |
| Search-R1 Qwen2.5-7B | 52.0 / 62.0 | 34.0 / 36.7 | 34.0 / 36.7 |

Final job IDs by column/model order are `732809-732812`,
`732813,733090,733091,733705`, and `732817-732820`.

Every output contains 50 ordered IDs with SHA-256
`4b1ccfa4a255e99559d159fa82c27a78da531819d30a7fdfc8abe041010df316`.
Counterfactual counts are 0/50/50. The last two columns intentionally use the
same inputs at this smoke size; full 1,000-example runs distinguish 100 vs 500.
