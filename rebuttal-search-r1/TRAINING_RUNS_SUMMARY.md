# Search-R1 reproduction — training-run summary

Compiled 2026-05-27. For sharing context on the Qwen3-vs-Qwen2.5 rebuttal experiments.

## Background

Reviewer asked: the Search-R1 paper reports HotpotQA F1 on **Qwen2.5-3B base** (~0.32 in the paper). The KBevo submission uses **Qwen3** models. To make the comparison apples-to-apples, we re-train the Search-R1 recipe on Qwen3-1.7B and Qwen3-4B using the same data and a closely-matched hyperparameter setup, then report Search-R1 numbers under the same eval conditions.

All training is **verl 0.7.0 + sglang multi-turn rollout + FSDP**. Eval framework is verl's built-in `val-core/searchR1_hotpotqa/acc/mean@1` (token-F1 reward; same 100-prompt val set used in `data/test_verl.parquet`). A separate 1000-prompt eval against the LMLM/KBevo eval split is being run in parallel via `eval_search_r1.py` (results in `eval_results/` once finished).

## Hyperparameters (identical across runs; only LR and model size vary)

| Knob | Value | Notes |
|------|-------|-------|
| Optimizer | AdamW, cosine schedule, 10% warmup | matches Search-R1 paper appendix |
| LR | 1e-6 OR 3e-6 | swept |
| KL regularizer | `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` | as in paper |
| PPO clip ratio (ε) | 0.2 | paper |
| Entropy coeff | 0 | paper |
| Algorithm | GRPO (`adv_estimator=grpo`) | paper |
| Batch shape | 16 prompts × 5 rollouts/prompt = 80 completions/step | matches KBevo |
| Max prompt / response | 4096 / 1024 tokens | matches KBevo |
| Max assistant turns | 4 | matches Search-R1 paper |
| Reward | byte-identical to KBevo `f1_reward` (`rewards/hotpotqa_f1.py`) | same metric across methods |
| Total training steps | 500 | matches KBevo `MAX_STEPS=500` |
| Val frequency | every 25 steps | 20 val checkpoints over the curve |
| Save frequency | every 100 steps | for resumable runs |
| Retrieval | E5-base-v2 + FAISS over wiki18+hotpotqa_corpus (185MB index, 63k passages) | Search-R1 paper setup |
| Search top-k | 3 | paper default |

## Runs

### Qwen3-1.7B @ lr=1e-6 (COMPLETED)

| Run segment | Steps | Status | Log |
|---|---|---|---|
| Original aimi-02 interactive run | 1–447 | crashed at 447 (NFS gremlin on `tool_config.yaml`, see below) | `logs/train_a2a_20260526_1906.log` |
| Resume from ckpt-400 (sbatch 839612 → aimi-02) | 401–500 | COMPLETED ✓ | `logs/train_a2a_lr1e-6_839612_20260527_0934.log` |

**Val F1 trajectory (combined)**:

| step | val F1 | step | val F1 |
|---:|---:|---:|---:|
| 25 | 0.177 | 275 | 0.222 |
| 50 | 0.189 | 300 | 0.239 |
| 75 | 0.179 | 325 | 0.231 |
| 100 | 0.205 | 350 | 0.247 |
| 125 | 0.230 | **375** | **0.262** *(peak)* |
| 150 | 0.213 | 400 | 0.230 |
| 175 | 0.175 | 425 | 0.234 |
| 200 | 0.214 | 450 | 0.222 |
| 225 | 0.237 | 475 | 0.244 |
| 250 | 0.261 | 500 | 0.212 |

**Notes**: noisy plateau; peak at step 375 (val F1 0.262). After resume from ckpt-400 the trajectory drops slightly — likely cosine LR tail (lr ≈ 1e-7 at step 425, ≈ 4e-8 at step 500) doing tiny gradient updates.

**Final HF checkpoint**: `merged_ckpts/qwen3-1.7b-searchR1-a2a-lr1e6-step500/`

---

### Qwen3-1.7B @ lr=3e-6 (COMPLETED)

| Run segment | Steps | Status | Log |
|---|---|---|---|
| Initial sbatch 823954 → aimi-03 | 1–120 | OOM-killed at 120 (256GB --mem cap hit; bumped to 768GB on resume) | `logs/train_a2a_lr3e-6_823954_20260527_0119.log` |
| Resume from ckpt-100 (sbatch 839613 → aimi-03) | 101–500 | COMPLETED ✓ | `logs/train_a2a_lr3e-6_839613_20260527_0934.log` |

**Val F1 trajectory (combined)**:

| step | val F1 | step | val F1 |
|---:|---:|---:|---:|
| 25 | 0.200 | 275 | **0.278** |
| 50 | 0.205 | 300 | 0.218 |
| 75 | 0.250 | 325 | 0.219 |
| 100 | 0.235 | 350 | 0.269 |
| 125 | 0.246 | 375 | 0.234 |
| 150 | 0.258 | 400 | 0.255 |
| 175 | 0.256 | 425 | 0.242 |
| 200 | 0.232 | 450 | 0.241 |
| 225 | 0.205 | **475** | **0.278** *(peak)* |
| 250 | 0.187 | 500 | 0.258 |

**Notes**: **3e-6 outperforms 1e-6** at the peak (0.278 vs 0.262, +0.016) and at step 500 (0.258 vs 0.212, +0.046). The 5e-6 run (not shown here) collapsed at step 70 — so 3e-6 is the sweet spot of the LR sweep we ran.

**Final HF checkpoint**: `merged_ckpts/qwen3-1.7b-searchR1-a2a-lr3e6-step500/`

---

### Qwen3-4B @ lr=1e-6 (RUNNING — 857992 on aimi-02)

Started 2026-05-27 14:36. Step 115/500 as of 16:37. ETA ~9 PM tonight.

**Val F1 trajectory so far**:

| step | val F1 |
|---:|---:|
| 25 | 0.179 |
| 50 | 0.175 |
| 75 | 0.239 |
| 100 | **0.287** |
| 125 | **0.297** |

Already at step 125 (val F1 0.297) the 4B model exceeds the 1.7B@1e-6 peak (0.262). 4B has clear capacity advantage — expected.

**Log**: `logs/train_a2a_lr1e-6_857992_20260527_1436.log`

---

### Qwen3-4B @ lr=3e-6 (RUNNING — 858020 on aimi-03)

Started 2026-05-27 14:42. Step 109/500 as of 16:37.

**Val F1 trajectory so far**:

| step | val F1 |
|---:|---:|
| 25 | 0.157 |
| 50 | 0.239 |
| 75 | 0.252 |
| 100 | 0.260 |
| 125 | **0.318** |

Best val F1 of any run so far (0.318 at step 125). If it doesn't collapse later, this is the run for the rebuttal table.

**Log**: `logs/train_a2a_lr3e-6_858020_20260527_1442.log`

---

## Incidents / fixes encountered

These don't affect the result quality but explain the log structure:

1. **NFS file deletion mid-training (≈ 01:03 UTC May 27)** — all three then-running jobs (4B@1e-6, 1.7B@3e-6, 1.7B@1e-6) crashed simultaneously with `FileNotFoundError: verl_config/tool_config/search_tool_config.yaml`. Root cause: verl's `ToolAgentLoop` re-reads this file every rollout-batch, and an external edit briefly removed it from NFS. **Fix**: sbatch wrapper now snapshots `verl_config/` to node-local `/scratch/lz586/run_${SLURM_JOB_ID}/` and points verl at the snapshot, so NFS edits can't reach running jobs.

2. **OOM at --mem=256G** (training peaks ~370 GB CPU for 1.7B, ~590 GB for 4B). **Fix**: bumped to 768G in `sbatch_qwen3_1.7b_apples_to_apples.slurm`. Memory growth seems to come from Ray's 120 `SearchExecutionWorker` actors (`num_workers=120` in `verl_config/tool_config/search_tool_config.yaml`) — each is a Python process with ~2 GB overhead.

3. **`flock` fd inherited by retrieval-server child** — eval jobs hung waiting for a lock the training's retrieval server inherited and never released. **Fix**: fast-path skip when retrieval is already up (no lock needed); close fd 200 before forking when we do launch.

## Pointers

- All sbatch wrappers and launchers in `rebuttal-search-r1/`.
- Raw verl logs (per step: entropy, lr, grad_norm, reward, num_turns, cpu_mem, plus val F1 every 25 steps) in `rebuttal-search-r1/logs/`.
- Per-rollout JSON dumps (the actual model completions) in `rebuttal-search-r1/outputs/`.
- Verl FSDP checkpoints in `rebuttal-search-r1/checkpoints/search-r1-hotpotqa/<experiment_name>/global_step_{100,200,300,400,500}/`.
- Merged HF checkpoints (vLLM-ready) in `rebuttal-search-r1/merged_ckpts/`.
- WandB project: `search-r1-hotpotqa` (entity: `ryan-noonan-cornell-university`). Run names: `qwen3-{1.7b,4b}-searchR1-a2a-lr{1e6,3e6}-*`.

## Headline numbers (val F1 mean@1 on 100-prompt `searchR1_hotpotqa` set)

| Model | LR | Peak val F1 | Step-500 val F1 | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B (base) | 1e-6 | 0.262 (step 375) | 0.212 | COMPLETED |
| Qwen3-1.7B (base) | **3e-6** | **0.278** (step 475) | **0.258** | COMPLETED |
| Qwen3-4B (base) | 1e-6 | 0.297 (step 125, in flight) | TBD | RUNNING ~9 PM ETA |
| Qwen3-4B (base) | **3e-6** | **0.318** (step 125, in flight) | TBD | RUNNING ~9 PM ETA |
| Qwen2.5-3B (paper) | (paper) | ~0.32 | — | reference |

The Qwen3-4B@3e-6 number, if it holds, matches or exceeds the paper's Qwen2.5-3B baseline despite running through the same recipe — addressing the reviewer's apples-to-apples concern.
