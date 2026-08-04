# Search-R1 baseline on B200 — overnight status

**Date:** 2026-05-25 ~04:35
**Branch:** `searchr1-rebuttal-training`
**Node:** aimi-compute-02 (4× B200 grabbed; now released)
**Env:** `/scratch/lz586/envs/searchr1-sgl` (torch 2.8 cu128 + sglang 0.5.2 + verl 0.7 + flashinfer)

## TL;DR

Everything is staged: env built, data adapted, retrieval server works, training launches and gets all the way through SGLang CUDA-graph capture. It then fails consistently at the **first `release_memory_occupation` call** from verl's hybrid engine, before any training step runs. I applied one upstream patch (PR #21035) targeted at this code path; it did not fix the symptom. Per our agreed contract, I stopped iterating and left a clean box.

No training steps have completed. No wandb run was started successfully.

## The blocker (exact failure)

After CUDA graph capture finishes (`Capturing batches ... 100%`), verl calls `AgentLoopManager.sleep()` to put the SGLang hybrid engine to sleep so FSDP can take the GPUs. That calls `_engine.release_memory_occupation(tags=["kv_cache","weights"])` via HTTP to the per-GPU SGLang HTTP server. All 4 workers log:

```
WARNING:Connection error for release_memory_occupation (attempt 1)
WARNING:Connection error for release_memory_occupation (attempt 3)
RuntimeError: Failed to complete async request to release_memory_occupation after 3 attempts
```

So the SGLang HTTP server is either crashing or hanging when `release_memory_occupation` is invoked. Stack trace ends in
`verl/workers/rollout/sglang_rollout/sglang_rollout.py:156 → http_server_engine.py:718`.

Reference log: `/tmp/claude-1733861/-home-lz586-icl-Multi-Hop-Reasoning/371e35dc-8f95-43c0-822d-4bc2a9b74d6e/tasks/br9n2jzwj.output`

## What was tried

1. **PR #21035 patch** — wraps `_import_static_state` in `torch.inference_mode()` (file: `scheduler_update_weights_mixin.py`). Applied; still fails. In hindsight this PR fixes `resume_memory_occupation` (which calls `_import_static_state`), not `release_memory_occupation` (which calls `_export_static_state`). So the patch was for the wrong direction of the round-trip — explains why the symptom persisted.

2. **flashinfer attention backend** for SGLang (B200 / SM 100 is not supported by FA3). Confirmed via `+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer`. CUDA graph capture completes cleanly with flashinfer, so this part is fine.

3. **`attn_implementation=sdpa`** override to dodge the HF kernels-hub repo_type bug for B200. Works.

4. **cu12 ABI alignment** to torch 2.8 cu128's exact pins (nvjitlink-cu12 12.8.x, cusparseLt etc). Resolved earlier `libnvJitLink` and NCCL crashes.

## Plan options for morning (ranked)

**Plan B — disable `enable_memory_saver` for sglang in verl (likely fastest).** SGLang issue #21036 notes that avoiding torch_memory_saver sidesteps the release/resume round-trip entirely on Blackwell. Cost is more GPU memory (rollout weights stay resident across training), so we may need to drop `gpu_memory_utilization` from 0.5 → ~0.35 or `rollout.n` from 5 → 3 to fit. Edit candidate: `verl/workers/rollout/sglang_rollout/async_sglang_server.py` and/or the launch-time `enable_memory_saver=True` flag.

**Plan C — build sglang from `main` with the actual fix.** I never confirmed which sglang commit fixes `release_memory_occupation` on Blackwell. Need ~20 min to bisect issues #21036/#21037/#21100 area. Risk: building sglang+sgl-kernel from source on B200 may pull in newer CUDA assumptions that break our cu128 stack. Medium risk.

**Plan D — switch rollout backend to vLLM.** verl 0.7 supports `actor_rollout_ref.rollout.name=vllm`, but the `search_r1_like` recipe's tool integration (`multi_turn.tool_config_path`) is wired through SGLang's tool-calling protocol. We'd need to check whether vllm's agent loop supports the same `search_tool_config.yaml`. Higher uncertainty.

**Plan A (not recommended) — keep band-aiding sglang patches.** Diminishing returns; the failure mode is in the HTTP path, not just the static-state copy.

My recommendation: try Plan B first. If we can simply skip `enable_memory_saver`, hybrid engine still works (slower memory thrash but functional), and we keep this baseline on apples-to-apples ground (Qwen3 + SGLang + GRPO + multi-turn search).

## Current state — what's ready and what's not

Ready:
- Data: `data/train_verl.parquet`, `data/test_verl.parquet`, `data/index/e5_Flat.index`, `data/hotpotqa_corpus.jsonl` — all in verl schema.
- Retrieval: `retrieval_server_verl.py` + `launch_retrieval_verl.sh` — CPU FAISS, `/retrieve` endpoint, verified responding.
- Configs: `verl_config/search_multiturn_grpo.yaml` (hydra `pkg://verl.trainer.config`), `verl_config/tool_config/search_tool_config.yaml`.
- Launcher: `run_qwen3_1.7b_search_multiturn.sh` (4× GPU, ppo_mini=128, n=5, max_turns=4).
- Env: `/scratch/lz586/envs/searchr1-sgl` — torch 2.8 cu128 + sglang 0.5.2 + verl 0.7 + flashinfer + faiss-cpu.

Patches applied in-env (need to be re-applied if env is rebuilt):
- `/scratch/lz586/envs/searchr1-sgl/lib/python3.11/site-packages/sglang/srt/managers/scheduler_update_weights_mixin.py` — `_import_static_state` wrapped in `torch.inference_mode()` (PR #21035; insufficient for our failure mode but harmless).
- `/scratch/lz586/envs/searchr1-sgl/lib/python3.11/site-packages/kernels/deps.py` — `str | None` → `Optional[str]` (huggingface_hub strict validator on Python 3.11).

Box cleanup:
- Retrieval server killed (PID 4179864).
- All 4 GPUs at 1 MiB used.
- No `main_ppo` / `ray::` / `sglang` processes left.

## To resume

1. Plan B sketch (most likely):
   ```bash
   # 1. Find and flip enable_memory_saver=True → False in verl's sglang launch
   grep -rn "enable_memory_saver" /scratch/lz586/envs/searchr1-sgl/lib/python3.11/site-packages/verl/workers/rollout/sglang_rollout/
   # 2. Lower memory util to compensate
   #    edit run_qwen3_1.7b_search_multiturn.sh:  gpu_memory_utilization=0.35
   # 3. Relaunch retrieval + training
   bash rebuttal-search-r1/launch_retrieval_verl.sh &
   bash rebuttal-search-r1/run_qwen3_1.7b_search_multiturn.sh
   ```

2. If Plan B works, training should hit step 1 in ~3–5 min after launch. wandb project: `search-r1-hotpotqa-rebuttal`.

3. If Plan B fails (e.g., enable_memory_saver is required by the hybrid engine code path), fall back to Plan C and budget ~30 min.
