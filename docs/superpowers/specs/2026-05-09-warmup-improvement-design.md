# Warmup improvement — design

**Date:** 2026-05-09
**Status:** approved (pending spec review)

## Goal

Extend `scripts/warmup.py` so a single `make warmup` after first server boot pre-JITs every Triton kernel that real Claude Code / Codex traffic against the configured DGX Spark + Gemma-4 26B + DFlash stack will plausibly hit. After this lands, a fresh `~/.triton/cache` followed by `make serve` then `make warmup` should leave the running server free of `Triton kernel JIT compilation during inference: …` warnings under normal single-user agent traffic.

## Motivation

After running `make warmup`, the running server still emits warnings of the form:

```
[jit_monitor.py:103] Triton kernel JIT compilation during inference: <kernel>
```

for these kernels:

| Kernel | Why current warmup misses it |
|---|---|
| `_topk_topp_kernel` | `warmup.py` and `warmup_diag.py` both use `temperature=0` — the top-k/top-p sampling path never runs |
| `eagle_prepare_inputs_padded_kernel` | Padded variants for batch>1 — `warmup.py` only hits batch=1 |
| `eagle_prepare_next_token_padded_kernel` | Same — small-batch padded shapes not exercised |
| `copy_and_expand_dflash_inputs_kernel` | DFlash spec-decode kernel hit on shapes the 2-prompt warmup skips |
| `expand_kernel` | Same |
| `_compute_slot_mapping_kernel` | Triggered at prefill/decode shape boundaries the 2-prompt warmup skips |
| `fused_moe_kernel` | M-bin coverage exists in `warmup_diag.py` but not in `warmup.py` |

Each warning corresponds to a latency spike when the kernel JIT-compiles mid-request. The warnings disappear once the kernel is cached, so a sufficiently broad warmup makes all of them go away on subsequent serves on the same machine.

## Scope

**In scope**

- Replace `scripts/warmup.py` body with a four-phase warmup covering prefill shape diversity, sampling, small-batch concurrent decode, and sustained decode.
- Update README "First-run warmup" paragraph (time estimate + coverage).

**Out of scope**

- `scripts/warmup_diag.py` stays as-is (still the per-step Triton-cache-snapshot diagnostic).
- No log scraping / reactive replay. New JIT warnings, if any, get added to the script manually.
- No multimodal / vision kernel coverage — `serve` config disables image/video.
- No batch>3 — past that the kernel shapes either reuse smaller-batch padded variants or land in seldom-hit regions for single-user dev traffic.

## Coverage matrix

The script runs four phases. Each phase targets a class of kernels.

| Phase | Steps | Sampling | Silences |
|---|---|---|---|
| **A. Prefill M-bin sweep** | 8 sequential single-request calls, prompt sizes ~16, 72, 120, 400, 2000, 8000, 14000, 20000 tokens. `max_tokens=8` for all. The 20000-token step exercises sustained long prefill within the configured `--max-num-batched-tokens 32768` budget. | greedy | `fused_moe_kernel`, `_compute_slot_mapping_kernel`, `expand_kernel`, `copy_and_expand_dflash_inputs_kernel` |
| **B. Sampling sweep** | 3 sequential single-request calls, ~120-token prompt, `max_tokens=64`. (i) `temperature=0.7, top_p=0.95`, (ii) `temperature=1.0, top_k=50`, (iii) `temperature=0.8, top_p=0.9, presence_penalty=0.5`. | non-greedy | `_topk_topp_kernel`, sampling-side variants of the spec-decode kernels |
| **C. Concurrent decode** | (i) batch=2 with two ~120-token prompts at `temperature=0.7, top_p=0.95, max_tokens=64`. (ii) batch=3 with three ~400-token prompts greedy `max_tokens=64`. | mixed | `eagle_prepare_inputs_padded_kernel`, `eagle_prepare_next_token_padded_kernel` (padded variants for batch>1) |
| **D. Sustained decode** | 1 sequential call, ~120-token prompt, `temperature=0.7, top_p=0.95, max_tokens=256`. | non-greedy | Long-loop variants of the DFlash spec-decode kernels — keeps Eagle engaged across many decode steps |

Total: 14 steps / 17 HTTP requests, estimated ~3–5 minutes wall-clock on Spark.

## Code shape

Self-contained script. No new shared module — `warmup_diag.py` keeps its own copies of `post_chat` / `make_prompt`; the duplication is cheaper than a third file.

```
scripts/warmup.py
├── post_chat(content, max_tokens, *, temperature=0.0, top_p=None,
│             top_k=None, presence_penalty=None) -> (elapsed, text)
│      Builds the OpenAI-shaped JSON body, including only the
│      sampling fields that are not None / non-default. Same
│      urllib request flow as today.
├── make_prompt(approx_tokens) -> str
│      Copy of the helper in warmup_diag.py.
├── run_concurrent(prompts_and_kwargs) -> list[(elapsed, text)]
│      ThreadPoolExecutor wrapper around post_chat for batch>1.
├── phase_prefill_sweep()
├── phase_sampling_sweep()
├── phase_concurrent_decode()
├── phase_sustained_decode()
└── main()
       Calls each phase in order. Prints one line per step:
       "  <label> ... ok (Xs) -> <first-48-chars-of-reply>".
       At end: "warmup complete (total: Ys, steps: 14)".
       Failure: any URLError -> print FAIL, exit 1.
```

Reads the same env vars as today: `VLLM_BASE_URL`, `VLLM_API_KEY`, `SERVED_MODEL_NAME`. Defaults unchanged.

## Makefile + README

- **Makefile:** no change. The `warmup` target already runs `scripts/warmup.py`.
- **README "First-run warmup" paragraph:** bump time estimate from "~1–2 minutes" to "~3–5 minutes". Add one sentence noting coverage now includes sampling and small-batch concurrent decode in addition to prefill shapes. The `warmup-diag` paragraph is unchanged.

## Verification

After implementation:

1. Clear Triton cache: `rm -rf ~/.triton/cache`
2. `make serve` — wait for `Application startup complete`
3. `make warmup` — confirm exit 0, total runtime in 3–5 min range
4. Drive one Claude Code turn against the server (`make cc`, ask anything that runs ~hundred decode tokens with non-greedy sampling)
5. Inspect server log — confirm zero `Triton kernel JIT compilation during inference: …` lines after warmup completes

If a warning still appears for a kernel we expected to cover, treat that kernel-and-shape as an unhandled case and extend the relevant phase.

## Risks

- **Sampling stability.** vLLM's `_topk_topp_kernel` is the same kernel across (top_p, top_k) combos but can pick different launch configs at certain extreme values. The three sampling configs in phase B were chosen to span "common" agent traffic; if Codex/CC start sending values outside this range, a residual JIT may still occur. Acceptable — easy to extend.
- **Runtime budget.** Each batch=2/3 step waits on the slowest concurrent request. On a fully cold cache the first batch-N step compiles all the spec-decode padded variants and can take 30–60s. Total runtime envelope is 3–5 min — generous, but if it grows past 7–8 min on a real run we should split into a fast `warmup` and a thorough `warmup-full` rather than slow down the default path.
