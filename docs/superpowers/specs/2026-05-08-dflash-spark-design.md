# DFlash + DGX Spark vLLM setup — design

**Date:** 2026-05-08
**Status:** approved (pending spec review)

## Goal

Make `coding-agent-vllm` install and run successfully on a DGX Spark (aarch64, NVIDIA GB10, Blackwell SM_121, Ubuntu 24.04), serving Gemma-4 26B-A4B with the `z-lab/gemma-4-26B-A4B-it-DFlash` draft for speculative decoding. The current `Makefile` source-builds vLLM from PR #41703 and does not apply the GB10-specific patches needed for Spark; this work replaces that path.

## Scope

**In scope**

- DGX Spark only. The portable x86_64 path is removed.
- Gemma-4 26B-A4B at BF16 (current model pair preserved).
- Single-stream serving (Claude Code / Codex coding-agent workloads).
- The three runtime file-patches from `meanaverage/gemma4-dflash-spark-vllm`, vendored verbatim.

**Out of scope (deferred)**

- 31B-FP8 path (the reference's verified config) — useful for benchmarking later.
- PR #41703's batched-verification correctness fix — only matters under heavy concurrent batching.
- Multi-GPU / tensor parallelism (Spark is single-GPU).
- Benchmarking and tuning `num_speculative_tokens`.
- Multimodal serving — disabled deliberately.

## Reference material

- DGX Spark + DFlash setup: <https://github.com/meanaverage/gemma4-dflash-spark-vllm>, Apache-2.0.
- Patcher script (vendored): `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` from the above repo, `main` branch.
- DFlash batched-verification fix (intentionally *not* applied): <https://github.com/vllm-project/vllm/pull/41703>.

## Architecture

Three layers, each isolated and independently testable:

```
┌──────────────────────────────────────────────────────────┐
│ Makefile                                                 │
│  install → uv pip install vllm==<pin>; then make patch   │
│  patch   → python scripts/patch_vllm_gb10_...py          │
│  serve   → VLLM_DISABLE_COMPILE_CACHE=1 vllm serve …     │
│  chat / cc / codex → unchanged client-side targets       │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│ scripts/patch_vllm_gb10_gemma4_dflash_runtime.py         │
│  (vendored verbatim, Apache-2.0)                         │
│  - patch_w8a8_utils       force CUTLASS FP8 unsupported  │
│  - patch_triton_backend   add supports_non_causal()      │
│  - patch_qwen3_dflash     pin FlashAttentionBackend      │
│  - clear_python_caches    drop stale __pycache__         │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│ vLLM nightly (pinned), installed in .venv via uv pip     │
│  Modified by patcher in-place (filesystem rewrite of 3   │
│  .py files). Idempotent — re-running is safe.            │
└──────────────────────────────────────────────────────────┘
```

The patcher is *not* runtime monkey-patching. It rewrites three files in the installed `vllm/` site-packages tree. This means:

- A subsequent `pip install -U vllm` overwrites the patches → user must re-run `make patch`.
- `make patch` is idempotent (the patcher prints `already-patched` on re-runs and skips writes).
- Pinning the vLLM version (one place in `Makefile` and `pyproject.toml`) prevents accidental upgrades from silently reverting patches.

## Components

### 1. `Makefile`

| Target | Behavior |
|---|---|
| `help` | Lists available targets. |
| `venv` | `uv venv` (unchanged). |
| `install` | `uv pip install -U --pre --torch-backend=auto --extra-index-url $(VLLM_INDEX_URL) "vllm==$(VLLM_PIN)"` followed by `$(MAKE) patch`. |
| `patch` | Runs `.venv/bin/python scripts/patch_vllm_gb10_gemma4_dflash_runtime.py`. Idempotent. |
| `serve` | New invocation (see below). |
| `chat` | Unchanged. |
| `cc` | Unchanged. |
| `codex` | Unchanged. |

**Removed:** `serve-no-speculative` (currently in working-tree diff, not committed; not part of the documented project flow).

**Variables:** existing `SERVED_MODEL_NAME`, `TOOL_CALL_PARSER`, `REASONING_PARSER`, `VLLM_BASE_URL`, `VLLM_API_KEY` are preserved. New variables (overridable from CLI):

- `VLLM_PIN ?= 0.19.2rc1.dev21` — initial candidate (the version the reference repo verified on Spark). Implementation step 1 verifies this resolves on aarch64+sm_121; if not, it bumps to the next-newer working nightly and updates this default.
- `VLLM_INDEX_URL ?= https://wheels.vllm.ai/nightly` — vLLM ships nightlies through their own wheel index, not PyPI. Implementation verifies this URL serves an aarch64 wheel for the chosen pin.

**`make serve` command line:**

```sh
VLLM_DISABLE_COMPILE_CACHE=1 \
.venv/bin/vllm serve google/gemma-4-26B-A4B-it \
    --served-model-name $(SERVED_MODEL_NAME) \
    --enable-auto-tool-choice \
    --tool-call-parser $(TOOL_CALL_PARSER) \
    --reasoning-parser $(REASONING_PARSER) \
    --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15}' \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --trust-remote-code
```

Differences vs. current Makefile and the rationale:

| Change | Why |
|---|---|
| Drop `--attention-backend triton_attn` | After the Triton non-causal patch, vLLM's auto-selector is correct; pinning to a flag we don't need narrows recovery if upstream evolves. |
| Drop `attention_backend: flash_attn` from speculative config | The `qwen3_dflash.py` patch hardcodes `FlashAttentionBackend` for the draft layer — JSON arg is redundant. |
| Add `--enforce-eager` | Blackwell SM_121 torch.compile / CUDA-graph paths in vLLM nightly are still being stabilized; eager mode is what the reference verified. |
| Add `VLLM_DISABLE_COMPILE_CACHE=1` | Belt-and-braces with `--enforce-eager`; prevents a stale compile cache from sneaking into a run. |
| Add `--max-model-len 16384` | Without it, vLLM provisions KV cache for Gemma-4's full 262K context, which OOMs Spark's 96 GB GPU partition. |
| Add `--gpu-memory-utilization 0.80` | Reference value. Spark uses unified memory; leaving headroom for the system is conservative-safe. |
| Reduce `--max-num-batched-tokens` 32768 → 16384 | Match `max-model-len`; mirrors reference. |
| Add `--limit-mm-per-prompt '{"image":0,"video":0}'` | Gemma-4 26B-A4B is a VLM; this is a coding agent — disabling image/video reclaims memory and skips vision-tower load. |
| Keep `num_speculative_tokens: 15` | User's existing choice; reference uses 8 for 31B. Tunable later. |

### 2. `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py`

Vendored verbatim from `meanaverage/gemma4-dflash-spark-vllm@main`, Apache-2.0. Three patches applied to the installed `vllm` package:

1. **`model_executor/layers/quantization/utils/w8a8_utils.py`** — force `cutlass_fp8_supported()` and `cutlass_block_fp8_supported()` to return `False`; force module-level `CUTLASS_FP8_SUPPORTED` and `CUTLASS_BLOCK_FP8_SUPPORTED` to `False`. Effect on the BF16 26B path: no-op at runtime (we never reach FP8 code paths), but harmless to apply and keeps the patcher uniform with the reference.

2. **`v1/attention/backends/triton_attn.py`** — add a `supports_non_causal()` classmethod returning `True`. Required because DFlash needs non-causal attention during draft initialization, which vLLM's selector blocks for Triton by default.

3. **`model_executor/models/qwen3_dflash.py`** — import `FlashAttentionBackend` and pass `attn_backend=FlashAttentionBackend` to the `Attention()` constructor in the DFlash draft model. Forces the draft layer onto FlashAttention regardless of selector decisions.

After patching, all `__pycache__` directories under the three patched subtrees are removed so the next Python run picks up the rewritten source.

The script's header gets a vendor block prepended:

```python
# Vendored from https://github.com/meanaverage/gemma4-dflash-spark-vllm
# @ <SHA — filled during implementation step 1 by `git ls-remote`>
# SPDX-License-Identifier: Apache-2.0
# Original copyright retained per Apache-2.0 §4(b)/(d).
```

The body of the script is otherwise unchanged.

### 3. `NOTICE`

New file at the repo root, one short paragraph required by Apache-2.0 §4(d):

```
This product includes software ("scripts/patch_vllm_gb10_gemma4_dflash_runtime.py")
developed by meanaverage (https://github.com/meanaverage/gemma4-dflash-spark-vllm),
licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).
```

### 4. `pyproject.toml`

Replace `dependencies = []` with `dependencies = ["vllm==0.19.2rc1.dev21"]` (matching the `Makefile`'s `VLLM_PIN` default) so any future lock-file or `uv sync` workflow honors the same pin. If implementation step 1 changes the pin, it must update both places.

Note: the vLLM wheel index isn't on PyPI, so `uv pip install` against this `pyproject.toml` *also* needs `--extra-index-url`. The `Makefile` provides this; running `pip install` directly won't work without that flag. We accept this trade-off — the project is Makefile-driven and the README documents the install path.

### 5. `README.md`

Rewritten Setup, Run, and Configuration sections:

- Remove references to PR #41703.
- Add Spark prerequisites (aarch64, GB10, Blackwell SM_121, Ubuntu 24.04).
- Document the two-step `make install` (which auto-runs `make patch`) and call out that re-running `pip install -U vllm` requires re-running `make patch`.
- Update the configuration table to remove `--attention-backend triton_attn` and `attention_backend: flash_attn` rows; add `--enforce-eager`, `--max-model-len 16384`, `--limit-mm-per-prompt`.

## Data flow at startup

```
make install
  └─ uv pip install vllm==<pin>      # downloads aarch64+sm121 wheel
  └─ make patch                      # Python ⇒ vllm.__file__ ⇒ rewrite 3 files
make serve
  └─ VLLM_DISABLE_COMPILE_CACHE=1 vllm serve …
       ├─ vllm imports patched modules
       │    ├─ w8a8_utils:    CUTLASS_FP8_SUPPORTED = False
       │    ├─ triton_attn:   supports_non_causal() = True
       │    └─ qwen3_dflash:  draft Attention pinned to FlashAttention
       ├─ loads Gemma-4 26B-A4B target on Triton backend
       ├─ loads DFlash draft on FlashAttention backend
       └─ listens on 0.0.0.0:8000 (OpenAI + Anthropic-compat paths)
make cc / make codex
  └─ client points ANTHROPIC_BASE_URL / OpenAI base_url at :8000
```

## Validation

Implementation must verify these in order. Stop and investigate on first failure.

1. **Install completes.** `make install` exits 0; `.venv/bin/python -c "import vllm; print(vllm.__version__)"` matches the pin. *If wheel doesn't exist for aarch64+sm_121 at the default pin:* browse the wheels index (`VLLM_INDEX_URL`), pick the next-newer nightly that ships an aarch64 wheel, update `VLLM_PIN` default in `Makefile` and `pyproject.toml`, retry.

2. **Patch applies cleanly, idempotent.** First `make patch` run prints three `patched` lines. Second run prints three `already-patched` lines and writes nothing.

3. **Patch markers are still valid against the pinned version.** *Most likely failure:* `patch_qwen3_dflash` raising `patch marker not found` because the 26B DFlash architecture file differs structurally from 31B. Recovery: read the installed `qwen3_dflash.py`, adapt the marker string to match the local function, document.

4. **Server starts.** `make serve` reaches "server listening" without exception; vLLM log mentions "DFlash" and "speculative" and reports both target and draft models loaded.

5. **End-to-end completion works.** `make chat` returns coherent responses to a simple prompt.

6. **DFlash actually engaged.** vLLM log line shows non-zero speculative-token acceptance rate (DFlash didn't silently fall back to autoregressive). If the metric isn't logged at default verbosity, raise log level briefly to confirm and revert.

7. **Coding agent integration unchanged.** `make cc` and `make codex` still launch and complete a simple turn against the local server.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 26B DFlash file structure differs from 31B → `patch_qwen3_dflash` marker mismatch | Medium | Implementation step (4): read installed file first, adapt marker if needed, document the new marker in the patcher header. |
| No aarch64+sm_121 wheel for the pinned nightly | Medium | Implementation tries 2–3 candidate nightlies (starting with `0.19.2rc1.dev21`); documents the working pin in `Makefile` *and* `pyproject.toml`. |
| `https://wheels.vllm.ai/nightly` URL incorrect or aarch64 wheels published elsewhere | Low-Medium | Implementation step 1 verifies the index URL by listing what's served; if wrong, finds the canonical index and updates `VLLM_INDEX_URL`. |
| Future `pip install -U vllm` silently reverts patches | Low | `make patch` is idempotent and called by `make install`; README documents that any manual upgrade requires `make patch` after. The pin in `pyproject.toml` discourages accidental upgrades. |
| FP8 patches no-op for our BF16 path turn out to *not* be no-ops | Low | If module-level `CUTLASS_FP8_SUPPORTED = False` somehow affects non-FP8 paths, drop that part of the patch — but the reference applies it without harm at the same vLLM version. |
| `--enforce-eager` is significantly slower than torch.compile path | Medium-perf, Low-correctness | Acceptable for first working version; revisit when benchmarking. |

## Implementation order (preview for the plan step)

1. Verify `VLLM_INDEX_URL` resolves and the `VLLM_PIN` default ships an aarch64+sm_121 wheel; bump pin if not. Capture `meanaverage/gemma4-dflash-spark-vllm@main` SHA via `git ls-remote`.
2. Create `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` with the vendor header (commit SHA filled in).
3. Create `NOTICE`.
4. Edit `pyproject.toml` to add the vLLM pin.
5. Rewrite `Makefile` (install, patch, serve targets; remove `serve-no-speculative`).
6. Rewrite affected `README.md` sections.
7. Run validation steps 1–7 above; debug as needed (most likely: adjust `patch_qwen3_dflash` marker for 26B).
8. Commit.

## Acceptance criteria

- `make install` on a clean DGX Spark venv succeeds end-to-end.
- `make patch` is idempotent.
- `make serve` starts and serves Gemma-4 26B-A4B with DFlash draft engaged (acceptance rate > 0 in logs).
- `make cc` and `make codex` complete at least one turn against the local server.
- `README.md` matches the actual flow; no stale references to PR #41703 or the x86_64 source build.
- `NOTICE` present, patcher script carries vendor header.
