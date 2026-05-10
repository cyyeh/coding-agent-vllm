# coding-agent-vllm

> Run Claude Code or Codex against a self-hosted vLLM backend on NVIDIA DGX Spark — Gemma 4 26B accelerated with DFlash block-diffusion speculative decoding.

Local vLLM server for [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) with [`z-lab/gemma-4-26B-A4B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) as the speculative draft model.

DFlash uses block diffusion to draft multiple tokens in parallel, reportedly delivering up to ~3.7x speedup over autoregressive decoding when paired with this verifier.

## Prerequisites

- NVIDIA DGX Spark (aarch64, GB10 / Blackwell SM_121, Ubuntu 24.04)
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Hugging Face account with access granted to `google/gemma-4-26B-A4B-it`

This setup targets DGX Spark only. For traditional x86_64 + CUDA hosts, see upstream vLLM directly.

## Setup

```bash
cp .env.example .env       # then fill in HF_TOKEN
make venv
make install
```

`make install` pulls a pinned vLLM nightly wheel from the vLLM nightly index, then automatically runs `make patch` to adapt the installed `vllm` package for GB10 / Blackwell SM_121 + DFlash. See [Patches](#patches) for details.

## Patches

`scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` rewrites three files inside the installed `vllm` site-packages tree (it edits source on disk and clears `__pycache__` — not runtime monkey-patching). The script is vendored from [`meanaverage/gemma4-dflash-spark-vllm`](https://github.com/meanaverage/gemma4-dflash-spark-vllm) (Apache-2.0; see `NOTICE`), with one local change: an idempotency guard in `patch_w8a8_utils` so re-running `make patch` is safe.

What each patch does:

- **`patch_w8a8_utils`** — forces `cutlass_fp8_supported()` and the `CUTLASS_FP8_SUPPORTED` module flag to `False`. Only relevant if you switch the target to FP8 quantization (which we don't on the BF16 26B path) — at that point the patch routes around CUTLASS FP8 kernels that don't have GB10 prebuilt coverage. No-op at runtime for the default config; kept for parity with upstream.

- **`patch_triton_backend`** — adds `supports_non_causal()` returning `True` on `TritonAttnBackend`. DFlash's initialization probes the backend for non-causal support; the unpatched class returns `False`, which aborts startup before any model is loaded. Required even though the Triton backend is *also* what vLLM auto-selects for the target verifier (because Gemma-4 has heterogeneous head dimensions — vLLM hard-pins target attention to Triton to avoid mixed-backend numerical divergence).

- **`patch_qwen3_dflash`** — injects `attn_backend=FlashAttentionBackend` into the DFlash draft model's `Attention()` constructor. Pins the draft layer to FlashAttention v2 regardless of vLLM's selector (which would otherwise route the draft to Triton via the same heterogeneous-head-dim policy that locks the target). The DFlash draft is the only place FlashAttention runs in this setup.

The patcher is idempotent — re-running `make patch` against an already-patched install prints `already-patched` for each file and writes nothing.

If you ever run `uv pip install -U vllm` (or any operation that reinstalls the wheel), the patched files are overwritten. Run `make patch` again afterward.

## Run

```bash
make serve
```

The Makefile auto-loads `.env`, so `HF_TOKEN` is exported to the vLLM process. The OpenAI-compatible API listens on `http://0.0.0.0:8000` and is also served under the Anthropic-compatible path expected by Claude Code, so the same server feeds both `make cc` and `make codex`.

### First-run warmup (optional)

After the server reports `Application startup complete`, in another shell:

```bash
make warmup
```

This sends a short and a ~5K-token prompt at the running server to JIT-compile Triton kernels that vLLM's internal startup warmup misses (DFlash spec-decoding kernels, longer-context prefill attention). One-time cost is ~1–2 minutes; the compiled kernels land in `~/.triton/cache`, so subsequent `make serve` boots on the same machine reuse them and don't need this step.

For broader coverage, run:

```bash
make warmup-diag
```

This sweeps every M-bin that `fused_moe_kernel` selects a distinct `(BLOCK_SIZE_M, num_warps, num_stages)` tuple for (≤32, 33–96, 97–128, 129–512, >512 small/medium/large), then runs a ~20K-token prefill and a batch=2 concurrent decode, snapshots `~/.triton/cache` between every step, and prints each new compile by kernel name and config. Use when a real workload still triggers a JIT compile that `make warmup` didn't catch, or to verify that a new vLLM/DFlash build hits the same kernel set as before.

## LMCache (optional KV-cache offload)

Set `LMCACHE=1` to enable [LMCache](https://github.com/LMCache/LMCache), a CPU + local-disk KV-cache offload tier that layers on top of vLLM's GPU prefix cache. Useful here because Claude Code and Codex both repeat long system prompts and conversation prefixes across requests — the GPU prefix cache catches the in-memory tail, and LMCache extends it through host RAM and disk so hits survive eviction and server restarts.

```bash
LMCACHE=1 make serve            # spec-decode path with LMCache
LMCACHE=1 make serve-no-spec    # no-spec path with LMCache
```

The toggle works identically on both serve targets. With `LMCACHE=0` (the default), the existing serve commands are unchanged — the only addition vs. the pre-LMCache recipes is an unconditional `mkdir -p $(LMCACHE_DISK_PATH)` prelude.

### Tunables

| Variable                      | Default               | Meaning                                                             |
| ----------------------------- | --------------------- | ------------------------------------------------------------------- |
| `LMCACHE`                     | `0`                   | Toggle. Anything other than `1` disables LMCache.                   |
| `LMCACHE_LOCAL_CPU`           | `false`               | Enable the in-memory CPU tier (pinned host RAM) above the disk tier. |
| `LMCACHE_MAX_LOCAL_CPU_SIZE`  | `20`                  | GB of host RAM dedicated to the CPU tier (when `LMCACHE_LOCAL_CPU=true`). |
| `LMCACHE_DISK_PATH`           | `<repo>/data/lmcache` | Persistent disk-tier directory. Lives under the gitignored `data/`. |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | `50`                  | GB of disk dedicated to the persistent tier.                        |
| `LMCACHE_USE_LAYERWISE`       | `true`                | Overlap KV transfer with per-layer forward instead of one shot per request. Requires PIECEWISE CUDA graphs. |
| `LMCACHE_ENABLE_ASYNC_LOADING` | `true`               | Prefetch from the disk tier asynchronously instead of blocking per chunk. |
| `LMCACHE_SAVE_DECODE_CACHE`   | `true`                | Cache decode-phase (assistant) tokens in addition to prefill, so prior turns hit on the next request. |
| `LMCACHE_BLOCKING_TIMEOUT_SECS` | `30`                | Max seconds a blocking lookup waits before falling back to recompute. |

The chunk size is fixed at 256 tokens. To change it, edit `LMCACHE_CHUNK_SIZE` inside `LMCACHE_ENV` in the Makefile.

Override at invocation, e.g. `LMCACHE=1 LMCACHE_MAX_LOCAL_CPU_SIZE=40 make serve`.

### Verifying it is live

The disk tier populates `data/lmcache/` as the cache warms — an empty directory after several requests is a sign LMCache did not initialize. Server logs typically also include `LMCache` lines on hit/miss; grep the serve output. After a restart, hits against the same prompts should arrive faster than a fresh boot, since the disk tier persists.

### Caveats

- `make bench` numbers are not directly comparable across `LMCACHE=0` and `LMCACHE=1`. Pick one setting and stick with it for any spec-vs-no-spec comparison.
- A model swap (different `HF_MODEL`) leaves stale entries under `data/lmcache/` that count against the `LMCACHE_MAX_LOCAL_DISK_SIZE` cap. Clear with `rm -rf data/lmcache` between model swaps.

## Use with Claude Code

In a second shell, with the server running:

```bash
make cc
```

This starts Claude Code with `ANTHROPIC_BASE_URL` pointed at the local server and all three model tiers mapped to the served name `vllm-model`. See the [vLLM × Claude Code guide](https://docs.vllm.ai/en/stable/serving/integrations/claude_code/) for details.

## Use with Codex

With [Codex](https://github.com/openai/codex) installed, in a second shell with the server running:

```bash
make codex
```

This launches Codex with `model_provider=vllm` and the OpenAI Responses API wire format pointing at the local vLLM server. Provider settings are passed inline via Codex `-c` flags, so no `~/.codex/config.toml` edits are needed. See the [vLLM × Codex guide](https://docs.vllm.ai/en/latest/serving/integrations/codex/) for details.

## Benchmark

With the server running in another shell, measure live request throughput, time-to-first-token (TTFT), and inter-token latency (ITL) using `vllm bench serve` via:

```bash
make bench           # synthetic random prompts, no download
make bench-sharegpt  # real multi-turn chat workload, downloads ShareGPT V3 on first run
```

Run `make warmup` first — otherwise the first ~10 requests trigger Triton JIT compiles and skew TTFT/ITL.

### `make bench` (random dataset)

Hits `/v1/completions` with synthetic random tokens. Fastest path to a number; no filesystem state.

Tunables (defaults shown):

| Variable                 | Default | Meaning                                                  |
| ------------------------ | ------- | -------------------------------------------------------- |
| `BENCH_NUM_PROMPTS`      | `200`   | Total requests sent during the run.                      |
| `BENCH_REQUEST_RATE`     | `inf`   | Requests per second. `inf` = max throughput, no pacing.  |
| `BENCH_RANDOM_INPUT_LEN` | `1024`  | Synthetic prompt length in tokens.                       |
| `BENCH_RANDOM_OUTPUT_LEN` | `256`   | Synthetic decode length in tokens.                       |

Example: `make bench BENCH_NUM_PROMPTS=500 BENCH_REQUEST_RATE=8`.

### `make bench-sharegpt` (real chat workload)

Hits `/v1/chat/completions` with ShareGPT V3 multi-turn conversations — exercises chat templating overhead on the same model Claude Code and Codex use. Use this when the number you want is "what will real agent traffic look like".

The dataset (~620 MB JSON) is downloaded on first invocation to `data/ShareGPT_V3_unfiltered_cleaned_split.json` and cached. `data/` is gitignored. Override `SHAREGPT_PATH` to use an existing copy, or `SHAREGPT_URL` to pull from a different mirror.

`BENCH_NUM_PROMPTS` and `BENCH_REQUEST_RATE` apply here too.

### Compare with vs without speculative decoding

To quantify the DFlash speedup on this stack, run `make bench` (or `make bench-sharegpt`) twice — once with `make serve` running, once with `make serve-no-spec` running. The delta in `Output token throughput` and `Mean ITL` reported by `vllm bench serve` is the speedup attributable to DFlash speculative decoding.

## Configuration

| Setting                     | Value                              |
| --------------------------- | ---------------------------------- |
| Served model name           | `vllm-model`                       |
| Speculative method          | `dflash`                           |
| Speculative tokens per step | 8                                  |
| Max model length            | 32768                              |
| Max batched tokens          | 32768                              |
| GPU memory utilization      | 0.80                               |
| Multimodal                  | disabled (text-only)               |
| Tool calling                | auto                               |
| Tool-call parser            | `gemma4`                           |
| Reasoning parser            | `gemma4`                           |
| `trust_remote_code`         | enabled                            |

Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, `make codex VLLM_BASE_URL=http://remote:8000`, `make bench BENCH_NUM_PROMPTS=500`, or `LMCACHE=1 make serve`.

## License

[MIT](LICENSE) © 2026 ChihYu Yeh

Vendored components retain their original licenses; see `NOTICE`.
