# LMCache opt-in KV-cache offload — design

**Date:** 2026-05-10
**Status:** approved (pending spec review)

## Goal

Add an opt-in path — `LMCACHE=1 make serve` (and `LMCACHE=1 make serve-no-spec`) — that launches the local vLLM server with [LMCache](https://github.com/LMCache/LMCache) enabled for CPU + local-disk KV-cache offload. Default `make serve` is unchanged. After this lands, an operator can flip the env var, get a prefix cache that survives GPU eviction (and server restarts, via the disk tier), and verify hits in the server log.

## Motivation

Claude Code and Codex both repeat long system prompts and conversation prefixes across requests. vLLM's GPU prefix cache catches the in-memory tail of that, but is bounded by GPU memory and does not survive eviction or restart. LMCache extends the prefix cache into host RAM and disk, which is the right tier for this single-host, single-user setup. The pinned vLLM (`0.20.2rc1.dev140`) already ships the connector glue under `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`; only the `lmcache` Python package and a few flags are missing.

## Scope

**In scope**

- Pin and install `lmcache` as part of `make install` (single-line addition; not opt-in at install time).
- New Makefile env var `LMCACHE` and three tunable knobs (`LMCACHE_MAX_LOCAL_CPU_SIZE`, `LMCACHE_DISK_PATH`, `LMCACHE_MAX_LOCAL_DISK_SIZE`).
- Two reusable Make fragments (`LMCACHE_ENV`, `LMCACHE_FLAGS`) that expand to the env vars and the `--kv-transfer-config` arg when `LMCACHE=1` and to nothing otherwise.
- `mkdir -p $(LMCACHE_DISK_PATH)` prelude on both serve recipes (no-op when LMCache is off).
- New "## LMCache (optional KV-cache offload)" section in README between **Run** and **Use with Claude Code**.

**Out of scope**

- `LMCacheMPConnector` (multi-process). Only useful with worker-per-GPU sharding; this is a single-instance setup.
- Remote backends (Redis, NIXL P2P, S3). No multi-host or multi-tenant requirement.
- A separate `make serve-lmcache` target. The user asked for an env-var toggle on the existing serve targets.
- A separate `make install-lmcache` target. The user asked for `lmcache` in the base install.
- Auto-priming the LMCache during `make warmup`. The existing warmup already pushes a long prompt, which seeds LMCache as a side-effect when `LMCACHE=1` is set on the serve.
- Adjusting `make bench` for LMCache. The two settings are not directly comparable; documented as a caveat in the README.

## Install change

A second `uv pip install` line gets appended to the `install` target, gated by a new `LMCACHE_PIN` variable near the existing pins:

```makefile
LMCACHE_PIN ?= 0.4.4
```

```makefile
install:
	uv pip install -U --pre --torch-backend=auto \
		--extra-index-url $(VLLM_INDEX_URL) \
		"vllm==$(VLLM_PIN)"
	uv pip install "lmcache==$(LMCACHE_PIN)"
	$(MAKE) patch
```

`lmcache==0.4.4` is the version that resolves on aarch64 today via uv's standard index. Pinning matters because the install pulls `cupy-cuda12x`, `nixl-cu12`, and downgrades `numpy 2.3.5 → 2.2.6` plus several `opentelemetry-*` packages relative to vLLM's resolved set. Pinning lets that downgrade surface stay reproducible across re-installs of the venv.

## Serve change

### Variables

Added to the existing variable block at the top of the Makefile:

| Variable | Default | Notes |
|---|---|---|
| `LMCACHE` | `0` | Toggle. Anything other than `1` leaves serve behavior unchanged. |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | `20` | GB of host RAM dedicated to the CPU tier. |
| `LMCACHE_DISK_PATH` | `$(CURDIR)/data/lmcache` | Local-disk tier directory. Lives under the already-gitignored `data/`. |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | `50` | GB of disk dedicated to the persistent tier. |
| `LMCACHE_PIN` | `0.4.4` | Pinned `lmcache` wheel version (used by `install`). |

All overridable from the command line, e.g. `LMCACHE=1 LMCACHE_MAX_LOCAL_CPU_SIZE=40 make serve`.

### Fragments

Two reusable fragments, defined once near the variables and consumed by both serve recipes:

```makefile
LMCACHE_ENV = $(if $(filter 1,$(LMCACHE)), \
	LMCACHE_LOCAL_CPU=true \
	LMCACHE_MAX_LOCAL_CPU_SIZE=$(LMCACHE_MAX_LOCAL_CPU_SIZE) \
	LMCACHE_LOCAL_DISK=file://$(LMCACHE_DISK_PATH) \
	LMCACHE_MAX_LOCAL_DISK_SIZE=$(LMCACHE_MAX_LOCAL_DISK_SIZE) \
	LMCACHE_CHUNK_SIZE=256,)

LMCACHE_FLAGS = $(if $(filter 1,$(LMCACHE)), \
	--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',)
```

When `LMCACHE != 1`, both expand to empty strings — the serve recipes behave exactly as today.

`kv_role: "kv_both"` makes the connector both produce and consume KV blocks, which is the correct shape for single-instance prefix-cache offload (vs. the disaggregated prefill / decode pair used in multi-instance LMCache deployments).

`LMCACHE_CHUNK_SIZE=256` matches the LMCache default; pinned here so it does not silently change with future `lmcache` versions.

### Serve targets

Both `serve` and `serve-no-spec` get the same two changes:

1. Prepend `mkdir -p $(LMCACHE_DISK_PATH)` to the recipe (cheap and unconditional — directory creation is a no-op when LMCache is off, and it never costs more than a stat call).
2. Prepend `$(LMCACHE_ENV)` to the existing `PATH=...` env block, and append `$(LMCACHE_FLAGS)` to the trailing arg list.

Sketch for `serve` (analogous edit for `serve-no-spec`):

```makefile
serve:
	mkdir -p $(LMCACHE_DISK_PATH)
	$(LMCACHE_ENV) \
	PATH=$(CURDIR)/.venv/bin:$$PATH \
	$(VLLM) serve $(HF_MODEL) \
		--served-model-name $(SERVED_MODEL_NAME) \
		--enable-auto-tool-choice \
		--tool-call-parser $(TOOL_CALL_PARSER) \
		--reasoning-parser $(REASONING_PARSER) \
		--speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 8}' \
		--max-model-len 32768 \
		--max-num-batched-tokens 32768 \
		--gpu-memory-utilization 0.80 \
		--limit-mm-per-prompt '{"image":0,"video":0}' \
		--trust-remote-code \
		$(LMCACHE_FLAGS)
```

### Bookkeeping

- `data/` is already gitignored (added by the bench targets work), so `data/lmcache/` does not need a separate entry.
- `make help` gains one line near the existing `serve` description noting the `LMCACHE=1` opt-in.
- `.PHONY` does not change.

## README

A new "## LMCache (optional KV-cache offload)" section is added between **Run** and **Use with Claude Code**. Structure:

1. One-paragraph intro: what LMCache does, why it helps Claude Code / Codex specifically (long repeated system prompts), and the one-line opt-in: `LMCACHE=1 make serve`.
2. Tunables table (the four `LMCACHE_*` runtime vars from the Variables section above).
3. How to verify it is working: server logs include `LMCache` lines on hit/miss, and the configured disk path will populate under `data/lmcache/` as the cache warms.
4. One short caveat: `make bench` results from `LMCACHE=1` and `LMCACHE=0` are not directly comparable; pick one and stick with it for any spec-vs-no-spec comparison.

The existing "Override any of these on the command line, e.g. ..." example list under **Configuration** gains one LMCache example.

## Verification

After implementation:

1. `make help` mentions the `LMCACHE=1` toggle.
2. `make install` from a clean venv installs `lmcache==0.4.4` alongside the pinned vLLM, then `make patch` runs and prints `already-patched` for each file on a re-run.
3. `make serve` (no env var) boots and serves identically to today — `vllm serve` log shows no LMCache lines, no extra env vars in the process.
4. `LMCACHE=1 make serve` boots, the server log includes an "Initializing latest dev LMCache connector" line (or equivalent), and `data/lmcache/` exists after startup.
5. With `LMCACHE=1 make serve` running, `make warmup` completes; a second identical prompt issued via `make chat` shows a TTFT drop on the second submission and an LMCache hit line in the server log.
6. Killing and restarting the server with `LMCACHE=1 make serve` retains the disk tier across restarts (the ~5K-token warmup prompt is still a hit).
7. `git status` after the verification run shows no new tracked files under `data/`.

## Risks

- **Dep churn against vLLM nightly.** Installing `lmcache` downgrades numpy and several opentelemetry packages relative to vLLM's preferred resolution. Mitigation: pin `lmcache==0.4.4`, and add a verification step (#3 above) that `make serve` still boots cleanly after the install. If a future vLLM nightly bump conflicts, bump `LMCACHE_PIN` in lockstep.
- **`LMCACHE_LOCAL_DISK=file://…` URL format unverified for this LMCache version.** This is the documented convention for upstream LMCache, but the bundled adapter delegates env-var parsing to the `lmcache` package itself, which is not yet installed. Verification step #4 catches this on first boot. Fallback: write a `lmcache.yaml` and point `LMCACHE_CONFIG_FILE` at it instead — same Makefile shape, different env var.
- **CPU memory pressure.** 20 GB cap is a default, not a measured value. DGX Spark unified memory is shared with the GPU pool (set to 0.80 utilization in `serve`). If host paging starts during long bench runs, drop `LMCACHE_MAX_LOCAL_CPU_SIZE` — the env var is the override.
- **Disk tier persistence is also a foot-gun.** A model swap (different HF_MODEL) without clearing `data/lmcache/` will accumulate dead entries that LMCache's keying should ignore but that still consume the 50 GB cap. Documented in README; mitigation is `rm -rf data/lmcache` between model swaps.
- **`patch` step is unaffected.** The DGX Spark patches operate on vLLM source files only; LMCache code paths are independent and do not need patching for GB10 / SM_121.
