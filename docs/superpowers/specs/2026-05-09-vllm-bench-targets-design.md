# vLLM benchmark Makefile targets — design

**Date:** 2026-05-09
**Status:** approved (pending spec review)

## Goal

Add Makefile targets and README documentation for benchmarking the running vLLM server with `vllm bench serve`, so that the speedup of the DGX Spark + Gemma-4 26B + DFlash setup over plain autoregressive decode is measurable from the same checkout that runs the server. After this lands, a user with `make serve` (or `make serve-no-spec`) running in one shell can run `make bench` or `make bench-sharegpt` in another and get throughput / TTFT / ITL numbers for the live server.

## Motivation

The repo currently exposes serve, warmup, chat, cc, and codex targets, but no way to quantify the DFlash speedup. The project README claims "up to ~3.7x speedup" but provides no in-repo path to reproduce a measurement. Reviewers and operators reaching for those numbers should not have to assemble the bench command line themselves.

## Scope

**In scope**

- Two new Makefile targets — `bench` (synthetic, no download) and `bench-sharegpt` (real chat workload, downloads ShareGPT V3 once into `data/`).
- Six new overridable Makefile variables for tuning the bench load.
- A file-target rule to download the ShareGPT JSON on first use.
- `data/` added to `.gitignore`.
- New "Benchmark" section in README documenting both targets, when to use each, and how to compare spec vs no-spec.

**Out of scope**

- `vllm bench throughput` (offline mode). Loads its own model copy and would conflict with the live serve's GPU memory budget; not useful in this single-GPU setup.
- `vllm bench mm-processor`. The serve config disables image/video.
- Automated comparison harness across spec / no-spec runs. Two manual invocations with different `make serve*` targets are clearer than a wrapper that has to manage two server lifecycles.
- Result persistence / plotting. `vllm bench serve` already supports `--save-result` and `--plot-timeline`; users who want them can pass them through a future `BENCH_EXTRA_ARGS` extension. Not added now to keep the surface small.

## Targets

### `make bench` — synthetic smoke test

Random-token dataset, hits `/v1/completions` with `--backend vllm`. No filesystem state, no download. Fastest path to a number.

```makefile
bench:
	$(VLLM) bench serve \
		--backend vllm \
		--model $(SERVED_MODEL_NAME) \
		--base-url $(VLLM_BASE_URL) \
		--endpoint /v1/completions \
		--dataset-name random \
		--random-input-len $(BENCH_RANDOM_INPUT_LEN) \
		--random-output-len $(BENCH_RANDOM_OUTPUT_LEN) \
		--num-prompts $(BENCH_NUM_PROMPTS) \
		--request-rate $(BENCH_REQUEST_RATE)
```

### `make bench-sharegpt` — real chat workload

ShareGPT V3 multi-turn conversations, hits `/v1/chat/completions` with `--backend openai-chat`. This is the same code path Claude Code and Codex use, so numbers from this target reflect deployed behavior including chat templating overhead.

The ShareGPT JSON is fetched on first invocation via a file-target rule and cached in `./data/`.

```makefile
$(SHAREGPT_PATH):
	mkdir -p $(dir $(SHAREGPT_PATH))
	curl -L --fail -o $@ $(SHAREGPT_URL)

bench-sharegpt: $(SHAREGPT_PATH)
	$(VLLM) bench serve \
		--backend openai-chat \
		--model $(SERVED_MODEL_NAME) \
		--base-url $(VLLM_BASE_URL) \
		--endpoint /v1/chat/completions \
		--dataset-name sharegpt \
		--dataset-path $(SHAREGPT_PATH) \
		--num-prompts $(BENCH_NUM_PROMPTS) \
		--request-rate $(BENCH_REQUEST_RATE)
```

### Variables

Added near the existing variable block at the top of the Makefile:

| Variable | Default | Notes |
|---|---|---|
| `BENCH_NUM_PROMPTS` | `200` | Total requests sent during the run. |
| `BENCH_REQUEST_RATE` | `inf` | Requests per second. `inf` = max throughput, no pacing. |
| `BENCH_RANDOM_INPUT_LEN` | `1024` | Synthetic prompt length, tokens. Used by `bench` only. |
| `BENCH_RANDOM_OUTPUT_LEN` | `256` | Synthetic decode length, tokens. Used by `bench` only. |
| `SHAREGPT_PATH` | `data/ShareGPT_V3_unfiltered_cleaned_split.json` | Cached dataset location. |
| `SHAREGPT_URL` | `https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` | Source of the cached download. |

All overridable from the command line, e.g. `make bench BENCH_NUM_PROMPTS=500 BENCH_REQUEST_RATE=8`.

### Bookkeeping

- `.PHONY` line at the top of the Makefile gains `bench bench-sharegpt`. The file-target rule for `$(SHAREGPT_PATH)` is intentionally NOT phony — that's how make caches the download.
- `make help` gains two lines describing the new targets.
- `.gitignore` gains a `data/` entry so the ~620 MB ShareGPT JSON never lands in git.

## README

A new "Benchmark" subsection lands between the existing "Use with Codex" and "Configuration" sections. Structure:

1. One-paragraph intro: what the targets do, that they require `make serve` (or `make serve-no-spec`) running in another shell.
2. Two short subsections, one per target — exact command, what it measures, when to use it, the override variables that apply.
3. A "Compare spec vs no-spec" paragraph: run `make bench` once with `make serve` running, then again with `make serve-no-spec` running — the difference in `Output token throughput` and `Mean ITL` is the DFlash speedup. No new code for this; it's just instructions.
4. One sentence noting that the ShareGPT download is ~620 MB and lands in `data/` (gitignored), pulled on first `make bench-sharegpt`.

The existing "Override any of these on the command line, e.g. ..." example list under "Configuration" gains one bench example.

## Verification

After implementation:

1. `make help` lists `bench` and `bench-sharegpt`.
2. With `make serve` running in another shell, `make bench BENCH_NUM_PROMPTS=20 BENCH_REQUEST_RATE=inf` completes and prints a `vllm bench serve` summary table including request throughput and TTFT.
3. `make bench-sharegpt BENCH_NUM_PROMPTS=20` on a fresh checkout downloads the JSON to `data/` (one-time), then completes and prints the summary table.
4. A second `make bench-sharegpt` does NOT re-download (file-target rule satisfied).
5. `git status` after step 3 shows no `data/` entries (gitignored).
6. README renders cleanly; the new section sits between "Use with Codex" and "Configuration".

## Risks

- **ShareGPT URL drift.** The HF resolve link points at an unaffiliated user's dataset mirror. If it 404s, `make bench-sharegpt` fails at the curl step. Mitigation: `SHAREGPT_URL` is overridable, so a user with their own copy can point at it; the failure mode is loud (`curl --fail`) rather than silent.
- **First-run latency on `bench`.** If the user runs `make bench` before `make warmup`, the first ~10 requests will trigger Triton JITs and skew TTFT/ITL numbers. Mitigation: README's "Benchmark" intro paragraph notes that `make warmup` should be run first, same as for normal serving traffic.
- **`--request-rate inf` saturates the server.** With default `BENCH_NUM_PROMPTS=200` and `inf` rate, all 200 requests queue up at once. That's the right default for measuring max throughput, but a user who wants to measure latency-under-load should pass a finite rate. Documented in the README.
