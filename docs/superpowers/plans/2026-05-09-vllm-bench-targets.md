# vLLM Benchmark Makefile Targets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `make bench` (synthetic) and `make bench-sharegpt` (real chat workload) so users can measure throughput / TTFT / ITL of the running vLLM server, and document both in the README.

**Architecture:** Two new Makefile targets that shell out to `vllm bench serve` against the running server. `bench` uses the synthetic `random` dataset (no setup); `bench-sharegpt` depends on a file-target rule that lazily downloads ShareGPT V3 to `data/` (gitignored). README gets a new "Benchmark" section between "Use with Codex" and "Configuration".

**Tech Stack:** GNU make, vLLM CLI (`vllm bench serve`), curl, ShareGPT V3 dataset.

**Spec:** `docs/superpowers/specs/2026-05-09-vllm-bench-targets-design.md`

---

## File Structure

| File | Change |
|---|---|
| `.gitignore` | Add `data/` line so the cached ShareGPT JSON never enters git. |
| `Makefile` | Add 6 variables, 2 targets, 1 file-target rule. Update `.PHONY` and `help`. |
| `README.md` | Add "Benchmark" section. Add one example to the "Override any of these on the command line, e.g. ..." bullet under "Configuration". |

The Makefile holds all behavior; README holds all documentation; gitignore is one line. No new scripts, no Python.

---

## Verification Notes (read once before starting)

This repo has no test framework — it's an ops/serving project. Verification is two-tiered:

1. **Static (works anywhere)** — uses `make -n <target>` (dry run, prints commands without executing) and `make help` (lists targets). These confirm the Makefile is wired correctly without needing a running server.
2. **Live (DGX Spark only)** — actually runs `vllm bench serve` against `make serve`. Documented as a manual step at the end; the engineer doing the implementation may not have access to the hardware, so live verification is not gating per-task.

When a step says "verify", it means run the static check unless explicitly noted as live.

---

## Task 1: Add `data/` to `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add the entry**

Append a new section to `.gitignore`:

```
# Benchmark dataset cache (ShareGPT V3 JSON, ~620 MB)
data/
```

The existing `.gitignore` ends after `.env`. Add a blank line, then the comment + `data/` line.

- [ ] **Step 2: Verify**

Run: `cat .gitignore`
Expected: includes a `data/` line under a `# Benchmark dataset cache ...` comment.

Run: `mkdir -p data && touch data/test.json && git status --short data/`
Expected: empty output (file is gitignored).
Then: `rm data/test.json && rmdir data` (cleanup).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "Ignore data/ dataset cache directory"
```

---

## Task 2: Add `make bench` target (synthetic, no download)

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the four bench-load variables to the variable block**

The existing variable block in `Makefile` ends with `VLLM_INDEX_URL ?= https://wheels.vllm.ai/nightly`. Append these four lines immediately after that line (before the blank line preceding the `help:` target):

```makefile
BENCH_NUM_PROMPTS ?= 200
BENCH_REQUEST_RATE ?= inf
BENCH_RANDOM_INPUT_LEN ?= 1024
BENCH_RANDOM_OUTPUT_LEN ?= 256
```

- [ ] **Step 2: Add `bench` to the `.PHONY` line**

Locate the first line of the Makefile:

```makefile
.PHONY: help venv install patch serve serve-no-spec warmup warmup-diag chat cc codex
```

Replace it with:

```makefile
.PHONY: help venv install patch serve serve-no-spec warmup warmup-diag chat cc codex bench bench-sharegpt
```

(Both `bench` and `bench-sharegpt` are added in this single edit, even though `bench-sharegpt` arrives in Task 3 — it's harmless to declare a phony target before it exists, and it avoids a second edit to the same line.)

- [ ] **Step 3: Add the `bench` help line**

Inside the `help:` target, the existing block ends with:

```makefile
	@echo "  codex        Launch Codex routed to the local vLLM server"
```

Add immediately after that line (before the blank line preceding `venv:`):

```makefile
	@echo "  bench        Benchmark the running server with vllm bench serve (random dataset)"
```

- [ ] **Step 4: Add the `bench` target at the end of the file**

Append to the bottom of `Makefile` (after the final `codex:` block):

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

Note: indentation is a real TAB, not spaces — make is whitespace-sensitive.

- [ ] **Step 5: Verify (static)**

Run: `make help | grep bench`
Expected: line `  bench        Benchmark the running server with vllm bench serve (random dataset)`

Run: `make -n bench`
Expected: prints a single command starting with `.venv/bin/vllm bench serve --backend vllm --model vllm-model --base-url http://localhost:8000 --endpoint /v1/completions --dataset-name random --random-input-len 1024 --random-output-len 256 --num-prompts 200 --request-rate inf` (line-wrapped — confirm all flags are present).

Run: `make -n bench BENCH_NUM_PROMPTS=50 BENCH_REQUEST_RATE=4`
Expected: same command but with `--num-prompts 50 --request-rate 4`. This confirms the variable overrides work.

- [ ] **Step 6: Commit**

```bash
git add Makefile
git commit -m "Add make bench target for synthetic vllm bench serve runs"
```

---

## Task 3: Add `make bench-sharegpt` target (downloads ShareGPT V3 on first use)

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the two ShareGPT variables to the variable block**

Append these two lines immediately after the `BENCH_RANDOM_OUTPUT_LEN ?= 256` line added in Task 2:

```makefile
SHAREGPT_PATH ?= data/ShareGPT_V3_unfiltered_cleaned_split.json
SHAREGPT_URL ?= https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

- [ ] **Step 2: Add the `bench-sharegpt` help line**

Inside the `help:` target, immediately after the `bench` help line added in Task 2, add:

```makefile
	@echo "  bench-sharegpt Benchmark the running server with vllm bench serve (ShareGPT V3, downloaded on first run)"
```

- [ ] **Step 3: Add the file-target rule and the `bench-sharegpt` target at the end of the file**

Append to the bottom of `Makefile` (after the `bench:` block from Task 2):

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

Note: TAB indentation for both rules. The `$(SHAREGPT_PATH)` rule is intentionally NOT in the `.PHONY` list — it's a real file target so make caches the download.

- [ ] **Step 4: Verify (static)**

Run: `make help | grep bench-sharegpt`
Expected: line `  bench-sharegpt Benchmark the running server with vllm bench serve (ShareGPT V3, downloaded on first run)`

Run: `make -n bench-sharegpt`
Expected: two commands printed in order — first `mkdir -p data/` and `curl -L --fail -o data/ShareGPT_V3_unfiltered_cleaned_split.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json`, then `.venv/bin/vllm bench serve --backend openai-chat --model vllm-model --base-url http://localhost:8000 --endpoint /v1/chat/completions --dataset-name sharegpt --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 200 --request-rate inf`.

Run: `mkdir -p data && touch data/ShareGPT_V3_unfiltered_cleaned_split.json && make -n bench-sharegpt`
Expected: ONLY the `vllm bench serve ...` command. The mkdir + curl lines must NOT appear (the file-target rule is satisfied by the existing file). This confirms the caching behavior.
Then: `rm data/ShareGPT_V3_unfiltered_cleaned_split.json && rmdir data` (cleanup).

- [ ] **Step 5: Commit**

```bash
git add Makefile
git commit -m "Add make bench-sharegpt target with on-demand dataset download"
```

---

## Task 4: Add "Benchmark" section to README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the Benchmark section**

The existing README has the structure: "Use with Codex" → "Configuration" → "License". Insert a new "## Benchmark" section between "Use with Codex" and "Configuration".

Locate the line in `README.md` that reads `## Configuration`.

Immediately before that line (with a blank line separator), insert the following Markdown literally. The block uses fenced code blocks internally — keep the inner triple-backtick fences exactly as shown:

````markdown
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
| `BENCH_RANDOM_OUTPUT_LEN`| `256`   | Synthetic decode length in tokens.                       |

Example: `make bench BENCH_NUM_PROMPTS=500 BENCH_REQUEST_RATE=8`.

### `make bench-sharegpt` (real chat workload)

Hits `/v1/chat/completions` with ShareGPT V3 multi-turn conversations — the same code path Claude Code and Codex use, including chat templating overhead. Use this when the number you want is "what will real agent traffic look like".

The dataset (~620 MB JSON) is downloaded on first invocation to `data/ShareGPT_V3_unfiltered_cleaned_split.json` and cached. `data/` is gitignored. Override `SHAREGPT_PATH` to use an existing copy, or `SHAREGPT_URL` to pull from a different mirror.

`BENCH_NUM_PROMPTS` and `BENCH_REQUEST_RATE` apply here too.

### Compare with vs without speculative decoding

To quantify the DFlash speedup on this stack, run `make bench` (or `make bench-sharegpt`) twice — once with `make serve` running, once with `make serve-no-spec` running. The delta in `Output token throughput` and `Mean ITL` reported by `vllm bench serve` is the speedup attributable to DFlash speculative decoding.
````

- [ ] **Step 2: Add a bench example to the Configuration override line**

Locate the paragraph at the end of the existing "Configuration" section that reads:

```
Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, or `make codex VLLM_BASE_URL=http://remote:8000`.
```

Replace it with:

```
Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, `make codex VLLM_BASE_URL=http://remote:8000`, or `make bench BENCH_NUM_PROMPTS=500`.
```

- [ ] **Step 3: Verify (visual)**

Run: `grep -n "^## " README.md`
Expected: section headers in order — Prerequisites, Setup, Patches, Run (or its current heading), Use with Claude Code, Use with Codex, **Benchmark** (new), Configuration, License.

Run: `grep -c "make bench" README.md`
Expected: a count of at least 6 (header bullets, subsections, configuration example, compare paragraph).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document make bench / bench-sharegpt in README"
```

---

## Final Live Verification (manual, on DGX Spark)

These steps require the actual hardware and a built `.venv`. Run them by hand once after merging; they are not gating per-task.

1. With `make serve` running in another shell, run:
   ```
   make bench BENCH_NUM_PROMPTS=20 BENCH_REQUEST_RATE=inf
   ```
   Expected: completes in well under a minute, prints a `vllm bench serve` summary table with `Successful requests: 20`, plus `Output token throughput`, `Mean TTFT`, `Mean ITL` rows.

2. From a fresh checkout state (no `data/` directory), run:
   ```
   make bench-sharegpt BENCH_NUM_PROMPTS=20
   ```
   Expected: first invocation downloads `data/ShareGPT_V3_unfiltered_cleaned_split.json` (~620 MB), then runs the bench. Summary table reports `Successful requests: 20`.

3. Run the same `make bench-sharegpt BENCH_NUM_PROMPTS=20` again.
   Expected: NO download this time — the file-target rule is satisfied. The bench runs immediately.

4. Run `git status`.
   Expected: `data/` does not appear in the listing (gitignored).

5. Stop `make serve`. Start `make serve-no-spec`. Re-run `make bench BENCH_NUM_PROMPTS=20`.
   Expected: completes; `Output token throughput` is meaningfully lower than the spec-decode run from step 1, demonstrating the DFlash speedup is measurable end-to-end.
