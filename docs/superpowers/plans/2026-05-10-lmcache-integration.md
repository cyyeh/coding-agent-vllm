# LMCache Opt-in KV-Cache Offload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `LMCACHE=1 make serve` (and `LMCACHE=1 make serve-no-spec`) path that launches vLLM with LMCache enabled for CPU + local-disk prefix-cache offload. Default `make serve` is unchanged. Base `make install` adds the pinned `lmcache` package.

**Architecture:** Two reusable Make fragments (`LMCACHE_ENV`, `LMCACHE_FLAGS`) expand to LMCache env vars and a `--kv-transfer-config` arg when `LMCACHE=1` and to empty strings otherwise. Both serve recipes consume the fragments uniformly. `make install` gains a single `uv pip install "lmcache==$(LMCACHE_PIN)"` line. README gets a new "LMCache (optional KV-cache offload)" section between **Run** and **Use with Claude Code**.

**Tech Stack:** GNU make, vLLM CLI (`vllm serve` with `--kv-transfer-config`), uv (pip install), [LMCache](https://github.com/LMCache/LMCache) (`LMCacheConnectorV1` shipped with vLLM).

**Spec:** `docs/superpowers/specs/2026-05-10-lmcache-integration-design.md`

---

## File Structure

| File | Change |
|---|---|
| `Makefile` | Add 5 variables, 2 conditional fragments, 1 install line. Edit both serve recipes. Edit `help:` block. |
| `README.md` | Add "LMCache (optional KV-cache offload)" section between **Run** and **Use with Claude Code**. Add one example to the Configuration override line. |

No new files, no new scripts, no new Python. All behavior lives in the Makefile; all documentation lives in the README.

---

## Verification Notes (read once before starting)

This repo has no test framework — it is an ops/serving project, identical to the prior bench-targets work. Verification is two-tiered:

1. **Static (works anywhere)** — `make -n <target>` (dry run, prints commands without executing) and `make help`. These confirm the Makefile is wired correctly without a running server, a built venv, or DGX Spark hardware.
2. **Live (DGX Spark only)** — actually runs `vllm serve` with LMCache. Documented as a manual step at the end; not gating per-task.

When a step says "verify", it means run the static check unless explicitly noted as live.

A note on `make -n` and the `$(if ...)` fragments: GNU make collapses unused branches of `$(if ...)` to empty strings, so `make -n serve` (with `LMCACHE` unset or `LMCACHE=0`) should print a command that is byte-for-byte identical to today's `make -n serve`, modulo the leading `mkdir -p ...` line. This identity is the regression check across Tasks 3 and 4.

---

## Task 1: Pin and install `lmcache` in `make install`

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the `LMCACHE_PIN` variable**

The existing variable block in `Makefile` ends with `SHAREGPT_URL ?= https://huggingface.co/...` (the line added by the prior bench-targets work). Locate the line:

```makefile
VLLM_INDEX_URL ?= https://wheels.vllm.ai/nightly
```

Add immediately after it (before the `BENCH_NUM_PROMPTS` line that follows):

```makefile
LMCACHE_PIN ?= 0.4.4
```

Placing `LMCACHE_PIN` next to `VLLM_PIN` and `VLLM_INDEX_URL` keeps the install-time pins grouped together and separates them from the runtime tunables that arrive in Task 2.

- [ ] **Step 2: Add the `lmcache` install line**

Locate the `install:` recipe (currently 4 lines):

```makefile
install:
	uv pip install -U --pre --torch-backend=auto \
		--extra-index-url $(VLLM_INDEX_URL) \
		"vllm==$(VLLM_PIN)"
	$(MAKE) patch
```

Replace it with:

```makefile
install:
	uv pip install -U --pre --torch-backend=auto \
		--extra-index-url $(VLLM_INDEX_URL) \
		"vllm==$(VLLM_PIN)"
	uv pip install "lmcache==$(LMCACHE_PIN)"
	$(MAKE) patch
```

The `lmcache` install lands between the vLLM install and the `patch` step. `lmcache` is published on the standard PyPI index, so it does NOT need `--extra-index-url $(VLLM_INDEX_URL)` and does NOT need `--pre`.

Indentation is a real TAB, not spaces. Re-check after pasting.

- [ ] **Step 3: Verify (static)**

Run: `make -n install`
Expected: prints three commands in order — the existing `uv pip install -U --pre --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly vllm==0.20.2rc1.dev140` (line-wrapped, no quoting issues), then `uv pip install lmcache==0.4.4`, then `make patch`.

Run: `make -n install LMCACHE_PIN=0.5.0`
Expected: same as above except the second line reads `uv pip install lmcache==0.5.0`. This confirms the variable override.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "Pin and install lmcache as part of make install"
```

---

## Task 2: Add LMCache runtime variables and conditional fragments

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the four runtime variables to the variable block**

The existing variable block (after Task 1) ends with the `SHAREGPT_URL ?= ...` line. Append immediately after it (before the blank line preceding `help:`):

```makefile
LMCACHE ?= 0
LMCACHE_MAX_LOCAL_CPU_SIZE ?= 20
LMCACHE_DISK_PATH ?= $(CURDIR)/data/lmcache
LMCACHE_MAX_LOCAL_DISK_SIZE ?= 50
```

These are the runtime knobs (consumed by the serve recipes), as opposed to `LMCACHE_PIN` which is the install-time pin.

- [ ] **Step 2: Add a `comma` constant and the two conditional fragments**

Append immediately after the four variables from Step 1:

```makefile
comma := ,
LMCACHE_ENV = $(if $(filter 1,$(LMCACHE)),\
	LMCACHE_LOCAL_CPU=true \
	LMCACHE_MAX_LOCAL_CPU_SIZE=$(LMCACHE_MAX_LOCAL_CPU_SIZE) \
	LMCACHE_LOCAL_DISK=file://$(LMCACHE_DISK_PATH) \
	LMCACHE_MAX_LOCAL_DISK_SIZE=$(LMCACHE_MAX_LOCAL_DISK_SIZE) \
	LMCACHE_CHUNK_SIZE=256,)

LMCACHE_FLAGS = $(if $(filter 1,$(LMCACHE)),--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1"$(comma)"kv_role":"kv_both"}',)
```

Notes:
- These are **recursive (`=`) variables**, not simply-expanded (`:=`). The `$(if ...)` must be re-evaluated each time the fragment is consumed so a per-invocation `LMCACHE=1` override takes effect.
- Backslash-newline continues the `LMCACHE_ENV` definition across lines. The trailing `,)` closes the `$(if ...)` with an empty else-branch.
- `comma := ,` is the standard GNU make idiom for embedding a literal comma inside a function-call argument. The `$(if cond,then,else)` function splits its arguments on commas before any shell quoting is considered, so the comma between `"LMCacheConnectorV1"` and `"kv_role"` in the JSON would otherwise be parsed as the then/else delimiter and silently truncate the value. The `$(comma)` substitution suppresses that.
- Continuation lines of `LMCACHE_ENV` use TAB indentation in the snippet above. Make joins backslash-continued lines and collapses runs of leading whitespace, so the indentation is purely for readability — TAB matches the surrounding Makefile style and is what the project uses elsewhere.

- [ ] **Step 3: Verify (static)**

Run: `make -n install`
Expected: identical to the output from Task 1 Step 3. Adding the new variables and fragments must NOT change any existing target behavior. This is the key regression check.

Run: `make -p 2>/dev/null | grep -E '^LMCACHE'`
Expected: shows `LMCACHE = 0`, `LMCACHE_MAX_LOCAL_CPU_SIZE = 20`, `LMCACHE_DISK_PATH = <cwd>/data/lmcache`, `LMCACHE_MAX_LOCAL_DISK_SIZE = 50`, `LMCACHE_PIN = 0.4.4`, `LMCACHE_ENV = $(if ...)`, `LMCACHE_FLAGS = $(if ...)`. The exact `=` vs `?=` and the un-expanded `$(if ...)` strings confirm the variables are loaded with the right flavors.

Run the following test to verify the comma fix in Step 2 is in place — i.e., `LMCACHE_FLAGS` is not being truncated by `$(if)` argument splitting. This guards against a regression if someone removes `comma := ,` or the `$(comma)` substitution:

```bash
make -p 2>/dev/null | grep '^LMCACHE_FLAGS' | head -1
```

Expected output (one line):

```
LMCACHE_FLAGS = $(if $(filter 1,$(LMCACHE)),--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1"$(comma)"kv_role":"kv_both"}',)
```

The `$(comma)` substitution must be visible in the un-expanded value. If you see a literal `,` between `"LMCacheConnectorV1"` and `"kv_role"` instead, the comma fix has been undone.

For an end-to-end check that the expanded value is also intact, drop a tiny include-style file that uses `$(info)` (which writes to make's own stderr, not via the shell, so it preserves inner double-quotes):

```bash
cat > /tmp/lm_show.mk << 'EOF'
include Makefile
$(info LMCACHE_FLAGS=[$(LMCACHE_FLAGS)])
show: ; @true
EOF
echo "--- LMCACHE=0 ---"
make -f /tmp/lm_show.mk show LMCACHE=0 2>&1 | grep '^LMCACHE_FLAGS'
echo "--- LMCACHE=1 ---"
make -f /tmp/lm_show.mk show LMCACHE=1 2>&1 | grep '^LMCACHE_FLAGS'
rm /tmp/lm_show.mk
```

Expected:
- `LMCACHE=0`: `LMCACHE_FLAGS=[]` — empty.
- `LMCACHE=1`: `LMCACHE_FLAGS=[--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}']` — full JSON, all four `"` quotes present, comma intact between the two key-value pairs.

If `LMCACHE=1` shows truncated output (e.g., cuts off at `LMCacheConnectorV1`), the comma fix has been undone.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "Add LMCache runtime variables and conditional Make fragments"
```

---

## Task 3: Wire LMCache into the `serve` recipe

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Replace the `serve` recipe**

Locate the `serve:` recipe (currently 13 lines, ending with `--trust-remote-code`):

```makefile
serve:
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
		--trust-remote-code
```

Replace it with:

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

Three changes vs today:
1. New first line: `mkdir -p $(LMCACHE_DISK_PATH)`. Unconditional (cheap; just creates `data/lmcache/` if missing).
2. New second line: `$(LMCACHE_ENV) \`. Expands to the LMCache env vars when `LMCACHE=1`, to empty otherwise.
3. New trailing line: `$(LMCACHE_FLAGS)` appended after `--trust-remote-code`. Expands to the `--kv-transfer-config` arg when `LMCACHE=1`, to empty otherwise.

Indentation: TAB everywhere. The `mkdir` line is a separate make recipe step (its own subshell); the long line starting with `$(LMCACHE_ENV) \` and continuing through `$(LMCACHE_FLAGS)` is one logical command (one subshell), as it was before.

- [ ] **Step 2: Verify default behavior is unchanged (static)**

Run: `make -n serve`
Expected: prints two commands. The first is `mkdir -p <cwd>/data/lmcache`. The second is the existing `vllm serve google/gemma-4-26B-A4B-it ...` command — verify it includes every existing flag (`--served-model-name vllm-model`, `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, `--reasoning-parser gemma4`, the `--speculative-config '{"method": "dflash", ...}'` block, `--max-model-len 32768`, `--max-num-batched-tokens 32768`, `--gpu-memory-utilization 0.80`, `--limit-mm-per-prompt '{"image":0,"video":0}'`, `--trust-remote-code`) and contains NO `--kv-transfer-config` flag and NO `LMCACHE_*` env-var prefixes.

This is the no-op regression check: `LMCACHE` defaults to `0`, so the fragments must collapse to empty.

- [ ] **Step 3: Verify opt-in behavior (static)**

Run: `LMCACHE=1 make -n serve`
Expected: prints two commands. The first is `mkdir -p <cwd>/data/lmcache` (unchanged). The second now starts with `LMCACHE_LOCAL_CPU=true LMCACHE_MAX_LOCAL_CPU_SIZE=20 LMCACHE_LOCAL_DISK=file://<cwd>/data/lmcache LMCACHE_MAX_LOCAL_DISK_SIZE=50 LMCACHE_CHUNK_SIZE=256 PATH=<cwd>/.venv/bin:$PATH .venv/bin/vllm serve google/gemma-4-26B-A4B-it ...` and ends with `... --trust-remote-code --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.

Run: `LMCACHE=1 LMCACHE_MAX_LOCAL_CPU_SIZE=40 LMCACHE_DISK_PATH=/tmp/lm make -n serve`
Expected: the first command is `mkdir -p /tmp/lm`. The env-var prefix in the second command includes `LMCACHE_MAX_LOCAL_CPU_SIZE=40 LMCACHE_LOCAL_DISK=file:///tmp/lm`. This confirms variable overrides flow through both the prelude and the env block.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "Wire LMCache opt-in into make serve"
```

---

## Task 4: Wire LMCache into the `serve-no-spec` recipe

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Replace the `serve-no-spec` recipe**

Locate the `serve-no-spec:` recipe (currently 12 lines, ending with `--trust-remote-code`):

```makefile
serve-no-spec:
	PATH=$(CURDIR)/.venv/bin:$$PATH \
	$(VLLM) serve $(HF_MODEL) \
		--served-model-name $(SERVED_MODEL_NAME) \
		--enable-auto-tool-choice \
		--tool-call-parser $(TOOL_CALL_PARSER) \
		--reasoning-parser $(REASONING_PARSER) \
		--max-model-len 32768 \
		--max-num-batched-tokens 32768 \
		--gpu-memory-utilization 0.80 \
		--limit-mm-per-prompt '{"image":0,"video":0}' \
		--trust-remote-code
```

Replace it with:

```makefile
serve-no-spec:
	mkdir -p $(LMCACHE_DISK_PATH)
	$(LMCACHE_ENV) \
	PATH=$(CURDIR)/.venv/bin:$$PATH \
	$(VLLM) serve $(HF_MODEL) \
		--served-model-name $(SERVED_MODEL_NAME) \
		--enable-auto-tool-choice \
		--tool-call-parser $(TOOL_CALL_PARSER) \
		--reasoning-parser $(REASONING_PARSER) \
		--max-model-len 32768 \
		--max-num-batched-tokens 32768 \
		--gpu-memory-utilization 0.80 \
		--limit-mm-per-prompt '{"image":0,"video":0}' \
		--trust-remote-code \
		$(LMCACHE_FLAGS)
```

Same three changes as in Task 3 — `mkdir` prelude, `$(LMCACHE_ENV)` prefix, `$(LMCACHE_FLAGS)` suffix. The only difference vs the `serve` recipe is the absence of the `--speculative-config` line, which is preserved as-is (this is what makes it the no-spec path).

Indentation: TAB everywhere.

- [ ] **Step 2: Verify default behavior is unchanged (static)**

Run: `make -n serve-no-spec`
Expected: prints two commands. The first is `mkdir -p <cwd>/data/lmcache`. The second is the existing `vllm serve google/gemma-4-26B-A4B-it ...` command — verify it includes every existing flag EXCEPT the `--speculative-config` block (this recipe is the no-spec variant), and contains NO `--kv-transfer-config` flag and NO `LMCACHE_*` env-var prefixes.

- [ ] **Step 3: Verify opt-in behavior (static)**

Run: `LMCACHE=1 make -n serve-no-spec`
Expected: same env-var prefix and `--kv-transfer-config` suffix as `LMCACHE=1 make -n serve` from Task 3, but without the `--speculative-config` block. The two recipes should differ in exactly one flag block.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "Wire LMCache opt-in into make serve-no-spec"
```

---

## Task 5: Update `make help` to mention the LMCache toggle

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Update the `serve` help line**

Inside the `help:` target, locate the line:

```makefile
	@echo "  serve        Launch vLLM serving Gemma 4 26B with DFlash on DGX Spark"
```

Replace it with:

```makefile
	@echo "  serve        Launch vLLM serving Gemma 4 26B with DFlash on DGX Spark (LMCACHE=1 for KV-cache offload)"
```

Mentioning the toggle on the existing line keeps the help block from growing — there is no separate `serve-lmcache` target to describe.

- [ ] **Step 2: Verify (static)**

Run: `make help | grep "serve  "`
Expected: line `  serve        Launch vLLM serving Gemma 4 26B with DFlash on DGX Spark (LMCACHE=1 for KV-cache offload)`

Run: `make help | grep -c LMCACHE`
Expected: `1` (single mention, on the `serve` help line).

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "Document LMCACHE=1 toggle in make help"
```

---

## Task 6: Add "LMCache (optional KV-cache offload)" section to README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the LMCache section**

The existing README has the structure: ... → "Run" → "Use with Claude Code" → ... Insert a new "## LMCache (optional KV-cache offload)" section between "Run" (which currently ends with the `make warmup-diag` paragraph) and "Use with Claude Code".

Locate the line in `README.md` that reads `## Use with Claude Code`.

Immediately before that line (with a blank line separator), insert the following Markdown literally. The block uses fenced code blocks internally — keep the inner triple-backtick fences exactly as shown:

````markdown
## LMCache (optional KV-cache offload)

Set `LMCACHE=1` to launch vLLM with [LMCache](https://github.com/LMCache/LMCache) enabled for a CPU + local-disk KV-cache offload tier on top of vLLM's GPU prefix cache. Useful here because Claude Code and Codex both repeat long system prompts and conversation prefixes across requests — the GPU prefix cache catches the in-memory tail, and LMCache extends it through host RAM and disk so hits survive eviction and server restarts.

```bash
LMCACHE=1 make serve            # spec-decode path with LMCache
LMCACHE=1 make serve-no-spec    # no-spec path with LMCache
```

The toggle works identically on both serve targets. `LMCACHE=0` (the default) leaves the recipes byte-for-byte unchanged.

### Tunables

| Variable                      | Default               | Meaning                                                             |
| ----------------------------- | --------------------- | ------------------------------------------------------------------- |
| `LMCACHE`                     | `0`                   | Toggle. Anything other than `1` disables LMCache.                   |
| `LMCACHE_MAX_LOCAL_CPU_SIZE`  | `20`                  | GB of host RAM dedicated to the CPU tier.                           |
| `LMCACHE_DISK_PATH`           | `<repo>/data/lmcache` | Persistent disk-tier directory. Lives under the gitignored `data/`. |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | `50`                  | GB of disk dedicated to the persistent tier.                        |

Override at invocation, e.g. `LMCACHE=1 LMCACHE_MAX_LOCAL_CPU_SIZE=40 make serve`.

### Verifying it is live

Server logs include `LMCache` lines on hit/miss; grep the serve output. The disk tier populates `data/lmcache/` as the cache warms — an empty directory after several requests is a sign LMCache did not initialize. After a restart, hits against the same prompts should arrive faster than a fresh boot, since the disk tier persists.

### Caveats

- `make bench` numbers are not directly comparable across `LMCACHE=0` and `LMCACHE=1`. Pick one setting and stick with it for any spec-vs-no-spec comparison.
- A model swap (different `HF_MODEL`) leaves stale entries under `data/lmcache/` that count against the `LMCACHE_MAX_LOCAL_DISK_SIZE` cap. Clear with `rm -rf data/lmcache` between model swaps.
````

- [ ] **Step 2: Add an LMCache example to the Configuration override line**

Locate the paragraph at the end of the existing "Configuration" section that currently reads (one line, ending with the bench example added by the prior bench-targets work):

```
Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, `make codex VLLM_BASE_URL=http://remote:8000`, or `make bench BENCH_NUM_PROMPTS=500`.
```

Replace it with:

```
Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, `make codex VLLM_BASE_URL=http://remote:8000`, `make bench BENCH_NUM_PROMPTS=500`, or `LMCACHE=1 make serve`.
```

- [ ] **Step 3: Verify (visual + grep)**

Run: `grep -n "^## " README.md`
Expected: section headers in order — Prerequisites, Setup, Patches, Run, **LMCache (optional KV-cache offload)** (new), Use with Claude Code, Use with Codex, Benchmark, Configuration, License.

Run: `grep -c "LMCACHE" README.md`
Expected: at least 7 (toggle bullets, tunable rows, override example).

Run: `grep -n "LMCACHE=1 make serve" README.md`
Expected: at least 3 hits (intro code block, override example, possibly the verifying paragraph).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document LMCACHE=1 opt-in in README"
```

---

## Final Live Verification (manual, on DGX Spark)

These steps require the actual hardware and a built `.venv`. Run them by hand once after merging; they are not gating per-task.

1. From a fresh venv, run `make install`.
   Expected: vLLM nightly resolves and installs, then `lmcache==0.4.4` installs (note: the install will downgrade `numpy` from `2.3.5` to `2.2.6` and several `opentelemetry-*` packages — this is expected; see spec risks). `make patch` then runs and prints `already-patched` for each file on a re-run, or applies the three patches on first run.

2. Run plain `make serve` (no env var).
   Expected: server boots and serves identically to before. Grep the server log for `LMCache`: zero matches. The `data/lmcache/` directory is created (by the unconditional `mkdir -p`) but stays empty.

3. Stop the server. Run `LMCACHE=1 make serve`.
   Expected: server boots; the log includes a line like `Initializing latest dev LMCache connector` (or equivalent). `data/lmcache/` exists and is writable.

4. With `LMCACHE=1 make serve` running, run `make warmup` in another shell, then send the same ~5K-token prompt a second time via `make chat`.
   Expected: second submission shows a TTFT drop and the server log includes an LMCache hit line.

5. Stop and restart the server with `LMCACHE=1 make serve`. Re-submit the same prompt from step 4.
   Expected: still a hit (disk tier persists across restarts). `ls data/lmcache/` shows non-empty cache files.

6. Run `git status`.
   Expected: no new tracked files under `data/`.

7. **Sanity check the `LMCACHE_LOCAL_DISK=file://...` URL form.** This is the documented LMCache convention but the bundled adapter delegates env-var parsing to the installed `lmcache` package. If step 3 fails to initialize the disk tier (an "invalid URL" or "unsupported scheme" log line), fall back to authoring a `lmcache.yaml` and pointing `LMCACHE_CONFIG_FILE` at it via `LMCACHE_ENV` — same Makefile shape, different env var. Update the spec and this plan if so.
