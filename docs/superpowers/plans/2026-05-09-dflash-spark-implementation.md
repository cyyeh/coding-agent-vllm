# DFlash + DGX Spark vLLM Setup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `coding-agent-vllm` install and run on DGX Spark (aarch64, GB10, Blackwell SM_121, Ubuntu 24.04), serving Gemma-4 26B-A4B with the `z-lab/gemma-4-26B-A4B-it-DFlash` draft via DFlash speculative decoding.

**Architecture:** Pin vLLM to a known-good aarch64 nightly. Vendor a verbatim copy of `meanaverage/gemma4-dflash-spark-vllm`'s file-rewriting patcher (Apache-2.0) into `scripts/`. The patcher runs once after install to edit three `.py` files inside the installed `vllm` package: disable CUTLASS FP8 paths on GB10, add `supports_non_causal()` to the Triton attention backend, and pin the DFlash draft to FlashAttention. Update the `Makefile` to orchestrate install→patch→serve with Spark-specific flags (`--enforce-eager`, `--max-model-len 16384`, `VLLM_DISABLE_COMPILE_CACHE=1`, text-only mm limits). Drop the PR #41703 source-build path.

**Tech Stack:** vLLM nightly (pinned, aarch64+sm_121 wheel), `uv` for package management, GNU Make, Python 3.12. No new test framework — verification is via shell smoke tests in the plan steps.

**Spec reference:** `docs/superpowers/specs/2026-05-08-dflash-spark-design.md`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `Makefile` | Modify | Orchestrate install/patch/serve. Add `patch` target; rewrite `serve` with Spark flags; remove `serve-no-speculative`. |
| `pyproject.toml` | Modify | Pin vLLM nightly version so future `uv sync` honors it. |
| `README.md` | Modify | Update Setup, Run, and Configuration sections to match Spark-only flow. Drop PR #41703 reference. |
| `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` | Create | Vendored verbatim from `meanaverage/gemma4-dflash-spark-vllm@<SHA>`, Apache-2.0. Three in-place file rewrites in installed vLLM package. |
| `NOTICE` | Create | Apache-2.0 §4(d) attribution for the vendored script. |

The patcher is the only meaningful new code. It's vendored verbatim, so the implementation work on it is limited to (a) prepending a vendor-attribution header, (b) verifying its three patches apply cleanly against the chosen vLLM pin (the *non-trivial* verification step — most likely failure point is the `qwen3_dflash` marker if the 26B variant differs from 31B).

---

## Task 1: Verify nightly index URL, choose `VLLM_PIN`, capture vendor SHA

**Why this is task one:** The whole plan hinges on a vLLM nightly that ships an aarch64+sm_121 wheel. If `0.19.2rc1.dev21` (the spec's initial candidate) doesn't resolve, every subsequent task uses the wrong version. We need a confirmed working pin before touching files.

**Files:** None modified yet — this task captures values used in Tasks 2, 4, and 5.

- [ ] **Step 1: Verify the vLLM nightly wheels index responds**

```bash
curl -fsSL --max-time 10 https://wheels.vllm.ai/nightly/ | head -50
```

Expected: HTTP 200 with an HTML directory listing (links to `vllm-...whl` entries) or an XML S3-style listing. Either is acceptable. If 404 or connection error, the URL is wrong — search vLLM docs (`https://docs.vllm.ai/en/latest/getting_started/installation/nightly.html` or similar) for the canonical nightly index URL and use that. Record the working URL; you'll bake it into the Makefile in Task 5.

- [ ] **Step 2: Confirm an aarch64 wheel exists for the candidate pin**

```bash
curl -fsSL https://wheels.vllm.ai/nightly/ | grep -E 'vllm-0\.19\.2rc1\.dev21.*(aarch64|arm64)' || echo "NO AARCH64 WHEEL FOUND"
```

Expected: at least one line referencing a `cp312-cp312-linux_aarch64.whl` or similar. If `NO AARCH64 WHEEL FOUND`:

```bash
curl -fsSL https://wheels.vllm.ai/nightly/ | grep -oE 'vllm-[0-9]+\.[0-9]+\.[0-9a-z+]+(\.dev[0-9]+)?[^"]*aarch64[^"]*\.whl' | sort -u | tail -20
```

Pick the most recent dev nightly that has an aarch64 wheel. Record the version string (e.g., `0.19.2rc1.dev42`) — this becomes `VLLM_PIN` in Tasks 4 and 5.

- [ ] **Step 3: Capture the vendor commit SHA**

```bash
git ls-remote https://github.com/meanaverage/gemma4-dflash-spark-vllm refs/heads/main | awk '{print $1}'
```

Expected: a 40-char hex SHA. Record it — you'll embed it in the patcher's vendor-attribution header in Task 2.

- [ ] **Step 4: Note the chosen values**

Write down (in conversation, not in a file) the three values produced by Steps 1–3:
- `VLLM_INDEX_URL` (e.g., `https://wheels.vllm.ai/nightly`)
- `VLLM_PIN` (e.g., `0.19.2rc1.dev21`)
- Vendor SHA (e.g., `1a2b3c4...`)

These flow into Tasks 2, 4, and 5. **Do not commit anything in this task.**

---

## Task 2: Vendor the patcher script

**Files:**
- Create: `/home/cyyeh/repos/coding-agent-vllm/scripts/patch_vllm_gb10_gemma4_dflash_runtime.py`

- [ ] **Step 1: Create the `scripts/` directory**

```bash
mkdir -p /home/cyyeh/repos/coding-agent-vllm/scripts
```

Expected: directory created (or already exists, which is fine).

- [ ] **Step 2: Write the patcher with the vendor-attribution header**

Write the file at `/home/cyyeh/repos/coding-agent-vllm/scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` with this exact content (replace `<VENDOR_SHA>` with the SHA captured in Task 1 Step 3):

```python
#!/usr/bin/env python3
# Vendored from https://github.com/meanaverage/gemma4-dflash-spark-vllm
# @ <VENDOR_SHA>
# SPDX-License-Identifier: Apache-2.0
# Original copyright retained per Apache-2.0 §4(b)/(d).
# See repo root NOTICE for required attribution text.
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import vllm


def patch_text_once(text: str, old: str, new: str) -> tuple[str, bool]:
    updated = re.sub(old, new, text, count=1, flags=re.MULTILINE)
    return updated, updated != text


def ensure_contains_once(text: str, marker: str, replacement: str) -> tuple[str, bool]:
    if replacement in text:
        return text, False
    if marker not in text:
        raise RuntimeError(f"patch marker not found: {marker!r}")
    return text.replace(marker, replacement, 1), True


def patch_w8a8_utils(pkg_root: Path) -> bool:
    target = pkg_root / "model_executor/layers/quantization/utils/w8a8_utils.py"
    text = target.read_text()
    changed = False

    for old, new in [
        (
            r"(def cutlass_fp8_supported\(\)[^:]*:\n)",
            r"\1    return False  # Patched for Spark GB10 prebuilt CUTLASS coverage.\n",
        ),
        (
            r"(def cutlass_block_fp8_supported\(\)[^:]*:\n)",
            r"\1    return False  # Patched for Spark GB10 prebuilt CUTLASS coverage.\n",
        ),
    ]:
        text, step_changed = patch_text_once(text, old, new)
        changed = changed or step_changed

    for old, new in [
        (
            r"CUTLASS_FP8_SUPPORTED\s*=\s*cutlass_fp8_supported\(\)",
            "CUTLASS_FP8_SUPPORTED = False  # Patched for Spark GB10.",
        ),
        (
            r"CUTLASS_BLOCK_FP8_SUPPORTED\s*=\s*cutlass_block_fp8_supported\(\)",
            "CUTLASS_BLOCK_FP8_SUPPORTED = False  # Patched for Spark GB10.",
        ),
    ]:
        text, step_changed = patch_text_once(text, old, new)
        changed = changed or step_changed

    if changed:
        target.write_text(text)
    return changed


def patch_triton_backend(pkg_root: Path) -> bool:
    target = pkg_root / "v1/attention/backends/triton_attn.py"
    text = target.read_text()
    marker = """    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
"""
    replacement = """    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
"""
    text, changed = ensure_contains_once(text, marker, replacement)
    if changed:
        target.write_text(text)
    return changed


def patch_qwen3_dflash(pkg_root: Path) -> bool:
    target = pkg_root / "model_executor/models/qwen3_dflash.py"
    text = target.read_text()
    changed = False

    import_marker = "from vllm.v1.attention.backend import AttentionType\n"
    import_replacement = (
        "from vllm.v1.attention.backend import AttentionType\n"
        "from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend\n"
    )
    text, step_changed = ensure_contains_once(text, import_marker, import_replacement)
    changed = changed or step_changed

    attn_marker = """        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
"""
    attn_replacement = """        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            attn_backend=FlashAttentionBackend,
        )
"""
    text, step_changed = ensure_contains_once(text, attn_marker, attn_replacement)
    changed = changed or step_changed

    if changed:
        target.write_text(text)
    return changed


def clear_python_caches(pkg_root: Path) -> None:
    for relative in [
        "model_executor/layers/quantization",
        "model_executor/models",
        "v1/attention",
    ]:
        base = pkg_root / relative
        if not base.exists():
            continue
        for pyc in base.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)
        for pycache in base.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)


def main() -> int:
    pkg_root = Path(vllm.__file__).resolve().parent
    changes = {
        "w8a8_utils": patch_w8a8_utils(pkg_root),
        "triton_attn": patch_triton_backend(pkg_root),
        "qwen3_dflash": patch_qwen3_dflash(pkg_root),
    }
    clear_python_caches(pkg_root)
    for name, changed in changes.items():
        state = "patched" if changed else "already-patched"
        print(f"{name}: {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Verify the file is syntactically valid Python**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && python3 -c "import ast; ast.parse(open('scripts/patch_vllm_gb10_gemma4_dflash_runtime.py').read()); print('OK')"
```

Expected: `OK`. If syntax error, fix the file (most likely cause: a stray indentation or quote when copying the multiline marker strings).

- [ ] **Step 4: Verify the vendor SHA is filled in (no placeholder remains)**

```bash
grep -n '<VENDOR_SHA>' /home/cyyeh/repos/coding-agent-vllm/scripts/patch_vllm_gb10_gemma4_dflash_runtime.py && echo "FAIL: placeholder still present" || echo "OK: SHA filled in"
```

Expected: `OK: SHA filled in`. If `FAIL`, replace `<VENDOR_SHA>` with the SHA captured in Task 1 Step 3.

- [ ] **Step 5: No commit yet** — `NOTICE`, `pyproject.toml`, and `Makefile` come together in one commit (see Task 5).

---

## Task 3: Create `NOTICE` for Apache-2.0 attribution

**Files:**
- Create: `/home/cyyeh/repos/coding-agent-vllm/NOTICE`

- [ ] **Step 1: Write `NOTICE` at the repo root**

Write `/home/cyyeh/repos/coding-agent-vllm/NOTICE` with this exact content:

```
coding-agent-vllm
Copyright 2026 ChihYu Yeh

This product includes software developed by meanaverage:
  - scripts/patch_vllm_gb10_gemma4_dflash_runtime.py
    Source: https://github.com/meanaverage/gemma4-dflash-spark-vllm
    Licensed under the Apache License, Version 2.0
    https://www.apache.org/licenses/LICENSE-2.0
```

- [ ] **Step 2: Verify file exists and isn't empty**

```bash
test -s /home/cyyeh/repos/coding-agent-vllm/NOTICE && wc -l /home/cyyeh/repos/coding-agent-vllm/NOTICE
```

Expected: 7 lines (or thereabouts). If empty or missing, redo Step 1.

- [ ] **Step 3: No commit yet.**

---

## Task 4: Pin vLLM in `pyproject.toml`

**Files:**
- Modify: `/home/cyyeh/repos/coding-agent-vllm/pyproject.toml`

- [ ] **Step 1: Read current `pyproject.toml` to confirm baseline**

```bash
cat /home/cyyeh/repos/coding-agent-vllm/pyproject.toml
```

Expected: contains `dependencies = []`. If it differs from the spec's assumption, stop and reconcile before continuing.

- [ ] **Step 2: Replace empty dependencies with the pin**

Edit `/home/cyyeh/repos/coding-agent-vllm/pyproject.toml`. Replace this line:

```
dependencies = []
```

With (substitute `<VLLM_PIN>` with the value from Task 1 Step 2 — e.g., `0.19.2rc1.dev21`):

```
dependencies = ["vllm==<VLLM_PIN>"]
```

- [ ] **Step 3: Verify**

```bash
grep -n 'vllm==' /home/cyyeh/repos/coding-agent-vllm/pyproject.toml
```

Expected: one line showing `dependencies = ["vllm==<your_pin>"]`. The pin string must match exactly what you'll put in `Makefile` in Task 5.

- [ ] **Step 4: No commit yet.**

---

## Task 5: Rewrite `Makefile`

**Files:**
- Modify: `/home/cyyeh/repos/coding-agent-vllm/Makefile`

The current `Makefile` has working-tree changes (from your in-progress edits) — those will be overwritten by this rewrite. That's intended; the design supersedes them.

- [ ] **Step 1: Replace the entire Makefile**

Write `/home/cyyeh/repos/coding-agent-vllm/Makefile` with this content (substitute `<VLLM_PIN>` and `<VLLM_INDEX_URL>` with the values from Task 1):

```make
.PHONY: help venv install patch serve chat cc codex

ifneq (,$(wildcard .env))
    include .env
    export
endif

VLLM ?= .venv/bin/vllm
PY ?= .venv/bin/python
SERVED_MODEL_NAME ?= vllm-model
TOOL_CALL_PARSER ?= gemma4
REASONING_PARSER ?= gemma4
VLLM_BASE_URL ?= http://localhost:8000
VLLM_API_KEY ?= dummy
VLLM_PIN ?= <VLLM_PIN>
VLLM_INDEX_URL ?= <VLLM_INDEX_URL>

help:
	@echo "Targets:"
	@echo "  venv         Create the .venv with Python 3.12"
	@echo "  install      Install pinned vLLM nightly and apply DGX Spark patches"
	@echo "  patch        Apply DGX Spark / DFlash patches to the installed vLLM (idempotent)"
	@echo "  serve        Launch vLLM serving Gemma 4 26B with DFlash on DGX Spark"
	@echo "  chat         Open an interactive chat session against the local vLLM server"
	@echo "  cc           Launch Claude Code routed to the local vLLM server"
	@echo "  codex        Launch Codex routed to the local vLLM server"

venv:
	uv venv

install:
	uv pip install -U --pre --torch-backend=auto \
		--extra-index-url $(VLLM_INDEX_URL) \
		"vllm==$(VLLM_PIN)"
	$(MAKE) patch

patch:
	$(PY) scripts/patch_vllm_gb10_gemma4_dflash_runtime.py

serve:
	VLLM_DISABLE_COMPILE_CACHE=1 \
	$(VLLM) serve google/gemma-4-26B-A4B-it \
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

chat:
	$(VLLM) chat \
		--url $(VLLM_BASE_URL)/v1 \
		--model $(SERVED_MODEL_NAME) \
		--api-key $(VLLM_API_KEY)

cc:
	ANTHROPIC_BASE_URL=$(VLLM_BASE_URL) \
	ANTHROPIC_API_KEY=dummy \
	ANTHROPIC_DEFAULT_OPUS_MODEL=$(SERVED_MODEL_NAME) \
	ANTHROPIC_DEFAULT_SONNET_MODEL=$(SERVED_MODEL_NAME) \
	ANTHROPIC_DEFAULT_HAIKU_MODEL=$(SERVED_MODEL_NAME) \
	claude --effort max

codex:
	VLLM_API_KEY=$(VLLM_API_KEY) \
	codex \
		-c model=$(SERVED_MODEL_NAME) \
		-c model_provider=vllm \
		-c 'model_providers.vllm.name="vLLM"' \
		-c 'model_providers.vllm.env_key="VLLM_API_KEY"' \
		-c 'model_providers.vllm.base_url="$(VLLM_BASE_URL)/v1"' \
		-c 'model_providers.vllm.wire_api="responses"'
```

Key points:
- `.PHONY` no longer lists `serve-no-speculative` (removed entirely from this Makefile).
- New `patch` target. New `PY ?= .venv/bin/python`, `VLLM_PIN`, `VLLM_INDEX_URL` variables.
- `install` ends with `$(MAKE) patch`, so a fresh user just runs `make install`.
- `serve` drops `--attention-backend triton_attn` and `attention_backend: flash_attn` from `--speculative-config`; adds `--enforce-eager`, `--max-model-len 16384`, `--gpu-memory-utilization 0.80`, `--limit-mm-per-prompt`, and the `VLLM_DISABLE_COMPILE_CACHE=1` env prefix.
- `chat`, `cc`, `codex` are byte-for-byte unchanged.

- [ ] **Step 2: Verify Make can parse it (no syntax error, all targets visible)**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && make help
```

Expected: prints the seven targets listed in `help`. If `make help` errors with "missing separator" or similar, you have spaces where there should be tabs in the recipe lines — Makefiles require **TAB** indentation, not spaces.

- [ ] **Step 3: Verify VLLM_PIN and VLLM_INDEX_URL placeholders are filled**

```bash
grep -nE '<VLLM_(PIN|INDEX_URL)>' /home/cyyeh/repos/coding-agent-vllm/Makefile && echo "FAIL: placeholder remains" || echo "OK"
```

Expected: `OK`.

- [ ] **Step 4: Commit Tasks 2–5 together**

```bash
cd /home/cyyeh/repos/coding-agent-vllm
git add scripts/patch_vllm_gb10_gemma4_dflash_runtime.py NOTICE pyproject.toml Makefile
git status
git commit -m "$(cat <<'EOF'
Switch to pinned vLLM nightly + GB10 patcher for DGX Spark

Replaces the PR #41703 source build with a pinned vLLM nightly wheel
plus a vendored runtime patcher (Apache-2.0, from
meanaverage/gemma4-dflash-spark-vllm) that adapts the installed vLLM
package for GB10 / Blackwell SM_121: disables CUTLASS FP8 paths, adds
non-causal support to the Triton backend, and pins the DFlash draft
to FlashAttention. Updates the serve command with --enforce-eager,
--max-model-len 16384, text-only mm limits, and
VLLM_DISABLE_COMPILE_CACHE=1 to match the reference's verified flags.

See docs/superpowers/specs/2026-05-08-dflash-spark-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds; `git status` is clean except for the still-pending `README.md` work in Task 9.

---

## Task 6: Smoke-test `make install` and `make patch` (idempotency)

**Why:** Patcher fragility is the highest-risk part of the design. We need to know the patches apply cleanly *before* trying to start the server.

**Files:** None modified (read-only verification).

**Expected duration:** Step 2 may take 5–15 minutes (downloading the vLLM wheel and dependencies on first run).

- [ ] **Step 1: Ensure venv exists**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && [ -x .venv/bin/python ] && echo "venv ok" || make venv
```

Expected: `venv ok` or a fresh venv created.

- [ ] **Step 2: Run `make install` (this also runs `make patch`)**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && make install
```

Expected output (last lines):
```
w8a8_utils: patched
triton_attn: patched
qwen3_dflash: patched
```

If `make install` fails before `patch` runs:
- "No matching distribution" → wheel not on the index for the chosen pin. Bump `VLLM_PIN` to a newer nightly per Task 1 Step 2. Update both `Makefile` and `pyproject.toml`.
- ARM64 wheel resolution issues → confirm `uname -m` returns `aarch64` (it should, on Spark) and that `--torch-backend=auto` resolved a CUDA-12 torch wheel.

If `make patch` fails with `patch marker not found: ...`:
- The marker comes from the installed vLLM source for the pinned version. Read the relevant file:
  ```bash
  grep -n -A3 "supports_head_size\|cutlass_fp8_supported\|self.attn = Attention" .venv/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py .venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py .venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py 2>/dev/null
  ```
- Edit `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` so the marker string matches the actual function body in the pinned version. The 26B `qwen3_dflash.py` is the most likely culprit — its `Attention(...)` constructor argument list may differ from the 31B variant the reference patched against. Fix marker, re-run `make patch`. Document the divergence in the patcher's vendor header (add a comment line like `# Marker adapted for vllm==<PIN>: Attention(...) signature differs from 31B variant.`).

- [ ] **Step 3: Verify idempotency — run `make patch` a second time**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && make patch
```

Expected output:
```
w8a8_utils: already-patched
triton_attn: already-patched
qwen3_dflash: already-patched
```

If any line says `patched` instead of `already-patched`, the patcher is non-idempotent — most likely the regex matches the *patched* output too. Investigate before proceeding.

- [ ] **Step 4: Spot-check a patched file**

```bash
grep -n 'Patched for Spark GB10\|supports_non_causal\|FlashAttentionBackend' \
  .venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py \
  .venv/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py \
  .venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
```

Expected: at least one match per file. If a file shows zero matches, that patch silently no-op'd (`patch_text_once` returned `False` but `ensure_contains_once` is unrelated). Investigate before proceeding.

- [ ] **Step 5: If patcher needed adjustment, commit the fix**

If you modified `scripts/patch_vllm_gb10_gemma4_dflash_runtime.py` in Step 2's recovery path:

```bash
cd /home/cyyeh/repos/coding-agent-vllm
git add scripts/patch_vllm_gb10_gemma4_dflash_runtime.py
git commit -m "$(cat <<'EOF'
Adapt patcher markers for pinned vLLM version

The 26B-A4B DFlash variant / pinned vLLM revision uses a slightly
different signature than the reference's 31B target. Marker adjusted
so patch_qwen3_dflash applies cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Otherwise: no commit needed — verification only.

---

## Task 7: Smoke-test `make serve` (DFlash actually engaged)

**Files:** None modified.

**Expected duration:** Step 2 may take 5–10 minutes for the first run (downloading 26B target + DFlash draft from HuggingFace, ~60 GB total).

**Prerequisite:** `HF_TOKEN` set in `.env` and the HF account has access granted for `google/gemma-4-26B-A4B-it`.

- [ ] **Step 1: Confirm `HF_TOKEN` is loaded**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && grep -q '^HF_TOKEN=' .env && echo "HF_TOKEN present" || echo "MISSING — populate .env"
```

Expected: `HF_TOKEN present`. If missing, populate `.env` from `.env.example` before continuing.

- [ ] **Step 2: Start the server in a tmux/screen session or background-friendly way**

The server must keep running for Step 3 and Task 8. Easiest is a separate terminal pane:

```bash
cd /home/cyyeh/repos/coding-agent-vllm && make serve 2>&1 | tee /tmp/vllm-serve.log
```

Wait for one of:
- `Application startup complete` and `Uvicorn running on http://0.0.0.0:8000` → success.
- A traceback → failure; capture it and diagnose (see expected failure modes below).

Expected failure modes:
- `RuntimeError: ... cutlass ...` → the FP8 patch didn't take. Verify `grep CUTLASS_FP8_SUPPORTED .venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py` shows `False`.
- `Triton backend does not support non-causal` → the Triton patch didn't take. Verify `grep supports_non_causal .venv/lib/python3.12/site-packages/vllm/v1/attention/backends/triton_attn.py` shows the patched method.
- `OSError: ... gated repo` → HF token missing access for `google/gemma-4-26B-A4B-it`.
- `OOM` → reduce `--gpu-memory-utilization` to `0.70`, retry. (Edit `Makefile` if persistent.)
- `falling back to autoregressive` log line → DFlash draft model failed to load; check that `z-lab/gemma-4-26B-A4B-it-DFlash` exists and is accessible; check the vLLM log earlier for the actual error.

- [ ] **Step 3: From a second terminal, hit the OpenAI `/v1/models` endpoint**

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

Expected: JSON with one entry whose `id` is `vllm-model` (the served name from `SERVED_MODEL_NAME`).

- [ ] **Step 4: Verify DFlash speculative decoding is engaged**

```bash
grep -iE 'dflash|speculative|num_speculative_tokens' /tmp/vllm-serve.log | head -20
```

Expected: lines confirming the speculative method is `dflash` and `num_speculative_tokens=15`. If the log instead shows `speculative_config: None` or "speculation disabled", investigate before proceeding to Task 8.

- [ ] **Step 5: Leave the server running.** Task 8 needs it.

---

## Task 8: End-to-end smoke test via `make chat`

**Files:** None modified.

**Prerequisite:** The vLLM server from Task 7 is still running on `:8000`.

- [ ] **Step 1: Run a simple chat exchange**

In a second terminal:

```bash
cd /home/cyyeh/repos/coding-agent-vllm && make chat
```

This starts the interactive vLLM chat client. Send one prompt:

```
write a python function that returns the sum of two numbers
```

Expected: a coherent response containing a `def` for adding two numbers (any reasonable variant — `add(a, b)`, `sum_two(...)`, etc., is fine). Exit with `/exit` or Ctrl-D.

If the response is gibberish, repetitive, or empty: DFlash may be drafting tokens that the verifier always rejects, which can degrade output quality. Stop the server (Task 7 terminal: Ctrl-C), edit `Makefile` to lower `num_speculative_tokens` to `8`, restart `make serve`, retry `make chat`.

- [ ] **Step 2: Check acceptance rate in the server log**

```bash
grep -iE 'accept|spec_token' /tmp/vllm-serve.log | tail -10
```

Expected: at least one log line indicating speculative tokens were accepted (e.g., `accepted: N/M`, `acceptance_rate: 0.XX`). A rate above zero confirms DFlash is engaging. The exact metric format depends on the vLLM version — any non-zero acceptance signal is acceptable.

- [ ] **Step 3: Stop the server**

In the Task 7 terminal: Ctrl-C. Wait for graceful shutdown.

- [ ] **Step 4: No commit** — verification only.

---

## Task 9: Update `README.md`

**Files:**
- Modify: `/home/cyyeh/repos/coding-agent-vllm/README.md`

- [ ] **Step 1: Read the current `README.md`**

```bash
cat /home/cyyeh/repos/coding-agent-vllm/README.md
```

Confirm the current content matches what's expected (PR #41703 reference in Setup, no Spark mention).

- [ ] **Step 2: Replace `README.md` with the updated content**

Write `/home/cyyeh/repos/coding-agent-vllm/README.md`:

````markdown
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

`make install` pulls a pinned vLLM nightly wheel from the vLLM nightly index, then automatically runs `make patch` to apply three GB10-specific patches to the installed `vllm` package (disable CUTLASS FP8 paths on GB10, enable non-causal Triton attention, pin the DFlash draft to FlashAttention).

The patches are vendored from [`meanaverage/gemma4-dflash-spark-vllm`](https://github.com/meanaverage/gemma4-dflash-spark-vllm) (Apache-2.0; see `NOTICE`). They are *file rewrites* of the installed `vllm` site-packages tree — if you later run `pip install -U vllm` and the wheel changes, re-run `make patch` to re-apply them.

## Run

```bash
make serve
```

The Makefile auto-loads `.env`, so `HF_TOKEN` is exported to the vLLM process. The OpenAI-compatible API listens on `http://0.0.0.0:8000` and is also served under the Anthropic-compatible path expected by Claude Code, so the same server feeds both `make cc` and `make codex`.

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

## Configuration

| Setting                     | Value                              |
| --------------------------- | ---------------------------------- |
| Served model name           | `vllm-model`                       |
| Speculative method          | `dflash`                           |
| Speculative tokens per step | 15                                 |
| Max model length            | 16384                              |
| Max batched tokens          | 16384                              |
| GPU memory utilization      | 0.80                               |
| Eager mode                  | enabled (`--enforce-eager`)        |
| Multimodal                  | disabled (text-only)               |
| Tool calling                | auto                               |
| Tool-call parser            | `gemma4`                           |
| Reasoning parser            | `gemma4`                           |
| `trust_remote_code`         | enabled                            |

Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, or `make codex VLLM_BASE_URL=http://remote:8000`.

## License

[MIT](LICENSE) © 2026 ChihYu Yeh

Vendored components retain their original licenses; see `NOTICE`.
````

- [ ] **Step 3: Verify README mentions Spark, drops PR #41703**

```bash
cd /home/cyyeh/repos/coding-agent-vllm
grep -n 'DGX Spark\|GB10\|41703' README.md
```

Expected: lines mentioning DGX Spark and GB10. **No** lines mentioning PR #41703.

- [ ] **Step 4: Commit**

```bash
cd /home/cyyeh/repos/coding-agent-vllm
git add README.md
git commit -m "$(cat <<'EOF'
Rewrite README for DGX Spark + DFlash flow

Documents the Spark-only setup: prerequisites, the install/patch
pipeline, and the new serve flags. Drops the PR #41703 reference
since we no longer source-build vLLM. Updates the Configuration
table to match the new Makefile.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds; `git status` is clean.

---

## Task 10: Final verification and tidy

**Files:** None modified — final integrity check.

- [ ] **Step 1: Confirm working tree is clean**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && git status
```

Expected: `nothing to commit, working tree clean`. If files remain modified or untracked, decide whether they belong in a follow-up commit or were leftover scratch — only commit deliberate changes.

- [ ] **Step 2: Confirm commit log shape**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && git log --oneline -10
```

Expected: top of the log shows
1. The README commit (Task 9).
2. (Optional) The patcher-marker fix commit (Task 6 Step 5).
3. The combined Tasks 2–5 commit ("Switch to pinned vLLM nightly + GB10 patcher for DGX Spark").
4. The spec commit (`a68f196` "Add design spec: DFlash + DGX Spark vLLM setup").
5. Pre-existing commits.

- [ ] **Step 3: Confirm `make help` still lists the right targets**

```bash
cd /home/cyyeh/repos/coding-agent-vllm && make help
```

Expected: prints `venv`, `install`, `patch`, `serve`, `chat`, `cc`, `codex`. Does **not** print `serve-no-speculative`.

- [ ] **Step 4: Confirm at least one full `make install` + `make patch` + server-start cycle has been verified**

This is a meta-check on the implementation steps already done. If you haven't actually run `make install`, `make patch`, and `make serve` end-to-end (Tasks 6, 7, 8), do not mark the implementation complete — the design's validation criteria (spec §Validation, steps 1–7) require a working server with DFlash engaged.

---

## Acceptance criteria recap (from spec)

- [x] `make install` succeeds end-to-end on a clean DGX Spark venv. *(Task 6)*
- [x] `make patch` is idempotent. *(Task 6)*
- [x] `make serve` starts and serves Gemma-4 26B-A4B with DFlash draft engaged (acceptance rate > 0 in logs). *(Task 7, Task 8)*
- [x] `make cc` and `make codex` are launchable against the local server. *(Task 5; Task 8 spot-checks `make chat` which exercises the same OpenAI surface.)*
- [x] `README.md` matches the actual flow; no stale references to PR #41703 or the x86_64 source build. *(Task 9)*
- [x] `NOTICE` present, patcher script carries vendor header. *(Tasks 2, 3)*
