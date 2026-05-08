# Warmup improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `scripts/warmup.py` so a single `make warmup` after first server boot pre-JITs every Triton kernel real Claude Code / Codex traffic on the configured DGX Spark + Gemma-4 26B + DFlash stack will plausibly hit, silencing the seven `Triton kernel JIT compilation during inference: …` warnings observed today.

**Architecture:** Self-contained Python script. Four phase functions (prefill sweep, sampling sweep, concurrent decode, sustained decode) called sequentially from `main()`, each driving HTTP requests against the locally-running vLLM OpenAI-compatible API via `urllib`. Helpers `post_chat`, `make_prompt`, `run_concurrent` shared across phases. No new dependencies, no test framework added (project has none, scripts are integration-tested by execution).

**Tech Stack:** Python 3.12 stdlib only (`urllib`, `concurrent.futures`, `json`, `time`). Reads env vars `VLLM_BASE_URL`, `VLLM_API_KEY`, `SERVED_MODEL_NAME` (unchanged).

**Spec:** [`docs/superpowers/specs/2026-05-09-warmup-improvement-design.md`](../specs/2026-05-09-warmup-improvement-design.md)

**Prerequisites:** `make serve` must be running in another shell with the configured Gemma-4 26B + DFlash setup. Verification steps in Tasks 2–5 hit the live server.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `scripts/warmup.py` | Modify (rewrite body) | Four-phase Triton kernel warmup |
| `README.md` | Modify (one paragraph) | "First-run warmup" docs — bump time estimate, add coverage note |

---

### Task 1: Refactor helpers (preserve existing behavior)

Add the helpers needed by the new phases. Keep existing `main()` for now so `make warmup` still does the original 2-prompt warmup — this is a refactor task, not a behavior change.

**Files:**
- Modify: `scripts/warmup.py`

- [ ] **Step 1: Replace the `post_chat` helper with a sampling-aware version**

Replace lines 32–55 of `scripts/warmup.py` with:

```python
def post_chat(
    content: str,
    max_tokens: int,
    *,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    presence_penalty: float | None = None,
) -> tuple[float, str]:
    payload: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload_resp = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    text = payload_resp["choices"][0]["message"]["content"]
    return elapsed, text
```

The signature is backwards-compatible: existing callers pass `(content, max_tokens)` positionally, sampling kwargs default to greedy (`temperature=0.0`, others None / omitted).

- [ ] **Step 2: Add `make_prompt` helper after `post_chat`**

Insert after the `post_chat` function:

```python
def make_prompt(approx_tokens: int) -> str:
    """Build a prompt of approximately ``approx_tokens`` tokens."""
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    n = max(1, int(approx_tokens * 3.6 / len(filler)) + 1)
    body = (filler * n)[: int(approx_tokens * 3.6)]
    return f"{body}\n\nReply with the single word: ok."
```

This is a verbatim copy of the helper in `scripts/warmup_diag.py`. The duplication is intentional — the two scripts are independent and we don't want to introduce a shared-helpers module for two callers.

- [ ] **Step 3: Add `run_concurrent` helper after `make_prompt`**

Insert after the `make_prompt` function:

```python
def run_concurrent(
    prompts_and_kwargs: list[tuple[str, dict]],
) -> list[tuple[float, str]]:
    """Submit each (prompt, sampling_kwargs) pair concurrently and gather results.

    Each kwargs dict must include a ``max_tokens`` key; the rest are passed
    through to ``post_chat`` as sampling kwargs. Input dicts are not mutated.
    Returns results in submission order (not completion order).
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prompts_and_kwargs)
    ) as ex:
        futures = []
        for prompt, kwargs in prompts_and_kwargs:
            kw = dict(kwargs)
            max_tokens = kw.pop("max_tokens")
            futures.append(ex.submit(post_chat, prompt, max_tokens, **kw))
        return [f.result() for f in futures]
```

- [ ] **Step 4: Add `concurrent.futures` import**

In the import block at the top, add `import concurrent.futures` alongside the existing imports. The full import block becomes:

```python
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
```

- [ ] **Step 5: Verify script still parses and runs unchanged**

```bash
python -c "import ast; ast.parse(open('scripts/warmup.py').read())" && echo OK
```

Expected: `OK`

Then with the server running:

```bash
make warmup
```

Expected: same output as before (existing 2-prompt warmup), exit 0. Verifies the `post_chat` refactor is backwards-compatible.

- [ ] **Step 6: Commit**

```bash
git add scripts/warmup.py
git commit -m "Refactor warmup.py helpers: sampling-aware post_chat, make_prompt, run_concurrent"
```

---

### Task 2: Replace main with phase A (prefill sweep)

Switch `main()` to a phase-based structure and implement Phase A. After this task, `make warmup` runs the 8-step prefill sweep instead of the old 2-prompt warmup.

**Files:**
- Modify: `scripts/warmup.py`

- [ ] **Step 1: Add `phase_prefill_sweep` function**

Insert after the helper functions, before `main()`:

```python
PREFILL_BIN_TARGETS: list[tuple[str, int]] = [
    ("A.1 prefill ~16 tokens (M-bin <=32)",      16),
    ("A.2 prefill ~72 tokens (M-bin 33-96)",     72),
    ("A.3 prefill ~120 tokens (M-bin 97-128)",  120),
    ("A.4 prefill ~400 tokens (M-bin 129-512)", 400),
    ("A.5 prefill ~2000 tokens (M-bin >512 sm)", 2_000),
    ("A.6 prefill ~8000 tokens (M-bin >512 md)", 8_000),
    ("A.7 prefill ~14000 tokens (M-bin >512 lg)", 14_000),
    ("A.8 prefill ~20000 tokens (long sustained)", 20_000),
]


def phase_prefill_sweep() -> None:
    print("phase A: prefill M-bin sweep")
    for label, n in PREFILL_BIN_TARGETS:
        print(f"  [{label}] ...", end=" ", flush=True)
        elapsed, text = post_chat(make_prompt(n), max_tokens=8)
        print(f"ok ({elapsed:.1f}s) -> {text.strip()[:48]!r}")
```

- [ ] **Step 2: Replace `main()` body with new structure**

Replace the existing `main()` (currently lines ~58–82) with:

```python
def main() -> int:
    print(f"warming up {BASE_URL} (model={MODEL})", flush=True)
    t_start = time.monotonic()
    try:
        phase_prefill_sweep()
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    total = time.monotonic() - t_start
    print(f"warmup complete (total: {total:.1f}s, steps: 8)")
    return 0
```

- [ ] **Step 3: Update the module docstring**

Replace the existing docstring (lines 2–17) with:

```python
"""Warm up Triton/JIT kernels on the running vLLM server.

Drives the locally-running vLLM with a four-phase request schedule that
pre-JITs every Triton kernel real Claude Code / Codex traffic against the
configured DGX Spark + Gemma-4 26B + DFlash stack will plausibly hit:

    Phase A — prefill M-bin sweep (greedy, batch=1)
    Phase B — sampling sweep (non-greedy, batch=1)
    Phase C — concurrent decode (batch=2 and batch=3)
    Phase D — sustained decode (long generation, non-greedy)

The first cold run JITs each Triton kernel for the shape/dtype/sampling
config it sees and writes the result to ~/.triton/cache, so subsequent
server starts on the same machine reuse the compiled artifacts and don't
need this script.

Reads VLLM_BASE_URL, VLLM_API_KEY, and SERVED_MODEL_NAME from env with
the same defaults the Makefile uses.
"""
```

- [ ] **Step 4: Run against live server and verify**

With `make serve` running in another shell:

```bash
make warmup
```

Expected output (rough shape; timings will vary):

```
warming up http://localhost:8000 (model=vllm-model)
phase A: prefill M-bin sweep
  [A.1 prefill ~16 tokens (M-bin <=32)] ... ok (1.2s) -> 'ok'
  [A.2 prefill ~72 tokens (M-bin 33-96)] ... ok (1.4s) -> 'ok'
  [A.3 prefill ~120 tokens (M-bin 97-128)] ... ok (1.5s) -> 'ok'
  [A.4 prefill ~400 tokens (M-bin 129-512)] ... ok (1.8s) -> 'ok'
  [A.5 prefill ~2000 tokens (M-bin >512 sm)] ... ok (3.0s) -> 'ok'
  [A.6 prefill ~8000 tokens (M-bin >512 md)] ... ok (6.0s) -> 'ok'
  [A.7 prefill ~14000 tokens (M-bin >512 lg)] ... ok (10.0s) -> 'ok'
  [A.8 prefill ~20000 tokens (long sustained)] ... ok (14.0s) -> 'ok'
warmup complete (total: 39.0s, steps: 8)
```

Expected exit code: 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/warmup.py
git commit -m "Replace warmup.py main with phase A: prefill M-bin sweep"
```

---

### Task 3: Add phase B (sampling sweep)

Add the sampling phase and wire it into `main()`. After this task, `make warmup` runs phases A + B = 11 steps.

**Files:**
- Modify: `scripts/warmup.py`

- [ ] **Step 1: Add `phase_sampling_sweep` function**

Insert after `phase_prefill_sweep`:

```python
SAMPLING_CONFIGS: list[tuple[str, dict]] = [
    (
        "B.1 sampling temp=0.7 top_p=0.95",
        {"temperature": 0.7, "top_p": 0.95, "max_tokens": 64},
    ),
    (
        "B.2 sampling temp=1.0 top_k=50",
        {"temperature": 1.0, "top_k": 50, "max_tokens": 64},
    ),
    (
        "B.3 sampling temp=0.8 top_p=0.9 presence=0.5",
        {
            "temperature": 0.8,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "max_tokens": 64,
        },
    ),
]


def phase_sampling_sweep() -> None:
    print("phase B: sampling sweep")
    prompt = make_prompt(120)
    for label, kwargs in SAMPLING_CONFIGS:
        print(f"  [{label}] ...", end=" ", flush=True)
        kwargs_copy = dict(kwargs)
        max_tokens = kwargs_copy.pop("max_tokens")
        elapsed, text = post_chat(prompt, max_tokens=max_tokens, **kwargs_copy)
        print(f"ok ({elapsed:.1f}s) -> {text.strip()[:48]!r}")
```

- [ ] **Step 2: Wire `phase_sampling_sweep` into `main()`**

In `main()`, add the call after `phase_prefill_sweep()` and update the step count:

```python
    try:
        phase_prefill_sweep()
        phase_sampling_sweep()
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    total = time.monotonic() - t_start
    print(f"warmup complete (total: {total:.1f}s, steps: 11)")
```

- [ ] **Step 3: Run against live server and verify**

```bash
make warmup
```

Expected: phase A output (8 steps) followed by phase B output (3 steps), exit 0. Total time should be roughly phase-A-time + 15–30s.

```
phase B: sampling sweep
  [B.1 sampling temp=0.7 top_p=0.95] ... ok (5.2s) -> '...'
  [B.2 sampling temp=1.0 top_k=50] ... ok (4.8s) -> '...'
  [B.3 sampling temp=0.8 top_p=0.9 presence=0.5] ... ok (5.0s) -> '...'
warmup complete (total: 54.0s, steps: 11)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/warmup.py
git commit -m "Add phase B (sampling sweep) to warmup.py"
```

---

### Task 4: Add phase C (concurrent decode)

Add the concurrent batch phase. After this task, `make warmup` runs phases A + B + C = 13 steps.

**Files:**
- Modify: `scripts/warmup.py`

- [ ] **Step 1: Add `phase_concurrent_decode` function**

Insert after `phase_sampling_sweep`:

```python
def phase_concurrent_decode() -> None:
    print("phase C: concurrent decode")

    print("  [C.1 batch=2 concurrent (~120 tokens, temp=0.7 top_p=0.95)] ...",
          end=" ", flush=True)
    prompt_120 = make_prompt(120)
    t0 = time.monotonic()
    results = run_concurrent([
        (prompt_120, {"temperature": 0.7, "top_p": 0.95, "max_tokens": 64}),
        (prompt_120, {"temperature": 0.7, "top_p": 0.95, "max_tokens": 64}),
    ])
    elapsed = time.monotonic() - t0
    summary = " | ".join(text.strip()[:24] for _, text in results)
    print(f"ok ({elapsed:.1f}s) -> {summary!r}")

    print("  [C.2 batch=3 concurrent (~400 tokens, greedy)] ...",
          end=" ", flush=True)
    prompt_400 = make_prompt(400)
    t0 = time.monotonic()
    results = run_concurrent([
        (prompt_400, {"max_tokens": 64}),
        (prompt_400, {"max_tokens": 64}),
        (prompt_400, {"max_tokens": 64}),
    ])
    elapsed = time.monotonic() - t0
    summary = " | ".join(text.strip()[:24] for _, text in results)
    print(f"ok ({elapsed:.1f}s) -> {summary!r}")
```

- [ ] **Step 2: Wire `phase_concurrent_decode` into `main()`**

Update the try block and step count:

```python
    try:
        phase_prefill_sweep()
        phase_sampling_sweep()
        phase_concurrent_decode()
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    total = time.monotonic() - t_start
    print(f"warmup complete (total: {total:.1f}s, steps: 13)")
```

- [ ] **Step 3: Run against live server and verify**

```bash
make warmup
```

Expected: phases A + B + C output, exit 0. Phase C may take 15–30s on a cold cache (it compiles the padded batch>1 spec-decode kernels) and noticeably less on a warm cache.

```
phase C: concurrent decode
  [C.1 batch=2 concurrent (~120 tokens, temp=0.7 top_p=0.95)] ... ok (8.5s) -> '...'
  [C.2 batch=3 concurrent (~400 tokens, greedy)] ... ok (10.2s) -> '...'
warmup complete (total: 73.0s, steps: 13)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/warmup.py
git commit -m "Add phase C (concurrent decode) to warmup.py"
```

---

### Task 5: Add phase D (sustained decode)

Add the final phase. After this task, `make warmup` runs all 4 phases = 14 steps.

**Files:**
- Modify: `scripts/warmup.py`

- [ ] **Step 1: Add `phase_sustained_decode` function**

Insert after `phase_concurrent_decode`:

```python
def phase_sustained_decode() -> None:
    print("phase D: sustained decode")
    print("  [D.1 sustained decode (~120 tokens, temp=0.7 top_p=0.95, max_tokens=256)] ...",
          end=" ", flush=True)
    elapsed, text = post_chat(
        make_prompt(120),
        max_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )
    print(f"ok ({elapsed:.1f}s) -> {text.strip()[:48]!r}")
```

- [ ] **Step 2: Wire `phase_sustained_decode` into `main()`**

Update the try block and step count:

```python
    try:
        phase_prefill_sweep()
        phase_sampling_sweep()
        phase_concurrent_decode()
        phase_sustained_decode()
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    total = time.monotonic() - t_start
    print(f"warmup complete (total: {total:.1f}s, steps: 14)")
```

- [ ] **Step 3: Run against live server and verify**

```bash
make warmup
```

Expected: phases A + B + C + D output, exit 0. Total wall-clock should land in the 3–5 minute range on a cold Triton cache; substantially less on a warm one.

```
phase D: sustained decode
  [D.1 sustained decode (~120 tokens, temp=0.7 top_p=0.95, max_tokens=256)] ... ok (15.8s) -> '...'
warmup complete (total: 89.0s, steps: 14)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/warmup.py
git commit -m "Add phase D (sustained decode) to warmup.py"
```

---

### Task 6: Update README

Update the "First-run warmup" paragraph to reflect the new coverage and runtime.

**Files:**
- Modify: `README.md` (around line 60, the "First-run warmup" paragraph)

- [ ] **Step 1: Replace the existing paragraph**

Current text in `README.md` after the `make warmup` code block:

```
This sends a short and a ~5K-token prompt at the running server to JIT-compile Triton kernels that vLLM's internal startup warmup misses (DFlash spec-decoding kernels, longer-context prefill attention). One-time cost is ~1–2 minutes; the compiled kernels land in `~/.triton/cache`, so subsequent `make serve` boots on the same machine reuse them and don't need this step.
```

Replace with:

```
This drives the running server through four phases — a prefill M-bin sweep, a sampling sweep (greedy, top-p, top-k, presence-penalty), a small-batch concurrent decode (batch=2 and batch=3), and a sustained decode (max_tokens=256) — to JIT-compile every Triton kernel that real Claude Code / Codex traffic on this stack plausibly hits, including the DFlash spec-decoding and `_topk_topp_kernel` paths that vLLM's internal startup warmup misses. One-time cost is ~3–5 minutes; the compiled kernels land in `~/.triton/cache`, so subsequent `make serve` boots on the same machine reuse them and don't need this step.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Update README: warmup runtime and four-phase coverage"
```

---

### Task 7: End-to-end verification (manual)

This is a verification procedure, not a code change. Confirm the new warmup actually silences the seven JIT warnings in a from-cold scenario.

**Files:** None modified. Inspection only.

- [ ] **Step 1: Stop the running server**

In the `make serve` shell, Ctrl+C and wait for graceful shutdown.

- [ ] **Step 2: Clear the Triton cache**

```bash
rm -rf ~/.triton/cache
```

This forces every kernel to JIT-compile fresh. (If `TRITON_CACHE_DIR` is set in the environment, clear that path instead.)

- [ ] **Step 3: Start a fresh server and capture its log**

In one shell:

```bash
make serve 2>&1 | tee /tmp/vllm-server.log
```

Wait for the line `Application startup complete`.

- [ ] **Step 4: Run warmup**

In a second shell:

```bash
time make warmup
```

Expected: exit 0. Wall-clock from `time` should land in the ~3–5 min range.

- [ ] **Step 5: Drive one Claude Code turn**

In a third shell:

```bash
make cc
```

Send one prompt that exercises ~100 decode tokens, e.g. `Write a Python function that returns the Fibonacci sequence up to n.` Wait for the reply, then exit Claude Code.

- [ ] **Step 6: Inspect the server log for residual JIT warnings**

```bash
grep 'Triton kernel JIT compilation during inference' /tmp/vllm-server.log
```

Expected: any matches must come from log lines emitted **before** the "warmup complete" message in the `make warmup` output. Cross-reference timestamps:
- Get the warmup completion timestamp from the second shell's `time make warmup` output.
- Confirm every JIT warning in the server log has a timestamp earlier than that completion (i.e. the warning fired *during* warmup, not during the `make cc` turn).

Pass criterion: zero `Triton kernel JIT compilation during inference: …` lines in the server log with a timestamp **after** warmup completed.

- [ ] **Step 7: If a kernel still triggers post-warmup, document and stop**

If a JIT warning fires during the `make cc` turn for a kernel not yet covered:

1. Capture the kernel name and the request that triggered it.
2. Open a follow-up issue or note — do NOT modify the plan in this iteration. The verification has revealed a coverage gap; addressing it is a separate planning cycle.

If the verification passes, the implementation is complete.
