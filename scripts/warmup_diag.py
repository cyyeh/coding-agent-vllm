#!/usr/bin/env python3
"""Diagnose Triton kernel JIT compilation during warmup.

Sends a parameterized series of chat completions covering the M-bin
boundaries that ``fused_moe_kernel`` cares about for single-user
(batch 1-2) serving, and snapshots the on-disk Triton cache before
and after each step so every new compile is attributable to one
specific request shape.

Background:
    ``fused_moe_kernel`` is ``@triton.jit`` (no autotune), with its
    ``BLOCK_SIZE_M/N/K``, ``num_warps``, ``num_stages`` chosen by
    ``get_default_config(M, ...)`` in
    ``vllm/model_executor/layers/fused_moe/fused_moe.py``.  M-bins
    yield distinct (BLOCK_*, num_warps, num_stages) tuples and
    therefore distinct Triton compiles:

        M <=   32  -> BLOCK_M=16,  num_warps=4, num_stages=4
        M  33- 96  -> BLOCK_M=32,  num_warps=4, num_stages=3
        M  97-128  -> BLOCK_M=64,  num_warps=4, num_stages=3
        M 129-512  -> BLOCK_M=64,  num_warps=8, num_stages=3
        M  >  512  -> BLOCK_M=128, num_warps=8, num_stages=3

    With DFlash ``num_speculative_tokens=8`` decode steps see
    ``M = batch * 9`` (always in M<=32 for batch 1-2), so the
    interesting variation is *prefill chunk size*.

Reads the same env vars as scripts/warmup.py:
    VLLM_BASE_URL, VLLM_API_KEY, SERVED_MODEL_NAME.

Cache directory is ``$TRITON_CACHE_DIR`` if set, else ~/.triton/cache.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
MODEL = os.environ.get("SERVED_MODEL_NAME", "vllm-model")
CACHE_DIR = Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))


def post_chat(content: str, max_tokens: int) -> tuple[float, str]:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
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
        payload = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    text = payload["choices"][0]["message"]["content"]
    return elapsed, text


# Approx 3.6 chars/token for Gemma-4-style tokenizers on lorem ipsum.
def make_prompt(approx_tokens: int) -> str:
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    n = max(1, int(approx_tokens * 3.6 / len(filler)) + 1)
    body = (filler * n)[: int(approx_tokens * 3.6)]
    return f"{body}\n\nReply with the single word: ok."


def snapshot_cache() -> set[str]:
    if not CACHE_DIR.exists():
        return set()
    return {p.name for p in CACHE_DIR.iterdir() if p.is_dir()}


def describe_entry(dir_hash: str) -> dict:
    """Return {name, num_warps, num_stages, shared} for a cache dir."""
    d = CACHE_DIR / dir_hash
    out = {"hash": dir_hash, "name": "?", "num_warps": "?", "num_stages": "?", "shared": "?"}
    # The kernel's metadata JSON lives alongside compiled artifacts as
    # <kernel_name>.json (not __grp__... which only points at children).
    for p in d.glob("*.json"):
        if p.name.startswith("__grp__"):
            continue
        try:
            meta = json.loads(p.read_text())
        except Exception:
            continue
        out["name"] = meta.get("name", "?")
        out["num_warps"] = meta.get("num_warps", "?")
        out["num_stages"] = meta.get("num_stages", "?")
        out["shared"] = meta.get("shared", "?")
        break
    return out


def report_new(label: str, before: set[str], after: set[str]) -> set[str]:
    new = sorted(after - before)
    if not new:
        print(f"    [cache] no new compiles")
        return set(new)
    print(f"    [cache] {len(new)} new compile(s):")
    for h in new:
        info = describe_entry(h)
        print(
            f"      - {info['name']:<32} "
            f"num_warps={info['num_warps']} "
            f"num_stages={info['num_stages']} "
            f"shared={info['shared']:>6} "
            f"hash={h[:16]}..."
        )
    return set(new)


def step_sequential(label: str, approx_tokens: int, max_tokens: int = 8) -> set[str]:
    print(f"\n[{label}] prefill ~{approx_tokens} tokens, max_tokens={max_tokens}")
    before = snapshot_cache()
    try:
        elapsed, text = post_chat(make_prompt(approx_tokens), max_tokens=max_tokens)
    except urllib.error.URLError as exc:
        print(f"    FAIL: {exc}")
        return set()
    print(f"    ok ({elapsed:.2f}s) -> {text.strip()[:48]!r}")
    after = snapshot_cache()
    return report_new(label, before, after)


def step_concurrent_batch2() -> set[str]:
    print("\n[batch=2] two concurrent ~120-token prompts, max_tokens=64 (overlapping decode)")
    before = snapshot_cache()
    p = make_prompt(120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(post_chat, p, 64) for _ in range(2)]
        results = []
        for f in concurrent.futures.as_completed(futs):
            try:
                results.append(f.result())
            except urllib.error.URLError as exc:
                print(f"    FAIL: {exc}")
                return set()
    for i, (elapsed, text) in enumerate(results):
        print(f"    req{i}: ok ({elapsed:.2f}s) -> {text.strip()[:48]!r}")
    after = snapshot_cache()
    return report_new("batch=2", before, after)


def main() -> int:
    print(f"warmup-diag: server={BASE_URL} model={MODEL} cache={CACHE_DIR}")
    print(f"  cache entries before: {len(snapshot_cache())}")

    initial = snapshot_cache()
    all_new: set[str] = set()

    # Prefill M-bin sweep. Lengths are chosen so the prompt token count
    # (after the chat template adds ~10-20 tokens) lands inside each
    # M-bin used by get_default_config.
    bin_targets = [
        ("prefill bin <=32",     16),
        ("prefill bin 33-96",    72),
        ("prefill bin 97-128",  120),
        ("prefill bin 129-512", 400),
        ("prefill bin >512 (sm)", 2_000),
        ("prefill bin >512 (md)", 8_000),
        ("prefill bin >512 (lg)", 14_000),
    ]
    for label, n in bin_targets:
        all_new |= step_sequential(label, n)

    # Chunked prefill: prompt > max_num_batched_tokens=16384 forces a
    # second prefill chunk of ~3.6K tokens.
    all_new |= step_sequential("chunked prefill (>16384)", 20_000)

    # Concurrent decode (batch=2). M=18 still falls in bin <=32, but
    # this exercises the scheduler/scatter path with batched decode.
    all_new |= step_concurrent_batch2()

    final = snapshot_cache()
    print("\n=== summary ===")
    print(f"  cache entries: {len(initial)} -> {len(final)} (+{len(final - initial)})")

    moe_compiles = []
    for h in sorted(final - initial):
        info = describe_entry(h)
        if "fused_moe" in info["name"]:
            moe_compiles.append(info)
    if moe_compiles:
        print(f"  fused_moe* compiles triggered by warmup: {len(moe_compiles)}")
        seen = set()
        for info in moe_compiles:
            key = (info["name"], info["num_warps"], info["num_stages"])
            if key in seen:
                continue
            seen.add(key)
            print(f"    - {info['name']} num_warps={info['num_warps']} num_stages={info['num_stages']}")
    else:
        print("  no fused_moe* compiles triggered (all shapes already cached)")

    print("\nNext: run a real workload (e.g. one Codex/CC turn) and watch the")
    print("server log for any new 'Triton kernel JIT compilation during")
    print("inference: ...' lines — those identify shapes this script missed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
