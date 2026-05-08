#!/usr/bin/env python3
"""Warm up Triton/JIT kernels on the running vLLM server.

Sends two chat completions to the locally-running vLLM:
  1. A short prompt (covers DFlash spec-decoding kernels and the
     common decode-path shapes).
  2. A ~5K-token prompt (covers prefill-attention kernels at sizes
     vLLM's internal startup warmup typically misses).

The first cold run JITs each Triton kernel for the shape/dtype it
sees and writes the result to ~/.triton/cache, so subsequent server
starts on the same machine reuse the compiled artifacts and don't
need this script.

Reads VLLM_BASE_URL, VLLM_API_KEY, and SERVED_MODEL_NAME from env
with the same defaults the Makefile uses.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
MODEL = os.environ.get("SERVED_MODEL_NAME", "vllm-model")


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


def make_prompt(approx_tokens: int) -> str:
    """Build a prompt of approximately ``approx_tokens`` tokens."""
    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit "
    n = max(1, int(approx_tokens * 3.6 / len(filler)) + 1)
    body = (filler * n)[: int(approx_tokens * 3.6)]
    return f"{body}\n\nReply with the single word: ok."


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


def main() -> int:
    print(f"warming up {BASE_URL} (model={MODEL})", flush=True)

    print("  1/2 short prompt ...", end=" ", flush=True)
    try:
        elapsed, text = post_chat("Say hi in one word.", max_tokens=16)
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"ok ({elapsed:.1f}s) -> {text.strip()!r}")

    filler = "lorem ipsum dolor sit amet consectetur adipiscing elit. " * 400
    long_prompt = f"{filler}\n\nIn five words, summarize the above."
    approx_tokens = len(long_prompt) // 4

    print(f"  2/2 long prompt (~{approx_tokens} tokens) ...", end=" ", flush=True)
    try:
        elapsed, text = post_chat(long_prompt, max_tokens=32)
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"ok ({elapsed:.1f}s) -> {text.strip()[:60]!r}")

    print("warmup complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
