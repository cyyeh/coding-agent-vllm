#!/usr/bin/env python3
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


def main() -> int:
    print(f"warming up {BASE_URL} (model={MODEL})", flush=True)
    t_start = time.monotonic()
    try:
        phase_prefill_sweep()
        phase_sampling_sweep()
        phase_concurrent_decode()
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}")
        return 1
    total = time.monotonic() - t_start
    print(f"warmup complete (total: {total:.1f}s, steps: 13)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
