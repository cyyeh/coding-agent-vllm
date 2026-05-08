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
