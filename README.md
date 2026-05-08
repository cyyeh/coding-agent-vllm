# coding-agent-vllm

> Run Claude Code or Codex against a self-hosted vLLM backend — Gemma 4 26B accelerated with DFlash block-diffusion speculative decoding.

Local vLLM server for [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) with [`z-lab/gemma-4-26B-A4B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) as the speculative draft model.

DFlash uses block diffusion to draft multiple tokens in parallel, reportedly delivering up to ~3.7x speedup over autoregressive decoding when paired with this verifier.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU with recent CUDA drivers
- Hugging Face account with access granted to `google/gemma-4-26B-A4B-it`

## Setup

```bash
cp .env.example .env       # then fill in HF_TOKEN
make venv
make install
```

`make install` builds vLLM from [PR #41703](https://github.com/vllm-project/vllm/pull/41703), which adds the `dflash` speculative method.

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

| Setting                       | Value          |
| ----------------------------- | -------------- |
| Served model name             | `vllm-model`   |
| Speculative method            | `dflash`       |
| Speculative tokens per step   | 15             |
| Draft attention backend       | `flash_attn`   |
| Target attention backend      | `triton_attn`  |
| Max batched tokens            | 32768          |
| Tool calling                  | auto           |
| Tool-call parser              | `pythonic`     |
| `trust_remote_code`           | enabled        |

Override any of these on the command line, e.g. `make serve TOOL_CALL_PARSER=hermes`, `make cc SERVED_MODEL_NAME=my-model`, or `make codex VLLM_BASE_URL=http://remote:8000`.

## License

[MIT](LICENSE) © 2026 ChihYu Yeh
