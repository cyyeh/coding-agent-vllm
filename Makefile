.PHONY: help venv install serve cc codex

ifneq (,$(wildcard .env))
    include .env
    export
endif

VLLM ?= .venv/bin/vllm
SERVED_MODEL_NAME ?= vllm-model
TOOL_CALL_PARSER ?= pythonic
VLLM_BASE_URL ?= http://localhost:8000
VLLM_API_KEY ?= dummy

help:
	@echo "Targets:"
	@echo "  venv         Create the .venv with Python 3.12"
	@echo "  install      Install vLLM from PR #41703 (DFlash speculative decoding support)"
	@echo "  serve        Launch vLLM serving Gemma 4 26B with the DFlash draft model"
	@echo "  cc           Launch Claude Code routed to the local vLLM server"
	@echo "  codex        Launch Codex routed to the local vLLM server"

venv:
	uv venv

install:
	uv pip install -U --torch-backend=auto \
		"vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/41703/head"

serve:
	$(VLLM) serve google/gemma-4-26B-A4B-it \
		--served-model-name $(SERVED_MODEL_NAME) \
		--enable-auto-tool-choice \
		--tool-call-parser $(TOOL_CALL_PARSER) \
		--speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
		--attention-backend triton_attn \
		--max-num-batched-tokens 32768 \
		--trust-remote-code

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
