.PHONY: help venv install patch serve serve-no-spec warmup warmup-diag chat cc codex

ifneq (,$(wildcard .env))
    include .env
    export HF_TOKEN
endif

VLLM ?= .venv/bin/vllm
PY ?= .venv/bin/python
SERVED_MODEL_NAME ?= vllm-model
TOOL_CALL_PARSER ?= gemma4
REASONING_PARSER ?= gemma4
VLLM_BASE_URL ?= http://localhost:8000
VLLM_API_KEY ?= dummy
VLLM_PIN ?= 0.20.2rc1.dev140
VLLM_INDEX_URL ?= https://wheels.vllm.ai/nightly

help:
	@echo "Targets:"
	@echo "  venv         Create the .venv with Python 3.12"
	@echo "  install      Install pinned vLLM nightly and apply DGX Spark patches"
	@echo "  patch        Apply DGX Spark / DFlash patches to the installed vLLM (idempotent)"
	@echo "  serve        Launch vLLM serving Gemma 4 26B with DFlash on DGX Spark"
	@echo "  serve-no-spec Launch vLLM serving Gemma 4 26B without speculative-config"
	@echo "  warmup       JIT-compile Triton kernels by hitting the running server (run once after first serve)"
	@echo "  warmup-diag  Like warmup but covers M-bin sweep + batch=2 and reports each new Triton compile"
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
	PATH=$(CURDIR)/.venv/bin:$$PATH \
	$(VLLM) serve google/gemma-4-26B-A4B-it \
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

serve-no-spec:
	PATH=$(CURDIR)/.venv/bin:$$PATH \
	$(VLLM) serve google/gemma-4-26B-A4B-it \
		--served-model-name $(SERVED_MODEL_NAME) \
		--enable-auto-tool-choice \
		--tool-call-parser $(TOOL_CALL_PARSER) \
		--reasoning-parser $(REASONING_PARSER) \
		--max-model-len 32768 \
		--max-num-batched-tokens 32768 \
		--gpu-memory-utilization 0.80 \
		--limit-mm-per-prompt '{"image":0,"video":0}' \
		--trust-remote-code

warmup:
	$(PY) scripts/warmup.py

warmup-diag:
	$(PY) scripts/warmup_diag.py

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
	CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192 \
	CLAUDE_CODE_MAX_CONTEXT_TOKENS=30720 \
	claude --bare

codex:
	VLLM_API_KEY=$(VLLM_API_KEY) \
	codex \
		-c model=$(SERVED_MODEL_NAME) \
		-c model_provider=vllm \
		-c 'model_catalog_json="$(CURDIR)/codex_catalog.json"' \
		-c 'model_providers.vllm.name="vLLM"' \
		-c 'model_providers.vllm.env_key="VLLM_API_KEY"' \
		-c 'model_providers.vllm.base_url="$(VLLM_BASE_URL)/v1"' \
		-c 'model_providers.vllm.wire_api="responses"'
