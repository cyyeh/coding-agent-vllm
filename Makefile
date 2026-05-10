.PHONY: help venv install patch serve serve-no-spec warmup warmup-diag chat cc codex bench bench-sharegpt
.DELETE_ON_ERROR:

ifneq (,$(wildcard .env))
    include .env
    export HF_TOKEN
endif

VLLM ?= .venv/bin/vllm
PY ?= .venv/bin/python
HF_MODEL ?= google/gemma-4-26B-A4B-it
SERVED_MODEL_NAME ?= vllm-model
TOOL_CALL_PARSER ?= gemma4
REASONING_PARSER ?= gemma4
VLLM_BASE_URL ?= http://localhost:8000
VLLM_API_KEY ?= dummy
VLLM_PIN ?= 0.20.2rc1.dev140
VLLM_INDEX_URL ?= https://wheels.vllm.ai/nightly
LMCACHE_PIN ?= 0.4.4
CUDA_HOME ?= /usr/local/cuda
comma := ,
BENCH_NUM_PROMPTS ?= 200
BENCH_REQUEST_RATE ?= inf
BENCH_RANDOM_INPUT_LEN ?= 1024
BENCH_RANDOM_OUTPUT_LEN ?= 256
SHAREGPT_PATH ?= data/ShareGPT_V3_unfiltered_cleaned_split.json
SHAREGPT_URL ?= https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
LMCACHE ?= 0
LMCACHE_MAX_LOCAL_CPU_SIZE ?= 20
LMCACHE_DISK_PATH ?= $(CURDIR)/data/lmcache
LMCACHE_MAX_LOCAL_DISK_SIZE ?= 50
LMCACHE_ENV = $(if $(filter 1,$(LMCACHE)),\
	LMCACHE_LOCAL_CPU=true \
	LMCACHE_MAX_LOCAL_CPU_SIZE=$(LMCACHE_MAX_LOCAL_CPU_SIZE) \
	LMCACHE_LOCAL_DISK=file://$(LMCACHE_DISK_PATH) \
	LMCACHE_MAX_LOCAL_DISK_SIZE=$(LMCACHE_MAX_LOCAL_DISK_SIZE) \
	LMCACHE_CHUNK_SIZE=256,)

LMCACHE_FLAGS = $(if $(filter 1,$(LMCACHE)),--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1"$(comma)"kv_role":"kv_both"}',)

help:
	@echo "Targets:"
	@echo "  venv         Create the .venv with Python 3.12"
	@echo "  install      Install pinned vLLM nightly + lmcache and apply DGX Spark patches"
	@echo "  patch        Apply DGX Spark / DFlash patches to the installed vLLM (idempotent)"
	@echo "  serve        Launch vLLM serving Gemma 4 26B with DFlash on DGX Spark (LMCACHE=1 for KV-cache offload)"
	@echo "  serve-no-spec Launch vLLM serving Gemma 4 26B without speculative-config"
	@echo "  warmup       JIT-compile Triton kernels by hitting the running server (run once after first serve)"
	@echo "  warmup-diag  Like warmup but covers M-bin sweep + batch=2 and reports each new Triton compile"
	@echo "  chat         Open an interactive chat session against the local vLLM server"
	@echo "  cc           Launch Claude Code routed to the local vLLM server"
	@echo "  codex        Launch Codex routed to the local vLLM server"
	@echo "  bench        Benchmark the running server with vllm bench serve (random dataset)"
	@echo "  bench-sharegpt Benchmark the running server with vllm bench serve (ShareGPT V3, downloaded on first run)"

venv:
	uv venv

install:
	uv pip install -U --pre --torch-backend=auto \
		--extra-index-url $(VLLM_INDEX_URL) \
		"vllm==$(VLLM_PIN)"
	CUDA_HOME=$(CUDA_HOME) uv pip install --no-build-isolation-package lmcache "lmcache==$(LMCACHE_PIN)"
	# lmcache pulls nixl-cu12 transitively, but this host is CUDA 13. Swap to the cu13
	# wheel so vLLM's optional `import nixl_ep` doesn't crash on a missing libcudart.so.12.
	uv pip uninstall nixl-cu12 || true
	uv pip install nixl-cu13
	$(MAKE) patch

patch:
	$(PY) scripts/patch_vllm_gb10_gemma4_dflash_runtime.py

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

bench:
	$(VLLM) bench serve \
		--backend vllm \
		--model $(HF_MODEL) \
		--served-model-name $(SERVED_MODEL_NAME) \
		--base-url $(VLLM_BASE_URL) \
		--endpoint /v1/completions \
		--dataset-name random \
		--random-input-len $(BENCH_RANDOM_INPUT_LEN) \
		--random-output-len $(BENCH_RANDOM_OUTPUT_LEN) \
		--num-prompts $(BENCH_NUM_PROMPTS) \
		--request-rate $(BENCH_REQUEST_RATE)

$(SHAREGPT_PATH):
	mkdir -p $(dir $(SHAREGPT_PATH))
	curl -L --fail -o $@ $(SHAREGPT_URL)

bench-sharegpt: $(SHAREGPT_PATH)
	$(VLLM) bench serve \
		--backend openai-chat \
		--model $(HF_MODEL) \
		--served-model-name $(SERVED_MODEL_NAME) \
		--base-url $(VLLM_BASE_URL) \
		--endpoint /v1/chat/completions \
		--dataset-name sharegpt \
		--dataset-path $(SHAREGPT_PATH) \
		--num-prompts $(BENCH_NUM_PROMPTS) \
		--request-rate $(BENCH_REQUEST_RATE)
