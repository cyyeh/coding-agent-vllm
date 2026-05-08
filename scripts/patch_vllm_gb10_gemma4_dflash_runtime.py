#!/usr/bin/env python3
# Vendored from https://github.com/meanaverage/gemma4-dflash-spark-vllm
# @ 76aa933e8ad6d451c153df1433cc39e01b05b807
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
