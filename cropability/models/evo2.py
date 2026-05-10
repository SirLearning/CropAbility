"""Evo2 模型下载检查、下载与推理运行。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


DEFAULT_EVO2_REPO = "arcinstitute/evo2_7b"


def check_model_downloadable(
    repo_id: str = DEFAULT_EVO2_REPO,
    token: Optional[str] = None,
) -> tuple[bool, str]:
    """检查 Hugging Face 上的模型元数据是否可访问。"""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False, "missing dependency: huggingface-hub"

    try:
        effective_token = token or os.getenv("HF_TOKEN")
        info = HfApi().model_info(repo_id=repo_id, token=effective_token)
        return True, f"model accessible: {info.id}"
    except Exception as exc:
        return False, f"model not accessible: {exc}"


def download_model(
    repo_id: str = DEFAULT_EVO2_REPO,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """下载（或复用缓存）Evo2 模型文件。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("missing dependency: huggingface-hub") from exc

    effective_token = token or os.getenv("HF_TOKEN")
    download_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=effective_token,
        local_dir_use_symlinks=False,
    )
    return Path(download_path)


def run_evo2(
    prompt: str,
    repo_id: str = DEFAULT_EVO2_REPO,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> str:
    """运行 Evo2 文本生成。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency: transformers and torch are required for evo2 run"
        ) from exc

    effective_token = token or os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        token=effective_token,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        token=effective_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        first_param = next(model.parameters())
        model_device = first_param.device
    except StopIteration:
        model_device = torch.device("cpu")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)
