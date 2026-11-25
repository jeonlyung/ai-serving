"""모델 가중치 사전 다운로드 — 첫 기동 때 수십 GB 를 받느라 헬스체크가 밀리는 걸 막는다."""

from __future__ import annotations

import os

from .catalog import Catalog


def pull_models(cat: Catalog, dest: str | None = None, only: list[str] | None = None) -> list[str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise SystemExit("huggingface_hub 가 필요합니다:  pip install 'ai-serving[hf]'") from e
    done = []
    for m in cat.models:
        if only and m.id not in only and not set(m.aliases) & set(only):
            continue
        if m.engine == "mock":
            continue
        path = snapshot_download(
            m.hf_repo,
            cache_dir=dest,
            token=os.environ.get("HF_TOKEN") or None,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.tiktoken",
                "tokenizer*",
            ],
        )
        done.append(f"{m.hf_repo} → {path}")
    return done
