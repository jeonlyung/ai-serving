from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    aliases: list[str] = Field(default_factory=list)
    upstreams: list[str]
    fallback: str | None = None
    kind: str = "chat"
    health_path: str = "/health"


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routes: list[Route]
    api_keys_env: str = "ROUTER_API_KEYS"  # 쉼표로 구분. 비면 인증 없음(개발용)
    connect_timeout: float = 5.0
    request_timeout: float = 300.0  # 긴 생성 허용. 스트리밍은 청크 간 간격 기준
    max_body_bytes: int = 8 * 1024 * 1024
    attempts_per_model: int = 2
    eject_after_failures: int = 3
    eject_seconds: float = 15.0
    probe_interval: float = 5.0


def load_router_config(path: str | Path) -> RouterConfig:
    return RouterConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
