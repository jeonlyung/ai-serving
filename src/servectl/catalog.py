"""모델 카탈로그 — 이 프로젝트의 단일 진실 공급원(single source of truth).

카탈로그(YAML) 한 장에서 docker-compose, Kubernetes 매니페스트, 라우터 설정,
GPU 메모리 산정이 모두 파생된다. 모델을 추가·교체하려면 이 파일만 고친다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Engine = Literal["vllm", "sglang", "llamacpp", "mock"]


class Arch(BaseModel):
    """메모리 산정용 모델 구조. config.json 의 값을 옮겨 적는다."""

    model_config = ConfigDict(extra="forbid")

    params_b: float = Field(gt=0, description="파라미터 수 (십억 단위)")
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)


class Gpu(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "A100-80G"
    count: int = Field(1, ge=0)  # 0 = CPU 전용(llama.cpp)
    memory_gb: float = 80
    node_selector: dict[str, str] = Field(default_factory=dict)  # K8s 노드 라벨


class Autoscale(BaseModel):
    """대기열 기반 자동 확장 (K8s + KEDA). GPU 사용률이 아니라 '밀린 요청 수'로 늘린다."""

    model_config = ConfigDict(extra="forbid")

    min: int = Field(1, ge=0)
    max: int = Field(4, ge=1)
    waiting_threshold: int = Field(8, ge=1, description="복제본당 대기 요청 수가 넘으면 확장")


class EngineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str
    port: int
    metrics_path: str = "/metrics"
    health_path: str = "/health"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="서빙 이름. 클라이언트가 보내는 model 값 (보통 HF repo id)")
    hf: str | None = Field(None, description="HF repo. 비면 id 와 같다")
    aliases: list[str] = Field(default_factory=list, description="용도별 별칭 (answer 등)")
    engine: Engine = "vllm"
    kind: Literal["chat", "embedding"] = "chat"
    replicas: int = Field(1, ge=0)
    gpu: Gpu = Field(default_factory=Gpu)
    tensor_parallel: int = Field(1, ge=1)
    max_model_len: int = Field(8192, ge=256)
    max_num_seqs: int = Field(64, ge=1)
    gpu_memory_utilization: float = Field(0.90, gt=0.1, le=0.99)
    dtype: str = "auto"
    quantization: Literal["fp8", "awq", "gptq", "int8"] | None = None
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    tool_call_parser: str | None = Field(None, description="예: hermes, llama3_json")
    reasoning_parser: str | None = None
    gguf: str | None = Field(None, description="llama.cpp 용 'repo:quant' (예: unsloth/..:Q4_K_M)")
    fallback: str | None = Field(None, description="이 모델이 죽으면 대신 보낼 모델 id")
    arch: Arch | None = None
    autoscale: Autoscale | None = None
    extra_args: list[str] = Field(default_factory=list)

    @property
    def hf_repo(self) -> str:
        return self.hf or self.id

    @model_validator(mode="after")
    def _check(self) -> Model:
        if self.tensor_parallel > max(self.gpu.count, 1) and self.engine in ("vllm", "sglang"):
            raise ValueError(
                f"{self.id}: tensor_parallel({self.tensor_parallel}) 이 gpu.count({self.gpu.count}) 보다 큼"
            )
        if self.engine in ("vllm", "sglang") and self.gpu.count == 0:
            raise ValueError(f"{self.id}: {self.engine} 는 GPU 가 필요합니다 (gpu.count>=1)")
        return self


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    engines: dict[str, EngineSpec]
    models: list[Model]

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        ids = [m.id for m in self.models]
        names = ids + [a for m in self.models for a in m.aliases]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"id/alias 중복: {sorted(dup)}")
        for m in self.models:
            if m.fallback and m.fallback not in ids:
                raise ValueError(f"{m.id}: fallback '{m.fallback}' 가 카탈로그에 없음")
            if m.engine not in self.engines:
                raise ValueError(f"{m.id}: engines 에 '{m.engine}' 정의가 없음")
        return self

    def model(self, ident: str) -> Model:
        for m in self.models:
            if ident == m.id or ident in m.aliases:
                return m
        raise KeyError(ident)


def load_catalog(path: str | Path) -> Catalog:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Catalog.model_validate(data)


def slug(model_id: str) -> str:
    """서비스·리소스 이름으로 쓸 수 있게 정리 (DNS-1123)."""
    out = "".join(c if c.isalnum() else "-" for c in model_id.lower()).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:50].strip("-")
