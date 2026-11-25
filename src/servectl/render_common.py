from __future__ import annotations

import json
from importlib import resources
from typing import Any

import yaml

from .catalog import Catalog, Model, slug


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # 앵커(&id) 없이 풀어서 쓴다
        return True


def _str_presenter(dumper: yaml.Dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_presenter)


def dump(obj: Any) -> str:
    return yaml.dump(obj, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=120)


def dump_all(objs: list[Any]) -> str:
    return "---\n".join(dump(o) for o in objs)


def asset(name: str) -> str:
    return (resources.files("servectl") / "assets" / name).read_text(encoding="utf-8")


def with_engine(cat: Catalog, engine: str) -> Catalog:
    """모든 모델의 엔진을 바꾼 사본 (mock 스택 / CPU 스택 렌더링용)."""
    models = []
    for m in cat.models:
        upd: dict[str, Any] = {"engine": engine}
        if engine in ("mock", "llamacpp"):
            upd["gpu"] = m.gpu.model_copy(update={"count": 0})
            upd["tensor_parallel"] = 1
            upd["quantization"] = None
            upd["kv_cache_dtype"] = "auto"
        models.append(m.model_copy(update=upd))
    return cat.model_copy(update={"models": models})


def replica_names(m: Model) -> list[str]:
    base = slug(m.id)
    if m.replicas <= 1:
        return [base]
    return [f"{base}-{i}" for i in range(1, m.replicas + 1)]


def router_config(cat: Catalog, upstream_url, *, per_replica: bool = True) -> dict[str, Any]:
    """upstream_url(model, replica_name) -> 라우터가 접속할 주소.

    per_replica=False 면 모델당 업스트림 1개 (k8s: 복제본 분산은 Service 가 한다).
    """
    routes = []
    for m in cat.models:
        r: dict[str, Any] = {
            "id": m.id,
            "upstreams": [
                upstream_url(m, n) for n in (replica_names(m) if per_replica else [slug(m.id)])
            ],
        }
        if m.aliases:
            r["aliases"] = m.aliases
        if m.fallback:
            r["fallback"] = m.fallback
        if m.kind != "chat":
            r["kind"] = m.kind
        routes.append(r)
    return {"routes": routes}


def prometheus_config(
    cat: Catalog, targets: dict[str, list[str]], router_target: str, *, dcgm: str | None
) -> str:
    """targets: 모델 id → ['host:port', ...] (엔진 /metrics)."""
    jobs: list[dict[str, Any]] = []
    metrics_paths: dict[str, list[tuple[str, list[str]]]] = {}
    for m in cat.models:
        path = cat.engines[m.engine].metrics_path
        metrics_paths.setdefault(path, []).append((m.id, targets[m.id]))
    for path, items in metrics_paths.items():
        jobs.append(
            {
                "job_name": "engines",
                "metrics_path": path,
                "static_configs": [{"targets": t, "labels": {"model": mid}} for mid, t in items],
            }
        )
    jobs.append({"job_name": "router", "static_configs": [{"targets": [router_target]}]})
    if dcgm:
        jobs.append({"job_name": "dcgm", "static_configs": [{"targets": [dcgm]}]})
    return dump(
        {
            "global": {"scrape_interval": "15s", "evaluation_interval": "15s"},
            "rule_files": ["/etc/prometheus/alerts.yml"],
            "scrape_configs": jobs,
        }
    )


def dashboard_json() -> str:
    return (
        json.dumps(json.loads(asset("grafana/overview.json")), ensure_ascii=False, indent=1) + "\n"
    )
