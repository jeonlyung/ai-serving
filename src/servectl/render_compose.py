"""docker-compose 렌더러 — 모델 카탈로그 → 실행 가능한 스택(엔진 + 라우터 + 모니터링)."""

from __future__ import annotations

from typing import Any

from . import render_common as rc
from .catalog import Catalog
from .engines import HF_CACHE, container_args, container_env, is_gpu

ROUTER_PORT = 8000


def _healthcheck(engine: str, port: int, path: str) -> dict[str, Any]:
    if engine == "llamacpp":
        test = ["CMD", "curl", "-sf", f"http://localhost:{port}{path}"]
    else:
        test = [
            "CMD",
            "python3",
            "-c",
            f"import urllib.request as u;u.urlopen('http://localhost:{port}{path}',timeout=3)",
        ]
    # 대형 모델은 가중치 로딩에 수 분이 걸린다 — start_period 를 넉넉히
    return {"test": test, "interval": "15s", "timeout": "5s", "retries": 5, "start_period": "600s"}


def render_compose(
    cat: Catalog, *, build_context: str = "../..", mock: bool = False
) -> dict[str, str]:
    if mock:
        cat = rc.with_engine(cat, "mock")
    services: dict[str, Any] = {}
    targets: dict[str, list[str]] = {}
    gpu_cursor = 0
    any_gpu = False

    for m in cat.models:
        spec = cat.engines[m.engine]
        targets[m.id] = []
        for name in rc.replica_names(m):
            args = container_args(m, cat)
            svc: dict[str, Any] = {
                "image": spec.image,
                "restart": "unless-stopped",
                "environment": container_env(m),
                "healthcheck": _healthcheck(m.engine, spec.port, spec.health_path),
                "networks": ["serving"],
            }
            if m.engine == "sglang":
                svc["entrypoint"] = []  # 이미지 기본 entrypoint 를 비우고 python -m 으로 직접 실행
            svc["command"] = args
            if m.engine == "mock":
                svc["pull_policy"] = "never"
                svc["environment"] = {}
            else:
                svc["volumes"] = [f"hf-cache:{HF_CACHE}"]
            if is_gpu(m):
                any_gpu = True
                ids = [str(gpu_cursor + i) for i in range(m.gpu.count)]
                gpu_cursor += m.gpu.count
                svc["ipc"] = "host"  # NCCL/공유메모리 (vLLM 권장)
                svc["deploy"] = {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {"driver": "nvidia", "device_ids": ids, "capabilities": ["gpu"]}
                            ]
                        }
                    }
                }
            services[name] = svc
            targets[m.id].append(f"{name}:{spec.port}")

    router_deps = {n: {"condition": "service_started"} for n in services}
    services["router"] = {
        "build": {"context": build_context, "dockerfile": "Dockerfile"},
        "image": "ai-serving:local",
        "restart": "unless-stopped",
        "command": [
            "uvicorn",
            "servectl.router.app:app_from_env",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            str(ROUTER_PORT),
            "--no-access-log",
        ],
        "environment": {
            "ROUTER_CONFIG": "/etc/router/router.yaml",
            "ROUTER_API_KEYS": "${ROUTER_API_KEYS:-}",
        },
        "volumes": ["./router.yaml:/etc/router/router.yaml:ro"],
        "ports": [f"${{ROUTER_PORT:-{ROUTER_PORT}}}:{ROUTER_PORT}"],
        "depends_on": router_deps,
        "healthcheck": {
            "test": [
                "CMD",
                "python",
                "-c",
                f"import urllib.request as u;u.urlopen('http://localhost:{ROUTER_PORT}/healthz',timeout=3)",
            ],
            "interval": "10s",
            "timeout": "5s",
            "retries": 3,
        },
        "networks": ["serving"],
    }
    services["prometheus"] = {
        "image": "prom/prometheus:v3.5.0",
        "restart": "unless-stopped",
        "command": [
            "--config.file=/etc/prometheus/prometheus.yml",
            "--storage.tsdb.retention.time=15d",
            "--web.enable-lifecycle",
        ],
        "volumes": [
            "./prometheus.yml:/etc/prometheus/prometheus.yml:ro",
            "./alerts.yml:/etc/prometheus/alerts.yml:ro",
            "prom-data:/prometheus",
        ],
        "ports": ["127.0.0.1:9090:9090"],
        "networks": ["serving"],
    }
    services["grafana"] = {
        "image": "grafana/grafana:12.1.0",
        "restart": "unless-stopped",
        "environment": {
            "GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD:-admin}",
            "GF_USERS_ALLOW_SIGN_UP": "false",
        },
        "volumes": [
            "./grafana/provisioning:/etc/grafana/provisioning:ro",
            "./grafana/dashboards:/var/lib/grafana/dashboards:ro",
            "grafana-data:/var/lib/grafana",
        ],
        "ports": ["${GRAFANA_PORT:-3000}:3000"],
        "depends_on": ["prometheus"],
        "networks": ["serving"],
    }
    dcgm = None
    if any_gpu:
        services["dcgm-exporter"] = {
            "image": "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.0-4.5.0-ubuntu22.04",
            "restart": "unless-stopped",
            "cap_add": ["SYS_ADMIN"],
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]
                    }
                }
            },
            "networks": ["serving"],
        }
        dcgm = "dcgm-exporter:9400"

    compose = {
        "name": "ai-serving",
        "services": services,
        "volumes": {"hf-cache": {}, "prom-data": {}, "grafana-data": {}},
        "networks": {"serving": {}},
    }
    header = (
        "# 자동 생성 — 직접 고치지 말고 catalog/models.yaml 을 수정한 뒤 `servectl render compose`\n"
        f"# 모델 {len(cat.models)}종 · GPU {gpu_cursor}장 필요"
        + (" · mock 엔진\n" if mock else "\n")
    )
    return {
        "docker-compose.yml": header + rc.dump(compose),
        "router.yaml": rc.dump(
            rc.router_config(cat, lambda m, n: f"http://{n}:{cat.engines[m.engine].port}")
        ),
        "prometheus.yml": rc.prometheus_config(cat, targets, "router:8000", dcgm=dcgm),
        "alerts.yml": rc.asset("alerts.yml"),
        "grafana/provisioning/datasources/datasource.yml": rc.asset("grafana/datasource.yml"),
        "grafana/provisioning/dashboards/dashboards.yml": rc.asset("grafana/dashboards.yml"),
        "grafana/dashboards/overview.json": rc.dashboard_json(),
        ".env.example": (
            "# cp .env.example .env\n"
            "HF_TOKEN=                 # gated 모델(Gemma/Llama) 다운로드용 Hugging Face 토큰\n"
            "ROUTER_API_KEYS=          # 쉼표 구분. 비우면 인증 없음 — 운영에서는 반드시 설정\n"
            "ROUTER_PORT=8000\n"
            "GRAFANA_PASSWORD=admin\n"
            "GRAFANA_PORT=3000\n"
        ),
    }
