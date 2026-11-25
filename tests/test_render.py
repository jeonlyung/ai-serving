from pathlib import Path

import yaml

from servectl.catalog import load_catalog
from servectl.render_compose import render_compose
from servectl.render_k8s import render_k8s

ROOT = Path(__file__).resolve().parent.parent


def test_compose_gpu_devices_never_overlap(catalog):
    dc = yaml.safe_load(render_compose(catalog)["docker-compose.yml"])
    seen: list[str] = []
    for svc in dc["services"].values():
        for d in (
            svc.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
        ):
            if "device_ids" in d:
                seen += d["device_ids"]
    # 27b(2) + 4b(1×2) + qwen(1) = 5장, 중복 배정 없음
    assert sorted(seen) == ["0", "1", "2", "3", "4"]


def test_compose_vllm_flags(catalog):
    dc = yaml.safe_load(render_compose(catalog)["docker-compose.yml"])
    cmd = dc["services"]["qwen-qwen3-32b"]["command"]
    assert cmd[0] == "Qwen/Qwen3-32B"
    for flag in ("--enable-prefix-caching", "--enable-auto-tool-choice", "--reasoning-parser"):
        assert flag in cmd
    assert cmd[cmd.index("--quantization") + 1] == "fp8"
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3-32B"
    assert "agent" in cmd  # 별칭도 served-model-name 에 포함


def test_router_yaml_matches_replicas_and_aliases(catalog):
    files = render_compose(catalog)
    routes = {r["id"]: r for r in yaml.safe_load(files["router.yaml"])["routes"]}
    r = routes["google/gemma-3-4b-it"]
    assert r["upstreams"] == [
        "http://google-gemma-3-4b-it-1:8000",
        "http://google-gemma-3-4b-it-2:8000",
    ]
    assert routes["google/gemma-3-27b-it"]["fallback"] == "google/gemma-3-4b-it"
    assert "answer" in routes["google/gemma-3-27b-it"]["aliases"]
    # 라우터가 가리키는 호스트는 전부 compose 서비스로 존재해야 한다
    svcs = yaml.safe_load(files["docker-compose.yml"])["services"]
    for route in routes.values():
        for u in route["upstreams"]:
            assert u.split("//")[1].split(":")[0] in svcs


def test_prometheus_scrapes_every_replica(catalog):
    prom = yaml.safe_load(render_compose(catalog)["prometheus.yml"])
    targets = {
        t
        for j in prom["scrape_configs"]
        if j["job_name"] == "engines"
        for sc in j["static_configs"]
        for t in sc["targets"]
    }
    assert "google-gemma-3-4b-it-2:8000" in targets and "qwen-qwen3-32b:8000" in targets
    assert any(j["job_name"] == "dcgm" for j in prom["scrape_configs"])


def test_mock_stack_has_no_gpu_requirements(catalog):
    files = render_compose(catalog, mock=True)
    dc = yaml.safe_load(files["docker-compose.yml"])
    assert "nvidia" not in files["docker-compose.yml"]
    assert "dcgm-exporter" not in dc["services"]
    assert dc["services"]["qwen-qwen3-32b"]["command"][:2] == ["servectl", "mock-engine"]


def test_k8s_manifests(catalog):
    files = render_k8s(catalog)
    docs = [d for t in files.values() for d in yaml.safe_load_all(t) if d]
    kinds = [d["kind"] for d in docs]
    for k in (
        "Deployment",
        "Service",
        "PersistentVolumeClaim",
        "ScaledObject",
        "PodDisruptionBudget",
        "PrometheusRule",
        "HorizontalPodAutoscaler",
        "ConfigMap",
    ):
        assert k in kinds, k
    dep = next(
        d
        for d in docs
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "google-gemma-3-27b-it"
    )
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 2
    assert dep["spec"]["strategy"]["rollingUpdate"]["maxSurge"] == 0  # GPU 는 먼저 내리고 올린다
    assert c["startupProbe"]["failureThreshold"] >= 100  # 모델 로딩 대기
    ks = yaml.safe_load(files["kustomization.yaml"])
    assert "secret.example.yaml" not in ks["resources"]  # 예시 시크릿은 적용 대상이 아니다
    for r in ks["resources"]:
        assert r in files


def test_k8s_router_points_at_service_dns(catalog):
    files = render_k8s(catalog, namespace="llm")
    cm = next(d for d in yaml.safe_load_all(files["router.yaml"]) if d["kind"] == "ConfigMap")
    cfg = yaml.safe_load(cm["data"]["router.yaml"])
    assert all("llm.svc" in u for r in cfg["routes"] for u in r["upstreams"])
    assert all(len(r["upstreams"]) == 1 for r in cfg["routes"])  # 분산은 k8s Service 가 담당


def test_committed_deploy_is_fresh(catalog):
    """deploy/ 산출물이 카탈로그와 어긋나면 실패 — `servectl render all` 을 다시 돌린다."""
    for rel, text in render_compose(load_catalog(ROOT / "catalog/models.yaml")).items():
        assert (ROOT / "deploy/compose" / rel).read_text(encoding="utf-8") == text, rel
    for rel, text in render_k8s(catalog).items():
        assert (ROOT / "deploy/k8s" / rel).read_text(encoding="utf-8") == text, rel
