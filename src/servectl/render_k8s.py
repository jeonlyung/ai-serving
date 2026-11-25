"""Kubernetes 렌더러 — Kustomize 로 바로 적용 가능한 매니페스트를 만든다.

모델 1종 = Deployment + Service (+ PDB, + KEDA ScaledObject).
GPU 는 비싸고 모자라므로 롤링 업데이트는 "먼저 내리고 올린다"(maxSurge 0).
모델 로딩이 길어 startupProbe 를 따로 두고, 그동안 라우터는 헬스체크로 트래픽을 보내지 않는다.
"""

from __future__ import annotations

from typing import Any

import yaml

from . import render_common as rc
from .catalog import Catalog, Model, slug
from .engines import HF_CACHE, container_args, container_env, is_gpu

LABEL = "app.kubernetes.io"


def _labels(name: str, component: str) -> dict[str, str]:
    return {
        f"{LABEL}/name": name,
        f"{LABEL}/component": component,
        f"{LABEL}/part-of": "ai-serving",
    }


def _probe_http(path: str, port: int) -> dict[str, Any]:
    return {"httpGet": {"path": path, "port": port}}


def _env(m: Model) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k, v in container_env(m).items():
        if k == "HF_TOKEN":
            out.append(
                {
                    "name": k,
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "ai-serving-secrets",
                            "key": "HF_TOKEN",
                            "optional": True,
                        }
                    },
                }
            )
        else:
            out.append({"name": k, "value": v})
    return out


def _engine_workload(m: Model, cat: Catalog, storage_class: str | None) -> list[dict[str, Any]]:
    spec = cat.engines[m.engine]
    name = slug(m.id)
    port = spec.port
    gpu = is_gpu(m)
    args = container_args(m, cat)
    container: dict[str, Any] = {
        "name": "engine",
        "image": spec.image,
        "ports": [{"name": "http", "containerPort": port}],
        "env": _env(m),
        "startupProbe": {
            **_probe_http(spec.health_path, port),
            "periodSeconds": 10,
            "failureThreshold": 180,
        },  # 최대 30분 (대형 모델 로딩)
        "readinessProbe": {
            **_probe_http(spec.health_path, port),
            "periodSeconds": 10,
            "failureThreshold": 3,
        },
        "livenessProbe": {
            **_probe_http(spec.health_path, port),
            "periodSeconds": 20,
            "failureThreshold": 6,
            "timeoutSeconds": 10,
        },
        "volumeMounts": [
            {"name": "hf-cache", "mountPath": HF_CACHE},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
    }
    if m.engine == "sglang":
        container["command"] = args[:3]
        container["args"] = args[3:]
    elif m.engine == "vllm":
        container["args"] = args  # 이미지 ENTRYPOINT(vllm serve)에 인자만 전달
    else:
        container["args"] = args
    if gpu:
        container["resources"] = {"limits": {"nvidia.com/gpu": m.gpu.count}}

    pod_spec: dict[str, Any] = {
        "containers": [container],
        "volumes": [
            {"name": "hf-cache", "persistentVolumeClaim": {"claimName": "hf-cache"}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        ],
        "terminationGracePeriodSeconds": 60,
    }
    if gpu:
        pod_spec["tolerations"] = [
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ]
        if m.gpu.node_selector:
            pod_spec["nodeSelector"] = m.gpu.node_selector

    selector = {f"{LABEL}/name": name}
    deploy = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "labels": _labels(name, "engine")},
        "spec": {
            "replicas": m.replicas,
            "selector": {"matchLabels": selector},
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1},
            },
            "template": {
                "metadata": {
                    "labels": _labels(name, "engine") | {"model": name},
                    "annotations": {
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": str(port),
                        "prometheus.io/path": spec.metrics_path,
                    },
                },
                "spec": pod_spec,
            },
        },
    }
    svc = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "labels": _labels(name, "engine") | {"model": name, "ai-serving/model-id": slug(m.id)},
        },
        "spec": {
            "selector": selector,
            "ports": [{"name": "http", "port": port, "targetPort": "http"}],
        },
    }
    objs: list[dict[str, Any]] = [deploy, svc]
    if m.replicas > 1 or m.autoscale:
        objs.append(
            {
                "apiVersion": "policy/v1",
                "kind": "PodDisruptionBudget",
                "metadata": {"name": name, "labels": _labels(name, "engine")},
                "spec": {"minAvailable": 1, "selector": {"matchLabels": selector}},
            }
        )
    if m.autoscale:
        a = m.autoscale
        objs.append(
            {
                "apiVersion": "keda.sh/v1alpha1",
                "kind": "ScaledObject",
                "metadata": {"name": name, "labels": _labels(name, "autoscaler")},
                "spec": {
                    "scaleTargetRef": {"name": name},
                    "minReplicaCount": a.min,
                    "maxReplicaCount": a.max,
                    "cooldownPeriod": 600,  # GPU 모델 재기동 비용이 커서 천천히 줄인다
                    "triggers": [
                        {
                            "type": "prometheus",
                            "metadata": {
                                "serverAddress": "http://prometheus-operated.monitoring.svc:9090",
                                "query": f'sum(vllm:num_requests_waiting{{model_name="{m.id}"}})',
                                "threshold": str(a.waiting_threshold),
                            },
                        }
                    ],
                },
            }
        )
    return objs


def render_k8s(
    cat: Catalog,
    *,
    namespace: str = "ai-serving",
    router_replicas: int = 2,
    storage_class: str | None = None,
    cache_size: str = "500Gi",
    router_image: str = "ghcr.io/your-org/ai-serving:0.1.0",
) -> dict[str, str]:
    files: dict[str, str] = {}
    resources: list[str] = ["namespace.yaml", "pvc.yaml", "router.yaml", "monitoring.yaml"]

    files["namespace.yaml"] = rc.dump(
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}}
    )

    pvc_spec: dict[str, Any] = {
        "accessModes": ["ReadWriteMany"],
        "resources": {"requests": {"storage": cache_size}},
    }
    if storage_class:
        pvc_spec["storageClassName"] = storage_class
    files["pvc.yaml"] = rc.dump(
        {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "hf-cache", "labels": _labels("hf-cache", "storage")},
            "spec": pvc_spec,
        }
    )

    for m in cat.models:
        fname = f"engine-{slug(m.id)}.yaml"
        files[fname] = rc.dump_all(_engine_workload(m, cat, storage_class))
        resources.append(fname)

    # 라우터 ----------------------------------------------------------
    rcfg = rc.router_config(
        cat,
        lambda m, n: f"http://{slug(m.id)}.{namespace}.svc:{cat.engines[m.engine].port}",
        per_replica=False,
    )
    # k8s 는 Service 가 복제본을 이미 분산하므로 업스트림은 모델당 1개 (Service DNS)
    router_dep = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "router", "labels": _labels("router", "router")},
        "spec": {
            "replicas": router_replicas,
            "selector": {"matchLabels": {f"{LABEL}/name": "router"}},
            "template": {
                "metadata": {"labels": _labels("router", "router")},
                "spec": {
                    "containers": [
                        {
                            "name": "router",
                            "image": router_image,
                            "command": [
                                "uvicorn",
                                "servectl.router.app:app_from_env",
                                "--factory",
                                "--host",
                                "0.0.0.0",
                                "--port",
                                "8000",
                                "--no-access-log",
                            ],
                            "env": [
                                {"name": "ROUTER_CONFIG", "value": "/etc/router/router.yaml"},
                                {
                                    "name": "ROUTER_API_KEYS",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "ai-serving-secrets",
                                            "key": "ROUTER_API_KEYS",
                                            "optional": True,
                                        }
                                    },
                                },
                            ],
                            "ports": [{"name": "http", "containerPort": 8000}],
                            "readinessProbe": _probe_http("/readyz", 8000) | {"periodSeconds": 5},
                            "livenessProbe": _probe_http("/healthz", 8000) | {"periodSeconds": 10},
                            "resources": {
                                "requests": {"cpu": "250m", "memory": "256Mi"},
                                "limits": {"memory": "512Mi"},
                            },
                            "volumeMounts": [{"name": "cfg", "mountPath": "/etc/router"}],
                        }
                    ],
                    "volumes": [{"name": "cfg", "configMap": {"name": "router-config"}}],
                },
            },
        },
    }
    router_objs = [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "router-config"},
            "data": {"router.yaml": rc.dump(rcfg)},
        },
        router_dep,
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "router", "labels": _labels("router", "router")},
            "spec": {
                "selector": {f"{LABEL}/name": "router"},
                "ports": [{"name": "http", "port": 80, "targetPort": "http"}],
            },
        },
        {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {"name": "router"},
            "spec": {"minAvailable": 1, "selector": {"matchLabels": {f"{LABEL}/name": "router"}}},
        },
        {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": "router"},
            "spec": {
                "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": "router"},
                "minReplicas": router_replicas,
                "maxReplicas": max(router_replicas * 3, 4),
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": 70},
                        },
                    }
                ],
            },
        },
    ]
    files["router.yaml"] = rc.dump_all(router_objs)

    # 모니터링 (Prometheus Operator CRD) ---------------------------------
    alerts = yaml.safe_load(rc.asset("alerts.yml"))
    mon = [
        {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PodMonitor",
            "metadata": {"name": "engines", "labels": {"release": "prometheus"}},
            "spec": {
                "selector": {"matchLabels": {f"{LABEL}/component": "engine"}},
                "podTargetLabels": ["model"],
                "podMetricsEndpoints": [
                    {"port": "http", "path": p, "interval": "15s"}
                    for p in sorted({s.metrics_path for s in cat.engines.values()})
                ],
            },
        },
        {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {"name": "router", "labels": {"release": "prometheus"}},
            "spec": {
                "selector": {"matchLabels": {f"{LABEL}/name": "router"}},
                "endpoints": [{"port": "http", "path": "/metrics", "interval": "15s"}],
            },
        },
        {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PrometheusRule",
            "metadata": {"name": "ai-serving", "labels": {"release": "prometheus"}},
            "spec": alerts,
        },
    ]
    files["monitoring.yaml"] = rc.dump_all(mon)

    files["secret.example.yaml"] = rc.dump(
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "ai-serving-secrets", "namespace": namespace},
            "type": "Opaque",
            "stringData": {"HF_TOKEN": "hf_xxx", "ROUTER_API_KEYS": "key1,key2"},
        }
    )
    files["kustomization.yaml"] = rc.dump(
        {
            "apiVersion": "kustomize.config.k8s.io/v1beta1",
            "kind": "Kustomization",
            "namespace": namespace,
            "resources": resources,
            "labels": [{"pairs": {f"{LABEL}/managed-by": "servectl"}, "includeSelectors": False}],
        }
    )
    return files
