from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from servectl.catalog import load_catalog
from servectl.mockengine import create_app as mock_app
from servectl.router.app import create_app as router_app
from servectl.router.config import RouterConfig

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(ROOT / "catalog" / "models.yaml")


class HostTransport(httpx.AsyncBaseTransport):
    """호스트명별로 다른 ASGI 앱에 연결 — 네트워크 없이 '여러 엔진'을 흉내 낸다."""

    def __init__(self, apps: dict[str, object]):
        self.t = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}
        self.calls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.calls.append(f"{host}{request.url.path}")
        if host not in self.t:
            raise httpx.ConnectError(f"no route to {host}")
        return await self.t[host].handle_async_request(request)


@pytest.fixture
def stack():
    """big(2 복제본, 폴백→small) + small(1 복제본) 엔진과 라우터."""
    apps = {
        "big-1": mock_app("big", tokens_per_sec=5000, ttft_ms=1),
        "big-2": mock_app("big", tokens_per_sec=5000, ttft_ms=1),
        "small-1": mock_app("small", tokens_per_sec=5000, ttft_ms=1),
    }
    transport = HostTransport(apps)
    cfg = RouterConfig.model_validate(
        {
            "probe_interval": 0,
            "eject_after_failures": 2,
            "eject_seconds": 60,
            "routes": [
                {
                    "id": "big",
                    "aliases": ["answer"],
                    "upstreams": ["http://big-1", "http://big-2"],
                    "fallback": "small",
                },
                {"id": "small", "aliases": ["classify"], "upstreams": ["http://small-1"]},
            ],
        }
    )
    http = httpx.AsyncClient(transport=transport)
    app = router_app(cfg, client=http)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router")
    return type("Stack", (), {"client": client, "apps": apps, "transport": transport, "app": app})
