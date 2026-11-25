import json

import httpx
import pytest


def chat(model, **kw):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "안녕"}],
        "max_tokens": 8,
        **kw,
    }


async def test_alias_resolves_and_rewrites_model(stack):
    r = await stack.client.post("/v1/chat/completions", json=chat("answer"))
    assert r.status_code == 200
    assert r.headers["x-served-model"] == "big"
    assert r.json()["model"] == "big"  # 엔진은 실제 id 로 요청받는다
    assert r.headers["x-request-id"]


async def test_models_lists_ids_and_aliases(stack):
    ids = {m["id"] for m in (await stack.client.get("/v1/models")).json()["data"]}
    assert ids == {"big", "answer", "small", "classify"}


async def test_unknown_model_404_and_bad_body_400(stack):
    r = await stack.client.post("/v1/chat/completions", json=chat("nope"))
    assert r.status_code == 404 and r.json()["error"]["type"] == "model_not_found"
    r = await stack.client.post("/v1/chat/completions", content=b"not json")
    assert r.status_code == 400


async def test_streaming_passthrough_ends_with_done(stack):
    lines = []
    async with stack.client.stream(
        "POST", "/v1/chat/completions", json=chat("big", stream=True)
    ) as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                lines.append(line[6:])
    assert lines[-1] == "[DONE]"
    assert len(lines) > 3
    assert json.loads(lines[1])["choices"][0]["delta"]["content"]


async def test_load_spreads_across_replicas(stack):
    for _ in range(6):
        await stack.client.post("/v1/chat/completions", json=chat("big"))
    served = {u.url: u.served for u in stack.app.state.pools["big"].upstreams}
    assert all(n >= 2 for n in served.values()), served


async def test_retries_other_replica_on_5xx(stack):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stack.apps["big-1"]), base_url="http://x"
    ) as c:
        await c.post("/admin/chaos", json={"error_rate": 1.0})
    for _ in range(6):
        r = await stack.client.post("/v1/chat/completions", json=chat("big"))
        assert r.status_code == 200 and r.headers["x-served-model"] == "big"
        assert r.headers["x-upstream"] == "http://big-2"


async def test_connection_error_ejects_replica(stack):
    stack.transport.t.pop("big-1")  # 호스트 소멸 → ConnectError
    for _ in range(4):
        r = await stack.client.post("/v1/chat/completions", json=chat("big"))
        assert r.status_code == 200
    ups = {u.url: u for u in stack.app.state.pools["big"].upstreams}
    assert ups["http://big-1"].ejected_until > 0  # 연속 실패 → 제외됨
    calls_before = sum(1 for c in stack.transport.calls if c.startswith("big-1"))
    for _ in range(4):
        await stack.client.post("/v1/chat/completions", json=chat("big"))
    assert sum(1 for c in stack.transport.calls if c.startswith("big-1")) == calls_before


async def test_fallback_model_when_all_replicas_down(stack):
    stack.transport.t.pop("big-1")
    stack.transport.t.pop("big-2")
    r = await stack.client.post("/v1/chat/completions", json=chat("answer"))
    assert r.status_code == 200
    assert r.headers["x-served-model"] == "small"
    assert r.json()["model"] == "small"
    m = (await stack.client.get("/metrics")).text
    assert 'router_fallbacks_total{from_model="big",to_model="small"}' in m


async def test_503_when_everything_is_down(stack):
    for h in ("big-1", "big-2", "small-1"):
        stack.transport.t.pop(h)
    r = await stack.client.post("/v1/chat/completions", json=chat("big"))
    assert r.status_code == 503 and r.json()["error"]["type"] == "service_unavailable"


async def test_client_errors_are_not_retried(stack):  # 4xx 는 엔진 탓이 아니다
    r = await stack.client.post(
        "/v1/chat/completions", json={"model": "big"}
    )  # messages 없음 → mock 은 200
    assert r.status_code in (200, 400, 422)
    assert sum(1 for c in stack.transport.calls if "chat/completions" in c) == 1


async def test_readyz_and_metrics(stack):
    assert (await stack.client.get("/readyz")).status_code == 200
    await stack.client.post("/v1/chat/completions", json=chat("big"))
    m = (await stack.client.get("/metrics")).text
    assert 'router_requests_total{model="big",status="200"}' in m
    assert "router_request_seconds_bucket" in m


async def test_api_key_enforced(stack, monkeypatch):
    from servectl.router.app import create_app
    from servectl.router.config import RouterConfig

    monkeypatch.setenv("ROUTER_API_KEYS", "k1, k2")
    cfg = RouterConfig.model_validate(
        {"probe_interval": 0, "routes": [{"id": "big", "upstreams": ["http://big-1"]}]}
    )
    app = create_app(cfg, client=httpx.AsyncClient(transport=stack.transport))
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")
    assert (await c.post("/v1/chat/completions", json=chat("big"))).status_code == 401
    bad = await c.post(
        "/v1/chat/completions", json=chat("big"), headers={"Authorization": "Bearer zz"}
    )
    assert bad.status_code == 401
    ok = await c.post(
        "/v1/chat/completions", json=chat("big"), headers={"Authorization": "Bearer k2"}
    )
    assert ok.status_code == 200
    assert (await c.get("/healthz")).status_code == 200  # 헬스·메트릭은 인증 제외


async def test_body_size_limit(stack):
    from servectl.router.app import create_app
    from servectl.router.config import RouterConfig

    cfg = RouterConfig.model_validate(
        {
            "probe_interval": 0,
            "max_body_bytes": 100,
            "routes": [{"id": "big", "upstreams": ["http://big-1"]}],
        }
    )
    app = create_app(cfg, client=httpx.AsyncClient(transport=stack.transport))
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")
    r = await c.post("/v1/chat/completions", json=chat("big", pad="x" * 500))
    assert r.status_code == 413


@pytest.mark.parametrize("path", ["/v1/embeddings"])
async def test_embeddings_proxied(stack, path):
    r = await stack.client.post(path, json={"model": "small", "input": ["a", "b"]})
    assert r.status_code == 200 and len(r.json()["data"]) == 2
