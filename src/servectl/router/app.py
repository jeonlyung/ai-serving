"""라우터 앱.

하는 일
  · model(또는 별칭)로 업스트림 풀을 고르고 OpenAI 호환 요청을 그대로 전달 (SSE 스트리밍 포함)
  · 실패(연결 오류·5xx·429)는 스트림이 시작되기 전이면 다른 복제본 → 폴백 모델 순으로 재시도
  · API 키 인증, 요청 크기 제한, X-Request-ID 전파
  · 헬스체크로 죽은 복제본 자동 제외, /metrics 로 모델별 요청·지연·재시도 노출

하지 않는 일: 사용자 권한·프롬프트·워크플로우 — 그건 위쪽 ai-core 게이트웨이의 몫이다.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
import uuid
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .config import RouterConfig
from .pool import Pool, Upstream

PROXY_PATHS = ("chat/completions", "completions", "embeddings")
RETRY_STATUS = {429, 500, 502, 503, 504}
_HOP = {"content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding"}


def _err(status: int, msg: str, typ: str = "invalid_request_error", rid: str = "") -> JSONResponse:
    h = {"x-request-id": rid} if rid else None
    return JSONResponse({"error": {"message": msg, "type": typ, "code": status}}, status, headers=h)


def create_app(cfg: RouterConfig, client: httpx.AsyncClient | None = None) -> FastAPI:
    pools = {r.id: Pool(r) for r in cfg.routes}
    names: dict[str, str] = {}
    for r in cfg.routes:
        names[r.id] = r.id
        for a in r.aliases:
            names[a] = r.id
    keys = {k.strip() for k in os.environ.get(cfg.api_keys_env, "").split(",") if k.strip()}

    reg = CollectorRegistry()
    m_req = Counter("router_requests_total", "요청 수", ["model", "status"], registry=reg)
    m_lat = Histogram(
        "router_request_seconds",
        "첫 바이트까지/비스트림 전체 지연",
        ["model"],
        registry=reg,
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    )
    m_retry = Counter("router_retries_total", "재시도", ["model", "reason"], registry=reg)
    m_fb = Counter(
        "router_fallbacks_total", "폴백 모델 전환", ["from_model", "to_model"], registry=reg
    )
    g_healthy = Gauge("router_upstreams_healthy", "가용 복제본 수", ["model"], registry=reg)
    g_inflight = Gauge("router_inflight", "진행중 요청", ["model"], registry=reg)

    own_client = client is None
    http = client or httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.request_timeout, connect=cfg.connect_timeout),
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200),
    )

    async def probe_loop() -> None:
        while True:
            await asyncio.gather(*(_probe(p) for p in pools.values()), return_exceptions=True)
            for mid, p in pools.items():
                g_healthy.labels(mid).set(p.healthy_count())
            await asyncio.sleep(cfg.probe_interval)

    async def _probe(p: Pool) -> None:
        async def one(u: Upstream) -> None:
            try:
                r = await http.get(u.url + p.route.health_path, timeout=3)
                u.probe_ok = r.status_code < 500
            except httpx.HTTPError:
                u.probe_ok = False

        await asyncio.gather(*(one(u) for u in p.upstreams))

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(probe_loop()) if cfg.probe_interval > 0 else None
        for mid, p in pools.items():
            g_healthy.labels(mid).set(p.healthy_count())
        yield
        if task:
            task.cancel()
        if own_client:
            await http.aclose()

    app = FastAPI(title="ai-serving router", lifespan=lifespan)
    app.state.pools = pools

    @app.middleware("http")
    async def guard(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.rid = rid
        if request.url.path.startswith("/v1/") and keys:
            tok = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not any(secrets.compare_digest(tok, k) for k in keys):
                return _err(401, "invalid api key", "authentication_error", rid)
        resp = await call_next(request)
        resp.headers.setdefault("x-request-id", rid)
        return resp

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        ready = {mid: p.healthy_count() for mid, p in pools.items()}
        ok = any(ready.values())
        return JSONResponse({"ready": ok, "healthy_upstreams": ready}, 200 if ok else 503)

    @app.get("/metrics")
    async def metrics():
        for mid, p in pools.items():
            g_healthy.labels(mid).set(p.healthy_count())
            g_inflight.labels(mid).set(sum(u.inflight for u in p.upstreams))
        return PlainTextResponse(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models():
        now = int(time.time())
        data = []
        for r in cfg.routes:
            for name in [r.id, *r.aliases]:
                data.append(
                    {
                        "id": name,
                        "object": "model",
                        "created": now,
                        "owned_by": "ai-serving",
                        "root": r.id,
                    }
                )
        return {"object": "list", "data": data}

    @app.get("/admin/upstreams")
    async def admin_upstreams():
        now = time.monotonic()
        return {
            mid: [
                {
                    "url": u.url,
                    "available": u.available(now),
                    "inflight": u.inflight,
                    "served": u.served,
                    "probe_ok": u.probe_ok,
                }
                for u in p.upstreams
            ]
            for mid, p in pools.items()
        }

    async def _forward(path: str, body: dict, model_id: str, rid: str, accept: str):
        """model_id 풀 → 실패 시 fallback 체인 순서로 시도. (resp, 사용한 모델, 업스트림) 반환."""
        chain, seen = [], set()
        cur: str | None = model_id
        while cur and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = pools[cur].route.fallback if cur in pools else None
        last = "no healthy upstream"
        for idx, mid in enumerate(chain):
            pool = pools.get(mid)
            if not pool:
                continue
            if idx > 0:
                m_fb.labels(chain[idx - 1], mid).inc()
            tried: set[str] = set()
            for _ in range(cfg.attempts_per_model):
                up = pool.pick(tried)
                if up is None:
                    break
                tried.add(up.url)
                payload = {**body, "model": mid}  # 폴백 시 모델 id 로 다시 쓴다
                up.inflight += 1
                try:
                    req = http.build_request(
                        "POST",
                        f"{up.url}/v1/{path}",
                        json=payload,
                        headers={"x-request-id": rid, "accept": accept},
                    )
                    resp = await http.send(req, stream=True)
                except httpx.HTTPError as e:
                    up.inflight -= 1
                    pool.record(
                        up, False, threshold=cfg.eject_after_failures, eject_s=cfg.eject_seconds
                    )
                    m_retry.labels(mid, "connect").inc()
                    last = f"{up.url}: {type(e).__name__}"
                    continue
                if resp.status_code in RETRY_STATUS:
                    await resp.aclose()
                    up.inflight -= 1
                    pool.record(
                        up, False, threshold=cfg.eject_after_failures, eject_s=cfg.eject_seconds
                    )
                    m_retry.labels(mid, str(resp.status_code)).inc()
                    last = f"{up.url}: HTTP {resp.status_code}"
                    continue
                pool.record(up, True, threshold=cfg.eject_after_failures, eject_s=cfg.eject_seconds)
                return resp, mid, up
        return None, last, None

    async def proxy(request: Request, path: str):
        rid = request.state.rid
        raw = await request.body()
        if len(raw) > cfg.max_body_bytes:
            return _err(413, "request body too large", rid=rid)
        try:
            body = json.loads(raw)
            requested = body["model"]
        except (ValueError, KeyError, TypeError):
            return _err(400, "JSON body with 'model' is required", rid=rid)
        model_id = names.get(requested)
        if model_id is None:
            m_req.labels("unknown", "404").inc()
            return _err(404, f"model '{requested}' not found", "model_not_found", rid)

        t0 = time.perf_counter()
        accept = "text/event-stream" if body.get("stream") else "application/json"
        resp, used, up = await _forward(path, body, model_id, rid, accept)
        if resp is None:
            m_req.labels(model_id, "503").inc()
            return _err(503, f"all upstreams failed ({used})", "service_unavailable", rid)

        m_lat.labels(used).observe(time.perf_counter() - t0)
        m_req.labels(used, str(resp.status_code)).inc()
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP}
        headers.update({"x-request-id": rid, "x-served-model": used, "x-upstream": up.url})

        async def body_iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                up.inflight -= 1

        return StreamingResponse(
            body_iter(),
            status_code=resp.status_code,
            headers=headers,
            media_type=resp.headers.get("content-type"),
        )

    def _handler(path: str):
        async def handler(request: Request):
            return await proxy(request, path)

        return handler

    for p in PROXY_PATHS:
        app.add_api_route(f"/v1/{p}", _handler(p), methods=["POST"])

    return app


def app_from_env() -> FastAPI:
    """uvicorn --factory 진입점. ROUTER_CONFIG 경로의 YAML 을 읽는다."""
    from .config import load_router_config

    return create_app(
        load_router_config(os.environ.get("ROUTER_CONFIG", "/etc/router/router.yaml"))
    )
