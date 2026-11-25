"""모의 추론 엔진 — OpenAI 호환 API + vLLM 형식 /metrics.

GPU 없이 라우터·모니터링·대시보드·부하시험·CI 를 끝까지 돌리기 위한 것이다.
토큰 생성 속도·지연·오류율을 옵션으로 조절해 장애 상황도 흉내 낸다.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


def create_app(
    model: str = "mock-model",
    *,
    tokens_per_sec: float = 200.0,
    ttft_ms: float = 30.0,
    error_rate: float = 0.0,
    max_running: int = 64,
) -> FastAPI:
    app = FastAPI(title=f"mock-engine:{model}")
    reg = CollectorRegistry()
    # vLLM 과 같은 이름 — Grafana 대시보드·알림·KEDA 규칙이 그대로 동작한다
    running = Gauge("vllm:num_requests_running", "running", ["model_name"], registry=reg)
    waiting = Gauge("vllm:num_requests_waiting", "waiting", ["model_name"], registry=reg)
    kv_usage = Gauge("vllm:kv_cache_usage_perc", "kv usage 0-1", ["model_name"], registry=reg)
    prompt_tok = Counter("vllm:prompt_tokens_total", "prompt tokens", ["model_name"], registry=reg)
    gen_tok = Counter("vllm:generation_tokens_total", "gen tokens", ["model_name"], registry=reg)
    ok_req = Counter(
        "vllm:request_success_total", "ok", ["model_name", "finished_reason"], registry=reg
    )
    ttft = Histogram(
        "vllm:time_to_first_token_seconds",
        "ttft",
        ["model_name"],
        registry=reg,
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    e2e = Histogram(
        "vllm:e2e_request_latency_seconds",
        "e2e",
        ["model_name"],
        registry=reg,
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    )
    lab = {"model_name": model}
    sem = asyncio.Semaphore(max_running)
    state = {"running": 0, "waiting": 0, "ready": True, "error_rate": error_rate}

    def _sync() -> None:
        running.labels(**lab).set(state["running"])
        waiting.labels(**lab).set(state["waiting"])
        kv_usage.labels(**lab).set(min(state["running"] / max_running, 1.0) * 0.8)

    _sync()

    def _count(msgs: list[dict]) -> int:
        return max(1, sum(len(str(m.get("content", ""))) for m in msgs) // 3)

    def _reply(msgs: list[dict], n: int) -> list[str]:
        last = next((str(m["content"]) for m in reversed(msgs) if m.get("role") == "user"), "")
        words = f"[{model}] 모의 응답입니다. 입력: {last[:60]}".split()
        out = (words * (n // max(len(words), 1) + 1))[:n]
        return [w + " " for w in out]

    async def _generate(msgs: list[dict], max_tokens: int) -> AsyncIterator[str]:
        state["waiting"] += 1
        _sync()
        async with sem:
            state["waiting"] -= 1
            state["running"] += 1
            _sync()
            t0 = time.perf_counter()
            try:
                await asyncio.sleep(ttft_ms / 1000)
                ttft.labels(**lab).observe(time.perf_counter() - t0)
                for tok in _reply(msgs, min(max_tokens, 32)):
                    yield tok
                    gen_tok.labels(**lab).inc()
                    await asyncio.sleep(1 / tokens_per_sec)
                ok_req.labels(model, "stop").inc()
            finally:
                e2e.labels(**lab).observe(time.perf_counter() - t0)
                state["running"] -= 1
                _sync()

    def _fail() -> JSONResponse | None:
        if random.random() < state["error_rate"]:
            return JSONResponse(
                {"error": {"message": "mock injected failure", "type": "server_error"}}, 503
            )
        return None

    @app.get("/health")
    async def health():
        return PlainTextResponse("") if state["ready"] else PlainTextResponse("", 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "mock"}]}

    @app.get("/metrics")
    async def metrics():
        _sync()
        return PlainTextResponse(generate_latest(reg), media_type="text/plain; version=0.0.4")

    @app.post("/admin/chaos")  # 장애 주입: {"error_rate":0.5,"ready":false}
    async def chaos(req: Request):
        body = await req.json()
        state["error_rate"] = float(body.get("error_rate", state["error_rate"]))
        state["ready"] = bool(body.get("ready", state["ready"]))
        return state

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if (f := _fail()) is not None:
            return f
        body = await req.json()
        msgs = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 64)
        prompt_n = _count(msgs)
        prompt_tok.labels(**lab).inc(prompt_n)
        cid = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        if body.get("stream"):

            async def sse() -> AsyncIterator[bytes]:
                def chunk(delta: dict, finish: str | None = None) -> bytes:
                    payload = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

                yield chunk({"role": "assistant", "content": ""})
                async for tok in _generate(msgs, max_tokens):
                    yield chunk({"content": tok})
                yield chunk({}, "stop")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        parts = [t async for t in _generate(msgs, max_tokens)]
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "".join(parts).strip()},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_n,
                "completion_tokens": len(parts),
                "total_tokens": prompt_n + len(parts),
            },
        }

    @app.post("/v1/embeddings")
    async def embeddings(req: Request):
        body = await req.json()
        inp = body.get("input", [])
        inp = [inp] if isinstance(inp, str) else inp
        data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": [((hash((s, k)) % 2000) / 1000) - 1 for k in range(8)],
            }
            for i, s in enumerate(inp)
        ]
        return {
            "object": "list",
            "model": model,
            "data": data,
            "usage": {"prompt_tokens": sum(len(s) for s in inp), "total_tokens": 0},
        }

    return app
