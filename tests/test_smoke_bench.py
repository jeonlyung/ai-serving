import httpx

from servectl.bench import run_bench
from servectl.mockengine import create_app


async def test_mock_engine_metrics_follow_vllm_names():
    app = create_app("m", tokens_per_sec=5000, ttft_ms=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as c:
        await c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        text = (await c.get("/metrics")).text
    for name in (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_bucket",
    ):
        assert name in text, name


async def test_chaos_makes_engine_fail_and_unready():
    app = create_app("m", tokens_per_sec=5000, ttft_ms=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as c:
        await c.post("/admin/chaos", json={"error_rate": 1.0, "ready": False})
        assert (await c.get("/health")).status_code == 503
        r = await c.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert r.status_code == 503


def test_bench_report_math():
    from servectl.bench import Report, Sample

    rep = Report(
        "m",
        2,
        3,
        2.0,
        [Sample(True, 0.1, 1.0, 10), Sample(True, 0.3, 1.0, 10), Sample(False, error="x")],
    )
    s = rep.summary()
    assert s["errors"] == 1 and s["output_tok_per_s"] == 10.0 and s["req_per_s"] == 1.0
    assert s["error_rate"] == 0.3333


async def test_bench_against_asgi_engine(monkeypatch):
    app = create_app("m", tokens_per_sec=5000, ttft_ms=1)
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.ASGITransport(app=app)
        kw["base_url"] = "http://m"
        return real(*a, **kw)

    monkeypatch.setattr("servectl.bench.httpx.AsyncClient", factory)
    rep = await run_bench("http://m", "m", concurrency=4, requests=8, max_tokens=16)
    s = rep.summary()
    assert s["errors"] == 0 and s["output_tok_per_s"] > 0 and s["ttft_ms_p50"] > 0
