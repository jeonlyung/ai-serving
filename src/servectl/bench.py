"""간이 부하시험 — TTFT · 토큰 처리량 · 지연 분포 · 오류율.

배포 직후 "이 설정이 몇 동시 사용자를 견디나" 를 빠르게 보는 용도다.
정식 용량 산정은 `vllm bench serve` 또는 GuideLLM 으로 실제 데이터셋을 쓰자 (docs/OPERATIONS.md).
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx

PROMPT = "다음 문장을 이어서 자세히 설명해줘: 사내 규정 문서에서 근거를 찾아 답하는 시스템의 장점은"


@dataclass
class Sample:
    ok: bool
    ttft: float = 0.0
    total: float = 0.0
    tokens: int = 0
    error: str = ""


@dataclass
class Report:
    model: str
    concurrency: int
    requests: int
    wall: float
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.ok]

        def pct(v: list[float], p: float) -> float:
            if not v:
                return 0.0
            v = sorted(v)
            return v[min(len(v) - 1, int(len(v) * p))]

        toks = sum(s.tokens for s in ok)
        return {
            "model": self.model,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "errors": len(self.samples) - len(ok),
            "error_rate": round(1 - len(ok) / max(len(self.samples), 1), 4),
            "wall_s": round(self.wall, 2),
            "req_per_s": round(len(ok) / self.wall, 2) if self.wall else 0,
            "output_tok_per_s": round(toks / self.wall, 1) if self.wall else 0,
            "ttft_ms_p50": round(pct([s.ttft for s in ok], 0.5) * 1000, 1),
            "ttft_ms_p95": round(pct([s.ttft for s in ok], 0.95) * 1000, 1),
            "latency_ms_p50": round(pct([s.total for s in ok], 0.5) * 1000, 1),
            "latency_ms_p95": round(pct([s.total for s in ok], 0.95) * 1000, 1),
            "tok_per_s_per_req": round(
                statistics.mean([s.tokens / s.total for s in ok if s.total]), 1
            )
            if ok
            else 0,
        }


async def _one(c: httpx.AsyncClient, url: str, model: str, max_tokens: int) -> Sample:
    body = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    t0 = time.perf_counter()
    first, n = 0.0, 0
    try:
        async with c.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                return Sample(False, error=f"HTTP {r.status_code}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if delta.get("content"):
                    if not first:
                        first = time.perf_counter() - t0
                    n += 1  # 청크 ≈ 토큰 (vLLM 은 청크당 1토큰)
    except Exception as e:  # noqa: BLE001
        return Sample(False, error=type(e).__name__)
    return Sample(n > 0, first, time.perf_counter() - t0, n)


async def run_bench(
    url: str,
    model: str,
    *,
    concurrency: int = 8,
    requests: int = 40,
    max_tokens: int = 128,
    api_key: str = "",
) -> Report:
    h = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300, headers=h, base_url="") as c:

        async def guarded() -> Sample:
            async with sem:
                return await _one(c, url.rstrip("/"), model, max_tokens)

        t0 = time.perf_counter()
        samples = await asyncio.gather(*(guarded() for _ in range(requests)))
        return Report(model, concurrency, requests, time.perf_counter() - t0, list(samples))
