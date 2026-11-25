"""스모크 테스트 — 배포 직후 "정말 동작하나"를 30초 안에 확인한다."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def run_smoke(
    base_url: str, models: list[str], api_key: str = "", timeout: float = 120.0
) -> list[Check]:
    base = base_url.rstrip("/")
    h = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    out: list[Check] = []
    with httpx.Client(timeout=timeout, headers=h) as c:
        try:
            r = c.get(f"{base}/v1/models")
            served = {m["id"] for m in r.json().get("data", [])}
            out.append(Check("GET /v1/models", r.status_code == 200, f"{len(served)}개 노출"))
        except Exception as e:  # noqa: BLE001
            return [Check("GET /v1/models", False, f"{type(e).__name__}: {e}")]

        for mid in models:
            if mid not in served:
                out.append(
                    Check(f"모델 {mid}", False, "/v1/models 에 없음 (로딩 중이거나 라우팅 누락)")
                )
                continue
            body = {
                "model": mid,
                "messages": [{"role": "user", "content": "한 문장으로 인사해줘."}],
                "max_tokens": 32,
                "temperature": 0,
            }
            t0 = time.perf_counter()
            try:
                r = c.post(f"{base}/v1/chat/completions", json=body)
                ok = r.status_code == 200 and bool(
                    r.json()["choices"][0]["message"]["content"].strip()
                )
                out.append(
                    Check(
                        f"chat {mid}",
                        ok,
                        f"{(time.perf_counter() - t0) * 1000:.0f}ms"
                        + ("" if ok else f" HTTP {r.status_code} {r.text[:120]}"),
                    )
                )
            except Exception as e:  # noqa: BLE001
                out.append(Check(f"chat {mid}", False, f"{type(e).__name__}: {e}"))
                continue
            try:  # 스트리밍 — SSE 가 중간에 버퍼링되지 않고 [DONE] 으로 끝나는지
                chunks, done = 0, False
                with c.stream(
                    "POST", f"{base}/v1/chat/completions", json={**body, "stream": True}
                ) as s:
                    for line in s.iter_lines():
                        if line.startswith("data: "):
                            if line.strip() == "data: [DONE]":
                                done = True
                            else:
                                json.loads(line[6:])
                                chunks += 1
                out.append(Check(f"stream {mid}", done and chunks > 1, f"{chunks}청크"))
            except Exception as e:  # noqa: BLE001
                out.append(Check(f"stream {mid}", False, f"{type(e).__name__}: {e}"))
    return out
