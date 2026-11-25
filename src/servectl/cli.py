"""servectl — 카탈로그 검증 · 메모리 산정 · 배포 산출물 생성 · 스모크/부하 시험."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import __version__
from .catalog import load_catalog
from .planner import format_plans, plan_catalog

DEFAULT_CATALOG = "catalog/models.yaml"


def _write(files: dict[str, str], out: Path, check: bool) -> int:
    stale = []
    for rel, text in files.items():
        p = out / rel
        if check:
            if not p.exists() or p.read_text(encoding="utf-8") != text:
                stale.append(rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    if check:
        if stale:
            print(
                f"✗ 산출물이 카탈로그와 다릅니다 ({out}): {', '.join(stale)}\n  → servectl render 를 다시 실행하세요"
            )
            return 1
        print(f"✓ {out} 최신")
        return 0
    print(f"✓ {len(files)}개 파일 생성 → {out}")
    return 0


def cmd_validate(a) -> int:
    cat = load_catalog(a.catalog)
    for m in cat.models:
        gpus = m.gpu.count * m.replicas
        print(
            f"  {m.id:<28} {m.engine:<8} TP={m.tensor_parallel} ×{m.replicas}  GPU {gpus}장  "
            f"별칭={','.join(m.aliases) or '-'}" + (f"  폴백→{m.fallback}" if m.fallback else "")
        )
    print(
        f"✓ 카탈로그 유효 — 모델 {len(cat.models)}종, GPU 총 {sum(m.gpu.count * m.replicas for m in cat.models)}장"
    )
    return 0


def cmd_plan(a) -> int:
    cat = load_catalog(a.catalog)
    plans = plan_catalog(cat)
    print(format_plans(plans))
    return 1 if any(p.status == "FAIL" for p in plans) else 0


def cmd_render(a) -> int:
    from .render_compose import render_compose
    from .render_k8s import render_k8s

    cat = load_catalog(a.catalog)
    rc = 0
    if a.target in ("compose", "all"):
        out = Path(a.out or "deploy") / ("compose-mock" if a.mock else "compose")
        rc |= _write(render_compose(cat, mock=a.mock), out, a.check)
    if a.target in ("k8s", "all"):
        out = Path(a.out or "deploy") / "k8s"
        files = render_k8s(
            cat, namespace=a.namespace, storage_class=a.storage_class, router_image=a.router_image
        )
        rc |= _write(files, out, a.check)
    return rc


def cmd_router(a) -> int:
    import uvicorn

    from .router.app import create_app
    from .router.config import load_router_config

    uvicorn.run(
        create_app(load_router_config(a.config)), host=a.host, port=a.port, log_level="info"
    )
    return 0


def cmd_mock(a) -> int:
    import uvicorn

    from .mockengine import create_app

    app = create_app(a.model, tokens_per_sec=a.tps, ttft_ms=a.ttft_ms, error_rate=a.error_rate)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_smoke(a) -> int:
    from .smoke import run_smoke

    cat = load_catalog(a.catalog)
    models = a.model or [m.id for m in cat.models if m.kind == "chat"]
    results = run_smoke(a.url, models, a.api_key or os.environ.get("ROUTER_API_KEY", ""))
    for r in results:
        print(f"  {'✓' if r.ok else '✗'} {r.name:<42} {r.detail}")
    ok = all(r.ok for r in results)
    print("✓ 스모크 통과" if ok else "✗ 스모크 실패")
    return 0 if ok else 1


def cmd_bench(a) -> int:
    from .bench import run_bench

    rep = asyncio.run(
        run_bench(
            a.url,
            a.model,
            concurrency=a.concurrency,
            requests=a.requests,
            max_tokens=a.max_tokens,
            api_key=a.api_key or os.environ.get("ROUTER_API_KEY", ""),
        )
    )
    s = rep.summary()
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        for k, v in s.items():
            print(f"  {k:<20} {v}")
    return 0 if s["error_rate"] < 0.05 else 1


def cmd_pull(a) -> int:
    from .pull import pull_models

    for line in pull_models(load_catalog(a.catalog), a.dest, a.model or None):
        print("  ✓", line)
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="servectl", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-c", "--catalog", default=os.environ.get("SERVECTL_CATALOG", DEFAULT_CATALOG))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="카탈로그 검증").set_defaults(fn=cmd_validate)
    sub.add_parser("plan", help="GPU 메모리 산정 (FAIL 이 있으면 종료코드 1)").set_defaults(
        fn=cmd_plan
    )

    r = sub.add_parser("render", help="배포 산출물 생성")
    r.add_argument("target", choices=["compose", "k8s", "all"])
    r.add_argument(
        "--mock", action="store_true", help="모든 모델을 mock 엔진으로 (GPU 없이 스택 검증)"
    )
    r.add_argument("--out", help="출력 루트 (기본 deploy/)")
    r.add_argument("--check", action="store_true", help="쓰지 않고 최신 여부만 검사 (CI)")
    r.add_argument("--namespace", default="ai-serving")
    r.add_argument("--storage-class", default=None)
    r.add_argument("--router-image", default="ghcr.io/your-org/ai-serving:0.1.0")
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("router", help="모델 라우터 실행")
    s.add_argument("--config", default="router.yaml")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_router)

    s = sub.add_parser("mock-engine", help="모의 추론 엔진 실행")
    s.add_argument("--model", default="mock-model")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--tps", type=float, default=200.0)
    s.add_argument("--ttft-ms", type=float, default=30.0)
    s.add_argument("--error-rate", type=float, default=0.0)
    s.set_defaults(fn=cmd_mock)

    s = sub.add_parser("smoke", help="배포 후 스모크 테스트")
    s.add_argument("--url", default="http://localhost:8000")
    s.add_argument("--api-key", default="")
    s.add_argument("--model", action="append", help="미지정 시 카탈로그 전체")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("bench", help="간이 부하시험")
    s.add_argument("--url", default="http://localhost:8000")
    s.add_argument("--model", required=True)
    s.add_argument("--api-key", default="")
    s.add_argument("-n", "--requests", type=int, default=40)
    s.add_argument("-j", "--concurrency", type=int, default=8)
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("pull", help="모델 가중치 사전 다운로드 (HF)")
    s.add_argument("--dest", default=None)
    s.add_argument("--model", action="append")
    s.set_defaults(fn=cmd_pull)

    args = p.parse_args(argv)
    try:
        sys.exit(args.fn(args))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"오류: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
