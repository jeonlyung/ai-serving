"""업스트림 풀 — 최소 진행중 요청(least-inflight) + 수동·능동 헬스 제외."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from .config import Route


@dataclass
class Upstream:
    url: str
    inflight: int = 0
    failures: int = 0
    ejected_until: float = 0.0
    probe_ok: bool = True  # 능동 헬스체크 결과
    served: int = 0

    def available(self, now: float) -> bool:
        return self.probe_ok and now >= self.ejected_until


@dataclass
class Pool:
    route: Route
    upstreams: list[Upstream] = field(default_factory=list)
    _rr: itertools.count = field(default_factory=itertools.count)

    def __post_init__(self) -> None:
        self.upstreams = [Upstream(u.rstrip("/")) for u in self.route.upstreams]

    def pick(self, exclude: set[str] | None = None) -> Upstream | None:
        """가용 업스트림 중 inflight 가 가장 적은 것. 동률이면 라운드로빈.

        전부 제외 상태면 None → 호출자가 폴백 모델로 넘어간다.
        """
        exclude = exclude or set()
        now = time.monotonic()
        cands = [u for u in self.upstreams if u.url not in exclude and u.available(now)]
        if not cands:
            return None
        low = min(u.inflight for u in cands)
        tied = [u for u in cands if u.inflight == low]
        return tied[next(self._rr) % len(tied)]

    def healthy_count(self) -> int:
        now = time.monotonic()
        return sum(1 for u in self.upstreams if u.available(now))

    def record(self, up: Upstream, ok: bool, *, threshold: int, eject_s: float) -> None:
        if ok:
            up.failures = 0
            up.served += 1
            return
        up.failures += 1
        if up.failures >= threshold:
            up.ejected_until = time.monotonic() + eject_s
            up.failures = 0
