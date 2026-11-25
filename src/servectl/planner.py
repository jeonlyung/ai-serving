"""GPU 메모리 산정 — 올리기 전에 "이 모델이 이 GPU 에 들어가나"를 계산한다.

    사용 가능 = GPU메모리 × gpu_memory_utilization × TP 장수
    가중치    = 파라미터 × 바이트/파라미터(dtype·양자화)
    KV 풀    = 사용 가능 − 가중치 − 런타임 오버헤드(CUDA 그래프·활성값)
    토큰당 KV = 2(K,V) × layers × kv_heads × head_dim × 바이트

추정치다. 슬라이딩 윈도우 어텐션(Gemma 3 등)은 실제 KV 가 더 적게 든다 — 보수적으로 잡힌다.
실측은 vLLM 시작 로그의 "GPU KV cache size" 로 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog, Model

GIB = 1024**3
OVERHEAD_GIB_PER_GPU = 2.5  # CUDA 컨텍스트 + 그래프 + 활성값 (경험치)
AVG_CTX = 4096  # 동시성 판단에 쓰는 평균 컨텍스트 길이

_WEIGHT_BYTES = {"fp8": 1.0, "int8": 1.0, "awq": 0.56, "gptq": 0.56}
_DTYPE_BYTES = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0, "half": 2.0, "auto": 2.0}


@dataclass
class Plan:
    model: str
    status: str  # OK | WARN | FAIL | SKIP
    weights_gib: float = 0.0
    budget_gib: float = 0.0
    kv_pool_gib: float = 0.0
    kv_per_token_kib: float = 0.0
    max_tokens: int = 0
    full_len_seqs: float = 0.0
    avg_ctx_seqs: float = 0.0
    note: str = ""


def weight_bytes_per_param(m: Model) -> float:
    if m.quantization:
        return _WEIGHT_BYTES[m.quantization]
    return _DTYPE_BYTES.get(m.dtype, 2.0)


def plan_model(m: Model) -> Plan:
    if m.arch is None:
        return Plan(m.id, "SKIP", note="arch 가 없어 산정 불가 (catalog 에 arch 추가)")
    if m.engine == "mock":
        return Plan(m.id, "SKIP", note="mock 엔진")
    if m.gpu.count == 0:
        return Plan(m.id, "SKIP", note="CPU 모델 — 시스템 메모리로 판단")

    a = m.arch
    weights = a.params_b * 1e9 * weight_bytes_per_param(m) / GIB
    gpus = max(m.tensor_parallel, 1)
    budget = m.gpu.memory_gb * m.gpu_memory_utilization * gpus
    overhead = OVERHEAD_GIB_PER_GPU * gpus
    kv_pool = budget - weights - overhead
    kv_bytes = 1.0 if m.kv_cache_dtype == "fp8" else 2.0
    per_token = 2 * a.layers * a.kv_heads * a.head_dim * kv_bytes

    if kv_pool <= 0:
        return Plan(
            m.id,
            "FAIL",
            round(weights, 1),
            round(budget, 1),
            round(kv_pool, 1),
            round(per_token / 1024, 1),
            note=f"가중치({weights:.1f}GiB)가 예산({budget:.1f}GiB)을 못 넘김 — "
            "TP 늘리기 / 양자화 / 더 큰 GPU",
        )

    max_tokens = int(kv_pool * GIB / per_token)
    full = max_tokens / m.max_model_len
    avg = max_tokens / min(AVG_CTX, m.max_model_len)
    status, note = "OK", ""
    if full < 1:
        status = "FAIL"
        note = f"max_model_len({m.max_model_len}) 한 건도 못 담음 — max_model_len 줄이기 / FP8 KV"
    elif avg < m.max_num_seqs:
        status = "WARN"
        note = (
            f"평균 {AVG_CTX}토큰 기준 동시 {avg:.0f}건 — max_num_seqs({m.max_num_seqs}) 를 못 채움"
        )
    return Plan(
        m.id,
        status,
        round(weights, 1),
        round(budget, 1),
        round(kv_pool, 1),
        round(per_token / 1024, 1),
        max_tokens,
        round(full, 1),
        round(avg, 1),
        note,
    )


def plan_catalog(cat: Catalog) -> list[Plan]:
    return [plan_model(m) for m in cat.models]


def format_plans(plans: list[Plan]) -> str:
    head = f"{'모델':<28}{'판정':<6}{'가중치':>8}{'예산':>8}{'KV풀':>8}{'KV/tok':>9}{'최대토큰':>10}{'동시(풀길이)':>13}{'동시(평균)':>11}"
    lines = [head, "-" * len(head)]
    for p in plans:
        if p.status == "SKIP":
            lines.append(f"{p.model:<28}{p.status:<6}  {p.note}")
            continue
        lines.append(
            f"{p.model:<28}{p.status:<6}{p.weights_gib:>7.1f}G{p.budget_gib:>7.1f}G"
            f"{p.kv_pool_gib:>7.1f}G{p.kv_per_token_kib:>7.0f}Ki{p.max_tokens:>10,}"
            f"{p.full_len_seqs:>13.1f}{p.avg_ctx_seqs:>11.0f}"
        )
        if p.note:
            lines.append(f"{'':<34}↳ {p.note}")
    return "\n".join(lines)
