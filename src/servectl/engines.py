"""엔진별 실행 인자·환경변수 — compose 와 k8s 렌더러가 같이 쓴다."""

from __future__ import annotations

from .catalog import Catalog, Model

HF_CACHE = "/root/.cache/huggingface"


def container_args(m: Model, cat: Catalog, *, api_key_env: bool = False) -> list[str]:
    """컨테이너 command/args. vllm/vllm-openai 이미지는 ENTRYPOINT 가 `vllm serve` 이다."""
    port = cat.engines[m.engine].port
    if m.engine == "vllm":
        a = [
            m.hf_repo,
            "--served-model-name",
            m.id,
            *m.aliases,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--max-model-len",
            str(m.max_model_len),
            "--max-num-seqs",
            str(m.max_num_seqs),
            "--gpu-memory-utilization",
            str(m.gpu_memory_utilization),
            "--dtype",
            m.dtype,
            "--tensor-parallel-size",
            str(m.tensor_parallel),
            "--enable-prefix-caching",
            "--generation-config",
            "vllm",
        ]
        if m.quantization:
            a += ["--quantization", m.quantization]
        if m.kv_cache_dtype != "auto":
            a += ["--kv-cache-dtype", m.kv_cache_dtype]
        if m.tool_call_parser:
            a += ["--enable-auto-tool-choice", "--tool-call-parser", m.tool_call_parser]
        if m.reasoning_parser:
            a += ["--reasoning-parser", m.reasoning_parser]
        return a + m.extra_args
    if m.engine == "sglang":
        a = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            m.hf_repo,
            "--served-model-name",
            m.id,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--context-length",
            str(m.max_model_len),
            "--max-running-requests",
            str(m.max_num_seqs),
            "--mem-fraction-static",
            str(m.gpu_memory_utilization),
            "--tp-size",
            str(m.tensor_parallel),
            "--enable-metrics",
        ]
        if m.quantization:
            a += ["--quantization", m.quantization]
        if m.tool_call_parser:
            a += ["--tool-call-parser", m.tool_call_parser]
        if m.reasoning_parser:
            a += ["--reasoning-parser", m.reasoning_parser]
        return a + m.extra_args
    if m.engine == "llamacpp":
        if not m.gguf:
            raise ValueError(f"{m.id}: llamacpp 엔진은 gguf('repo:quant') 가 필요합니다")
        parallel = min(m.max_num_seqs, 8)  # -c 는 슬롯 전체 합 → 슬롯당 max_model_len 보장
        a = [
            "-hf",
            m.gguf,
            "--alias",
            m.id,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "-c",
            str(m.max_model_len * parallel),
            "--parallel",
            str(parallel),
            "--metrics",
            "--jinja",
        ]
        return a + m.extra_args
    if m.engine == "mock":
        return ["servectl", "mock-engine", "--model", m.id, "--port", str(port)]
    raise ValueError(m.engine)


def container_env(m: Model) -> dict[str, str]:
    env: dict[str, str] = {}
    if m.engine in ("vllm", "sglang", "llamacpp"):
        env["HF_HOME"] = HF_CACHE
        env["HF_TOKEN"] = "${HF_TOKEN:-}"  # compose 치환식. k8s 는 renderer 가 Secret 참조로 바꾼다
    if m.engine == "vllm":
        env["VLLM_LOGGING_LEVEL"] = "INFO"
    return env


def is_gpu(m: Model) -> bool:
    return m.engine in ("vllm", "sglang") and m.gpu.count > 0
