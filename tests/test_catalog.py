import pytest
from pydantic import ValidationError

from servectl.catalog import Catalog, slug

BASE = {
    "engines": {"vllm": {"image": "x", "port": 8000}},
    "models": [{"id": "a/b", "engine": "vllm", "gpu": {"count": 2}, "tensor_parallel": 2}],
}


def make(**patch):
    d = {**BASE, **patch}
    return Catalog.model_validate(d)


def test_default_catalog_loads(catalog):
    assert {m.id for m in catalog.models} >= {"google/gemma-3-27b-it"}
    assert catalog.model("answer").id == "google/gemma-3-27b-it"  # 별칭 조회


def test_tp_cannot_exceed_gpu_count():
    with pytest.raises(ValidationError, match="tensor_parallel"):
        make(models=[{"id": "a", "engine": "vllm", "gpu": {"count": 1}, "tensor_parallel": 2}])


def test_gpu_engine_requires_gpu():
    with pytest.raises(ValidationError, match="GPU"):
        make(models=[{"id": "a", "engine": "vllm", "gpu": {"count": 0}}])


def test_duplicate_alias_rejected():
    with pytest.raises(ValidationError, match="중복"):
        make(
            models=[
                {"id": "a", "aliases": ["x"], "gpu": {"count": 1}},
                {"id": "b", "aliases": ["x"], "gpu": {"count": 1}},
            ]
        )


def test_unknown_fallback_rejected():
    with pytest.raises(ValidationError, match="fallback"):
        make(models=[{"id": "a", "fallback": "nope", "gpu": {"count": 1}}])


def test_unknown_field_rejected():  # 오타(max_model_length 등)를 조용히 넘기지 않는다
    with pytest.raises(ValidationError):
        make(models=[{"id": "a", "gpu": {"count": 1}, "max_model_length": 10}])


def test_slug_is_dns_safe():
    s = slug("Qwen/Qwen3-32B")
    assert s == "qwen-qwen3-32b"
    assert slug("A__b//c") == "a-b-c"
