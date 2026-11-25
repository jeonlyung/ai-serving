from servectl.catalog import Arch, Gpu, Model
from servectl.planner import plan_model


def model(**kw):
    base = dict(
        id="m",
        gpu=Gpu(count=1, memory_gb=80),
        arch=Arch(params_b=8, layers=32, kv_heads=8, head_dim=128),
    )
    base.update(kw)
    return Model(**base)


def test_8b_bf16_fits_80g_with_known_kv_size():
    p = plan_model(model())
    assert p.status == "OK"
    assert p.kv_per_token_kib == 128.0  # 2*32*8*128*2B = 131072B
    assert 14.5 < p.weights_gib < 15.5  # 8e9*2B


def test_weights_larger_than_budget_fails():
    p = plan_model(model(arch=Arch(params_b=70, layers=80, kv_heads=8, head_dim=128)))
    assert p.status == "FAIL" and "TP" in p.note


def test_tp_and_fp8_make_70b_fit():
    arch = Arch(params_b=70, layers=80, kv_heads=8, head_dim=128)
    assert plan_model(model(arch=arch, gpu=Gpu(count=1, memory_gb=80))).status == "FAIL"
    p = plan_model(
        model(arch=arch, gpu=Gpu(count=2, memory_gb=80), tensor_parallel=2, quantization="fp8")
    )
    assert p.status in ("OK", "WARN")
    assert p.kv_pool_gib > 0


def test_fp8_kv_halves_per_token_cost():
    a = plan_model(model()).kv_per_token_kib
    b = plan_model(model(kv_cache_dtype="fp8")).kv_per_token_kib
    assert b == a / 2


def test_max_model_len_too_long_fails():
    p = plan_model(model(max_model_len=10_000_000))
    assert p.status == "FAIL" and "max_model_len" in p.note


def test_skip_without_arch_or_for_mock():
    assert plan_model(model(arch=None)).status == "SKIP"
    assert plan_model(model(engine="mock", gpu=Gpu(count=0))).status == "SKIP"
