# 아키텍처

```
 ai-core(게이트웨이·워크플로우) ──▶  router  ──▶ vLLM  gemma-3-27b (TP=2)
        OpenAI 호환 /v1             │  │      ├─▶ vLLM  gemma-3-4b ×2 (복제본)
                                    │  └─폴백─▶ vLLM  Qwen3-32B (FP8, tool-call)
                                    └─ /metrics
 Prometheus ◀── 엔진 /metrics · router /metrics · DCGM(GPU)  ──▶ Grafana · Alertmanager
 (K8s) KEDA ◀── vllm:num_requests_waiting ──▶ 엔진 복제본 수 조절
```

## 왜 이렇게 나눴나

| 결정 | 이유 |
|---|---|
| **엔진은 직접 만들지 않고 vLLM/SGLang/llama.cpp 를 그대로 쓴다** | 추론 엔진은 연속 배칭·PagedAttention·프리픽스 캐시 등 성능이 전부인 영역이라 오픈소스 최신 릴리스를 따라가는 게 맞다. 우리가 만드는 건 *배포·라우팅·관측* 이다. |
| **카탈로그 YAML 한 장이 단일 진실** | compose·K8s·라우터·Prometheus·메모리 산정이 같은 파일에서 나온다. 모델을 바꿀 때 4곳을 따로 고치다 어긋나는 사고를 막는다. 산출물이 어긋나면 CI(`render --check`)가 실패한다. |
| **라우터를 별도 컴포넌트로** | ai-core 는 base_url 하나만 안다. 모델명(별칭)→풀 라우팅, 복제본 분산, 죽은 복제본 제외, 폴백 모델, 키 인증을 엔진 앞단 한 곳에서 처리한다. |
| **라우터는 정책을 모른다** | 사용자 권한·마스킹·프롬프트는 ai-core 책임. 라우터는 "요청을 살아 있는 엔진에 전달"만 한다. 계층이 섞이지 않는다. |
| **오토스케일은 GPU 사용률이 아니라 대기열** | LLM 은 GPU 사용률이 일찍 100% 가 된다. 사용자가 느끼는 지연은 `num_requests_waiting` 에 먼저 나타난다. |
| **mock 엔진** | vLLM 과 같은 메트릭 이름을 내는 가짜 엔진. GPU 없이 라우터·대시보드·알림·CI 를 끝까지 검증한다. |

## 라우터 동작

1. `model`(또는 별칭) → 풀 선택. 모르면 404.
2. 풀에서 **진행중 요청이 가장 적은** 복제본 선택 (동률이면 라운드로빈).
3. 연결 실패·429·5xx → 스트림이 시작되기 **전이면** 다른 복제본으로 재시도, 연속 N회 실패하면 일정 시간 제외.
4. 모든 복제본이 실패하면 `fallback` 체인의 다음 모델로. (본문의 `model` 은 폴백 모델 id 로 다시 쓴다.)
5. 응답 헤더 `x-served-model`·`x-upstream`·`x-request-id` 로 어디서 처리됐는지 추적.
6. 별도로 5초마다 `/health` 를 찔러 모델 로딩 중인 복제본에는 트래픽을 보내지 않는다.

한계: 스트리밍이 **시작된 뒤** 엔진이 죽으면 재시도하지 않는다 (이미 일부 토큰이 나갔다). 클라이언트(ai-core)가 재시도·폴백한다.

## 메모리 산정 공식

```
KV 토큰당 = 2 × layers × kv_heads × head_dim × (2B | FP8 1B)
KV 풀     = GPU메모리 × utilization × TP − 가중치 − 2.5GiB × TP
동시 처리  ≈ KV 풀 ÷ KV 토큰당 ÷ 평균 컨텍스트
```
`servectl plan` 이 계산한다. 슬라이딩 윈도우 모델(Gemma 3)은 실제로 KV 가 더 적게 들어 보수적으로 나온다.
