# AI Serving — 오픈소스 LLM 서빙 스택

[ai-core](../ai-core)(LLM 게이트웨이·워크플로우) 아래에 놓이는 **모델 서빙 계층**입니다.
추론 엔진은 직접 만들지 않고 **vLLM · SGLang · llama.cpp** 를 그대로 쓰고,
이 프로젝트는 그 둘레의 *배포 · 라우팅 · 관측 · 용량 산정* 을 맡습니다.

```
ai-core ──(OpenAI 호환 /v1)──▶ router ──▶ vLLM gemma-3-27b (TP=2) ─ 죽으면 ─┐
                                 │   ├──▶ vLLM gemma-3-4b ×2                │ 폴백
                                 │   └──▶ vLLM Qwen3-32B (FP8, tool-call) ◀─┘
                                 └ /metrics
Prometheus ◀ 엔진·라우터·GPU(DCGM) ─▶ Grafana 대시보드 · 알림 · (K8s) KEDA 자동확장
```

**모델 카탈로그 YAML 한 장**([catalog/models.yaml](catalog/models.yaml))에서 모든 것이 나옵니다.

| 산출물 | 명령 |
|---|---|
| GPU 메모리 산정 (올리기 전 "들어가나?") | `servectl plan` |
| docker-compose 스택 (엔진+라우터+Prometheus+Grafana+DCGM) | `servectl render compose` |
| Kubernetes 매니페스트 (Kustomize, KEDA, PDB, PodMonitor, 알림 규칙) | `servectl render k8s` |
| GPU 없이 전체 스택 검증 (mock 엔진) | `servectl render compose --mock` |
| 배포 후 스모크 · 간이 부하시험 | `servectl smoke` · `servectl bench` |

## 5분 체험 (GPU 불필요)

```bash
make install
make mock-up                 # 라우터 :8000 · Grafana :3000(admin/admin) · Prometheus :9090
make smoke                   # 모델별 chat + 스트리밍 확인
.venv/bin/servectl bench --model answer -n 60 -j 12 --max-tokens 32
make mock-down
```

mock 엔진은 vLLM 과 **같은 메트릭 이름**(`vllm:num_requests_waiting` 등)을 내므로 대시보드·알림이 그대로 동작합니다.
`POST /admin/chaos {"error_rate":1}` 로 장애를 주입해 폴백을 직접 볼 수 있습니다.

## 실제 GPU 로 올리기

```bash
.venv/bin/servectl plan                       # FAIL 이 있으면 TP/양자화/max_model_len 조정
cp deploy/compose/.env.example deploy/compose/.env   # HF_TOKEN, ROUTER_API_KEYS
cd deploy/compose && docker compose up -d     # 가중치 로딩에 수 분 (healthcheck start_period 600s)
servectl smoke --api-key $ROUTER_API_KEY
```

호스트에 NVIDIA Container Toolkit 필요. K8s 는 `kubectl apply -k deploy/k8s`
(GPU Operator · Prometheus Operator · KEDA 전제, `secret.example.yaml` 참고).

## ai-core 연결

ai-core 코드는 **수정하지 않습니다.** 환경변수만 바꿉니다 (연동 확인 완료).

```ini
AICORE_LLM_MODE=openai
AICORE_LLM_BASE_URL=http://<router>:8000/v1
AICORE_LLM_API_KEY=<ROUTER_API_KEYS 중 하나>
AICORE_MODEL_ANSWER=google/gemma-3-27b-it     # 또는 별칭 answer
AICORE_MODEL_CLASSIFY=google/gemma-3-4b-it
AICORE_MODEL_FALLBACK=google/gemma-3-4b-it
```
라우터 폴백(서버측)과 ai-core 폴백(클라이언트측)이 겹치지만 문제 없습니다. 스트리밍 도중 끊긴 경우만 ai-core 가 처리합니다.

## 카탈로그

```yaml
- id: Qwen/Qwen3-32B            # 클라이언트가 보내는 model 값
  aliases: [agent]              # 용도별 별칭
  engine: vllm                  # vllm | sglang | llamacpp | mock
  gpu: { type: H100-80G, count: 1, memory_gb: 80 }
  quantization: fp8
  kv_cache_dtype: fp8
  max_model_len: 32768
  tool_call_parser: hermes      # 에이전트 도구 호출
  reasoning_parser: qwen3
  fallback: google/gemma-3-4b-it
  autoscale: { min: 1, max: 4, waiting_threshold: 8 }   # K8s + KEDA
  arch: { params_b: 32, layers: 64, kv_heads: 8, head_dim: 128 }   # plan 용
```
오타 필드·TP>GPU·id/별칭 중복·없는 폴백 대상은 `servectl validate` 에서 걸립니다.

## 구조

```
catalog/models.yaml       단일 진실 공급원
src/servectl/
  catalog.py planner.py   스키마·검증 / GPU 메모리 산정
  engines.py              엔진별 실행 인자 (vLLM·SGLang·llama.cpp)
  render_compose.py render_k8s.py render_common.py
  router/                 모델 라우터 (분산·재시도·폴백·인증·메트릭)
  mockengine.py           GPU 없는 OpenAI 호환 엔진
  smoke.py bench.py pull.py cli.py
  assets/                 Prometheus 알림 규칙 · Grafana 대시보드
deploy/                   렌더 결과 (커밋됨, CI 가 최신 여부 검사)
docs/                     ARCHITECTURE · OPERATIONS
```

## 검증 상태 (솔직하게)

| 항목 | 상태 |
|---|---|
| 단위·통합 테스트 39개 (라우터 폴백/제외/스트리밍/인증, 렌더링, 메모리 산정) | ✅ 통과 |
| mock 스택 Docker 기동 → 스모크·부하·폴백·Prometheus 수집·Grafana 프로비저닝 | ✅ 실제 확인 |
| `kubectl kustomize` 빌드, `docker-compose config` | ✅ |
| ai-core 가 라우터를 통해 chat/stream/health | ✅ |
| **실제 GPU 에서 vLLM/SGLang 기동** | ⚠️ **미검증** — 이 환경에 GPU 가 없음 |
| 이미지 태그(vLLM v0.34.0, SGLang v0.5.23) 존재 | ✅ 레지스트리 확인. 단, 엔진 플래그는 릴리스마다 바뀌므로 첫 기동 때 `--help` 로 대조 권장 |

`arch` 수치는 각 모델 `config.json` 기준 값을 옮겨 적은 것이니 새 모델을 넣을 때 직접 확인하세요.
라이선스 파일은 아직 없습니다 — 배포 전에 정해 주세요 (Gemma·Qwen 등 모델 라이선스는 별개).

문서: [아키텍처](docs/ARCHITECTURE.md) · [운영 가이드](docs/OPERATIONS.md)
