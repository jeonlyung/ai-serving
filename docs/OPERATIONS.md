# 운영 가이드

## 처음 올리기

```bash
cp deploy/compose/.env.example deploy/compose/.env    # HF_TOKEN(gated 모델), ROUTER_API_KEYS 채우기
servectl plan                                         # FAIL 이 없는지
servectl pull                                         # (선택) 가중치를 미리 받아 두기
cd deploy/compose && docker compose up -d
servectl smoke --api-key $KEY                         # 모델 로딩에 수 분 — 실패하면 잠시 후 재시도
```

NVIDIA Container Toolkit 이 호스트에 있어야 한다 (`docker run --gpus all ... nvidia-smi`).

## 모델 교체 · 추가

`catalog/models.yaml` 수정 → `make plan` → `make render` → 배포.
K8s 는 `kubectl apply -k deploy/k8s`. 롤링은 GPU 가 모자라므로 **먼저 내리고 올린다**(maxSurge 0)
— 복제본이 2개 이상이어야 무중단이다.

## 튜닝 순서 (효과 큰 순)

1. **`max_model_len`** 을 실제 필요한 길이로 줄인다. KV 풀이 가장 크게 늘어난다.
2. **FP8** (`quantization: fp8`, `kv_cache_dtype: fp8`) — Hopper/Ada 이상. 가중치·KV 모두 절반.
3. **`max_num_seqs`** — 동시 처리 상한. 크면 처리량↑ TTFT↑. `plan` 의 '동시(평균)' 이하로.
4. **프리픽스 캐시**(기본 on) — RAG 처럼 앞부분(시스템 프롬프트·문서)이 반복되는 요청에서 TTFT 가 크게 줄어든다. 프롬프트 앞쪽을 고정하자.
5. 더 큰 모델은 TP 를 늘리되 NVLink 가 없으면 TP 보다 양자화가 낫다.
6. 엔진 교체 (`engine: sglang`) — 구조화 출력·공유 프리픽스가 많은 워크로드에서 비교해 볼 만하다. 카탈로그 한 줄이다.

## 부하시험

간이: `servectl bench --model answer -j 16 -n 200`.
용량 산정은 실제 프롬프트 분포로 공식 도구를 쓴다.

```bash
# vLLM 내장
docker exec <vllm-container> vllm bench serve --model google/gemma-3-27b-it \
  --dataset-name random --random-input-len 2000 --random-output-len 300 \
  --request-rate 4 --num-prompts 200
# 또는 GuideLLM (SLO 기반 스윕)
pip install guidellm && guidellm benchmark --target http://localhost:8000 --rate-type sweep
```
SLO 예: TTFT p95 < 2s, 토큰 간 지연 < 50ms. 이를 만족하는 최대 동시 사용자 수가 복제본 1개의 용량이다.

## 알림이 울리면

| 알림 | 먼저 볼 것 |
|---|---|
| LLMEngineDown | `docker logs`. 로딩 중(수 분)이면 정상. OOM 이면 `gpu_memory_utilization`/`max_model_len` ↓ |
| LLMNoHealthyReplicas | `GET /admin/upstreams` (라우터). 폴백이 받아주는 중이라면 품질 저하 상태 |
| LLMRequestQueueBacklog | 복제본 증설 / `max_num_seqs` 상향. K8s 는 KEDA 가 자동 |
| LLMKVCacheNearFull | `max_model_len` ↓, FP8 KV, 복제본 증설. 방치하면 선점으로 지연이 튄다 |
| LLMHighTTFT | 긴 프롬프트 폭주(청크 프리필 `--enable-chunked-prefill` 확인), 대기열 동반 여부 |
| LLMFallbackActive | 주 모델 복구 후 사라지는지. 지속되면 사용자에게 답변 품질 영향 공지 |

## 보안 체크리스트

- [ ] `ROUTER_API_KEYS` 설정 (비면 인증 없음). 엔진 포트는 외부에 열지 않는다 (compose 는 엔진 포트를 publish 하지 않음).
- [ ] 라우터 앞단에 TLS 종단(Ingress/Nginx). 라우터 자체는 평문 HTTP.
- [ ] Prometheus 는 127.0.0.1 바인딩 — 외부 노출 시 인증 추가. Grafana 기본 비밀번호 변경.
- [ ] 폐쇄망: 이미지를 사내 레지스트리로 미러링하고 `HF_HUB_OFFLINE=1` + `servectl pull` 로 받아 둔 캐시를 PVC 에 올린다.
- [ ] 모델 라이선스(Gemma 약관, Qwen 등) 확인 — 사업 사용 조건이 다르다.

## 버전 올리기

`catalog/models.yaml` 의 `engines.*.image` 태그만 바꾸고 `make render`. 올리기 전에
`docker run --rm <image> --help` 로 사용하는 플래그(`--reasoning-parser` 등)가 남아 있는지 확인한다 —
엔진은 릴리스마다 플래그가 바뀌는 일이 잦다.
