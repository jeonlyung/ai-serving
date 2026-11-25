.PHONY: install test lint render check mock-up mock-down smoke plan
PY ?= .venv/bin

install:
	python3 -m venv .venv && $(PY)/pip install -q -e '.[dev]'

test:
	$(PY)/python -m pytest -q

lint:
	$(PY)/ruff check src tests && $(PY)/ruff format --check src tests

plan:          ## GPU 메모리 산정
	$(PY)/servectl plan

render:        ## 카탈로그 → deploy/ 산출물
	$(PY)/servectl render all && $(PY)/servectl render compose --mock

check:         ## 산출물이 최신인지 (CI)
	$(PY)/servectl render all --check && $(PY)/servectl render compose --mock --check

mock-up:       ## GPU 없이 전체 스택 기동 (라우터 :8000, Grafana :3000, Prometheus :9090)
	cd deploy/compose-mock && docker-compose up -d --build

mock-down:
	cd deploy/compose-mock && docker-compose down -v

smoke:
	$(PY)/servectl smoke
