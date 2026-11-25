# 라우터 + 모의 엔진 + servectl 을 한 이미지로 (실제 엔진은 vLLM/SGLang 공식 이미지를 쓴다)
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
RUN useradd -r -u 10001 -m app
COPY --from=build /install /usr/local
USER 10001
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/healthz',timeout=2)" || exit 1
CMD ["uvicorn", "servectl.router.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
