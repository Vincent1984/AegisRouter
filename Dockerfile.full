# =============================================================================
# AegisRouter — 完整版镜像（真实 LiteLLM 从 PyPI 安装）
# 用于真实环境验收测试
# =============================================================================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 安装 LiteLLM 从 PyPI（比 vendor 源码安装快得多）
# --------------------------------------------------------------------------
RUN pip install litellm[proxy]==1.72.6

# --------------------------------------------------------------------------
# 安装项目依赖（排除 litellm 因为已装）
# --------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt || true

# 安装 spaCy 中文模型
RUN pip install --upgrade pip && \
    pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl

# --------------------------------------------------------------------------
# 安装 routellm stub（真实路由测试可后续替换）
# --------------------------------------------------------------------------
COPY docker/stubs/routellm/ ./docker_stubs/routellm/
RUN pip install ./docker_stubs/routellm && rm -rf ./docker_stubs
COPY docker/stubs/clawvault/ ./docker_stubs/clawvault/
RUN pip install ./docker_stubs/clawvault && rm -rf ./docker_stubs

# --------------------------------------------------------------------------
# 复制应用代码
# --------------------------------------------------------------------------
COPY aegis_router/ ./aegis_router/
COPY config/ ./config/
COPY patterns/ ./patterns/
COPY start_litellm.py .
COPY pyproject.toml .
RUN pip install -e .

# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------
COPY supervisord.conf /etc/supervisord.conf

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=30s \
  CMD curl -sf -H "Authorization: Bearer ${AEGIS_MASTER_KEY}" http://localhost:8000/health/liveliness || exit 1

EXPOSE 8000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
