# =============================================================================
# AegisRouter — 智能安全 LLM 路由网关
# 使用 Supervisor 管理 ClawVault + LiteLLM Proxy 双进程
# 优化镜像大小：使用轻量 stub 包 + zh_core_web_sm 模型
# =============================================================================
FROM python:3.11-slim AS base

LABEL maintainer="AegisRouter Team"
LABEL version="0.1.0"
LABEL description="AegisRouter - Intelligent Security LLM Routing Gateway with PII masking"

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --------------------------------------------------------------------------
# 系统依赖：安装 curl（HEALTHCHECK 需要）和基础工具
# --------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 创建非 root 用户（安全最佳实践）
# --------------------------------------------------------------------------
RUN groupadd -r aegis && useradd -r -g aegis -d /app -s /sbin/nologin aegis

# --------------------------------------------------------------------------
# 阶段 1: 安装轻量 vendor stub 包（仅提供 aegis_router 实际使用的接口）
# 完整的 litellm/routellm/clawvault 作为外部服务运行，不需要打入镜像
# --------------------------------------------------------------------------
COPY docker/stubs/ ./docker_stubs/
RUN pip install ./docker_stubs/litellm && \
    pip install ./docker_stubs/routellm && \
    pip install ./docker_stubs/clawvault && \
    rm -rf ./docker_stubs

# --------------------------------------------------------------------------
# 阶段 2: 安装第三方依赖（requirements.txt 变更时才重建）
# --------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# --------------------------------------------------------------------------
# 阶段 3: 安装 spaCy 中文模型（使用 sm 模型以控制镜像大小）
# zh_core_web_sm (~15MB) vs zh_core_web_trf (~400MB+, 需要 PyTorch)
# --------------------------------------------------------------------------
RUN python -m spacy download zh_core_web_sm

# --------------------------------------------------------------------------
# 阶段 4: 复制应用代码和配置
# --------------------------------------------------------------------------
COPY aegis_router/ ./aegis_router/
COPY config/ ./config/
COPY patterns/ ./patterns/
COPY pyproject.toml .
RUN pip install -e .

# --------------------------------------------------------------------------
# 阶段 5: Supervisor 配置
# --------------------------------------------------------------------------
COPY supervisord.conf /etc/supervisord.conf

# --------------------------------------------------------------------------
# 清理不需要的文件，减小镜像体积
# --------------------------------------------------------------------------
RUN find /usr/local/lib/python3.11 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11 -name "*.pyc" -delete 2>/dev/null || true && \
    rm -rf /root/.cache /tmp/*

# --------------------------------------------------------------------------
# 设置文件权限，切换到非 root 用户
# --------------------------------------------------------------------------
RUN chown -R aegis:aegis /app
USER aegis

# --------------------------------------------------------------------------
# 健康检查：通过 curl 检测 LiteLLM Proxy 健康端点
# --------------------------------------------------------------------------
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
