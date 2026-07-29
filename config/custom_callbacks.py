"""LiteLLM Custom Callback — 由 litellm_settings.callbacks 加载

LiteLLM Proxy 通过 config.yaml 中 litellm_settings.callbacks 引用：
  callbacks: custom_callbacks.proxy_handler_instance

LiteLLM 在每个 worker 的 load_config 阶段执行此文件，
获取 proxy_handler_instance 并放入 litellm.callbacks。

同时注册 RequestLoggerCallback（请求日志回调）到 litellm.callbacks。
同时将 health_router 注册到 LiteLLM Proxy 的 FastAPI app 中。
"""

import sys
import litellm
from aegis_router.callbacks.plugin_loader import load_routing_plugin

print("[custom_callbacks] Loading routing plugin...", file=sys.stderr, flush=True)
proxy_handler_instance = load_routing_plugin(config_dir="/app/config")
print(f"[custom_callbacks] Plugin: {type(proxy_handler_instance).__name__}", file=sys.stderr, flush=True)

# 注册 health_router 到 LiteLLM Proxy 的 FastAPI app
try:
    from litellm.proxy.proxy_server import app
    from aegis_router.health import health_router
    app.include_router(health_router)
    print("[custom_callbacks] health_router: registered", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[custom_callbacks] health_router registration failed: {e}", file=sys.stderr, flush=True)

# 注册 RequestLoggerCallback（请求日志）
try:
    from aegis_router.observability.request_logger import (
        RequestLoggerCallback,
        load_request_logging_config,
    )
    req_log_config = load_request_logging_config("/app/config")
    if req_log_config.enabled:
        request_logger_instance = RequestLoggerCallback(config=req_log_config)
        litellm.callbacks.append(request_logger_instance)
        print(f"[custom_callbacks] RequestLogger: enabled", file=sys.stderr, flush=True)
    else:
        print("[custom_callbacks] RequestLogger: disabled", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[custom_callbacks] RequestLogger failed: {e}", file=sys.stderr, flush=True)
