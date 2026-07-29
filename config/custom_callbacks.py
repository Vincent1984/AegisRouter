"""LiteLLM Custom Callback — 由 litellm_settings.callbacks 加载

LiteLLM Proxy 通过 config.yaml 中 litellm_settings.callbacks 引用：
  callbacks: custom_callbacks.proxy_handler_instance

LiteLLM 在每个 worker 的 load_config 阶段执行此文件，
获取 proxy_handler_instance 并放入 litellm.callbacks。
"""

import sys
from aegis_router.callbacks.plugin_loader import load_routing_plugin

print("[custom_callbacks] Loading routing plugin...", file=sys.stderr, flush=True)
proxy_handler_instance = load_routing_plugin(config_dir="/app/config")
print(f"[custom_callbacks] Plugin: {type(proxy_handler_instance).__name__}", file=sys.stderr, flush=True)
