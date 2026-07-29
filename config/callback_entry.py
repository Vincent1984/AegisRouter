"""LiteLLM callback 入口点

LiteLLM 通过 litellm_settings.callbacks 引用此文件的 proxy_handler_instance 变量。
此文件根据 config.yaml 中的 routing_plugin 字段动态加载对应的路由插件。

LiteLLM Proxy 的 initialize_callbacks_on_proxy() 会：
1. 调用 get_instance_fn("callback_entry.proxy_handler_instance", config_file_path=...)
2. 加载本文件并获取 proxy_handler_instance
3. 将其放入 litellm.callbacks 列表
"""

import sys
from aegis_router.callbacks.plugin_loader import load_routing_plugin

print("[callback_entry] Loading routing plugin...", file=sys.stderr, flush=True)
proxy_handler_instance = load_routing_plugin(config_dir="./config")
print(f"[callback_entry] Plugin loaded: {type(proxy_handler_instance).__name__}", file=sys.stderr, flush=True)
