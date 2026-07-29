"""AegisRouter LiteLLM 启动脚本

替代直接调用 `litellm --config ...`。
不使用 --num_workers 参数，确保单进程模式下 callback 注册正确。
"""

import sys
from litellm.proxy.proxy_cli import run_server

sys.argv = ["litellm", "--config", "/app/config/config.yaml", "--port", "8000"]
run_server()
