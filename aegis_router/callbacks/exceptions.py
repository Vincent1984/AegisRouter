"""AegisRouter 回调异常定义。

自定义异常类，携带 LiteLLM Proxy 所需的属性（message, status_code, type），
使异常能正确映射为 HTTP 响应码。
"""

from __future__ import annotations


class TemplateNotFoundError(Exception):
    """引用不存在的路由模板时抛出 — 映射为 HTTP 400。

    属性:
        message: 错误描述（LiteLLM Proxy 通过 getattr(e, "message", ...) 读取）。
        status_code: HTTP 状态码（LiteLLM Proxy 通过 getattr(e, "status_code", 500) 读取）。
        type: 错误类型（LiteLLM Proxy 通过 getattr(e, "type", "None") 读取）。
    """

    def __init__(self, template_name: str) -> None:
        self.message = f"Template '{template_name}' not found"
        self.status_code = 400
        self.type = "invalid_request_error"
        super().__init__(self.message)
