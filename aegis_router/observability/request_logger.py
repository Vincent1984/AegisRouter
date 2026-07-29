"""请求日志模块

提供 RequestLoggerCallback 的配置模型和配置加载函数。
RequestLoggerCallback 是一个独立的 LiteLLM CustomLogger 回调，为通过 AegisRouter 的每个
LLM 请求记录完整的请求-响应生命周期（请求体、路由决策、响应内容、Token 用量和延迟）。

该模块包含:
- RequestLoggingConfig: 请求日志配置 Pydantic 模型
- load_request_logging_config: 从 config.yaml 加载 request_logging 段
- RequestLoggerCallback: 独立的请求日志 CustomLogger 回调
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel

from litellm.integrations.custom_logger import CustomLogger


logger = logging.getLogger(__name__)


class RequestLoggingConfig(BaseModel):
    """请求日志回调配置。"""

    enabled: bool = True
    output: Literal["stdout", "file", "both"] = "file"
    file_path: str = "./logs/request_log.jsonl"
    max_message_length: int = 4096  # 字符数；0 = 不截断
    retention_days: int = 30  # 文件日志保留天数
    log_level: str = "INFO"


def load_request_logging_config(
    config_dir: str | Path = "./config",
) -> RequestLoggingConfig:
    """从 config.yaml 加载 request_logging 段。

    Args:
        config_dir: 配置文件所在目录路径。

    Returns:
        RequestLoggingConfig 实例。配置文件缺失、格式错误或
        request_logging 段为空时返回 enabled=False 的默认配置。
    """
    config_path = Path(config_dir) / "config.yaml"
    if not config_path.exists():
        return RequestLoggingConfig(enabled=False)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to parse config.yaml for request_logging: %s", e)
        return RequestLoggingConfig(enabled=False)

    section = raw.get("request_logging", {})
    if not section:
        return RequestLoggingConfig(enabled=False)

    try:
        return RequestLoggingConfig(**section)
    except Exception as e:
        logger.warning("Invalid request_logging config: %s", e)
        return RequestLoggingConfig(enabled=False)


class _LogEntryBuilder:
    """从原始回调数据构建结构化日志条目。"""

    @staticmethod
    def _make_timestamp() -> str:
        """生成 UTC ISO-8601 毫秒精度时间戳，Z 后缀。"""
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _truncate_messages(
        messages: list, max_length: int
    ) -> list:
        """截断超过 max_message_length 的消息内容。"""
        if max_length <= 0:
            return messages

        truncated = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_length:
                content = content[:max_length] + " [truncated]"
            truncated.append({**msg, "content": content})
        return truncated

    @staticmethod
    def build_request_entry(data: dict, config: RequestLoggingConfig) -> dict:
        """构建 'request' 事件日志条目。"""
        metadata = data.get("metadata", {})
        request_id = metadata.get("request_id") or str(uuid.uuid4())
        session_id = metadata.get("session_id")

        messages = data.get("messages", [])
        truncated_messages = _LogEntryBuilder._truncate_messages(
            messages, config.max_message_length
        )

        model_requested = data.get("model")

        routing_decision = {
            "target_model": metadata.get("target_model"),
            "routing_plugin": metadata.get("routing_plugin"),
            "route_reason": metadata.get("route_reason"),
            "route_score": metadata.get("route_score"),
        }

        call_type = data.get("call_type")

        return {
            "ts": _LogEntryBuilder._make_timestamp(),
            "event_type": "request",
            "request_id": request_id,
            "session_id": session_id,
            "messages": truncated_messages,
            "model_requested": model_requested,
            "routing_decision": routing_decision,
            "call_type": call_type,
        }

    @staticmethod
    def build_success_entry(
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        config: RequestLoggingConfig,
    ) -> dict:
        """构建 'response_success' 事件日志条目。"""
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        if not metadata:
            metadata = kwargs.get("metadata", {})
        request_id = metadata.get("request_id") or str(uuid.uuid4())
        session_id = metadata.get("session_id")

        # Extract response text
        response_text: Optional[str] = None
        try:
            response_text = response_obj.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            if response_obj is not None:
                response_text = str(response_obj)

        # Apply truncation to response_text
        if (
            response_text is not None
            and config.max_message_length > 0
            and len(response_text) > config.max_message_length
        ):
            response_text = (
                response_text[: config.max_message_length] + " [truncated]"
            )

        # Extract model_used
        slo = kwargs.get("standard_logging_object", {}) or {}
        model_used = kwargs.get("model") or slo.get("model")

        # Extract usage from standard_logging_object
        usage = {
            "input_tokens": slo.get("prompt_tokens", 0),
            "output_tokens": slo.get("completion_tokens", 0),
            "total_tokens": slo.get("total_tokens", 0),
        }

        # Extract latency
        latency_ms = slo.get("response_time_ms") or slo.get(
            "completion_start_time_ms"
        )

        # Routing decision
        routing_decision = {
            "target_model": metadata.get("target_model"),
            "routing_plugin": metadata.get("routing_plugin"),
        }

        return {
            "ts": _LogEntryBuilder._make_timestamp(),
            "event_type": "response_success",
            "request_id": request_id,
            "session_id": session_id,
            "response_text": response_text,
            "model_used": model_used,
            "usage": usage,
            "latency_ms": latency_ms,
            "routing_decision": routing_decision,
        }

    @staticmethod
    def build_failure_entry(
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        config: RequestLoggingConfig,
    ) -> dict:
        """构建 'response_failure' 事件日志条目。"""
        metadata = kwargs.get("litellm_params", {}).get("metadata", {})
        if not metadata:
            metadata = kwargs.get("metadata", {})
        request_id = metadata.get("request_id") or str(uuid.uuid4())
        session_id = metadata.get("session_id")

        # Extract error info
        exception = kwargs.get("exception")
        if exception:
            error_message = str(exception)
            error_type = type(exception).__name__
        else:
            error_message = "Unknown error"
            error_type = "Unknown"

        # Extract model_used
        model_used = kwargs.get("model")

        # Extract usage and latency from standard_logging_object if available
        slo = kwargs.get("standard_logging_object")
        if slo:
            usage = {
                "input_tokens": slo.get("prompt_tokens", 0),
                "output_tokens": slo.get("completion_tokens", 0),
                "total_tokens": slo.get("total_tokens", 0),
            }
            latency_ms = slo.get("response_time_ms") or slo.get(
                "completion_start_time_ms"
            )
            incomplete_data = False
        else:
            usage = None
            latency_ms = None
            incomplete_data = True

        # Routing decision
        routing_decision = {
            "target_model": metadata.get("target_model"),
            "routing_plugin": metadata.get("routing_plugin"),
        }

        return {
            "ts": _LogEntryBuilder._make_timestamp(),
            "event_type": "response_failure",
            "request_id": request_id,
            "session_id": session_id,
            "error_message": error_message,
            "error_type": error_type,
            "model_used": model_used,
            "usage": usage,
            "latency_ms": latency_ms,
            "incomplete_data": incomplete_data,
            "routing_decision": routing_decision,
        }


class _JsonPassthroughFormatter(logging.Formatter):
    """极简 JSON 格式化器 — 直接输出 message 本身（已是 JSON 字符串）。"""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class RequestLoggerCallback(CustomLogger):
    """独立的请求日志回调 — 只观察和记录，不修改。

    继承 LiteLLM CustomLogger，为通过 AegisRouter 的每个 LLM 请求记录
    完整的请求-响应生命周期数据。使用独立的 logger 命名空间
    ``aegis_router.request_log``，与应用日志和审计日志分离。
    """

    def __init__(self, config: RequestLoggingConfig) -> None:
        super().__init__()
        self._config = config
        self._logger = logging.getLogger("aegis_router.request_log")
        self._configure_logger()

    def _configure_logger(self) -> None:
        """根据 output 配置设置 StreamHandler 和/或 TimedRotatingFileHandler。

        - 设置 log level 来自 config.log_level
        - propagate = False，避免日志向上传播
        - 清除已有 handler（幂等调用）
        - stdout/both: 添加 StreamHandler(sys.stdout)
        - file/both: 添加 TimedRotatingFileHandler
        """
        self._logger.setLevel(
            getattr(logging, self._config.log_level.upper(), logging.INFO)
        )
        self._logger.propagate = False
        self._logger.handlers.clear()

        formatter = _JsonPassthroughFormatter()

        if self._config.output in ("stdout", "both"):
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        if self._config.output in ("file", "both"):
            os.makedirs(
                os.path.dirname(self._config.file_path) or ".", exist_ok=True
            )
            handler = TimedRotatingFileHandler(
                self._config.file_path,
                when="midnight",
                interval=1,
                backupCount=self._config.retention_days,
                encoding="utf-8",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def _truncate_messages(self, messages: list, max_length: int) -> list:
        """截断超过 max_message_length 的消息内容。"""
        if max_length <= 0:
            return messages

        truncated = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_length:
                content = content[:max_length] + " [truncated]"
            truncated.append({**msg, "content": content})
        return truncated

    # ------------------------------------------------------------------
    # Hook placeholders — 后续 task 实现
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        """记录请求体和路由决策元数据。

        关键: 原样返回 data，绝不修改请求状态。
        """
        if not self._config.enabled:
            return data
        try:
            messages = data.get("messages")
            if not messages:
                return data
            # Inject call_type into data for the builder
            data_with_call_type = {**data, "call_type": call_type}
            entry = _LogEntryBuilder.build_request_entry(
                data_with_call_type, self._config
            )
            self._logger.info(json.dumps(entry, ensure_ascii=False))
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RequestLogger pre_call error: %s", e
            )
        return data

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """记录响应内容、Token 用量和延迟（来自 standard_logging_object）。"""
        if not self._config.enabled:
            return
        try:
            entry = _LogEntryBuilder.build_success_entry(
                kwargs, response_obj, start_time, end_time, self._config
            )
            self._logger.info(json.dumps(entry, ensure_ascii=False))
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RequestLogger success_event error: %s", e
            )

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """记录失败详情（包含可用的 standard_logging_object 数据）。"""
        if not self._config.enabled:
            return
        try:
            entry = _LogEntryBuilder.build_failure_entry(
                kwargs, response_obj, start_time, end_time, self._config
            )
            self._logger.info(json.dumps(entry, ensure_ascii=False))
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RequestLogger failure_event error: %s", e
            )
