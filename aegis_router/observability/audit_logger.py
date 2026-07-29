"""审计日志模块

提供结构化 JSON 审计日志记录，覆盖以下审计事件类型:
- route_decision: 路由决策事件
- compliance_check: 合规检测事件（注入检测、敏感词命中）
- degradation_change: 降级模式变更事件
- request_lifecycle: 请求全生命周期耗时记录

所有审计日志通过独立的 aegis_router.audit logger 输出，不包含任何原始 PII。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional


# Dedicated audit logger — separate from application logs
audit_logger = logging.getLogger("aegis_router.audit")


class _JsonFormatter(logging.Formatter):
    """极简 JSON 格式化器 — 直接输出 message 本身（已是 JSON 字符串）。"""

    def format(self, record: logging.LogRecord) -> str:
        # message 本身已经是完整的 JSON 字符串，直接返回
        return record.getMessage()


def configure_audit_handler(
    *,
    level: int = logging.INFO,
    stream: bool = True,
    file_path: Optional[str] = None,
) -> None:
    """配置 aegis_router.audit logger 的处理器。

    为审计 logger 设置专用 handler（stdout 和/或文件），使用 JSON 格式化器。
    调用此函数前审计日志仍可通过 caplog 等方式捕获，但不会输出到 stdout/file。

    Args:
        level: 日志级别，默认 INFO。
        stream: 是否添加 stdout handler，默认 True。
        file_path: 可选的审计日志文件路径。若提供则额外添加 FileHandler。
    """
    audit_logger.setLevel(level)
    # 避免日志向上传播到 root logger（防止重复输出）
    audit_logger.propagate = False

    # 清除已有 handler（幂等调用）
    audit_logger.handlers.clear()

    formatter = _JsonFormatter()

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        audit_logger.addHandler(stream_handler)

    if file_path:
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        audit_logger.addHandler(file_handler)


class AuditLogger:
    """结构化 JSON 审计日志记录器。

    记录路由决策、合规检测、降级变更、请求生命周期等事件，确保:
    - 每条记录为单行合法 JSON
    - 不包含任何原始 PII（仅 prompt_hash）
    - 包含完整的路由决策上下文（分数、候选列表、最终模型）
    """

    # ------------------------------------------------------------------
    # Route Decision
    # ------------------------------------------------------------------

    def log_route_decision(
        self,
        *,
        request_id: str,
        session_id: str,
        prompt_hash: str,
        prompt_length: int,
        route_score: Optional[float],
        candidates: list[str],
        target_model: str,
        route_reason: str,
        latency_mask_ms: float = 0.0,
        latency_route_ms: float = 0.0,
        entities_detected: Optional[list[str]] = None,
        api_key_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """记录一条路由决策审计日志。

        Args:
            request_id: 请求唯一标识
            session_id: 会话唯一标识
            prompt_hash: 原始 prompt 的 SHA-256 哈希（非原文）
            prompt_length: 原始 prompt 字符长度
            route_score: RouteLLM 分类器打分（0~1），规则命中时为 None
            candidates: 候选模型名称列表
            target_model: 最终选中的模型
            route_reason: 路由原因（如 single_match, overlap_lowest_cost 等）
            latency_mask_ms: PII 脱敏耗时（毫秒）
            latency_route_ms: 路由决策耗时（毫秒）
            entities_detected: 检测到的 PII 实体类型列表
            api_key_hash: API Key 的 SHA-256 哈希（可选）

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": "route_decision",
            "request_id": request_id,
            "session_id": session_id,
            "prompt_hash": prompt_hash,
            "prompt_length": prompt_length,
            "route_score": route_score,
            "candidates": candidates,
            "target_model": target_model,
            "route_reason": route_reason,
            "latency_mask_ms": round(latency_mask_ms, 2),
            "latency_route_ms": round(latency_route_ms, 2),
            "entities_detected": entities_detected or [],
        }

        if api_key_hash is not None:
            entry["api_key_hash"] = api_key_hash

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry

    # ------------------------------------------------------------------
    # Compliance Check
    # ------------------------------------------------------------------

    def log_compliance_event(
        self,
        *,
        request_id: str,
        session_id: str,
        check_type: str,
        passed: bool,
        mode: str,
        violations: Optional[list[str]] = None,
        details: Optional[str] = None,
    ) -> dict[str, Any]:
        """记录一条合规检测审计日志。

        Args:
            request_id: 请求唯一标识
            session_id: 会话唯一标识
            check_type: 检测类型 (如 "injection", "sensitive_word")
            passed: 检测是否通过 (True=通过, False=违规)
            mode: 合规模式 ("strict" / "permissive")
            violations: 违规项列表（如匹配到的注入模式或敏感词）
            details: 额外说明信息

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": "compliance_check",
            "request_id": request_id,
            "session_id": session_id,
            "check_type": check_type,
            "passed": passed,
            "mode": mode,
            "violations": violations or [],
        }

        if details is not None:
            entry["details"] = details

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry

    # ------------------------------------------------------------------
    # Degradation Change
    # ------------------------------------------------------------------

    def log_degradation_event(
        self,
        *,
        request_id: str,
        session_id: str,
        component: str,
        previous_state: str,
        current_state: str,
        action: str,
        fallback_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """记录一条降级模式变更审计日志。

        Args:
            request_id: 请求唯一标识
            session_id: 会话唯一标识
            component: 发生降级的组件 ("clawvault" / "redis" / "classifier")
            previous_state: 变更前的状态 ("healthy" / "unhealthy" / "unknown")
            current_state: 变更后的状态
            action: 降级动作描述 (如 "bypass_masking", "reject_request", "use_fallback_model")
            fallback_model: 降级后使用的兜底模型（仅 classifier 降级时提供）

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": "degradation_change",
            "request_id": request_id,
            "session_id": session_id,
            "component": component,
            "previous_state": previous_state,
            "current_state": current_state,
            "action": action,
        }

        if fallback_model is not None:
            entry["fallback_model"] = fallback_model

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry

    # ------------------------------------------------------------------
    # Request Lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Plan Generation (FR-8.1)
    # ------------------------------------------------------------------

    def log_plan_generation_event(
        self,
        *,
        trigger_reason: str,
        template_name: str,
        assignments: dict[str, str],
        total_agents: int,
    ) -> dict[str, Any]:
        """记录一条方案生成审计日志。

        在系统启动或配置变更时，为每个模板生成方案后记录。

        Args:
            trigger_reason: 触发方案生成的原因（如 "startup", "models.yaml" 等）
            template_name: 模板名称
            assignments: Agent 到模型的映射 {agent_name → model_name}
            total_agents: 模板中 Agent 总数

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "event": "plan_generation",
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trigger_reason": trigger_reason,
            "template_name": template_name,
            "assignments": assignments,
            "total_agents": total_agents,
        }

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry

    # ------------------------------------------------------------------
    # Transaction Dispatch (FR-8.2)
    # ------------------------------------------------------------------

    def log_dispatch_event(
        self,
        *,
        request_id: str,
        template: str,
        agent: str,
        assigned_model: str,
        reason: str,
        warnings: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """记录一条事务分发审计日志。

        每次事务级路由分发请求时记录。

        Args:
            request_id: 请求唯一标识
            template: 模板名称
            agent: Agent 名称
            assigned_model: 分配的模型名称
            reason: 分发原因（"plan", "failover", "fallback", "unknown"）
            warnings: 告警列表（如 UNKNOWN_AGENT）

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "event": "transaction_dispatch",
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": request_id,
            "template": template,
            "agent": agent,
            "assigned_model": assigned_model,
            "reason": reason,
            "warnings": warnings or [],
        }

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry

    # ------------------------------------------------------------------
    # Config Change (FR-8.3)
    # ------------------------------------------------------------------

    def log_config_change_event(
        self,
        *,
        changed_files: list[str],
        trigger_reason: str,
        plan_diff_summary: dict[str, Any],
        total_changes: int,
    ) -> dict[str, Any]:
        """记录一条配置变更审计日志。

        当配置变更触发方案重算时记录。

        Args:
            changed_files: 变更的文件列表
            trigger_reason: 触发原因（变更文件名拼接）
            plan_diff_summary: 方案差异摘要，包含:
                - added_templates: 新增模板列表
                - removed_templates: 删除模板列表
                - changed_assignments: 变更的分配列表
                  [{template, agent, old_model, new_model}, ...]
            total_changes: 所有变更的总计数

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "event": "config_change",
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "changed_files": changed_files,
            "trigger_reason": trigger_reason,
            "plan_diff_summary": plan_diff_summary,
            "total_changes": total_changes,
        }

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry

    # ------------------------------------------------------------------
    # Request Lifecycle
    # ------------------------------------------------------------------

    def log_request_lifecycle(
        self,
        *,
        request_id: str,
        session_id: str,
        phase: str,
        target_model: Optional[str] = None,
        latency_mask_ms: float = 0.0,
        latency_route_ms: float = 0.0,
        latency_llm_ms: float = 0.0,
        latency_restore_ms: float = 0.0,
        latency_total_ms: float = 0.0,
        status: str = "success",
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        """记录一条请求全生命周期审计日志。

        涵盖分步骤耗时打点：脱敏耗时、路由决策耗时、LLM 响应耗时、还原耗时。

        Args:
            request_id: 请求唯一标识
            session_id: 会话唯一标识
            phase: 生命周期阶段 ("start" / "end" / "error")
            target_model: 最终路由到的模型
            latency_mask_ms: PII 脱敏耗时（毫秒）
            latency_route_ms: 路由决策耗时（毫秒）
            latency_llm_ms: LLM API 响应耗时（毫秒）
            latency_restore_ms: 占位符还原耗时（毫秒）
            latency_total_ms: 请求总耗时（毫秒）
            status: 请求状态 ("success" / "error" / "degraded")
            error: 错误信息（仅在 phase="error" 时提供）

        Returns:
            生成的审计日志条目字典
        """
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": "request_lifecycle",
            "request_id": request_id,
            "session_id": session_id,
            "phase": phase,
            "status": status,
            "latency_mask_ms": round(latency_mask_ms, 2),
            "latency_route_ms": round(latency_route_ms, 2),
            "latency_llm_ms": round(latency_llm_ms, 2),
            "latency_restore_ms": round(latency_restore_ms, 2),
            "latency_total_ms": round(latency_total_ms, 2),
        }

        if target_model is not None:
            entry["target_model"] = target_model

        if error is not None:
            entry["error"] = error

        audit_logger.info(json.dumps(entry, ensure_ascii=False))
        return entry
