"""路由插件基类 (BaseRouterCallback)

抽取公共管道逻辑：合规检测、PII 脱敏、响应还原。
所有路由策略插件（对话级、事务级等）继承此基类，
只需实现 `_execute_routing()` 方法即可。
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import time
import uuid
from abc import abstractmethod
from typing import Any, AsyncGenerator, Optional

from litellm.integrations.custom_logger import CustomLogger

from aegis_router.callbacks.degradation import (
    ComponentState,
    DegradationManager,
)
from aegis_router.callbacks.stream_rehydrator import StreamRehydrator
from aegis_router.callbacks.uds_pool import ClawVaultPool

logger = logging.getLogger(__name__)

# Compliance mode
COMPLIANCE_MODE = os.environ.get("AEGIS_COMPLIANCE_MODE", "strict")


class BaseRouterCallback(CustomLogger):
    """路由插件基类 — 封装公共管道，子类实现路由策略。

    公共管道（始终执行）:
      1. 合规检测 (check_compliance)
      2. PII 脱敏 (mask)
      3. 响应占位符还原 (restore)
      4. 流式响应还原 (streaming restore)

    子类只需实现:
      - `_execute_routing(data, masked_text, original_text, prompt_hash)`
    """

    def __init__(
        self,
        pool: ClawVaultPool | None = None,
        degradation_manager: DegradationManager | None = None,
    ) -> None:
        super().__init__()
        self._pool = pool if pool is not None else ClawVaultPool()
        self._degradation = degradation_manager or DegradationManager()

    # ------------------------------------------------------------------
    # Abstract Method — Subclass Must Implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute_routing(
        self,
        data: dict,
        masked_text: str,
        original_text: str,
        prompt_hash: str,
    ) -> None:
        """Execute routing-specific logic. Subclass must implement.

        Args:
            data: The LiteLLM request data dict (mutate to set target model).
            masked_text: The PII-masked prompt text.
            original_text: The original (unmasked) prompt text.
            prompt_hash: SHA-256 hash of the original prompt.
        """
        ...

    # ------------------------------------------------------------------
    # Public: Whether routing is enabled
    # ------------------------------------------------------------------

    @property
    def routing_enabled(self) -> bool:
        """Whether this plugin has routing enabled. Subclasses may override."""
        return True

    # ------------------------------------------------------------------
    # Common Pipeline: async_pre_call_hook
    # ------------------------------------------------------------------

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[dict]:
        """Called before each LLM API call.

        Executes the common pipeline:
          1. Extract/generate session_id and request_id
          2. Concatenate user messages for scanning
          3. Call check_compliance — block if strict mode violation
          4. Call mask — replace PII in messages
          5. Delegate to _execute_routing() for model selection

        Returns:
            Modified data dict (including updated model field from routing).
        """
        import sys
        print(f"[DEBUG] async_pre_call_hook CALLED, model={data.get('model')}", file=sys.stderr, flush=True)
        messages = data.get("messages")
        if not messages:
            return data

        # --- Extract/generate IDs ---
        metadata = data.get("metadata") or {}
        session_id = metadata.get("session_id") or str(uuid.uuid4())
        request_id = metadata.get("request_id") or str(uuid.uuid4())

        # Store IDs back into metadata for post-call retrieval
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["session_id"] = session_id
        data["metadata"]["request_id"] = request_id

        # --- Concatenate user message contents ---
        user_texts = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("content"):
                user_texts.append(msg["content"])

        full_text = "\n".join(user_texts)
        if not full_text.strip():
            return data

        t_start = time.perf_counter()

        # --- Compliance Check ---
        compliance_result = await self._pool.call(
            "check_compliance",
            {"text": full_text, "direction": "inbound"},
        )

        if compliance_result is None:
            # ClawVault unavailable — bypass compliance and PII masking,
            # but routing should still work using original text.
            self._degradation.report_clawvault_unhealthy()
            logger.critical(
                "ClawVault 不可用: 跳过合规检测和 PII 脱敏 (request_id=%s)",
                request_id,
            )
            if self.routing_enabled:
                prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
                data["metadata"]["prompt_hash"] = prompt_hash
                await self._execute_routing(data, full_text, full_text, prompt_hash)
            return data

        if not compliance_result.get("passed", True):
            mode = compliance_result.get("mode", COMPLIANCE_MODE)
            violations = compliance_result.get("violations", [])
            violation_desc = "; ".join(
                v.get("description", v.get("pattern", "unknown"))
                for v in violations
            )

            if mode == "strict":
                logger.warning(
                    "合规拦截 (strict): request_id=%s, violations=%s",
                    request_id,
                    violation_desc,
                )
                raise Exception(
                    f"Request blocked by compliance check: {violation_desc}"
                )
            elif mode == "permissive":
                logger.warning(
                    "合规告警 (permissive): request_id=%s, violations=%s",
                    request_id,
                    violation_desc,
                )
            else:
                # interactive or other — log and continue
                logger.warning(
                    "合规检测不通过 (mode=%s): request_id=%s, violations=%s",
                    mode,
                    request_id,
                    violation_desc,
                )

        # --- PII Masking ---
        mask_result = await self._pool.call(
            "mask",
            {
                "text": full_text,
                "session_id": session_id,
                "request_id": request_id,
            },
        )

        if mask_result is None:
            # ClawVault unavailable — bypass PII masking,
            # but routing should still work using original text.
            self._degradation.report_clawvault_unhealthy()
            logger.critical(
                "ClawVault 不可用: 跳过 PII 脱敏 (request_id=%s)",
                request_id,
            )
            if self.routing_enabled:
                prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
                data["metadata"]["prompt_hash"] = prompt_hash
                await self._execute_routing(data, full_text, full_text, prompt_hash)
            return data

        # ClawVault responded successfully — report healthy
        self._degradation.report_clawvault_healthy()

        masked_text = mask_result.get("masked_text", full_text)
        entities_found = mask_result.get("entities_found", [])

        # --- Redis Health Check (when PII detected) ---
        pii_detected = len(entities_found) > 0
        if pii_detected:
            redis_state = await self._degradation.check_redis_health()
            if redis_state == ComponentState.UNHEALTHY:
                # Reject request: PII detected but Redis unavailable
                self._degradation.enforce_redis_policy(
                    pii_detected=True,
                    request_id=request_id,
                )
                return data

        # --- Replace message contents with masked versions ---
        if len(user_texts) == 1:
            for msg in messages:
                if isinstance(msg, dict) and msg.get("content"):
                    msg["content"] = masked_text
                    break
        else:
            for msg in messages:
                if isinstance(msg, dict) and msg.get("content"):
                    individual_result = await self._pool.call(
                        "mask",
                        {
                            "text": msg["content"],
                            "session_id": session_id,
                            "request_id": request_id,
                        },
                    )
                    if individual_result:
                        msg["content"] = individual_result.get(
                            "masked_text", msg["content"]
                        )

        t_end = time.perf_counter()
        latency_mask_ms = (t_end - t_start) * 1000

        # --- Audit Logging ---
        prompt_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        entity_types = list(set(
            e.get("type", "UNKNOWN") for e in entities_found
        )) if entities_found else []

        logger.info(
            "pre_call_hook 完成: request_id=%s, session_id=%s, "
            "prompt_hash=%s, entities_detected=%s, "
            "latency_mask_ms=%.2f",
            request_id,
            session_id,
            prompt_hash[:16],
            entity_types,
            latency_mask_ms,
        )

        # Store timing in metadata for observability
        data["metadata"]["latency_mask_ms"] = latency_mask_ms
        data["metadata"]["entities_detected"] = entity_types
        data["metadata"]["prompt_hash"] = prompt_hash

        # ==================================================================
        # ROUTING: delegate to subclass
        # ==================================================================
        if self.routing_enabled:
            await self._execute_routing(data, masked_text, full_text, prompt_hash)

        return data

    # ------------------------------------------------------------------
    # Common Pipeline: async_log_success_event (Response Restoration)
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called after a successful LLM response.

        Extracts response text, calls ClawVault restore to rehydrate
        placeholders back to original PII values.
        """
        metadata = kwargs.get("metadata") or kwargs.get("litellm_params", {}).get("metadata") or {}
        request_id = metadata.get("request_id")
        session_id = metadata.get("session_id")

        if not request_id:
            logger.debug("async_log_success_event: no request_id, skipping restore")
            return

        response_text = self._extract_response_text(response_obj)
        if not response_text:
            return

        t_start = time.perf_counter()

        restore_result = await self._pool.call(
            "restore",
            {
                "text": response_text,
                "request_id": request_id,
                "session_id": session_id,
            },
        )

        if restore_result is None:
            logger.critical(
                "ClawVault 不可用: 无法还原占位符 (request_id=%s), 响应将包含占位符",
                request_id,
            )
            return

        restored_text = restore_result.get("restored_text", response_text)
        self._set_response_text(response_obj, restored_text)

        t_end = time.perf_counter()
        latency_restore_ms = (t_end - t_start) * 1000

        # --- Inject aegis_metadata into response (FR-7.3) ---
        self._inject_aegis_metadata(response_obj, metadata)

        logger.info(
            "async_log_success_event 还原完成: request_id=%s, "
            "latency_restore_ms=%.2f",
            request_id,
            latency_restore_ms,
        )

    # ------------------------------------------------------------------
    # Common Pipeline: Streaming Restoration
    # ------------------------------------------------------------------

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: dict,
        response,
        request_data: dict,
    ) -> AsyncGenerator:
        """Called for streaming responses to restore placeholders in SSE chunks."""
        metadata = request_data.get("metadata") or {}
        request_id = metadata.get("request_id")
        session_id = metadata.get("session_id")

        if not request_id:
            logger.debug(
                "async_post_call_streaming_iterator_hook: no request_id, bypassing"
            )
            async for chunk in response:
                yield chunk
            return

        t_start = time.perf_counter()

        mapping_result = await self._pool.call(
            "get_mapping",
            {"request_id": request_id, "session_id": session_id},
        )

        if mapping_result is None:
            logger.critical(
                "ClawVault 不可用: 无法获取映射表 (request_id=%s), 流式响应不还原",
                request_id,
            )
            async for chunk in response:
                yield chunk
            return

        mapping = mapping_result.get("mapping", {})

        if not mapping:
            logger.debug(
                "async_post_call_streaming_iterator_hook: 映射表为空, 跳过还原 (request_id=%s)",
                request_id,
            )
            async for chunk in response:
                yield chunk
            return

        # --- Stream with rehydration ---
        rehydrator = StreamRehydrator(mapping)
        last_chunk = None

        async for chunk in response:
            content = self._extract_streaming_content(chunk)

            if content is None:
                yield chunk
                continue

            restored = rehydrator.process_chunk(content)

            if restored:
                self._set_streaming_content(chunk, restored)
                last_chunk = chunk
                yield chunk
            else:
                last_chunk = chunk

        # --- Flush remaining buffer ---
        remaining = rehydrator.flush_remaining()
        if remaining:
            if last_chunk is not None:
                final_chunk = self._create_streaming_chunk_from(last_chunk, remaining)
                yield final_chunk
            else:
                logger.warning(
                    "flush_remaining has content but no last_chunk template "
                    "(request_id=%s)",
                    request_id,
                )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        logger.info(
            "async_post_call_streaming_iterator_hook 完成: request_id=%s, "
            "latency_total_ms=%.2f",
            request_id,
            latency_ms,
        )

    # ------------------------------------------------------------------
    # Common: Failure Hook
    # ------------------------------------------------------------------

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """异步失败回调 — 记录失败信息。"""
        logger.warning("async_failure_event: model=%s", kwargs.get("model"))

    # ------------------------------------------------------------------
    # Sync Hooks (backward compatibility)
    # ------------------------------------------------------------------

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        """LiteLLM pre-call 同步钩子 (非 async)。"""
        logger.debug("pre_api_call: model=%s", model)

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """LiteLLM 成功回调。"""
        logger.debug("success_event: model=%s", kwargs.get("model"))

    def log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """LiteLLM 失败回调。"""
        logger.warning("failure_event: model=%s", kwargs.get("model"))

    # ------------------------------------------------------------------
    # Helpers (shared by all plugins)
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_aegis_metadata(response_obj: Any, metadata: dict) -> None:
        """Inject aegis_metadata into the LLM response object (FR-7.3).

        aegis_metadata includes:
          - template: The transaction template name (if applicable)
          - agent: The transaction agent name (if applicable)
          - assigned_model: The model that was assigned for this request
          - routing_plugin: Which routing plugin handled this request
          - warnings: Any routing warnings (e.g., UNKNOWN_AGENT)
        """
        routing_plugin = metadata.get("routing_plugin", "")
        assigned_model = metadata.get("target_model", "")
        warnings = metadata.get("_routing_warnings", [])

        # Template and agent are only available for transaction routing
        template = metadata.get("transaction_template", "")
        agent = metadata.get("transaction_agent", "")

        aegis_metadata = {
            "template": template,
            "agent": agent,
            "assigned_model": assigned_model,
            "routing_plugin": routing_plugin,
            "warnings": warnings,
        }

        # Inject into response object
        if hasattr(response_obj, "__dict__"):
            # Pydantic model or object with attributes
            try:
                response_obj.aegis_metadata = aegis_metadata
            except (AttributeError, ValueError):
                # Some pydantic models don't allow extra fields;
                # try _hidden_params or model_extra
                if hasattr(response_obj, "_hidden_params"):
                    response_obj._hidden_params["aegis_metadata"] = aegis_metadata
                elif hasattr(response_obj, "model_extra") and isinstance(
                    response_obj.model_extra, dict
                ):
                    response_obj.model_extra["aegis_metadata"] = aegis_metadata

        if isinstance(response_obj, dict):
            response_obj["aegis_metadata"] = aegis_metadata

    @staticmethod
    def _extract_response_text(response_obj: Any) -> Optional[str]:
        """Extract the text content from a LiteLLM response object."""
        try:
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
        except (IndexError, AttributeError):
            pass

        if isinstance(response_obj, dict):
            choices = response_obj.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content")

        return None

    @staticmethod
    def _set_response_text(response_obj: Any, text: str) -> None:
        """Set the response text content in a LiteLLM response object."""
        try:
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    choice.message.content = text
                    return
        except (IndexError, AttributeError):
            pass

        if isinstance(response_obj, dict):
            choices = response_obj.get("choices", [])
            if choices:
                if "message" in choices[0]:
                    choices[0]["message"]["content"] = text

    @staticmethod
    def _extract_streaming_content(chunk: Any) -> Optional[str]:
        """Extract text content from a streaming chunk."""
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta"):
                    delta = choice.delta
                    if hasattr(delta, "content"):
                        return delta.content
        except (IndexError, AttributeError):
            pass

        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content")

        return None

    @staticmethod
    def _set_streaming_content(chunk: Any, text: str) -> None:
        """Set text content in a streaming chunk's delta."""
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    choice.delta.content = text
                    return
        except (IndexError, AttributeError):
            pass

        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                if "delta" in choices[0]:
                    choices[0]["delta"]["content"] = text

    @staticmethod
    def _create_streaming_chunk_from(template_chunk: Any, content: str) -> Any:
        """Create a new streaming chunk with given content, based on a template chunk."""
        new_chunk = copy.deepcopy(template_chunk)
        BaseRouterCallback._set_streaming_content(new_chunk, content)
        return new_chunk
