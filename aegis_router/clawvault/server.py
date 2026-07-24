"""ClawVault UDS Server 主进程

监听 Unix Domain Socket (Linux/macOS) 或 TCP (Windows 开发环境),
处理 JSON-RPC 2.0 请求，提供 PII 脱敏、占位符还原、合规检测等接口。

启动方式:
    python -m aegis_router.clawvault.server
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("clawvault.server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Socket path (Unix) or TCP host:port (Windows fallback)
DEFAULT_SOCKET_PATH = "/var/run/clawvault.sock"
CLAWVAULT_SOCKET_PATH = os.environ.get("CLAWVAULT_SOCKET_PATH", DEFAULT_SOCKET_PATH)
CLAWVAULT_TCP_HOST = os.environ.get("CLAWVAULT_TCP_HOST", "127.0.0.1")
CLAWVAULT_TCP_PORT = int(os.environ.get("CLAWVAULT_TCP_PORT", "9600"))

# Determine transport mode: "unix" or "tcp"
# Windows doesn't support AF_UNIX reliably, so default to TCP on Windows.
USE_TCP = os.environ.get("CLAWVAULT_USE_TCP", "auto").lower()
if USE_TCP == "auto":
    USE_TCP = sys.platform == "win32"
else:
    USE_TCP = USE_TCP in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 Error Codes
# ---------------------------------------------------------------------------

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# JSON-RPC Helpers
# ---------------------------------------------------------------------------


def _jsonrpc_error(code: int, message: str, request_id: Any = None) -> dict:
    """Construct a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id,
    }


def _jsonrpc_result(result: Any, request_id: Any) -> dict:
    """Construct a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id,
    }


# ---------------------------------------------------------------------------
# Method Handlers (Stubs — actual logic implemented in subsequent tasks)
# ---------------------------------------------------------------------------


_masker_instance = None
_masker_lock = asyncio.Lock()

_restorer_instance = None
_restorer_lock = asyncio.Lock()


async def _get_masker():
    """Lazily initialize the PIIMasker singleton (thread-safe via asyncio.Lock)."""
    global _masker_instance
    if _masker_instance is None:
        async with _masker_lock:
            if _masker_instance is None:
                from aegis_router.clawvault.masker import PIIMasker

                _masker_instance = PIIMasker(
                    redis_client=None,
                    language="en",
                    nlp_model="en_core_web_sm",
                    score_threshold=0.4,
                )
                logger.info("PIIMasker initialized (lazy load)")
    return _masker_instance


async def _get_restorer():
    """Lazily initialize the PIIRestorer singleton (thread-safe via asyncio.Lock)."""
    global _restorer_instance
    if _restorer_instance is None:
        async with _restorer_lock:
            if _restorer_instance is None:
                from aegis_router.clawvault.restorer import PIIRestorer
                from aegis_router.storage.redis_client import RedisClient

                redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
                redis_client = RedisClient(url=redis_url)
                _restorer_instance = PIIRestorer(redis_client=redis_client)
                logger.info("PIIRestorer initialized (lazy load)")
    return _restorer_instance


async def handle_mask(params: dict) -> dict:
    """PII 脱敏 — 使用 PIIMasker 进行实时 PII 检测与占位符替换。

    Input:  {text, session_id, request_id}
    Output: {masked_text, entities_found}
    """
    text = params.get("text", "")
    session_id = params.get("session_id") or str(uuid.uuid4())
    request_id = params.get("request_id") or str(uuid.uuid4())

    masker = await _get_masker()
    result = await masker.mask(text, session_id, request_id)

    return {
        "masked_text": result["masked_text"],
        "entities_found": result["entities_found"],
    }


async def handle_restore(params: dict) -> dict:
    """占位符还原 — 使用 PIIRestorer 从 Redis 读取映射并替换占位符。

    Input:  {text, request_id, session_id?}
    Output: {restored_text}
    """
    text = params.get("text", "")
    request_id = params.get("request_id", "")
    session_id = params.get("session_id")

    try:
        restorer = await _get_restorer()
        result = await restorer.restore(
            text=text,
            request_id=request_id,
            session_id=session_id,
        )
        return result
    except Exception as exc:
        # Graceful degradation: if Redis is unavailable, return text unchanged
        logger.warning("Restore failed (graceful degradation): %s", exc)
        return {"restored_text": text}


async def handle_restore_stream_chunk(params: dict) -> dict:
    """流式 chunk 还原 — 当前为占位实现。

    Input:  {chunk, request_id, buffer_state}
    Output: {flushed_text, new_buffer_state}
    """
    chunk = params.get("chunk", "")
    buffer_state = params.get("buffer_state", "")
    return {
        "flushed_text": chunk,
        "new_buffer_state": buffer_state,
    }


async def handle_get_mapping(params: dict) -> dict:
    """获取 PII 映射表（原始 dict）— 供流式还原引擎使用。

    与 restore 不同，此方法仅返回映射表本身，不执行占位符替换。
    流式还原引擎 (StreamRehydrator) 需要原始映射来逐 chunk 替换。

    Input:  {request_id, session_id?}
    Output: {mapping: {placeholder: original_value, ...}}
    """
    request_id = params.get("request_id", "")
    session_id = params.get("session_id")

    try:
        restorer = await _get_restorer()
        mapping = await restorer._redis.get_mapping(
            request_id=request_id,
            session_id=session_id,
        )
        return {"mapping": mapping}
    except Exception as exc:
        logger.warning("get_mapping failed (graceful degradation): %s", exc)
        return {"mapping": {}}


async def handle_check_compliance(params: dict) -> dict:
    """合规检测 — 调用 ComplianceEngine 进行 Prompt Injection 检测 + 敏感词过滤。

    Input:  {text, direction: "inbound"|"outbound", mode?: "strict"|"interactive"|"permissive"}
    Output: {passed: bool, violations: [], mode: str}
    """
    from aegis_router.clawvault.compliance import ComplianceEngine, get_compliance_engine

    text = params.get("text", "")
    direction = params.get("direction", "inbound")

    # 使用默认路径初始化引擎（如果尚未初始化）
    engine = get_compliance_engine(
        patterns_file=Path(__file__).resolve().parent.parent.parent
        / "config"
        / "compliance_rules"
        / "injection_patterns.yaml",
        sensitive_words_file=Path(__file__).resolve().parent.parent.parent
        / "config"
        / "compliance_rules"
        / "sensitive_words.txt",
        mode=params.get("mode", "strict"),
    )

    result = engine.check_compliance(text=text, direction=direction)
    return result.to_dict()


# Method dispatch table
METHOD_HANDLERS: dict[str, Callable[[dict], Coroutine[Any, Any, dict]]] = {
    "mask": handle_mask,
    "restore": handle_restore,
    "restore_stream_chunk": handle_restore_stream_chunk,
    "get_mapping": handle_get_mapping,
    "check_compliance": handle_check_compliance,
}


# ---------------------------------------------------------------------------
# Request Dispatcher
# ---------------------------------------------------------------------------


async def dispatch_request(raw: bytes) -> bytes:
    """Parse a raw JSON-RPC request and dispatch to the appropriate handler.

    Returns the JSON-RPC response as bytes (UTF-8 encoded JSON).
    """
    request_id = None

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        resp = _jsonrpc_error(JSONRPC_PARSE_ERROR, "Parse error")
        return json.dumps(resp).encode("utf-8")

    # Validate JSON-RPC 2.0 structure
    if not isinstance(data, dict):
        resp = _jsonrpc_error(JSONRPC_INVALID_REQUEST, "Invalid Request: expected object")
        return json.dumps(resp).encode("utf-8")

    request_id = data.get("id")
    jsonrpc_version = data.get("jsonrpc")
    method = data.get("method")
    params = data.get("params", {})

    if jsonrpc_version != "2.0":
        resp = _jsonrpc_error(JSONRPC_INVALID_REQUEST, "Invalid Request: jsonrpc must be '2.0'", request_id)
        return json.dumps(resp).encode("utf-8")

    if not isinstance(method, str) or not method:
        resp = _jsonrpc_error(JSONRPC_INVALID_REQUEST, "Invalid Request: method must be a non-empty string", request_id)
        return json.dumps(resp).encode("utf-8")

    handler = METHOD_HANDLERS.get(method)
    if handler is None:
        resp = _jsonrpc_error(JSONRPC_METHOD_NOT_FOUND, f"Method not found: {method}", request_id)
        return json.dumps(resp).encode("utf-8")

    if not isinstance(params, dict):
        resp = _jsonrpc_error(JSONRPC_INVALID_PARAMS, "Invalid params: expected object", request_id)
        return json.dumps(resp).encode("utf-8")

    try:
        result = await handler(params)
        resp = _jsonrpc_result(result, request_id)
    except Exception as exc:
        logger.exception("Internal error while handling method '%s'", method)
        resp = _jsonrpc_error(JSONRPC_INTERNAL_ERROR, f"Internal error: {exc}", request_id)

    return json.dumps(resp).encode("utf-8")


# ---------------------------------------------------------------------------
# Connection Handler
# ---------------------------------------------------------------------------


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single client connection.

    Protocol: newline-delimited JSON. Each request is a single JSON line,
    and the response is a single JSON line followed by newline.
    """
    peer = writer.get_extra_info("peername") or "unix-client"
    logger.debug("Client connected: %s", peer)

    try:
        while True:
            line = await reader.readline()
            if not line:
                # Client disconnected
                break

            line = line.strip()
            if not line:
                continue

            response_bytes = await dispatch_request(line)
            writer.write(response_bytes + b"\n")
            await writer.drain()
    except asyncio.CancelledError:
        pass
    except ConnectionResetError:
        logger.debug("Client disconnected abruptly: %s", peer)
    except Exception:
        logger.exception("Unexpected error handling client %s", peer)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.debug("Client disconnected: %s", peer)


# ---------------------------------------------------------------------------
# Server Lifecycle
# ---------------------------------------------------------------------------


class ClawVaultServer:
    """Manages the ClawVault JSON-RPC server lifecycle."""

    def __init__(self):
        self._server: asyncio.AbstractServer | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the server and listen for connections."""
        if USE_TCP:
            self._server = await asyncio.start_server(
                handle_connection,
                host=CLAWVAULT_TCP_HOST,
                port=CLAWVAULT_TCP_PORT,
            )
            logger.info(
                "ClawVault server listening on TCP %s:%s",
                CLAWVAULT_TCP_HOST,
                CLAWVAULT_TCP_PORT,
            )
        else:
            # Unix Domain Socket mode
            socket_path = Path(CLAWVAULT_SOCKET_PATH)

            # Clean up stale socket file
            if socket_path.exists():
                logger.warning("Removing stale socket file: %s", socket_path)
                socket_path.unlink()

            # Ensure parent directory exists
            socket_path.parent.mkdir(parents=True, exist_ok=True)

            self._server = await asyncio.start_unix_server(
                handle_connection,
                path=str(socket_path),
            )
            logger.info("ClawVault server listening on UDS %s", socket_path)

    async def serve_forever(self) -> None:
        """Serve until shutdown is requested."""
        if self._server is None:
            raise RuntimeError("Server not started. Call start() first.")

        async with self._server:
            await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        """Gracefully stop the server and clean up resources."""
        logger.info("Shutting down ClawVault server...")
        self._shutdown_event.set()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        # Clean up socket file (Unix mode)
        if not USE_TCP:
            socket_path = Path(CLAWVAULT_SOCKET_PATH)
            if socket_path.exists():
                socket_path.unlink()
                logger.info("Removed socket file: %s", socket_path)

        logger.info("ClawVault server stopped.")


# ---------------------------------------------------------------------------
# Signal Handling & Main Entry Point
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure logging for the ClawVault server process."""
    log_level = os.environ.get("CLAWVAULT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


async def _async_main() -> None:
    """Async entry point: start server with signal handling."""
    server = ClawVaultServer()

    loop = asyncio.get_running_loop()

    # Register signal handlers for graceful shutdown
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(server.shutdown()))
    else:
        # Windows: use signal.signal for SIGINT (Ctrl+C)
        # SIGTERM is not reliably supported on Windows, but we handle SIGINT.
        def _win_handler(signum, frame):
            asyncio.ensure_future(server.shutdown())

        signal.signal(signal.SIGINT, _win_handler)

    await server.start()

    logger.info(
        "ClawVault server ready (transport=%s, pid=%d)",
        "TCP" if USE_TCP else "UDS",
        os.getpid(),
    )

    await server.serve_forever()


def main() -> None:
    """Synchronous entry point for ``python -m aegis_router.clawvault.server``."""
    _setup_logging()
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.info("ClawVault server interrupted by user.")


if __name__ == "__main__":
    main()
