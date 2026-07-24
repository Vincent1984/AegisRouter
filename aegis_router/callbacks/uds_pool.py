"""ClawVault UDS/TCP 连接池

为 SmartRouterCallback 提供高效的异步连接池，支持：
- Unix Domain Socket (Linux) 和 TCP (Windows) 双传输模式
- 基于 asyncio.Queue 的连接复用
- 懒创建、自动重连、优雅关闭
- JSON-RPC 2.0 newline-delimited 协议
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAWVAULT_SOCKET_PATH = os.environ.get("CLAWVAULT_SOCKET_PATH", "/var/run/clawvault.sock")
CLAWVAULT_TCP_HOST = os.environ.get("CLAWVAULT_TCP_HOST", "127.0.0.1")
CLAWVAULT_TCP_PORT = int(os.environ.get("CLAWVAULT_TCP_PORT", "9600"))
CLAWVAULT_TIMEOUT = float(os.environ.get("CLAWVAULT_TIMEOUT", "5.0"))
CLAWVAULT_POOL_SIZE = int(os.environ.get("CLAWVAULT_POOL_SIZE", "10"))
CLAWVAULT_POOL_MIN = int(os.environ.get("CLAWVAULT_POOL_MIN", "2"))

_USE_TCP_ENV = os.environ.get("CLAWVAULT_USE_TCP", "auto").lower()
if _USE_TCP_ENV == "auto":
    USE_TCP = sys.platform == "win32"
else:
    USE_TCP = _USE_TCP_ENV in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Connection wrapper
# ---------------------------------------------------------------------------


class _Connection:
    """Thin wrapper around asyncio StreamReader/StreamWriter pair."""

    __slots__ = ("reader", "writer", "healthy")

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.healthy = True

    async def close(self) -> None:
        """Close the underlying transport."""
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass
        finally:
            self.healthy = False


# ---------------------------------------------------------------------------
# ClawVaultPool
# ---------------------------------------------------------------------------


class ClawVaultPool:
    """Async connection pool for communicating with ClawVault via UDS or TCP.

    Uses an asyncio.Queue to manage idle connections and a semaphore to
    bound the total number of open connections.

    Parameters
    ----------
    max_connections : int
        Maximum number of simultaneous connections (default from env CLAWVAULT_POOL_SIZE).
    min_connections : int
        Minimum idle connections to maintain (used for pre-warming, default from env CLAWVAULT_POOL_MIN).
    timeout : float
        Default timeout for RPC calls in seconds (default from env CLAWVAULT_TIMEOUT).
    socket_path : str
        UDS path (Linux).
    tcp_host : str
        TCP host (Windows).
    tcp_port : int
        TCP port (Windows).
    use_tcp : bool
        Force TCP transport.
    """

    def __init__(
        self,
        max_connections: int = CLAWVAULT_POOL_SIZE,
        min_connections: int = CLAWVAULT_POOL_MIN,
        timeout: float = CLAWVAULT_TIMEOUT,
        socket_path: str = CLAWVAULT_SOCKET_PATH,
        tcp_host: str = CLAWVAULT_TCP_HOST,
        tcp_port: int = CLAWVAULT_TCP_PORT,
        use_tcp: bool = USE_TCP,
    ) -> None:
        self._max_connections = max_connections
        self._min_connections = min_connections
        self._timeout = timeout
        self._socket_path = socket_path
        self._tcp_host = tcp_host
        self._tcp_port = tcp_port
        self._use_tcp = use_tcp

        # Pool internals
        self._pool: asyncio.Queue[_Connection] = asyncio.Queue(maxsize=max_connections)
        self._semaphore = asyncio.Semaphore(max_connections)
        self._created_count = 0
        self._closed = False

        logger.info(
            "ClawVaultPool initialized (transport=%s, max=%d, min=%d, timeout=%.1fs)",
            "TCP" if use_tcp else "UDS",
            max_connections,
            min_connections,
            timeout,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_connections(self) -> int:
        return self._max_connections

    @property
    def pool_size(self) -> int:
        """Number of idle connections currently in the pool."""
        return self._pool.qsize()

    @property
    def created_count(self) -> int:
        """Total connections created (including broken ones already discarded)."""
        return self._created_count

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _create_connection(self) -> _Connection:
        """Create a new connection to ClawVault."""
        if self._use_tcp:
            reader, writer = await asyncio.open_connection(self._tcp_host, self._tcp_port)
        else:
            reader, writer = await asyncio.open_unix_connection(self._socket_path)
        self._created_count += 1
        logger.debug(
            "ClawVaultPool: new connection created (total_created=%d)", self._created_count
        )
        return _Connection(reader, writer)

    @asynccontextmanager
    async def _acquire(self, timeout: float):
        """Context manager that acquires a connection from the pool.

        On normal exit the connection is returned to the pool.
        On error the connection is discarded.
        """
        conn: Optional[_Connection] = None
        await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        try:
            # Try to get an idle connection from the queue
            try:
                conn = self._pool.get_nowait()
                # Verify the connection is still healthy
                if not conn.healthy or conn.writer.is_closing():
                    await conn.close()
                    conn = None
            except asyncio.QueueEmpty:
                conn = None

            # Create new connection if needed
            if conn is None:
                conn = await asyncio.wait_for(self._create_connection(), timeout=timeout)

            yield conn

            # Return healthy connection to pool
            if conn.healthy and not conn.writer.is_closing():
                try:
                    self._pool.put_nowait(conn)
                    logger.debug(
                        "ClawVaultPool: connection returned (pool_size=%d)", self._pool.qsize()
                    )
                except asyncio.QueueFull:
                    # Pool is full, discard this connection
                    await conn.close()
            else:
                await conn.close()

        except BaseException:
            # Discard broken connection
            if conn is not None:
                conn.healthy = False
                await conn.close()
            raise
        finally:
            self._semaphore.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def call(
        self,
        method: str,
        params: dict,
        timeout: float | None = None,
    ) -> Optional[dict]:
        """Send a JSON-RPC 2.0 request to ClawVault and return the result.

        Parameters
        ----------
        method : str
            RPC method name (mask, restore, check_compliance).
        params : dict
            Method parameters.
        timeout : float | None
            Timeout in seconds. Defaults to pool-level timeout.

        Returns
        -------
        dict | None
            The 'result' field from the JSON-RPC response, or None if
            ClawVault is unavailable (graceful degradation).
        """
        if self._closed:
            logger.critical("ClawVaultPool is closed — 进入 bypass 模式")
            return None

        if timeout is None:
            timeout = self._timeout

        request_id = str(uuid.uuid4())
        request_payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }

        try:
            async with self._acquire(timeout) as conn:
                # Send newline-delimited JSON
                payload_bytes = json.dumps(request_payload).encode("utf-8") + b"\n"
                conn.writer.write(payload_bytes)
                await conn.writer.drain()

                # Read response line
                response_line = await asyncio.wait_for(
                    conn.reader.readline(),
                    timeout=timeout,
                )

                if not response_line:
                    logger.critical("ClawVault 返回空响应 — 进入 bypass 模式")
                    conn.healthy = False
                    return None

                response = json.loads(response_line)

                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(
                        f"ClawVault RPC error [{error.get('code')}]: {error.get('message')}"
                    )

                return response.get("result")

        except (OSError, ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as exc:
            logger.critical(
                "ClawVault 不可用 (连接失败): %s — 进入 bypass 模式", exc
            )
            return None
        except asyncio.TimeoutError:
            logger.critical("ClawVault 响应超时 (%.1fs) — 进入 bypass 模式", timeout)
            return None
        except json.JSONDecodeError as exc:
            logger.critical("ClawVault 返回无效 JSON: %s — 进入 bypass 模式", exc)
            return None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close all pooled connections."""
        self._closed = True
        closed_count = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
                closed_count += 1
            except asyncio.QueueEmpty:
                break
        logger.info("ClawVaultPool closed (%d connections cleaned up)", closed_count)

    async def __aenter__(self) -> "ClawVaultPool":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
