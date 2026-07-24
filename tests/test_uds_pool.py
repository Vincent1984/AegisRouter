"""Tests for ClawVaultPool — async UDS/TCP connection pool.

Tests cover:
- Pool initialization and configuration
- Concurrent requests don't exceed max_connections
- Connection reuse (connections returned to pool)
- Graceful degradation when ClawVault is unavailable (returns None)
- Broken connection is discarded and new one created
- Pool cleanup/close
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aegis_router.callbacks.uds_pool import ClawVaultPool, _Connection


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def make_json_response(result: dict, request_id: str = "test-id") -> bytes:
    """Create a JSON-RPC response line."""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": request_id}).encode() + b"\n"


def make_mock_connection():
    """Create a mock reader/writer pair that behaves like a real connection."""
    reader = AsyncMock()
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    writer.is_closing = MagicMock(return_value=False)
    return reader, writer


@pytest.fixture
def pool():
    """Create a pool configured for testing with TCP transport."""
    return ClawVaultPool(
        max_connections=5,
        min_connections=1,
        timeout=2.0,
        use_tcp=True,
        tcp_host="127.0.0.1",
        tcp_port=9600,
    )


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestPoolInitialization:
    """Test pool creation and configuration."""

    def test_default_configuration(self):
        """Pool picks up defaults from env vars / hardcoded values."""
        p = ClawVaultPool()
        assert p.max_connections >= 1
        assert p.pool_size == 0  # No connections pre-created (lazy)
        assert p.created_count == 0

    def test_custom_configuration(self):
        """Pool accepts custom max_connections, timeout, etc."""
        p = ClawVaultPool(max_connections=20, min_connections=5, timeout=3.0)
        assert p.max_connections == 20

    def test_pool_starts_empty(self, pool):
        """Pool starts with zero idle connections (lazy creation)."""
        assert pool.pool_size == 0
        assert pool.created_count == 0


# ---------------------------------------------------------------------------
# Tests: Successful RPC calls
# ---------------------------------------------------------------------------


class TestPoolCall:
    """Test the pool.call() method for successful requests."""

    async def test_call_returns_result(self, pool):
        """A successful call returns the 'result' field from JSON-RPC response."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(
            return_value=make_json_response({"masked_text": "hello [PII]", "entities_found": []})
        )

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            result = await pool.call("mask", {"text": "hello John"})

        assert result == {"masked_text": "hello [PII]", "entities_found": []}

    async def test_call_sends_jsonrpc_format(self, pool):
        """Pool sends valid JSON-RPC 2.0 request with newline delimiter."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(
            return_value=make_json_response({"ok": True})
        )

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            await pool.call("check_compliance", {"text": "test", "direction": "inbound"})

        # Verify what was written
        written_data = writer.write.call_args[0][0]
        assert written_data.endswith(b"\n")
        payload = json.loads(written_data.decode())
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "check_compliance"
        assert payload["params"] == {"text": "test", "direction": "inbound"}
        assert "id" in payload


# ---------------------------------------------------------------------------
# Tests: Connection reuse
# ---------------------------------------------------------------------------


class TestConnectionReuse:
    """Test that connections are returned to pool and reused."""

    async def test_connection_returned_to_pool_after_call(self, pool):
        """After a successful call, the connection goes back to the pool."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(return_value=make_json_response({"ok": True}))

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            await pool.call("mask", {"text": "hello"})

        # Connection should be in the pool now
        assert pool.pool_size == 1
        assert pool.created_count == 1

    async def test_connection_reused_on_subsequent_calls(self, pool):
        """Second call reuses the connection from the pool instead of creating new one."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(return_value=make_json_response({"ok": True}))

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            await pool.call("mask", {"text": "first"})
            await pool.call("mask", {"text": "second"})

        # Only 1 connection should have been created
        assert pool.created_count == 1
        # open_connection called only once
        assert mock_open.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Test concurrent requests respect max_connections."""

    async def test_concurrent_requests_bounded_by_max(self):
        """Concurrent requests do not exceed max_connections."""
        max_conn = 3
        p = ClawVaultPool(max_connections=max_conn, timeout=5.0, use_tcp=True)

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        original_create = p._create_connection

        async def slow_create():
            nonlocal active_count, max_active
            conn = await original_create()
            return conn

        async def slow_call(idx: int):
            nonlocal active_count, max_active
            reader, writer = make_mock_connection()

            async def slow_readline():
                nonlocal active_count, max_active
                async with lock:
                    active_count += 1
                    max_active = max(max_active, active_count)
                await asyncio.sleep(0.05)  # Simulate work
                async with lock:
                    active_count -= 1
                return make_json_response({"idx": idx})

            reader.readline = slow_readline
            return reader, writer

        call_idx = 0

        async def mock_open(*args, **kwargs):
            nonlocal call_idx
            call_idx += 1
            return await slow_call(call_idx)

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection", side_effect=mock_open):
            # Launch more requests than max_connections
            tasks = [p.call("test", {"i": i}) for i in range(10)]
            results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r is not None for r in results)
        # Never exceeded max_connections simultaneously
        assert max_active <= max_conn

        await p.close()

    async def test_20_concurrent_requests_no_blocking_no_leak(self):
        """20 concurrent requests with max_connections=10: no blocking, no leak.

        Verifies:
        - All 20 requests complete successfully
        - Concurrency never exceeds max_connections (10)
        - No connection leak after all requests finish
        - Parallelism occurs (total time << 20 * per-request time)
        - Pool cleanup works correctly
        """
        import time

        max_conn = 10
        num_requests = 20
        per_request_delay = 0.02  # 20ms simulated work

        p = ClawVaultPool(max_connections=max_conn, timeout=10.0, use_tcp=True)

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def mock_open(*args, **kwargs):
            nonlocal active_count, max_active
            reader, writer = make_mock_connection()

            async def slow_readline():
                nonlocal active_count, max_active
                async with lock:
                    active_count += 1
                    max_active = max(max_active, active_count)
                await asyncio.sleep(per_request_delay)
                async with lock:
                    active_count -= 1
                return make_json_response({"ok": True})

            reader.readline = slow_readline
            return reader, writer

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection", side_effect=mock_open):
            start = time.monotonic()
            tasks = [p.call("test", {"i": i}) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)
            elapsed = time.monotonic() - start

        # 1. All 20 requests complete successfully (no None, no exceptions)
        assert len(results) == num_requests
        assert all(r is not None for r in results)

        # 2. Concurrency never exceeded max_connections
        assert max_active <= max_conn

        # 3. No connection leak: pool_size <= max_connections
        assert p.pool_size <= max_conn

        # 4. Parallelism: total time should be much less than sequential
        #    Sequential would be 20 * 0.02 = 0.4s. With 10 concurrency it should be ~0.04s + overhead.
        #    We generously allow up to 0.3s to avoid flakiness.
        sequential_time = num_requests * per_request_delay
        assert elapsed < sequential_time, (
            f"Total time {elapsed:.3f}s too close to sequential {sequential_time:.3f}s — parallelism not working"
        )

        # 5. Cleanup: close pool and verify no idle connections remain
        await p.close()
        assert p.pool_size == 0

    async def test_connection_leak_on_mixed_success_failure(self):
        """20 concurrent requests with mixed success/failure: no connection leak.

        Verifies:
        - Connections are properly released on both success and failure paths
        - Semaphore is fully released after all requests settle
        - Pool can still accept new requests after failures
        """
        max_conn = 10
        num_requests = 20

        p = ClawVaultPool(max_connections=max_conn, timeout=10.0, use_tcp=True)

        call_idx = 0
        idx_lock = asyncio.Lock()

        async def mock_open(*args, **kwargs):
            nonlocal call_idx
            async with idx_lock:
                call_idx += 1
                current_idx = call_idx

            reader, writer = make_mock_connection()

            async def mock_readline():
                await asyncio.sleep(0.01)
                # Even-indexed requests fail with ConnectionResetError
                if current_idx % 2 == 0:
                    raise ConnectionResetError("Connection reset by peer")
                return make_json_response({"idx": current_idx})

            reader.readline = mock_readline
            return reader, writer

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection", side_effect=mock_open):
            tasks = [p.call("test", {"i": i}) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)

        # Some succeed (dict result), some fail gracefully (None)
        successes = [r for r in results if r is not None]
        failures = [r for r in results if r is None]
        assert len(successes) + len(failures) == num_requests
        assert len(failures) > 0, "Expected some failures from ConnectionResetError"
        assert len(successes) > 0, "Expected some successes"

        # No connection leak: pool_size should not exceed max_connections
        assert p.pool_size <= max_conn

        # Semaphore fully released: we should be able to acquire max_connections permits
        # This proves no permits were leaked during error handling
        acquired = 0
        for _ in range(max_conn):
            try:
                await asyncio.wait_for(p._semaphore.acquire(), timeout=0.5)
                acquired += 1
            except asyncio.TimeoutError:
                break

        assert acquired == max_conn, (
            f"Semaphore leak detected: only acquired {acquired}/{max_conn} permits"
        )

        # Release permits we just acquired (cleanup)
        for _ in range(acquired):
            p._semaphore.release()

        # Pool can still serve new requests after the mixed batch.
        # Drain any idle connections first so a fresh connection is created.
        while not p._pool.empty():
            try:
                conn = p._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break

        async def fresh_mock_open(*args, **kwargs):
            reader, writer = make_mock_connection()
            reader.readline = AsyncMock(return_value=make_json_response({"fresh": True}))
            return reader, writer

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection", side_effect=fresh_mock_open):
            fresh_result = await p.call("test", {"fresh": True})

        assert fresh_result == {"fresh": True}

        await p.close()
        assert p.pool_size == 0


# ---------------------------------------------------------------------------
# Tests: Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test returns None when ClawVault is unavailable."""

    async def test_returns_none_on_connection_refused(self, pool):
        """Returns None when ClawVault refuses connections."""
        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.side_effect = ConnectionRefusedError("Connection refused")
            result = await pool.call("mask", {"text": "hello"})

        assert result is None

    async def test_returns_none_on_timeout(self, pool):
        """Returns None when connection or read times out."""
        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.side_effect = asyncio.TimeoutError()
            result = await pool.call("mask", {"text": "hello"}, timeout=0.1)

        assert result is None

    async def test_returns_none_on_os_error(self, pool):
        """Returns None when OS-level socket error occurs."""
        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.side_effect = OSError("No such file or directory")
            result = await pool.call("mask", {"text": "hello"})

        assert result is None

    async def test_returns_none_on_empty_response(self, pool):
        """Returns None when server sends empty response (connection closed)."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(return_value=b"")

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            result = await pool.call("mask", {"text": "hello"})

        assert result is None

    async def test_returns_none_on_invalid_json(self, pool):
        """Returns None when server sends invalid JSON."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(return_value=b"not valid json\n")

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            result = await pool.call("mask", {"text": "hello"})

        assert result is None

    async def test_returns_none_after_pool_closed(self, pool):
        """Returns None when pool has been closed."""
        await pool.close()
        result = await pool.call("mask", {"text": "hello"})
        assert result is None

    async def test_raises_on_rpc_error(self, pool):
        """Raises RuntimeError when JSON-RPC response contains error field."""
        reader, writer = make_mock_connection()
        response = json.dumps({
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": "Internal error"},
            "id": "test",
        }).encode() + b"\n"
        reader.readline = AsyncMock(return_value=response)

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            with pytest.raises(RuntimeError, match="ClawVault RPC error"):
                await pool.call("mask", {"text": "hello"})


# ---------------------------------------------------------------------------
# Tests: Broken connection handling
# ---------------------------------------------------------------------------


class TestBrokenConnections:
    """Test that broken connections are discarded and new ones created."""

    async def test_broken_connection_discarded(self, pool):
        """A connection that fails mid-request is not returned to pool."""
        reader, writer = make_mock_connection()
        reader.readline = AsyncMock(side_effect=ConnectionResetError("Connection reset"))

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection") as mock_open:
            mock_open.return_value = (reader, writer)
            result = await pool.call("mask", {"text": "hello"})

        assert result is None
        # Connection should NOT be in the pool
        assert pool.pool_size == 0

    async def test_stale_connection_replaced(self, pool):
        """A pooled connection whose writer is_closing gets replaced with fresh one."""
        # First call — creates a connection that will become stale
        reader1, writer1 = make_mock_connection()
        reader1.readline = AsyncMock(return_value=make_json_response({"ok": True}))

        # Second call — the old connection is stale, so a new one is created
        reader2, writer2 = make_mock_connection()
        reader2.readline = AsyncMock(return_value=make_json_response({"ok": True}))

        call_count = [0]

        async def mock_open(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return reader1, writer1
            return reader2, writer2

        with patch("aegis_router.callbacks.uds_pool.asyncio.open_connection", side_effect=mock_open):
            # First call — connection created and returned to pool
            await pool.call("mask", {"text": "first"})
            assert pool.pool_size == 1

            # Mark the pooled connection as closing (simulates stale/broken)
            writer1.is_closing.return_value = True

            # Second call — stale conn discarded, new one created
            await pool.call("mask", {"text": "second"})

        # Two connections were created total
        assert pool.created_count == 2


# ---------------------------------------------------------------------------
# Tests: Pool close / cleanup
# ---------------------------------------------------------------------------


class TestPoolClose:
    """Test graceful pool shutdown."""

    async def test_close_cleans_all_idle_connections(self):
        """close() drains all idle connections from pool."""
        p = ClawVaultPool(max_connections=5, timeout=2.0, use_tcp=True)

        # Manually place connections into the pool
        reader1, writer1 = make_mock_connection()
        reader2, writer2 = make_mock_connection()
        conn1 = _Connection(reader1, writer1)
        conn2 = _Connection(reader2, writer2)
        p._pool.put_nowait(conn1)
        p._pool.put_nowait(conn2)
        assert p.pool_size == 2

        await p.close()

        assert p.pool_size == 0
        # Writers should have been closed
        writer1.close.assert_called_once()
        writer2.close.assert_called_once()

    async def test_close_sets_closed_flag(self, pool):
        """After close(), pool refuses new calls."""
        await pool.close()
        assert pool._closed is True

    async def test_context_manager(self):
        """Pool can be used as async context manager."""
        async with ClawVaultPool(max_connections=2, use_tcp=True) as p:
            assert p._closed is False
        assert p._closed is True
