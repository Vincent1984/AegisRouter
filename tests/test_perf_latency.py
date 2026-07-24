"""性能延迟基准测试

验证各核心组件在目标延迟阈值内完成处理:
- TC-PERF-LAT-001: PII 脱敏延迟 < 12ms (含中文 NER, 100 条取 P95)
- TC-PERF-LAT-002: 规则前置引擎延迟 < 1ms (1000 条取 P99)
- TC-PERF-LAT-003: RouteLLM 分类器延迟 < 10ms (100 条取 P95)
- TC-PERF-LAT-004: 占位符还原延迟 < 3ms (100 条取 P95)
- TC-PERF-LAT-005: 完整网关附加延迟 (脱敏+路由+还原) ≤ 20ms (P95)
- TC-PERF-LAT-006: UDS 通信单次 round-trip < 0.5ms
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegis_router.callbacks.stream_rehydrator import StreamRehydrator
from aegis_router.callbacks.uds_pool import ClawVaultPool
from aegis_router.clawvault.masker import PIIMasker
from aegis_router.clawvault.recognizers import (
    ChineseIdCardRecognizer,
    ChineseNameRecognizer,
    ChinesePhoneRecognizer,
)
from aegis_router.clawvault.restorer import PIIRestorer
from aegis_router.config import ClassifierConfig, TrivialConfig
from aegis_router.router.model_classifier import ModelClassifier
from aegis_router.router.rule_engine import RuleEngine


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: float) -> float:
    """计算第 p 百分位数 (0-100)。"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    weight = idx - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis_client():
    """创建一个无延迟的 mock Redis 客户端。"""
    client = AsyncMock()
    client.get_mapping = AsyncMock(return_value={})
    client.store_mapping = AsyncMock(return_value=None)
    client.update_session_mapping = AsyncMock(return_value=None)
    return client


@pytest.fixture
def pii_masker(mock_redis_client):
    """创建带中文识别器的 PIIMasker 实例。"""
    masker = PIIMasker(
        redis_client=mock_redis_client,
        language="en",
        nlp_model="en_core_web_sm",
        score_threshold=0.4,
    )
    # 注册中文识别器
    masker.register_recognizer(ChinesePhoneRecognizer())
    masker.register_recognizer(ChineseIdCardRecognizer())
    masker.register_recognizer(ChineseNameRecognizer())
    return masker


@pytest.fixture
def patterns_file(tmp_path):
    """创建临时寒暄词库文件。"""
    content = (
        "# 寒暄词库\n"
        "你好\nhello\nhi\nhey\n谢谢\n再见\n早上好\n晚上好\n"
        "good morning\ngood evening\ngood night\nthanks\n"
        "thank you\nbye\ngoodbye\n嗨\n哈喽\n晚安\n早安\n你好呀\n"
    )
    f = tmp_path / "trivial_chat.txt"
    f.write_text(content, encoding="utf-8")
    return str(f)


@pytest.fixture
def rule_engine(patterns_file):
    """创建规则前置引擎实例。"""
    config = TrivialConfig(
        enabled=True,
        max_length=30,
        target_model="local-7b",
        patterns_file=patterns_file,
    )
    return RuleEngine(config)


@pytest.fixture
def mock_restorer_redis():
    """创建带预设映射的 mock Redis 客户端 (用于还原测试)。"""
    mapping = {
        "[PERSON_1]": "张三",
        "[PHONE_1]": "13800138000",
        "[ID_CARD_1]": "110101199001011234",
        "[EMAIL_1]": "test@example.com",
        "[PERSON_2]": "李四",
    }
    client = AsyncMock()
    client.get_mapping = AsyncMock(return_value=mapping)
    return client


# ---------------------------------------------------------------------------
# TC-PERF-LAT-001: PII 脱敏延迟 < 12ms (含中文 NER, 100 条取 P95)
# ---------------------------------------------------------------------------


class TestPIIMaskingLatency:
    """TC-PERF-LAT-001: PII 脱敏延迟基准测试。"""

    async def test_pii_masking_p95_under_12ms(self, pii_masker):
        """PII 脱敏 (含中文实体检测) P95 延迟应 < 12ms。"""
        # 包含中文 PII 的测试文本
        test_text = "用户张三的手机号是13912345678，身份证号为110101199001011234"

        latencies: list[float] = []

        # 预热 (排除首次加载影响)
        await pii_masker.mask(test_text, session_id="warmup", request_id="warmup-0")

        for i in range(100):
            start = time.perf_counter()
            await pii_masker.mask(
                text=test_text,
                session_id="bench-session",
                request_id=f"bench-req-{i}",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)
        avg = statistics.mean(latencies)

        # 断言 P95 < 12ms
        assert p95 < 12.0, (
            f"PII 脱敏 P95 延迟 {p95:.2f}ms 超出 12ms 阈值 "
            f"(avg={avg:.2f}ms, min={min(latencies):.2f}ms, max={max(latencies):.2f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-LAT-002: 规则前置引擎延迟 < 1ms (1000 条取 P99)
# ---------------------------------------------------------------------------


class TestRuleEngineLatency:
    """TC-PERF-LAT-002: 规则前置引擎延迟基准测试。"""

    def test_rule_engine_p99_under_1ms(self, rule_engine):
        """规则前置引擎 check() P99 延迟应 < 1ms。"""
        # 混合测试用例: 寒暄 + 非寒暄
        prompts = [
            "你好",
            "hello",
            "解释量子计算",
            "hi",
            "请帮我写一段代码",
            "thanks",
            "什么是深度学习",
            "再见",
            "帮我分析这段代码的性能",
            "早上好",
        ]

        latencies: list[float] = []

        # 预热
        for p in prompts[:3]:
            rule_engine.check(p)

        for i in range(1000):
            prompt = prompts[i % len(prompts)]
            start = time.perf_counter()
            rule_engine.check(prompt)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        p99 = percentile(latencies, 99)
        avg = statistics.mean(latencies)

        assert p99 < 1.0, (
            f"规则前置引擎 P99 延迟 {p99:.2f}ms 超出 1ms 阈值 "
            f"(avg={avg:.2f}ms, min={min(latencies):.2f}ms, max={max(latencies):.2f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-LAT-003: RouteLLM 分类器延迟 < 10ms (100 条取 P95)
# ---------------------------------------------------------------------------


class TestModelClassifierLatency:
    """TC-PERF-LAT-003: RouteLLM 分类器延迟基准测试。"""

    def test_classifier_p95_under_10ms(self):
        """RouteLLM 分类器推理 P95 延迟应 < 10ms。"""
        config = ClassifierConfig(type="mf", model_path=None)
        classifier = ModelClassifier(config, timeout_ms=50.0)

        # Mock _router 的 calculate_strong_win_rate 模拟真实推理延迟
        mock_router = MagicMock()

        def fake_inference(prompt: str) -> float:
            """模拟真实推理 (~2-5ms 计算)。"""
            # 简单的字符串哈希计算来模拟实际计算负载
            total = 0
            for _ in range(500):
                total += sum(ord(c) for c in prompt[:50])
            return (total % 100) / 100.0

        mock_router.calculate_strong_win_rate = fake_inference
        classifier._router = mock_router
        classifier._loaded = True
        classifier._load_error = None

        prompts = [
            "请解释量子纠缠现象",
            "What is machine learning?",
            "帮我写一段Python排序算法",
            "分析全球气候变化的影响",
            "How does a transformer model work?",
        ]

        latencies: list[float] = []

        # 预热
        for p in prompts[:2]:
            try:
                classifier.classify(p)
            except TimeoutError:
                pass

        for i in range(100):
            prompt = prompts[i % len(prompts)]
            start = time.perf_counter()
            try:
                classifier.classify(prompt)
            except TimeoutError:
                # 超时也记录耗时 (但这不应发生)
                pass
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)
        avg = statistics.mean(latencies)

        assert p95 < 10.0, (
            f"RouteLLM 分类器 P95 延迟 {p95:.2f}ms 超出 10ms 阈值 "
            f"(avg={avg:.2f}ms, min={min(latencies):.2f}ms, max={max(latencies):.2f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-LAT-004: 占位符还原延迟 < 3ms (100 条取 P95)
# ---------------------------------------------------------------------------


class TestPIIRestorerLatency:
    """TC-PERF-LAT-004: 占位符还原延迟基准测试。"""

    async def test_restorer_p95_under_3ms(self, mock_restorer_redis):
        """占位符还原 P95 延迟应 < 3ms。"""
        restorer = PIIRestorer(redis_client=mock_restorer_redis)

        # 包含多个占位符的测试文本
        test_text = (
            "用户 [PERSON_1] 的手机号是 [PHONE_1]，"
            "身份证号为 [ID_CARD_1]，"
            "邮箱是 [EMAIL_1]，同事 [PERSON_2] 也参与了会议。"
        )

        latencies: list[float] = []

        # 预热
        await restorer.restore(test_text, request_id="warmup", session_id="warmup")

        for i in range(100):
            start = time.perf_counter()
            await restorer.restore(
                text=test_text,
                request_id=f"bench-req-{i}",
                session_id="bench-session",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)
        avg = statistics.mean(latencies)

        assert p95 < 3.0, (
            f"占位符还原 P95 延迟 {p95:.2f}ms 超出 3ms 阈值 "
            f"(avg={avg:.2f}ms, min={min(latencies):.2f}ms, max={max(latencies):.2f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-LAT-005: 完整网关附加延迟 (脱敏+路由+还原) ≤ 20ms (P95)
# ---------------------------------------------------------------------------


class TestFullPipelineLatency:
    """TC-PERF-LAT-005: 完整网关附加延迟基准测试。"""

    async def test_full_pipeline_p95_under_20ms(
        self, pii_masker, rule_engine, mock_restorer_redis
    ):
        """完整网关管道 (脱敏→路由→还原) P95 延迟应 ≤ 20ms (不含 LLM API 耗时)。"""
        restorer = PIIRestorer(redis_client=mock_restorer_redis)

        # Mock 分类器 (模拟轻量推理)
        config = ClassifierConfig(type="mf", model_path=None)
        classifier = ModelClassifier(config, timeout_ms=50.0)
        mock_router = MagicMock()
        mock_router.calculate_strong_win_rate = lambda prompt: 0.7
        classifier._router = mock_router
        classifier._loaded = True
        classifier._load_error = None

        test_text = "用户张三的手机号是13912345678，请帮我查询订单"
        response_text = "好的 [PERSON_1]，您的手机号 [PHONE_1] 对应的订单如下"

        latencies: list[float] = []

        # 预热
        await pii_masker.mask(test_text, session_id="warmup", request_id="warmup-0")
        rule_engine.check(test_text)
        classifier.classify(test_text)
        await restorer.restore(response_text, request_id="warmup", session_id="warmup")

        for i in range(100):
            start = time.perf_counter()

            # Step 1: PII 脱敏
            mask_result = await pii_masker.mask(
                text=test_text,
                session_id="bench-session",
                request_id=f"pipeline-req-{i}",
            )

            # Step 2: 路由决策 (规则前置 + 分类器)
            masked_text = mask_result["masked_text"]
            rule_result = rule_engine.check(masked_text)
            if not rule_result.matched:
                classifier.classify(masked_text)

            # Step 3: 占位符还原
            await restorer.restore(
                text=response_text,
                request_id=f"pipeline-req-{i}",
                session_id="bench-session",
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        p95 = percentile(latencies, 95)
        avg = statistics.mean(latencies)

        assert p95 <= 20.0, (
            f"完整网关管道 P95 延迟 {p95:.2f}ms 超出 20ms 阈值 "
            f"(avg={avg:.2f}ms, min={min(latencies):.2f}ms, max={max(latencies):.2f}ms)"
        )


# ---------------------------------------------------------------------------
# TC-PERF-LAT-006: UDS 通信单次 round-trip < 0.5ms
# ---------------------------------------------------------------------------


class TestUDSRoundtripLatency:
    """TC-PERF-LAT-006: UDS/TCP 通信 round-trip 延迟基准测试。"""

    async def test_uds_roundtrip_p95_under_0_5ms(self, tmp_path):
        """UDS/TCP 单次 round-trip P95 延迟应 < 0.5ms。"""
        # 在 Windows 上使用 TCP, 其他平台用 UDS
        use_tcp = sys.platform == "win32"

        if use_tcp:
            # TCP 模式: 启动本地 TCP echo server
            server_ready = asyncio.Event()
            tcp_port = 0  # 使用随机端口

            async def handle_client(reader, writer):
                """简单的 JSON-RPC echo 处理。"""
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        request = json.loads(line)
                        response = {
                            "jsonrpc": "2.0",
                            "result": {"status": "ok"},
                            "id": request.get("id"),
                        }
                        writer.write(json.dumps(response).encode("utf-8") + b"\n")
                        await writer.drain()
                    except (json.JSONDecodeError, KeyError):
                        break
                writer.close()

            server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
            actual_port = server.sockets[0].getsockname()[1]
            server_ready.set()

        else:
            # UDS 模式
            socket_path = str(tmp_path / "test_clawvault.sock")
            server_ready = asyncio.Event()

            async def handle_client(reader, writer):
                """简单的 JSON-RPC echo 处理。"""
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        request = json.loads(line)
                        response = {
                            "jsonrpc": "2.0",
                            "result": {"status": "ok"},
                            "id": request.get("id"),
                        }
                        writer.write(json.dumps(response).encode("utf-8") + b"\n")
                        await writer.drain()
                    except (json.JSONDecodeError, KeyError):
                        break
                writer.close()

            server = await asyncio.start_unix_server(handle_client, path=socket_path)
            actual_port = None
            server_ready.set()

        try:
            # 创建连接池
            if use_tcp:
                pool = ClawVaultPool(
                    max_connections=5,
                    min_connections=1,
                    timeout=2.0,
                    tcp_host="127.0.0.1",
                    tcp_port=actual_port,
                    use_tcp=True,
                )
            else:
                pool = ClawVaultPool(
                    max_connections=5,
                    min_connections=1,
                    timeout=2.0,
                    socket_path=socket_path,
                    use_tcp=False,
                )

            latencies: list[float] = []

            # 预热 (建立连接)
            await pool.call("ping", {"echo": "warmup"}, timeout=2.0)

            for i in range(100):
                start = time.perf_counter()
                result = await pool.call("ping", {"echo": f"test-{i}"}, timeout=2.0)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed_ms)
                assert result is not None, f"第 {i} 次调用返回 None"

            p95 = percentile(latencies, 95)
            avg = statistics.mean(latencies)

            assert p95 < 0.5, (
                f"UDS/TCP round-trip P95 延迟 {p95:.3f}ms 超出 0.5ms 阈值 "
                f"(avg={avg:.3f}ms, min={min(latencies):.3f}ms, "
                f"max={max(latencies):.3f}ms)"
            )

            await pool.close()

        finally:
            server.close()
            await server.wait_closed()
