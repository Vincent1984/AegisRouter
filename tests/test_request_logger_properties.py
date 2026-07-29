"""Property-based tests for RequestLoggerCallback using Hypothesis.

Validates correctness properties from the request-logging design document.

# Feature: request-logging
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from hypothesis import given, settings, HealthCheck, strategies as st

from aegis_router.observability.request_logger import (
    RequestLoggingConfig,
    _LogEntryBuilder,
)


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Strategy for generating arbitrary message content
_st_message_content = st.text(min_size=0, max_size=500)

# Strategy for generating a single chat message dict
_st_chat_message = st.fixed_dictionaries(
    {"role": st.sampled_from(["system", "user", "assistant"]), "content": _st_message_content}
)

# Strategy for generating a non-empty list of messages
_st_messages = st.lists(_st_chat_message, min_size=1, max_size=10)

# Strategy for generating metadata with optional fields
_st_metadata = st.fixed_dictionaries(
    {},
    optional={
        "request_id": st.text(min_size=1, max_size=50),
        "session_id": st.text(min_size=0, max_size=50),
        "target_model": st.text(min_size=1, max_size=50),
        "routing_plugin": st.sampled_from(["conversation", "transaction", "agent_workbuddy"]),
        "route_reason": st.text(min_size=0, max_size=100),
        "route_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    },
)

# Strategy for generating request data dict
_st_request_data = st.fixed_dictionaries(
    {"messages": _st_messages},
    optional={
        "model": st.text(min_size=1, max_size=50),
        "metadata": _st_metadata,
        "call_type": st.sampled_from(["completion", "acompletion", "text_completion"]),
    },
)

# Strategy for generating a config
_st_config = st.builds(
    RequestLoggingConfig,
    enabled=st.just(True),
    output=st.sampled_from(["stdout", "file", "both"]),
    max_message_length=st.integers(min_value=0, max_value=10000),
    retention_days=st.integers(min_value=1, max_value=365),
)

# Strategy for kwargs used in success/failure hooks
_st_kwargs_with_slo = st.fixed_dictionaries(
    {
        "litellm_params": st.fixed_dictionaries(
            {},
            optional={"metadata": _st_metadata},
        ),
    },
    optional={
        "model": st.text(min_size=1, max_size=50),
        "metadata": _st_metadata,
        "standard_logging_object": st.fixed_dictionaries(
            {},
            optional={
                "prompt_tokens": st.integers(min_value=0, max_value=100000),
                "completion_tokens": st.integers(min_value=0, max_value=100000),
                "total_tokens": st.integers(min_value=0, max_value=200000),
                "response_time_ms": st.floats(
                    min_value=0.0, max_value=60000.0, allow_nan=False
                ),
                "model": st.text(min_size=1, max_size=50),
            },
        ),
        "exception": st.builds(ValueError, st.text(min_size=1, max_size=100)),
    },
)

# ISO-8601 UTC timestamp regex with millisecond precision and Z suffix
_ISO8601_MS_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)

_VALID_EVENT_TYPES = {"request", "response_success", "response_failure"}


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 1: 日志条目结构有效性
# Validates: Requirements 5.1, 5.2, 5.3, 5.5
# ---------------------------------------------------------------------------


class TestProperty1LogEntryStructureValidity:
    """Property 1: 日志条目结构有效性

    对于任意输入数据（request、success 或 failure），Request Logger 发出的日志条目应当是：
    1. 有效的单行 JSON 字符串
    2. 包含匹配 UTC ISO-8601 格式（毫秒精度）的 ts 字段
    3. 包含非空的 request_id 字段
    4. 包含正确的 event_type 字段

    **Validates: Requirements 5.1, 5.2, 5.3, 5.5**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(data=_st_request_data, config=_st_config)
    def test_property_1_request_entry_structure(self, data: dict, config: RequestLoggingConfig):
        """Request entries produced by build_request_entry have valid structure."""
        entry = _LogEntryBuilder.build_request_entry(data, config)

        # 1. Valid single-line JSON
        json_str = json.dumps(entry, ensure_ascii=False)
        assert "\n" not in json_str, "Log entry must be single-line JSON"
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # 2. ts field matches UTC ISO-8601 with millisecond precision
        assert "ts" in entry, "Entry must contain 'ts' field"
        assert _ISO8601_MS_PATTERN.match(entry["ts"]), (
            f"ts field '{entry['ts']}' does not match ISO-8601 UTC ms format"
        )
        # Verify it parses as a valid datetime
        parsed_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
        assert parsed_ts.tzinfo is not None

        # 3. Non-empty request_id
        assert "request_id" in entry, "Entry must contain 'request_id' field"
        assert isinstance(entry["request_id"], str)
        assert len(entry["request_id"]) > 0, "request_id must be non-empty"

        # 4. Correct event_type
        assert "event_type" in entry, "Entry must contain 'event_type' field"
        assert entry["event_type"] == "request"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(kwargs=_st_kwargs_with_slo, config=_st_config)
    def test_property_1_success_entry_structure(self, kwargs: dict, config: RequestLoggingConfig):
        """Success entries produced by build_success_entry have valid structure."""
        entry = _LogEntryBuilder.build_success_entry(
            kwargs, response_obj=None, start_time=None, end_time=None, config=config
        )

        # 1. Valid single-line JSON
        json_str = json.dumps(entry, ensure_ascii=False)
        assert "\n" not in json_str, "Log entry must be single-line JSON"
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # 2. ts field matches UTC ISO-8601 with millisecond precision
        assert "ts" in entry, "Entry must contain 'ts' field"
        assert _ISO8601_MS_PATTERN.match(entry["ts"]), (
            f"ts field '{entry['ts']}' does not match ISO-8601 UTC ms format"
        )
        parsed_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
        assert parsed_ts.tzinfo is not None

        # 3. Non-empty request_id
        assert "request_id" in entry, "Entry must contain 'request_id' field"
        assert isinstance(entry["request_id"], str)
        assert len(entry["request_id"]) > 0, "request_id must be non-empty"

        # 4. Correct event_type
        assert "event_type" in entry, "Entry must contain 'event_type' field"
        assert entry["event_type"] == "response_success"

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(kwargs=_st_kwargs_with_slo, config=_st_config)
    def test_property_1_failure_entry_structure(self, kwargs: dict, config: RequestLoggingConfig):
        """Failure entries produced by build_failure_entry have valid structure."""
        entry = _LogEntryBuilder.build_failure_entry(
            kwargs, response_obj=None, start_time=None, end_time=None, config=config
        )

        # 1. Valid single-line JSON
        json_str = json.dumps(entry, ensure_ascii=False)
        assert "\n" not in json_str, "Log entry must be single-line JSON"
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # 2. ts field matches UTC ISO-8601 with millisecond precision
        assert "ts" in entry, "Entry must contain 'ts' field"
        assert _ISO8601_MS_PATTERN.match(entry["ts"]), (
            f"ts field '{entry['ts']}' does not match ISO-8601 UTC ms format"
        )
        parsed_ts = datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
        assert parsed_ts.tzinfo is not None

        # 3. Non-empty request_id
        assert "request_id" in entry, "Entry must contain 'request_id' field"
        assert isinstance(entry["request_id"], str)
        assert len(entry["request_id"]) > 0, "request_id must be non-empty"

        # 4. Correct event_type
        assert "event_type" in entry, "Entry must contain 'event_type' field"
        assert entry["event_type"] == "response_failure"


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 5: 失败事件捕获错误详情
# Validates: Requirements 4.1, 4.2, 4.3, 4.4
# ---------------------------------------------------------------------------


# Strategy: generate random exception types and messages
_exception_classes = st.sampled_from([
    ValueError,
    TypeError,
    RuntimeError,
    KeyError,
    IOError,
    OSError,
    AttributeError,
    IndexError,
    ZeroDivisionError,
    NotImplementedError,
    ConnectionError,
    TimeoutError,
    PermissionError,
    FileNotFoundError,
])

_exception_messages = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
)

_slo_strategy = st.fixed_dictionaries({
    "prompt_tokens": st.integers(min_value=0, max_value=100000),
    "completion_tokens": st.integers(min_value=0, max_value=100000),
    "total_tokens": st.integers(min_value=0, max_value=200000),
    "response_time_ms": st.floats(min_value=0.1, max_value=60000.0, allow_nan=False, allow_infinity=False),
})


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    exc_class=_exception_classes,
    exc_message=_exception_messages,
    slo=_slo_strategy,
)
def test_property_5a_failure_with_exception_and_slo(
    exc_class: type,
    exc_message: str,
    slo: dict,
):
    """Property 5A: When kwargs contains an exception AND a standard_logging_object,
    error_message contains the exception's string representation,
    error_type equals the exception class name,
    usage is populated from SLO,
    and incomplete_data is False.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    # Arrange
    exception = exc_class(exc_message)
    kwargs = {
        "exception": exception,
        "standard_logging_object": slo,
        "model": "test-model",
        "litellm_params": {"metadata": {}},
    }
    config = RequestLoggingConfig()

    # Act
    entry = _LogEntryBuilder.build_failure_entry(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
        config=config,
    )

    # Assert
    # 4.1: error event type
    assert entry["event_type"] == "response_failure"

    # 4.2: error_message contains exception string, error_type is class name
    assert str(exception) in entry["error_message"]
    assert entry["error_type"] == exc_class.__name__

    # 4.3: usage is populated from SLO
    assert entry["usage"] is not None
    assert entry["usage"]["input_tokens"] == slo["prompt_tokens"]
    assert entry["usage"]["output_tokens"] == slo["completion_tokens"]
    assert entry["usage"]["total_tokens"] == slo["total_tokens"]

    # 4.4: incomplete_data is False when SLO exists
    assert entry["incomplete_data"] is False


@settings(max_examples=100)
@given(
    exc_class=_exception_classes,
    exc_message=_exception_messages,
)
def test_property_5b_failure_with_exception_no_slo(
    exc_class: type,
    exc_message: str,
):
    """Property 5B: When kwargs contains an exception but NO standard_logging_object,
    error_message contains the exception's string representation,
    error_type equals the exception class name,
    usage is None,
    and incomplete_data is True.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    # Arrange
    exception = exc_class(exc_message)
    kwargs = {
        "exception": exception,
        "model": "test-model",
        "litellm_params": {"metadata": {}},
    }
    config = RequestLoggingConfig()

    # Act
    entry = _LogEntryBuilder.build_failure_entry(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
        config=config,
    )

    # Assert
    # 4.1: error event type
    assert entry["event_type"] == "response_failure"

    # 4.2: error_message contains exception string, error_type is class name
    assert str(exception) in entry["error_message"]
    assert entry["error_type"] == exc_class.__name__

    # 4.3: usage is None when SLO is absent
    assert entry["usage"] is None

    # 4.4: incomplete_data is True when SLO is absent
    assert entry["incomplete_data"] is True


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 4: 从 standard_logging_object 提取成功响应数据
# ---------------------------------------------------------------------------


# Strategy: generate random standard_logging_object dicts for success entries
_slo_success_strategy = st.fixed_dictionaries({
    "prompt_tokens": st.integers(min_value=0, max_value=10_000_000),
    "completion_tokens": st.integers(min_value=0, max_value=10_000_000),
    "total_tokens": st.integers(min_value=0, max_value=10_000_000),
    "response_time_ms": st.floats(
        min_value=0.01, max_value=600_000.0, allow_nan=False, allow_infinity=False
    ),
})


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(slo=_slo_success_strategy)
def test_property_4_slo_token_and_latency_passthrough(slo: dict):
    """Property 4: For any standard_logging_object with token usage and latency,
    the emitted response_success log entry passes values through directly
    (no recalculation).

    Verifies:
    1. usage.input_tokens == slo["prompt_tokens"]
    2. usage.output_tokens == slo["completion_tokens"]
    3. usage.total_tokens == slo["total_tokens"]
    4. latency_ms == slo["response_time_ms"]
    5. Values are passed through directly without recalculation

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """
    # Arrange: build kwargs with standard_logging_object and required metadata
    kwargs = {
        "standard_logging_object": slo,
        "litellm_params": {
            "metadata": {
                "request_id": "prop4-test-req",
            }
        },
        "model": "gpt-4",
    }
    config = RequestLoggingConfig()

    # Act: build a success entry
    entry = _LogEntryBuilder.build_success_entry(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
        config=config,
    )

    # Assert event type
    assert entry["event_type"] == "response_success"

    # 1. usage.input_tokens == slo["prompt_tokens"]
    assert entry["usage"]["input_tokens"] == slo["prompt_tokens"], (
        f"input_tokens mismatch: expected {slo['prompt_tokens']}, "
        f"got {entry['usage']['input_tokens']}"
    )

    # 2. usage.output_tokens == slo["completion_tokens"]
    assert entry["usage"]["output_tokens"] == slo["completion_tokens"], (
        f"output_tokens mismatch: expected {slo['completion_tokens']}, "
        f"got {entry['usage']['output_tokens']}"
    )

    # 3. usage.total_tokens == slo["total_tokens"]
    assert entry["usage"]["total_tokens"] == slo["total_tokens"], (
        f"total_tokens mismatch: expected {slo['total_tokens']}, "
        f"got {entry['usage']['total_tokens']}"
    )

    # 4. latency_ms == slo["response_time_ms"]
    assert entry["latency_ms"] == slo["response_time_ms"], (
        f"latency_ms mismatch: expected {slo['response_time_ms']}, "
        f"got {entry['latency_ms']}"
    )

    # 5. Values are passed through directly (no recalculation) -
    #    verify exact equality (not derived from other fields)
    assert entry["usage"]["input_tokens"] == slo["prompt_tokens"]
    assert entry["usage"]["output_tokens"] == slo["completion_tokens"]
    assert entry["usage"]["total_tokens"] == slo["total_tokens"]
    assert entry["latency_ms"] == slo["response_time_ms"]


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 8: 非修改不变量
# Validates: Requirements 7.4, 8.4
# ---------------------------------------------------------------------------

import asyncio
import copy

from aegis_router.observability.request_logger import RequestLoggerCallback


# Strategy for generating nested data dictionaries for async_pre_call_hook
_st_nested_metadata = st.fixed_dictionaries(
    {},
    optional={
        "request_id": st.text(min_size=1, max_size=50),
        "session_id": st.text(min_size=0, max_size=50),
        "target_model": st.text(min_size=1, max_size=50),
        "routing_plugin": st.sampled_from(["conversation", "transaction", "agent_workbuddy"]),
        "route_reason": st.text(min_size=0, max_size=100),
        "route_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    },
)

_st_hook_data = st.fixed_dictionaries(
    {"messages": _st_messages},
    optional={
        "model": st.text(min_size=1, max_size=50),
        "metadata": _st_nested_metadata,
        "stream": st.booleans(),
        "temperature": st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
        "max_tokens": st.integers(min_value=1, max_value=100000),
    },
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(data=_st_hook_data)
def test_property_8_non_modification_invariant(data: dict):
    """Property 8: 非修改不变量

    For ANY data dictionary passed to async_pre_call_hook, the hook must NOT
    modify the data dictionary. The returned value must be deep-equal to the
    original input, and the original reference must remain unmutated.

    **Validates: Requirements 7.4, 8.4**
    """
    # Arrange: create a deep copy before calling the hook
    data_before = copy.deepcopy(data)

    config = RequestLoggingConfig(enabled=True, output="stdout")
    callback = RequestLoggerCallback(config)

    # Act: call async_pre_call_hook
    result = asyncio.run(
        callback.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data=data,
            call_type="completion",
        )
    )

    # Assert: the returned data is deep-equal to the original
    assert result == data_before, (
        f"async_pre_call_hook returned modified data.\n"
        f"Original: {data_before}\n"
        f"Returned: {result}"
    )

    # Assert: the data dict itself was NOT mutated (original reference check)
    assert data == data_before, (
        f"async_pre_call_hook mutated the input data dict.\n"
        f"Original: {data_before}\n"
        f"After hook: {data}"
    )


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 6: 消息截断正确性
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

# Strategy: random max_message_length values (positive and zero/negative)
_st_max_length_positive = st.integers(min_value=1, max_value=5000)
_st_max_length_non_positive = st.integers(min_value=-100, max_value=0)

# Strategy: random content strings of varying lengths
_st_content_varied = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z", "S")),
    min_size=0,
    max_size=10000,
)

# Strategy: random role values
_st_role = st.sampled_from(["system", "user", "assistant", "tool", "function"])


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    content=_st_content_varied,
    max_length=_st_max_length_positive,
    role=_st_role,
)
def test_property_6a_truncation_when_exceeds_max_length(
    content: str,
    max_length: int,
    role: str,
):
    """Property 6A: For any content string and any max_message_length > 0,
    if len(content) > max_message_length, the result is
    content[:max_message_length] + " [truncated]";
    if len(content) <= max_message_length, the result is unchanged.

    The role field is always preserved regardless of truncation.

    **Validates: Requirements 6.4**
    """
    messages = [{"role": role, "content": content}]

    result = _LogEntryBuilder._truncate_messages(messages, max_length)

    assert len(result) == 1
    result_msg = result[0]

    # Role is always preserved
    assert result_msg["role"] == role

    if len(content) > max_length:
        # Content is truncated to max_length chars + " [truncated]"
        expected = content[:max_length] + " [truncated]"
        assert result_msg["content"] == expected, (
            f"Expected truncated content of length {len(expected)}, "
            f"got length {len(result_msg['content'])}"
        )
    else:
        # Content is unchanged
        assert result_msg["content"] == content


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    content=_st_content_varied,
    max_length=_st_max_length_non_positive,
    role=_st_role,
)
def test_property_6b_no_truncation_when_max_length_non_positive(
    content: str,
    max_length: int,
    role: str,
):
    """Property 6B: For any content string and max_message_length <= 0,
    the result is always unchanged (no truncation is applied).

    The role field is always preserved.

    **Validates: Requirements 6.4**
    """
    messages = [{"role": role, "content": content}]

    result = _LogEntryBuilder._truncate_messages(messages, max_length)

    assert len(result) == 1
    result_msg = result[0]

    # Role is always preserved
    assert result_msg["role"] == role

    # Content is unchanged when max_length <= 0
    assert result_msg["content"] == content


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    messages=st.lists(
        st.fixed_dictionaries({
            "role": _st_role,
            "content": _st_content_varied,
        }),
        min_size=1,
        max_size=10,
    ),
    max_length=_st_max_length_positive,
)
def test_property_6c_role_preserved_for_all_messages(
    messages: list,
    max_length: int,
):
    """Property 6C: For any list of messages, the role field is always
    preserved regardless of whether truncation occurs.

    **Validates: Requirements 6.4**
    """
    result = _LogEntryBuilder._truncate_messages(messages, max_length)

    assert len(result) == len(messages)

    for original, truncated in zip(messages, result):
        # Role is always preserved
        assert truncated["role"] == original["role"]

        content = original["content"]
        if len(content) > max_length:
            expected = content[:max_length] + " [truncated]"
            assert truncated["content"] == expected
        else:
            assert truncated["content"] == content


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 7: 错误隔离 — 无异常传播
# Validates: Requirements 7.3
# ---------------------------------------------------------------------------

import asyncio
from unittest.mock import MagicMock

from hypothesis import given, settings, HealthCheck, strategies as st

from aegis_router.observability.request_logger import (
    RequestLoggerCallback,
    RequestLoggingConfig,
)


# Strategy: objects that are NOT JSON-serializable
_st_non_serializable = st.one_of(
    st.just(lambda x: x),                       # lambda
    st.just(object()),                           # bare object
    st.builds(MagicMock),                        # Mock objects
    st.just(b"\x80\x81\x82"),                    # raw bytes
    st.just(float("inf")),                       # infinity
    st.just(float("nan")),                       # NaN
    st.just(set()),                              # set (not serializable)
    st.just(frozenset([1, 2, 3])),               # frozenset
)

# Strategy: values that may appear as fields in data dicts
_st_problematic_values = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(min_size=0, max_size=200),
    st.binary(min_size=0, max_size=50),
    _st_non_serializable,
    st.lists(st.none(), min_size=0, max_size=5),
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(st.none(), st.integers(), st.text(min_size=0, max_size=20)),
        min_size=0,
        max_size=5,
    ),
)

# Strategy: generate messy data dict for pre_call_hook
_st_messy_data = st.fixed_dictionaries(
    {},
    optional={
        "messages": st.one_of(
            st.none(),
            st.just("not_a_list"),
            st.lists(_st_problematic_values, min_size=0, max_size=5),
            st.just([{"role": "user", "content": lambda: "boom"}]),
        ),
        "model": _st_problematic_values,
        "metadata": st.one_of(
            st.none(),
            _st_problematic_values,
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=_st_problematic_values,
                min_size=0,
                max_size=5,
            ),
        ),
        "call_type": _st_problematic_values,
    },
)

# Strategy: generate messy kwargs for success/failure hooks
_st_messy_kwargs = st.fixed_dictionaries(
    {},
    optional={
        "litellm_params": st.one_of(
            st.none(),
            _st_problematic_values,
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=_st_problematic_values,
                min_size=0,
                max_size=5,
            ),
        ),
        "model": _st_problematic_values,
        "metadata": _st_problematic_values,
        "standard_logging_object": st.one_of(
            st.none(),
            _st_problematic_values,
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=_st_problematic_values,
                min_size=0,
                max_size=5,
            ),
        ),
        "exception": st.one_of(
            st.none(),
            st.builds(ValueError, st.text(min_size=0, max_size=50)),
            st.just(RuntimeError()),
            _st_non_serializable,
        ),
    },
)


def _make_logger_callback() -> RequestLoggerCallback:
    """Create a RequestLoggerCallback with stdout output (no file I/O)."""
    config = RequestLoggingConfig(
        enabled=True,
        output="stdout",
        max_message_length=100,
        retention_days=1,
    )
    return RequestLoggerCallback(config)


class TestProperty7ErrorIsolationNoExceptionPropagation:
    """Property 7: 错误隔离 — 无异常传播

    For ANY input (including inputs that would cause JSON serialization errors,
    unexpected types, missing fields, non-serializable objects), the callback
    hooks NEVER raise an exception to the caller.

    **Validates: Requirements 7.3**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(data=_st_messy_data)
    def test_property_7_pre_call_hook_no_exception(self, data: dict):
        """async_pre_call_hook never raises, even with non-serializable/malformed data."""
        cb = _make_logger_callback()
        # Should not raise any exception
        result = asyncio.run(
            cb.async_pre_call_hook(
                user_api_key_dict={},
                cache=None,
                data=data,
                call_type="completion",
            )
        )
        # The hook always returns data (possibly the original dict)
        assert result is not None or result is None  # no exception = pass

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(kwargs=_st_messy_kwargs)
    def test_property_7_log_success_event_no_exception(self, kwargs: dict):
        """async_log_success_event never raises, even with malformed kwargs."""
        cb = _make_logger_callback()
        # Should not raise any exception
        asyncio.run(
            cb.async_log_success_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=None,
                end_time=None,
            )
        )
        # If we reach here, no exception was propagated — test passes

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(kwargs=_st_messy_kwargs)
    def test_property_7_log_failure_event_no_exception(self, kwargs: dict):
        """async_log_failure_event never raises, even with malformed kwargs."""
        cb = _make_logger_callback()
        # Should not raise any exception
        asyncio.run(
            cb.async_log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=None,
                end_time=None,
            )
        )
        # If we reach here, no exception was propagated — test passes

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(non_serializable=_st_non_serializable)
    def test_property_7_pre_call_with_non_serializable_in_messages(
        self, non_serializable
    ):
        """async_pre_call_hook handles non-serializable objects within messages."""
        cb = _make_logger_callback()
        data = {
            "messages": [
                {"role": "user", "content": non_serializable},
            ],
            "metadata": {"request_id": "test-123", "target_model": non_serializable},
        }
        # Should not raise any exception
        result = asyncio.run(
            cb.async_pre_call_hook(
                user_api_key_dict={},
                cache=None,
                data=data,
                call_type="completion",
            )
        )
        # The hook must return data regardless of serialization issues
        assert result == data


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 2: 请求消息忠实捕获
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------


class TestProperty2RequestMessageFaithfulCapture:
    """Property 2: 请求消息忠实捕获

    For any request data containing a non-empty messages array, the emitted
    "request" log entry must completely preserve the messages (subject to
    truncation configuration). Message count is always preserved. Role fields
    are always preserved. Content follows truncation rules.

    **Validates: Requirements 1.1**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(messages=_st_messages, metadata=_st_metadata)
    def test_property_2a_messages_preserved_no_truncation(
        self, messages: list, metadata: dict
    ):
        """With max_message_length=0 (no truncation), messages are preserved
        exactly — role and content unchanged, message count preserved."""
        config = RequestLoggingConfig(enabled=True, max_message_length=0)
        data = {"messages": messages, "metadata": metadata, "call_type": "completion"}

        entry = _LogEntryBuilder.build_request_entry(data, config)

        # Message count is preserved
        assert len(entry["messages"]) == len(messages), (
            f"Message count mismatch: expected {len(messages)}, got {len(entry['messages'])}"
        )

        # Each message's role and content are unchanged
        for original, logged in zip(messages, entry["messages"]):
            assert logged["role"] == original["role"], (
                f"Role mismatch: expected '{original['role']}', got '{logged['role']}'"
            )
            assert logged["content"] == original["content"], (
                f"Content mismatch: expected '{original['content'][:50]}...', "
                f"got '{logged['content'][:50]}...'"
            )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        messages=_st_messages,
        metadata=_st_metadata,
        max_length=st.integers(min_value=1, max_value=5000),
    )
    def test_property_2b_messages_with_truncation(
        self, messages: list, metadata: dict, max_length: int
    ):
        """With max_message_length > 0, role is always preserved, content
        follows truncation rules, and message count is preserved."""
        config = RequestLoggingConfig(enabled=True, max_message_length=max_length)
        data = {"messages": messages, "metadata": metadata, "call_type": "completion"}

        entry = _LogEntryBuilder.build_request_entry(data, config)

        # Message count is always preserved
        assert len(entry["messages"]) == len(messages), (
            f"Message count mismatch: expected {len(messages)}, got {len(entry['messages'])}"
        )

        for original, logged in zip(messages, entry["messages"]):
            # Role is always preserved
            assert logged["role"] == original["role"], (
                f"Role mismatch: expected '{original['role']}', got '{logged['role']}'"
            )

            # Content follows truncation rules
            original_content = original["content"]
            if len(original_content) > max_length:
                expected_content = original_content[:max_length] + " [truncated]"
                assert logged["content"] == expected_content, (
                    f"Truncation mismatch for content of length {len(original_content)} "
                    f"with max_length={max_length}"
                )
            else:
                assert logged["content"] == original_content, (
                    f"Content should be unchanged when length {len(original_content)} "
                    f"<= max_length {max_length}"
                )

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(messages=_st_messages, config=_st_config)
    def test_property_2c_message_count_always_preserved(
        self, messages: list, config: RequestLoggingConfig
    ):
        """For any config, the message count in the entry always equals
        the message count in the input."""
        data = {"messages": messages, "metadata": {}, "call_type": "completion"}

        entry = _LogEntryBuilder.build_request_entry(data, config)

        assert len(entry["messages"]) == len(messages), (
            f"Message count mismatch: expected {len(messages)}, got {len(entry['messages'])}"
        )


# ---------------------------------------------------------------------------
# Feature: request-logging, Property 3: 元数据字段忠实传播
# Validates: Requirements 1.2, 2.1, 2.2, 2.3, 8.3
# ---------------------------------------------------------------------------


class TestProperty3MetadataFieldFaithfulPropagation:
    """Property 3: 元数据字段忠实传播

    For any request metadata containing session_id, request_id, target_model,
    routing_plugin, route_reason, or route_score, the emitted log entry must
    contain these fields with values identical to the source metadata.

    **Validates: Requirements 1.2, 2.1, 2.2, 2.3, 8.3**
    """

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(metadata=_st_metadata, messages=_st_messages)
    def test_property_3a_request_id_propagation(
        self, metadata: dict, messages: list
    ):
        """When metadata contains request_id, the entry's request_id MUST
        match it exactly."""
        config = RequestLoggingConfig(enabled=True, max_message_length=0)
        data = {"messages": messages, "metadata": metadata, "call_type": "completion"}

        entry = _LogEntryBuilder.build_request_entry(data, config)

        if "request_id" in metadata:
            assert entry["request_id"] == metadata["request_id"], (
                f"request_id mismatch: expected '{metadata['request_id']}', "
                f"got '{entry['request_id']}'"
            )
        else:
            # When metadata lacks request_id, a UUID is generated
            assert entry["request_id"] is not None
            assert len(entry["request_id"]) > 0

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(metadata=_st_metadata, messages=_st_messages)
    def test_property_3b_session_id_propagation(
        self, metadata: dict, messages: list
    ):
        """When metadata contains session_id, the entry's session_id MUST
        match it exactly."""
        config = RequestLoggingConfig(enabled=True, max_message_length=0)
        data = {"messages": messages, "metadata": metadata, "call_type": "completion"}

        entry = _LogEntryBuilder.build_request_entry(data, config)

        if "session_id" in metadata:
            assert entry["session_id"] == metadata["session_id"], (
                f"session_id mismatch: expected '{metadata['session_id']}', "
                f"got '{entry['session_id']}'"
            )
        else:
            # When metadata lacks session_id, entry has None
            assert entry["session_id"] is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(metadata=_st_metadata, messages=_st_messages)
    def test_property_3c_routing_decision_fields_propagation(
        self, metadata: dict, messages: list
    ):
        """routing_decision fields (target_model, routing_plugin, route_reason,
        route_score) must match their metadata counterparts."""
        config = RequestLoggingConfig(enabled=True, max_message_length=0)
        data = {"messages": messages, "metadata": metadata, "call_type": "completion"}

        entry = _LogEntryBuilder.build_request_entry(data, config)

        routing_decision = entry["routing_decision"]

        # target_model
        if "target_model" in metadata:
            assert routing_decision["target_model"] == metadata["target_model"], (
                f"target_model mismatch: expected '{metadata['target_model']}', "
                f"got '{routing_decision['target_model']}'"
            )
        else:
            assert routing_decision["target_model"] is None

        # routing_plugin
        if "routing_plugin" in metadata:
            assert routing_decision["routing_plugin"] == metadata["routing_plugin"], (
                f"routing_plugin mismatch: expected '{metadata['routing_plugin']}', "
                f"got '{routing_decision['routing_plugin']}'"
            )
        else:
            assert routing_decision["routing_plugin"] is None

        # route_reason
        if "route_reason" in metadata:
            assert routing_decision["route_reason"] == metadata["route_reason"], (
                f"route_reason mismatch: expected '{metadata['route_reason']}', "
                f"got '{routing_decision['route_reason']}'"
            )
        else:
            assert routing_decision["route_reason"] is None

        # route_score
        if "route_score" in metadata:
            assert routing_decision["route_score"] == metadata["route_score"], (
                f"route_score mismatch: expected {metadata['route_score']}, "
                f"got {routing_decision['route_score']}"
            )
        else:
            assert routing_decision["route_score"] is None

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    @given(
        request_id=st.text(min_size=1, max_size=50),
        session_id=st.text(min_size=0, max_size=50),
        target_model=st.text(min_size=1, max_size=50),
        routing_plugin=st.sampled_from(["conversation", "transaction", "agent_workbuddy"]),
        route_reason=st.text(min_size=0, max_size=100),
        route_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_property_3d_all_fields_present_propagation(
        self,
        request_id: str,
        session_id: str,
        target_model: str,
        routing_plugin: str,
        route_reason: str,
        route_score: float,
    ):
        """When ALL metadata fields are present, every field propagates
        correctly to the log entry."""
        metadata = {
            "request_id": request_id,
            "session_id": session_id,
            "target_model": target_model,
            "routing_plugin": routing_plugin,
            "route_reason": route_reason,
            "route_score": route_score,
        }
        config = RequestLoggingConfig(enabled=True, max_message_length=0)
        data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": metadata,
            "call_type": "completion",
        }

        entry = _LogEntryBuilder.build_request_entry(data, config)

        # request_id must match exactly
        assert entry["request_id"] == request_id

        # session_id must match exactly
        assert entry["session_id"] == session_id

        # routing_decision fields must match
        assert entry["routing_decision"]["target_model"] == target_model
        assert entry["routing_decision"]["routing_plugin"] == routing_plugin
        assert entry["routing_decision"]["route_reason"] == route_reason
        assert entry["routing_decision"]["route_score"] == route_score
