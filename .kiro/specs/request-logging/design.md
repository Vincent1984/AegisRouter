# 请求日志 — 技术设计文档

## 概述

Request Logger 是一个独立的 LiteLLM `CustomLogger` 回调，为通过 AegisRouter 的每个 LLM 请求记录完整的请求-响应生命周期。它独立于路由插件运行 — 不做路由决策，不修改请求/响应数据。它只是观察管道并发出结构化 JSON 日志条目，捕获：

- 请求体（PII 脱敏后）
- 路由决策元数据（由当前活跃的路由插件填充）
- 响应内容、Token 用量和延迟（来自 LiteLLM 的 `standard_logging_object`）
- 请求失败时的错误详情

**核心设计决策：**

1. **独立 CustomLogger，不继承 BaseRouterCallback** — Request Logger 没有路由逻辑、没有 PII 脱敏、没有合规检测。它只读取其他回调已经产生的数据。

2. **观察者模式** — 它从 `data["metadata"]`（路由插件填充）读取路由元数据，从 `kwargs["standard_logging_object"]`（LiteLLM 填充）读取响应指标。它从不写入这些结构。

3. **独立 logger 命名空间** — 使用 `aegis_router.request_log`，将请求日志与应用日志（`aegis_router.*`）和审计日志（`aegis_router.audit`）分离。

4. **发射即忘 + 错误隔离** — 所有日志操作都包裹在 try/except 中。日志失败永远不会传播到请求管道。

5. **通过 `custom_callbacks.py` 注册** — Request Logger 实例通过 LiteLLM 原生的多回调支持，被添加到 `litellm.callbacks` 列表中，与路由插件并行运行。

---

## 架构

### 回调链位置

```
litellm.callbacks = [
    routing_plugin_instance,      # TransactionRouterCallback 或 SmartRouterCallback
    request_logger_instance,      # RequestLoggerCallback（本功能）
]
```

两个回调接收相同的钩子调用。路由插件先执行（修改 `data["model"]` 并填充元数据），Request Logger 后执行（读取元数据并记录日志）。

### 数据流

```
请求到达
    │
    ▼
LiteLLM 对每个回调调用 async_pre_call_hook:
    │
    ├── [1] 路由插件: 合规 → PII 脱敏 → 路由 → 设置元数据
    │       data["metadata"]["target_model"] = "gpt-5.5"
    │       data["metadata"]["routing_plugin"] = "transaction"
    │       data["metadata"]["route_reason"] = "plan"
    │
    ├── [2] Request Logger: 读取 data["messages"]、data["metadata"]
    │       → 发出 "request" 日志条目
    │
    ▼
LiteLLM 转发到目标模型
    │
    ▼
LiteLLM 对每个回调调用 async_log_success_event（或 failure）:
    │
    ├── [1] 路由插件: 还原响应中的 PII 占位符
    │
    ├── [2] Request Logger: 读取 response_obj、kwargs["standard_logging_object"]
    │       → 发出 "response_success" 或 "response_failure" 日志条目
    │
    ▼
响应返回客户端
```

### 模块位置

```
aegis_router/
├── observability/
│   ├── __init__.py
│   ├── audit_logger.py          # 现有审计日志
│   ├── metrics.py               # 现有指标
│   └── request_logger.py        # 新增: RequestLoggerCallback + 配置

config/
├── config.yaml                  # 修改: 添加 request_logging 段
├── custom_callbacks.py          # 修改: 实例化并注册 RequestLoggerCallback
```

---

## 组件与接口

### 1. RequestLoggerCallback

```python
from litellm.integrations.custom_logger import CustomLogger

class RequestLoggerCallback(CustomLogger):
    """独立的请求日志回调 — 只观察和记录，不修改。"""

    def __init__(self, config: RequestLoggingConfig):
        super().__init__()
        self._config = config
        self._logger = logging.getLogger("aegis_router.request_log")
        self._configure_logger()

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
            self._emit_request_entry(data)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RequestLogger pre_call error: %s", e
            )
        return data

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """记录响应内容、Token 用量和延迟（来自 standard_logging_object）。"""
        if not self._config.enabled:
            return
        try:
            self._emit_success_entry(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RequestLogger success_event error: %s", e
            )

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """记录失败详情（包含可用的 standard_logging_object 数据）。"""
        if not self._config.enabled:
            return
        try:
            self._emit_failure_entry(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "RequestLogger failure_event error: %s", e
            )
```

### 2. RequestLoggingConfig（Pydantic 模型）

```python
class RequestLoggingConfig(BaseModel):
    """请求日志回调配置。"""
    
    enabled: bool = True
    output: Literal["stdout", "file", "both"] = "file"
    file_path: str = "./logs/request_log.jsonl"
    max_message_length: int = 4096          # 字符数；0 = 不截断
    retention_days: int = 30                 # 文件日志保留天数
    log_level: str = "INFO"
```

### 3. 日志条目构建器（内部）

```python
class _LogEntryBuilder:
    """从原始回调数据构建结构化日志条目。"""

    @staticmethod
    def build_request_entry(data: dict, config: RequestLoggingConfig) -> dict:
        """构建 'request' 事件日志条目。"""
        ...

    @staticmethod
    def build_success_entry(
        kwargs: dict, response_obj: Any, start_time, end_time, config: RequestLoggingConfig
    ) -> dict:
        """构建 'response_success' 事件日志条目。"""
        ...

    @staticmethod
    def build_failure_entry(
        kwargs: dict, response_obj: Any, start_time, end_time, config: RequestLoggingConfig
    ) -> dict:
        """构建 'response_failure' 事件日志条目。"""
        ...
```

### 4. 注册方式（custom_callbacks.py 修改）

```python
# config/custom_callbacks.py
import sys
import litellm
from aegis_router.callbacks.plugin_loader import load_routing_plugin
from aegis_router.observability.request_logger import (
    RequestLoggerCallback,
    load_request_logging_config,
)

# 加载路由插件（现有行为）
proxy_handler_instance = load_routing_plugin(config_dir="/app/config")

# 加载并注册请求日志器
request_logging_config = load_request_logging_config(config_dir="/app/config")
if request_logging_config.enabled:
    request_logger_instance = RequestLoggerCallback(config=request_logging_config)
    litellm.callbacks.append(request_logger_instance)
    print(f"[custom_callbacks] RequestLogger enabled", file=sys.stderr, flush=True)
```

**为什么用 `litellm.callbacks.append()` 而不是替换 `proxy_handler_instance`？**

LiteLLM 的 Proxy 加载 `proxy_handler_instance` 作为第一个回调。额外的回调通过 append 到 `litellm.callbacks`（一个列表）来注册。这种方式被 LiteLLM 自身的内部回调（限流器、缓存控制、告警）使用，是多回调设置的标准模式。

---

## 数据模型

### 日志条目 Schema

每条日志条目是一个单行 JSON 对象，包含公共信封和事件特定字段。

#### 公共信封

| 字段 | 类型 | 描述 |
|------|------|------|
| `ts` | string | UTC ISO-8601 时间戳，毫秒精度（如 `2025-01-15T10:30:45.123Z`） |
| `event_type` | string | 取值: `"request"`、`"response_success"`、`"response_failure"` |
| `request_id` | string | 唯一请求标识（来自元数据或生成的 UUID） |
| `session_id` | string \| null | 会话标识（来自元数据，可为 null） |

#### "request" 事件

| 字段 | 类型 | 描述 |
|------|------|------|
| `messages` | array | 消息数组（PII 脱敏后），按配置截断 |
| `model_requested` | string | 原始请求中的模型名称 |
| `routing_decision.target_model` | string \| null | 路由插件选择的模型 |
| `routing_decision.routing_plugin` | string \| null | 插件名称（conversation/transaction/agent_workbuddy） |
| `routing_decision.route_reason` | string \| null | 选择原因 |
| `routing_decision.route_score` | float \| null | 路由评分（如适用） |
| `call_type` | string | LiteLLM 调用类型（如 "completion"） |

#### "response_success" 事件

| 字段 | 类型 | 描述 |
|------|------|------|
| `response_text` | string \| null | 响应内容（按配置截断） |
| `model_used` | string | 实际服务请求的模型 |
| `usage.input_tokens` | int | 来自 standard_logging_object 的 Prompt Token 数 |
| `usage.output_tokens` | int | 来自 standard_logging_object 的 Completion Token 数 |
| `usage.total_tokens` | int | 总 Token 数 |
| `latency_ms` | float | 来自 standard_logging_object 的端到端延迟 |
| `routing_decision.target_model` | string \| null | 用于关联 |
| `routing_decision.routing_plugin` | string \| null | 用于关联 |

#### "response_failure" 事件

| 字段 | 类型 | 描述 |
|------|------|------|
| `error_message` | string | 异常消息或错误描述 |
| `error_type` | string | 异常类名 |
| `model_used` | string \| null | 尝试使用的模型 |
| `usage` | object \| null | 来自 standard_logging_object 的 Token 用量（如有） |
| `latency_ms` | float \| null | 延迟（如有） |
| `incomplete_data` | bool | standard_logging_object 不可用时为 true |
| `routing_decision.target_model` | string \| null | 用于关联 |
| `routing_decision.routing_plugin` | string \| null | 用于关联 |

### config.yaml 配置

```yaml
# config/config.yaml — 新增段
request_logging:
  enabled: true
  output: "file"                    # "stdout"、"file" 或 "both"
  file_path: "./logs/request_log.jsonl"
  max_message_length: 4096          # 0 = 不截断
  retention_days: 30
```

### 日志条目示例

**请求事件：**
```json
{"ts":"2025-01-15T10:30:45.123Z","event_type":"request","request_id":"abc-123","session_id":"sess-456","messages":[{"role":"user","content":"分析这段代码..."}],"model_requested":"placeholder","routing_decision":{"target_model":"gpt-5.5","routing_plugin":"transaction","route_reason":"plan","route_score":null},"call_type":"completion"}
```

**成功事件：**
```json
{"ts":"2025-01-15T10:30:47.456Z","event_type":"response_success","request_id":"abc-123","session_id":"sess-456","response_text":"以下是我的分析...","model_used":"gpt-5.5","usage":{"input_tokens":150,"output_tokens":320,"total_tokens":470},"latency_ms":2312.5,"routing_decision":{"target_model":"gpt-5.5","routing_plugin":"transaction"}}
```

**失败事件：**
```json
{"ts":"2025-01-15T10:30:47.789Z","event_type":"response_failure","request_id":"abc-123","session_id":"sess-456","error_message":"Rate limit exceeded","error_type":"RateLimitError","model_used":"gpt-5.5","usage":null,"latency_ms":1523.2,"incomplete_data":false,"routing_decision":{"target_model":"gpt-5.5","routing_plugin":"transaction"}}
```

---

## 实现细节

### 读取路由元数据

Request Logger 从 `data["metadata"]` 读取路由决策，该字典由活跃的路由插件在其 `async_pre_call_hook` 中填充。元数据 key 在所有路由插件类型中保持一致：

```python
metadata = data.get("metadata", {})
routing_decision = {
    "target_model": metadata.get("target_model"),
    "routing_plugin": metadata.get("routing_plugin"),
    "route_reason": metadata.get("route_reason"),
    "route_score": metadata.get("route_score"),
}
```

这对所有三种插件类型都有效，因为它们都写入相同的元数据 key（由 BaseRouterCallback 约定建立）。

### 使用 standard_logging_object

LiteLLM 在 success/failure 回调中提供 `kwargs["standard_logging_object"]`。这是一个包含预计算指标的字典：

```python
slo = kwargs.get("standard_logging_object", {})
usage = {
    "input_tokens": slo.get("prompt_tokens", 0),
    "output_tokens": slo.get("completion_tokens", 0),
    "total_tokens": slo.get("total_tokens", 0),
}
latency_ms = slo.get("response_time_ms") or slo.get("completion_start_time_ms")
```

Request Logger 绝不独立计算这些指标 — 它信任并转发 LiteLLM 提供的数据。

### 消息截断

```python
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
```

### Logger 配置

```python
def _configure_logger(self):
    """配置 aegis_router.request_log logger 的处理器。"""
    self._logger.setLevel(getattr(logging, self._config.log_level.upper()))
    self._logger.propagate = False
    self._logger.handlers.clear()

    formatter = _JsonPassthroughFormatter()  # 与 audit_logger 相同模式

    if self._config.output in ("stdout", "both"):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    if self._config.output in ("file", "both"):
        from logging.handlers import TimedRotatingFileHandler
        os.makedirs(os.path.dirname(self._config.file_path), exist_ok=True)
        handler = TimedRotatingFileHandler(
            self._config.file_path,
            when="midnight",
            interval=1,
            backupCount=self._config.retention_days,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
```

**日志保留:** 使用 Python 的 `TimedRotatingFileHandler`，`backupCount=retention_days`。文件每天午夜轮转，超出保留窗口的旧文件自动删除。

### 异步/非阻塞保证

回调钩子（`async_pre_call_hook`、`async_log_success_event`、`async_log_failure_event`）已经由 LiteLLM 在异步上下文中调用。日志本身是同步的但极其快速：

1. **JSON 序列化** — 对小字典执行 `json.dumps()`: < 0.1ms
2. **Logger 输出** — Python 的 logging 是线程安全和缓冲的
3. **文件 I/O** — 由 OS 写缓冲处理（小写入非阻塞）

pre_call_hook 路径中的总开销受 JSON 序列化时间限制（远低于 5ms）。无网络 I/O，无数据库调用，无等待外部服务。

### 配置加载

```python
def load_request_logging_config(config_dir: str | Path = "./config") -> RequestLoggingConfig:
    """从 config.yaml 加载 request_logging 段。"""
    config_path = Path(config_dir) / "config.yaml"
    if not config_path.exists():
        return RequestLoggingConfig(enabled=False)
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    
    section = raw.get("request_logging", {})
    if not section:
        return RequestLoggingConfig(enabled=False)
    
    return RequestLoggingConfig(**section)
```

---

## 正确性属性

*属性是在系统所有有效执行中都应成立的特征或行为 — 本质上是关于系统应该做什么的形式化陈述。属性是人类可读规范与机器可验证正确性保证之间的桥梁。*

### 属性 1: 日志条目结构有效性

*对于任意*输入数据（request、success 或 failure），Request Logger 发出的日志条目应当是有效的单行 JSON 字符串，包含：匹配 UTC ISO-8601 格式（毫秒精度）的 `ts` 字段、非空的 `request_id` 字段、以及值为 `"request"`、`"response_success"` 或 `"response_failure"` 的 `event_type` 字段（与产生它的钩子匹配）。

**验证: 需求 5.1, 5.2, 5.3, 5.5**

### 属性 2: 请求消息忠实捕获

*对于任意*包含非空消息数组的请求数据，发出的 "request" 日志条目应当包含完整的消息数组（受截断配置约束），所有消息的 role 和 content 字段应当被保留。

**验证: 需求 1.1**

### 属性 3: 元数据字段忠实传播

*对于任意*包含 session_id、request_id、target_model、routing_plugin、route_reason 或 route_score 的请求元数据，发出的日志条目应当包含这些字段中的每一个，其值与源元数据值相同。

**验证: 需求 1.2, 2.1, 2.2, 2.3, 8.3**

### 属性 4: 从 standard_logging_object 提取成功响应数据

*对于任意* `kwargs["standard_logging_object"]` 包含 Token 用量（prompt_tokens、completion_tokens、total_tokens）和延迟数据的成功事件，发出的 "response_success" 日志条目应当包含这些精确值（不独立重新计算），以及响应文本内容。

**验证: 需求 3.1, 3.2, 3.3, 3.4**

### 属性 5: 失败事件捕获错误详情

*对于任意*有关联异常或错误的失败事件，发出的 "response_failure" 日志条目应当包含错误消息字符串和异常类型名称，并应当在 standard_logging_object 存在时提取可用数据，在其不存在时设置 `incomplete_data: true`。

**验证: 需求 4.1, 4.2, 4.3, 4.4**

### 属性 6: 消息截断正确性

*对于任意*消息内容字符串和任意配置的 `max_message_length > 0`，如果内容长度超过 `max_message_length`，记录的内容应当恰好是 `max_message_length` 个字符后跟 `" [truncated]"`；如果内容长度在限制内，应当保持不变。

**验证: 需求 6.4**

### 属性 7: 错误隔离 — 无异常传播

*对于任意*在日志条目构建或发出过程中导致内部错误的输入（如不可序列化对象、I/O 错误），Request Logger 钩子应当正常返回而不抛出异常，并应当将错误记录到应用日志。

**验证: 需求 7.3**

### 属性 8: 非修改不变量

*对于任意*传递给 `async_pre_call_hook` 的请求数据字典，该字典在钩子执行前后应当结构相同（深度相等）。Request Logger 绝不修改请求数据、响应数据或路由元数据。

**验证: 需求 7.4, 8.4**

---

## 错误处理

| 场景 | 行为 |
|------|------|
| JSON 序列化失败（不可序列化字段） | 捕获异常，向应用日志记录 warning，跳过该日志条目 |
| 文件 I/O 错误（磁盘满、权限拒绝） | Python logging 模块优雅处理；调用 `handleError()`，不传播 |
| 缺少元数据字段（无 session_id、无 request_id） | session_id 使用 `None`；request_id 缺失时生成 UUID |
| 成功事件中缺少 standard_logging_object | 日志条目中 `usage: null`、`latency_ms: null` |
| 失败事件中缺少 standard_logging_object | 日志条目中 `incomplete_data: true` |
| 配置文件缺失或格式错误 | 默认 `enabled: false` — 不记录日志，不崩溃 |
| Request Logger 实例化失败 | 在 custom_callbacks.py 中捕获；路由插件继续工作，记录 warning |
| 日志文件轮转失败 | TimedRotatingFileHandler 内部处理；旧日志可能累积 |

**错误隔离原则:** 每个钩子方法将整个方法体包裹在 `try/except Exception` 中。except 块向 `logging.getLogger(__name__)`（应用日志，非请求日志）记录，然后优雅返回。

---

## 测试策略

### 基于属性的测试（Hypothesis）

基于属性的测试适合此功能，因为：
- 日志条目构建是纯转换：输入数据 → JSON 字符串
- 通用属性（格式有效性、字段保留、截断逻辑）在所有输入上都成立
- 输入空间很大（任意消息数组、元数据字典、Token 数）

**库:** [Hypothesis](https://hypothesis.readthedocs.io/)（Python PBT 框架）
**最小迭代次数:** 每个属性测试 100 次
**标签格式:** `# Feature: request-logging, Property {N}: {title}`

每个正确性属性映射到一个基于属性的测试：

| 属性 | 测试重点 | 生成器策略 |
|------|---------|-----------|
| P1: 结构有效性 | JSON 解析、时间戳格式、event_type | 随机消息 + 元数据 + 钩子类型 |
| P2: 消息捕获 | 消息数组保留 | 随机消息字典列表 |
| P3: 元数据传播 | 字段一致性 | 包含路由字段的随机元数据字典 |
| P4: SLO 提取 | Token/延迟直通 | 随机 standard_logging_object 字典 |
| P5: 失败详情 | 错误消息/类型捕获 | 随机异常类型和消息 |
| P6: 截断 | 长度 + 后缀行为 | 随机字符串 × 随机 max_length |
| P7: 错误隔离 | 无异常逃逸 | 随机不可序列化/格式错误的输入 |
| P8: 非修改 | 深度相等前后对比 | 随机嵌套数据字典 |

### 单元测试（pytest）

- **禁用模式**: 验证 `enabled=False` 时零处理
- **空消息跳过**: 验证 `data["messages"]` 为空/缺失时不产生条目
- **Logger 命名空间**: 验证使用 `aegis_router.request_log`
- **Handler 配置**: 验证 stdout/file/both 模式正确设置处理器
- **保留配置**: 验证 `backupCount` 匹配 `retention_days`
- **类继承**: 验证继承 `CustomLogger`，不继承 `BaseRouterCallback`
- **插件无关**: 使用 conversation、transaction 和 agent_workbuddy 插件的元数据进行测试

### 集成测试

- **文件输出**: 向临时文件写入条目，读取并解析
- **多回调链**: 同时注册路由插件和请求日志器，发送请求，验证两者都触发
- **性能基准**: 测量 `async_pre_call_hook` 执行时间（< 5ms 目标）
- **真实 standard_logging_object**: 在模拟回调场景中使用 LiteLLM 实际的 SLO 格式
