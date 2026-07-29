# 需求文档

## 简介

AegisRouter 是一个 LLM 路由代理，通过多种路由插件（conversation、transaction、agent_workbuddy）将请求分发到不同的 LLM 提供商。本功能为系统添加全局请求日志能力，记录完整的请求-响应生命周期 — 包括请求体、路由决策和响应内容 — 帮助运维人员验证路由策略是否正确执行。日志系统实现为一个 LiteLLM CustomLogger 回调，独立于路由插件运行，复用 LiteLLM 的 `standard_logging_object` 获取 Token 用量和延迟数据。

## 术语表

- **Request_Logger**: 负责记录请求-响应生命周期数据的 LiteLLM CustomLogger 回调，适用于所有路由插件。
- **Log_Entry**: JSONL 日志文件中的一行，是一个独立的结构化 JSON 对象，记录请求-响应生命周期中的一个事件（请求到达、成功响应或失败响应）。多条 Log_Entry 顺序追加到同一个 `.jsonl` 文件中。
- **Standard_Logging_Object**: LiteLLM 在 success/failure 回调中提供的 `kwargs["standard_logging_object"]` 字典，包含预计算的 Token 用量和延迟数据。
- **Routing_Decision**: 描述请求被路由到哪个模型的元数据，包括路由插件名称、目标模型和选择原因。
- **Log_Store**: 可配置的存储后端，用于持久化 Log_Entry 记录（基于文件的结构化 JSON 日志）。
- **Log_Retention_Policy**: 决定 Log_Entry 记录保留多长时间的配置。
- **AegisRouter**: 位于客户端和多个 LLM 提供商之间的 LLM 路由代理系统。
- **Routing_Plugin**: 实现模型选择逻辑的回调类（conversation、transaction 或 agent_workbuddy）。

## 需求

### 需求 1: 记录请求体

**用户故事:** 作为运维人员，我希望看到每个 LLM 请求的请求体，以便了解通过路由代理发送了哪些 prompt。

#### 验收标准

1. 当请求通过 pre-call 钩子时，Request_Logger 应当从请求数据中捕获完整的消息数组。
2. 当请求包含 session_id 和 request_id 元数据时，Request_Logger 应当在 Log_Entry 中包含 session_id 和 request_id。
3. Request_Logger 应当在基础管道完成 PII 脱敏之后记录请求体。
4. 当请求不包含消息时，Request_Logger 应当跳过该请求的日志记录。

### 需求 2: 记录路由决策

**用户故事:** 作为运维人员，我希望看到每个请求被路由到哪个模型以及原因，以便验证路由策略是否正确执行。

#### 验收标准

1. 当路由决策完成时，Request_Logger 应当在 Log_Entry 中记录目标模型名称。
2. 当路由决策完成时，Request_Logger 应当在 Log_Entry 中记录做出决策的 Routing_Plugin 名称。
3. 当路由元数据包含路由原因或评分时，Request_Logger 应当在 Log_Entry 中包含路由原因和评分。
4. Request_Logger 应当能够捕获所有三种 Routing_Plugin 类型（conversation、transaction、agent_workbuddy）的路由决策，无需插件特定逻辑。

### 需求 3: 记录响应内容

**用户故事:** 作为运维人员，我希望看到 LLM 响应内容，以便验证响应是否正确并将其与路由决策关联。

#### 验收标准

1. 当收到成功的 LLM 响应时，Request_Logger 应当在 Log_Entry 中记录响应文本内容。
2. 当收到成功的 LLM 响应时，Request_Logger 应当从 Standard_Logging_Object 中提取 Token 用量（输入 Token、输出 Token）。
3. 当收到成功的 LLM 响应时，Request_Logger 应当从 Standard_Logging_Object 中提取延迟数据。
4. 当 Standard_Logging_Object 提供数据时，Request_Logger 不应独立计算 Token 用量或延迟。

### 需求 4: 记录失败事件

**用户故事:** 作为运维人员，我希望看到失败的请求及其错误详情，以便诊断 LLM 提供商或路由的问题。

#### 验收标准

1. 当 LLM 请求失败时，Request_Logger 应当记录一条状态为 "error" 的 Log_Entry。
2. 当 LLM 请求失败时，Request_Logger 应当在 Log_Entry 中包含错误消息或异常类型。
3. 当 LLM 请求失败时，Request_Logger 应当从 Standard_Logging_Object 中提取可用数据（如果存在）。
4. 当 LLM 请求失败且 Standard_Logging_Object 不可用时，Request_Logger 应当记录包含部分数据的 Log_Entry，并标记数据不完整。

### 需求 5: 结构化日志格式

**用户故事:** 作为运维人员，我希望日志采用结构化、可查询的格式，以便高效地搜索和过滤请求日志。

#### 验收标准

1. Request_Logger 应当将每条 Log_Entry 输出为单行 JSON 对象。
2. Request_Logger 应当在每条 Log_Entry 中包含 UTC ISO-8601 格式的时间戳（毫秒精度）。
3. Request_Logger 应当在每条 Log_Entry 中包含唯一的 request_id，以便在 pre-call、success 和 failure 事件之间进行关联。
4. Request_Logger 应当使用独立的 Python logger 命名空间（aegis_router.request_log），与应用日志和现有审计日志分离。
5. Request_Logger 应当在每条 Log_Entry 中包含 "event_type" 字段，取值为 "request"、"response_success" 或 "response_failure"。

### 需求 6: 配置

**用户故事:** 作为运维人员，我希望能配置请求日志行为，以便控制日志详细程度、存储位置和保留策略。

#### 验收标准

1. Request_Logger 应当支持通过 config.yaml 中的配置标志启用或禁用请求日志。
2. Request_Logger 应当支持配置日志输出目标（stdout、文件路径或两者）。
3. 当配置了文件路径时，Request_Logger 应当将 Log_Entry 记录写入指定文件路径。
4. Request_Logger 应当支持可配置的最大消息体长度，截断超过限制的消息并追加 "[truncated]" 标记。
5. Request_Logger 应当支持可配置的文件日志保留天数。
6. 当请求日志被禁用时，Request_Logger 不应执行任何日志逻辑，且不应引入额外延迟。

### 需求 7: 性能与无干扰

**用户故事:** 作为运维人员，我希望请求日志对性能的影响最小，不会降低客户端体验到的请求延迟。

#### 验收标准

1. Request_Logger 应当异步执行日志操作，不应阻塞请求-响应管道。
2. Request_Logger 在 pre-call 钩子执行路径上增加的延迟不应超过 5 毫秒。
3. 如果 Request_Logger 在日志记录过程中遇到错误，应当将错误记录到应用日志，且不应将异常传播到请求管道。
4. Request_Logger 应当独立于路由插件运行，不应修改请求数据、响应数据或路由元数据。

### 需求 8: 与现有回调架构集成

**用户故事:** 作为开发者，我希望请求日志器能干净地集成到现有的 LiteLLM 回调系统中，与当前的路由和 PII 管道协同工作，避免代码重复。

#### 验收标准

1. Request_Logger 应当实现为独立的 LiteLLM CustomLogger 回调类，不继承 BaseRouterCallback。
2. Request_Logger 应当通过 config.yaml 在 LiteLLM 回调链中与现有回调一起注册。
3. Request_Logger 应当从路由插件填充的请求元数据字典中读取路由决策信息（target_model、routing_plugin、route_reason）。
4. Request_Logger 不应重复 BaseRouterCallback 中的任何逻辑（合规检测、PII 脱敏、响应还原）。
