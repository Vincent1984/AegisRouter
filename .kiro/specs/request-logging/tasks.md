# Implementation Plan: 请求日志 (Request Logging)

## Overview

将 RequestLoggerCallback 实现为独立的 LiteLLM CustomLogger 回调，记录完整的请求-响应生命周期（请求体、路由决策、响应内容、失败事件）。使用独立 logger 命名空间 `aegis_router.request_log`，通过 `TimedRotatingFileHandler` 实现日志轮转，通过 `litellm.callbacks.append()` 注册到回调链中。

## Tasks

- [x] 1. 创建 RequestLoggingConfig 和配置加载
  - [x] 1.1 实现 RequestLoggingConfig Pydantic 模型和 load_request_logging_config 函数
    - 创建 `aegis_router/observability/request_logger.py` 文件
    - 实现 `RequestLoggingConfig(BaseModel)` 包含字段: enabled, output, file_path, max_message_length, retention_days, log_level
    - 实现 `load_request_logging_config(config_dir)` 函数，从 config.yaml 读取 `request_logging` 段
    - 配置文件缺失或格式错误时返回 `enabled=False` 的默认配置
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [x] 1.2 在 config.yaml 中添加 request_logging 配置段
    - 在 `config/config.yaml` 末尾添加 `request_logging` 段
    - 包含: enabled, output, file_path, max_message_length, retention_days
    - _Requirements: 6.1, 6.2, 6.5_

- [x] 2. 实现 RequestLoggerCallback 核心类
  - [x] 2.1 实现 RequestLoggerCallback 类骨架和 logger 配置
    - 在 `aegis_router/observability/request_logger.py` 中实现 `RequestLoggerCallback(CustomLogger)`
    - 实现 `__init__(self, config: RequestLoggingConfig)` — 初始化 `aegis_router.request_log` logger
    - 实现 `_configure_logger()` — 根据 output 配置设置 StreamHandler 和/或 TimedRotatingFileHandler
    - 使用 `_JsonPassthroughFormatter` 直接输出 JSON 字符串（参照 audit_logger.py 模式）
    - TimedRotatingFileHandler: when="midnight", backupCount=retention_days
    - _Requirements: 5.4, 6.2, 6.3, 6.5, 8.1_

  - [x] 2.2 实现 _LogEntryBuilder 静态方法
    - 实现 `_LogEntryBuilder.build_request_entry(data, config)` — 构建 "request" 事件日志条目
    - 实现 `_LogEntryBuilder.build_success_entry(kwargs, response_obj, start_time, end_time, config)` — 构建 "response_success" 事件
    - 实现 `_LogEntryBuilder.build_failure_entry(kwargs, response_obj, start_time, end_time, config)` — 构建 "response_failure" 事件
    - 每条日志包含公共信封: ts (UTC ISO-8601 毫秒精度), event_type, request_id, session_id
    - request_id 缺失时生成 UUID
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 1.1, 1.2, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4_

  - [x] 2.3 实现消息截断逻辑
    - 在 RequestLoggerCallback 中实现 `_truncate_messages(messages, max_length)` 方法
    - 当 max_message_length > 0 且内容超出时，截断为 max_message_length 字符 + " [truncated]"
    - max_message_length <= 0 时不截断
    - _Requirements: 6.4_

  - [x] 2.4 实现 async_pre_call_hook 方法
    - 当 enabled=False 时直接返回 data，不执行任何逻辑
    - 从 `data["messages"]` 提取消息数组，无消息时跳过
    - 从 `data["metadata"]` 读取路由决策 (target_model, routing_plugin, route_reason, route_score)
    - 调用 `_LogEntryBuilder.build_request_entry()` 构建日志条目
    - 通过 `self._logger.info(json.dumps(entry))` 发出日志
    - 整个方法体包裹在 try/except 中，异常时记录 warning 到应用日志
    - 原样返回 data，绝不修改
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 6.6, 7.1, 7.2, 7.3, 7.4, 8.4_

  - [x] 2.5 实现 async_log_success_event 方法
    - 当 enabled=False 时直接返回
    - 从 `kwargs["standard_logging_object"]` 提取 prompt_tokens, completion_tokens, total_tokens, response_time_ms
    - 从 response_obj 提取响应文本
    - 调用 `_LogEntryBuilder.build_success_entry()` 构建日志条目
    - 不独立计算 Token 用量或延迟
    - 整个方法体包裹在 try/except 中
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.3_

  - [x] 2.6 实现 async_log_failure_event 方法
    - 当 enabled=False 时直接返回
    - 从 kwargs 提取异常信息（error_message, error_type）
    - 从 `kwargs["standard_logging_object"]` 提取可用数据（如果存在）
    - SLO 不可用时设置 `incomplete_data: true`
    - 整个方法体包裹在 try/except 中
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 7.3_

- [x] 3. Checkpoint - 确保核心实现完成
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. 注册回调和集成
  - [x] 4.1 修改 config/custom_callbacks.py 注册 RequestLoggerCallback
    - 导入 `RequestLoggerCallback` 和 `load_request_logging_config`
    - 加载请求日志配置
    - 当 enabled=True 时实例化并 append 到 `litellm.callbacks`
    - 实例化失败时捕获异常，记录 warning，不影响路由插件
    - _Requirements: 8.1, 8.2, 7.3_

  - [x] 4.2 更新 observability/__init__.py 导出新模块
    - 在 `aegis_router/observability/__init__.py` 中添加 RequestLoggerCallback 和 load_request_logging_config 的导出
    - _Requirements: 8.1_

- [x] 5. 属性测试 (Hypothesis)
  - [x]* 5.1 编写属性测试 P1: 日志条目结构有效性
    - **Property 1: 日志条目结构有效性**
    - 验证任意输入产生的日志是有效 JSON、包含 ISO-8601 ts、非空 request_id、正确 event_type
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5**

  - [x]* 5.2 编写属性测试 P2: 请求消息忠实捕获
    - **Property 2: 请求消息忠实捕获**
    - 验证非空消息数组被完整保留（受截断配置约束），role 和 content 字段不丢失
    - **Validates: Requirements 1.1**

  - [x]* 5.3 编写属性测试 P3: 元数据字段忠实传播
    - **Property 3: 元数据字段忠实传播**
    - 验证 session_id, request_id, target_model, routing_plugin, route_reason, route_score 正确传播
    - **Validates: Requirements 1.2, 2.1, 2.2, 2.3, 8.3**

  - [x]* 5.4 编写属性测试 P4: 从 standard_logging_object 提取成功响应数据
    - **Property 4: 从 standard_logging_object 提取成功响应数据**
    - 验证 Token 用量和延迟数据精确传递，不重新计算
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

  - [x]* 5.5 编写属性测试 P5: 失败事件捕获错误详情
    - **Property 5: 失败事件捕获错误详情**
    - 验证错误消息、异常类型被捕获，SLO 存在时提取数据，不存在时设置 incomplete_data
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

  - [x]* 5.6 编写属性测试 P6: 消息截断正确性
    - **Property 6: 消息截断正确性**
    - 验证超限内容截断为 max_message_length 字符 + " [truncated]"，未超限内容不变
    - **Validates: Requirements 6.4**

  - [x]* 5.7 编写属性测试 P7: 错误隔离 — 无异常传播
    - **Property 7: 错误隔离 — 无异常传播**
    - 验证不可序列化对象等异常输入不导致钩子抛出异常
    - **Validates: Requirements 7.3**

  - [x]* 5.8 编写属性测试 P8: 非修改不变量
    - **Property 8: 非修改不变量**
    - 验证 async_pre_call_hook 执行前后 data 字典深度相等
    - **Validates: Requirements 7.4, 8.4**

- [x] 6. 单元测试
  - [x]* 6.1 编写单元测试覆盖核心场景
    - 测试 enabled=False 时零处理
    - 测试空消息跳过
    - 测试 logger 命名空间为 `aegis_router.request_log`
    - 测试 handler 配置（stdout/file/both）
    - 测试 backupCount 匹配 retention_days
    - 测试类继承 CustomLogger，不继承 BaseRouterCallback
    - 测试 conversation、transaction、agent_workbuddy 三种插件的元数据
    - _Requirements: 1.4, 2.4, 5.4, 6.1, 6.6, 8.1, 8.3_

- [x] 7. Final checkpoint - 确保所有测试通过
  - Ensure all tests pass, ask the user if questions arise.

## 8. 回归测试

> 确认 RequestLoggerCallback 集成后不影响主功能（路由、PII 脱敏、Failover）

- [x] 8.1 重新打包镜像（包含 RequestLogger 注册）并启动
- [x] 8.2 事务路由回归
  - 发送 3 个不同 Agent 请求，确认路由到正确模型（与 Phase 8 结果一致）
  - 确认 RequestLogger 不影响 model 选择
- [x] 8.3 对话级路由回归
  - 切换 `routing_plugin: conversation`，重启，发请求确认正常
- [x] 8.4 PII 脱敏回归
  - 发送含中文 PII 的请求，确认响应中 PII 被正确还原
- [x] 8.5 异常隔离验证
  - 确认 RequestLogger 初始化失败时不影响路由（故意传错配置测试）
- [x] 8.6 日志输出验证
  - 确认请求日志文件生成且包含 JSON 格式日志条目
  - 确认日志包含 request_id、model、event_type 字段

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- 测试文件应放在 `tests/test_request_logger.py` 和 `tests/test_request_logger_properties.py`
- 所有属性测试使用 Hypothesis 框架，每个属性最少 100 次迭代

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6"] },
    { "id": 4, "tasks": ["4.1", "4.2"] },
    { "id": 5, "tasks": ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "6.1"] }
  ]
}
```
