# Implementation Plan: Agent-WorkBuddy 路由插件

## Overview

实现 AegisRouter 的第三个路由插件 Agent-WorkBuddy，采用单维度路由策略（agent_name → model_name），复用现有 CapabilityProfileManager 评分逻辑。启动时预计算方案表，运行时纯内存查表完成请求分发。

## Tasks

## Phase 1: 核心数据层

- [x] 1. 实现 AgentPlanStore（单维度内存查找表）
  - [x] 1.1 创建 `aegis_router/router/agent_plan_store.py`
    - 实现 `AgentPlanStore` 类，内部使用 `dict[str, str]` 存储
    - 实现 `set_model(agent: str, model: str)` — 设置 agent → model 映射
    - 实现 `get_model(agent: str) -> Optional[str]` — O(1) 哈希查找
    - 实现 `get_all_plans() -> dict[str, str]` — 返回完整映射（日志/调试用）
    - 实现 `__len__`、`__contains__` 方法
    - _需求: FR-7.1, FR-7.3, FR-7.4_

- [x] 2. 实现 AgentPlanGenerator（方案生成器）
  - [x] 2.1 创建 `aegis_router/router/agent_plan_generator.py`
    - 定义 `AgentWorkbuddyDef` dataclass（name, capability_profile, override_model, description）
    - 实现 `__init__` 接收 profile_manager, models 列表, fallback_model
    - 实现 `generate_all(agents: list[AgentWorkbuddyDef]) -> AgentPlanStore`
    - 优先级：override_model → Profile 评分选模型 → fallback
    - 处理 Profile 不存在：降级为 `medium` + PROFILE_NOT_FOUND 警告
    - 处理无候选模型：使用 fallback + NO_CANDIDATE 警告
    - 处理重复 agent 名称：后定义覆盖前面 + DUPLICATE_AGENT 警告
    - 校验 override_model 在 models.yaml 中存在
    - 生成完成后输出汇总表日志（agent、模型、profile、评分）
    - _需求: FR-2.1, FR-2.2, FR-2.3, FR-2.4, FR-2.5, FR-2.6, FR-2.7, FR-8.1_

### Phase 1 验证检查点
- [x] V1-1: AgentPlanStore 基本 set/get 操作正确
- [x] V1-2: `get_model` 对未知 agent 返回 None
- [x] V1-3: `generate_all()` 正确为各 Agent 分配模型
- [x] V1-4: override_model 优先于 Profile 评分
- [x] V1-5: 相同配置多次调用 `generate_all()` → 结果完全相同（确定性）
- [x] V1-6: 重复 agent 名称最后定义胜出


## Phase 2: 路由插件主类

- [x] 3. 实现 AgentWorkbuddyCallback（路由插件主入口）
  - [x] 3.1 创建 `aegis_router/callbacks/agent_workbuddy_router.py`
    - 继承 `BaseRouterCallback`，实现 `_execute_routing()` 方法
    - 实现 `_extract_agent(data: dict) -> Optional[str]` — 从最后一条 `role: "user"` 消息的 `agent` 字段提取
    - 实现 metadata.agent 备选逻辑
    - 实现 agent 名称校验（正则 `[a-zA-Z0-9_-]+`）
    - 实现方案表查找 + `data["model"]` 赋值
    - 实现缺失/非法/未知 agent 的 fallback 路由
    - 实现 aegis_metadata 注入（agent、assigned_model、routing_plugin="agent_workbuddy"、warnings）
    - 实现 failover 链支持（LLM 错误时尝试下一个模型，仅影响当次请求）
    - 实现 failover_enabled 开关
    - 构造参数：plan_store, fallback_model, failover_chains, failover_enabled, pool, degradation_manager, config_dir
    - _需求: FR-1.1~FR-1.6, FR-3.1~FR-3.3, FR-5.1~FR-5.3, FR-8.3_

### Phase 2 验证检查点
- [x] V2-1: 单条 user 消息正确提取 agent 字段
- [x] V2-2: 多条 user 消息取最后一条的 agent
- [x] V2-3: user 消息无 agent 时使用 metadata.agent 备选
- [x] V2-4: 均无 agent → fallback + NO_AGENT 警告
- [x] V2-5: 非法 agent 名称 → fallback + INVALID_AGENT 警告
- [x] V2-6: 未知 agent → fallback + UNKNOWN_AGENT 警告
- [x] V2-7: 已知 agent → 正确路由到方案表中的模型
- [x] V2-8: aegis_metadata 正确填充
- [x] V2-9: failover 链在 LLM 错误时触发，不修改全局方案
- [x] V2-10: failover_enabled=False 时不尝试替代模型


## Phase 3: 插件注册与配置

- [x] 4. 插件注册
  - [x] 4.1 修改 `aegis_router/callbacks/plugin_loader.py`
    - 在 `SUPPORTED_PLUGINS` 中新增 `"agent_workbuddy"` 条目，指向 `AgentWorkbuddyCallback`
    - 实现 `_initialize_agent_workbuddy_plugin()` 初始化函数（模式与 `_initialize_transaction_plugin()` 一致）
    - 在 `load_routing_plugin()` 中新增 agent_workbuddy 分支
    - 实现启动时方案表日志输出函数
    - _需求: FR-6.1, FR-6.2, FR-6.3, FR-6.4_

- [x] 5. 创建配置文件
  - [x] 5.1 创建 `config/agent_workbuddy.yaml`
    - 顶层字段 `agents`，包含 Agent 列表
    - 示例 Agent：intent_classifier, document_parser, reasoning_engine, code_assistant, general_assistant, heavy_analyst
    - 每个条目包含：name, capability_profile, 可选 override_model, 可选 description
    - 添加详细注释说明各字段含义
    - _需求: FR-2.1_

  - [x] 5.2 实现 YAML 加载函数
    - 创建 `load_agent_workbuddy_config(config_path: Path) -> list[AgentWorkbuddyDef]`
    - 解析 YAML 并转换为 `AgentWorkbuddyDef` dataclass 列表
    - 文件不存在：返回空列表 + 日志警告
    - YAML 语法错误：抛出明确错误信息
    - _需求: FR-2.1, FR-6.3_

### Phase 3 验证检查点
- [x] V3-1: `SUPPORTED_PLUGINS` 包含 "agent_workbuddy" 条目
- [x] V3-2: `routing_plugin: agent_workbuddy` 配置下正确加载 AgentWorkbuddyCallback
- [x] V3-3: agent_workbuddy.yaml 不存在时正常启动，方案表为空
- [x] V3-4: 插件互斥 — agent_workbuddy 激活时 conversation/transaction 不参与路由


## Phase 4: ConfigWatcher 集成（可选增强）

- [x] 6. 扩展 ConfigWatcher（宿主机环境可选，Docker 环境不依赖）
  - [x] 6.1 修改 `aegis_router/router/config_watcher.py`
    - 将 `agent_workbuddy.yaml` 加入监听文件列表
    - 创建 `AGENT_WORKBUDDY_PLAN_TRIGGER_FILES` 集合（models.yaml, capability_profiles.yaml, agent_workbuddy.yaml）
    - 新增 `on_agent_workbuddy_plan_updated` 回调参数
    - 实现 `_do_agent_workbuddy_plan_reload()` 方法（重载配置 → 重算方案 → 原子替换）
    - 重算后输出新旧方案对比日志
    - **注意**：Docker overlay fs 环境下 inotify 不触发，此功能仅在宿主机环境生效
    - _需求: 4.2, 4.4_

### Phase 4 验证检查点
- [x] V4-1: 重启进程后配置变更生效，方案正确重算
- [x] V4-2: YAML 语法错误时启动失败，输出明确错误
- [x] V4-3: 方案原子替换，无半成品状态
- [x] V4-4: （宿主机环境）ConfigWatcher 检测文件变更触发重算


## Phase 5: 测试

### 5.1 数据层单元测试

- [x] 7. AgentPlanStore 单元测试
  - TC-STORE-001: 基本 set/get 操作
  - TC-STORE-002: get_model 对未知 agent 返回 None
  - TC-STORE-003: `__contains__` 正确判断
  - TC-STORE-004: `__len__` 返回正确数量
  - TC-STORE-005: get_all_plans 返回完整映射
  - TC-STORE-006: 重复 set 同一 key 覆盖旧值，长度不变

### 5.2 方案生成器单元测试

- [x] 8. AgentPlanGenerator 单元测试
  - TC-GEN-001: 正常评分路径 — Agent 无 override 时选最优模型
  - TC-GEN-002: Override 路径 — override_model 跳过评分直接分配
  - TC-GEN-003: Profile 不存在 → 降级为 medium
  - TC-GEN-004: 无候选模型 → 使用 fallback_model
  - TC-GEN-005: 重复 agent 名称 → 后定义胜出
  - TC-GEN-006: override_model 不在模型列表中 → 触发警告
  - TC-GEN-007: 相同输入多次调用 → 结果完全一致（确定性）

### 5.3 路由回调单元测试

- [x] 9. AgentWorkbuddyCallback 单元测试
  - TC-WB-001: 单条 user 消息正确提取 agent
  - TC-WB-002: 多条 user 消息取最后一条的 agent
  - TC-WB-003: metadata.agent 备选提取
  - TC-WB-004: 缺失 agent → fallback + NO_AGENT
  - TC-WB-005: 非法 agent 名称 → fallback + INVALID_AGENT
  - TC-WB-006: 未知 agent → fallback + UNKNOWN_AGENT
  - TC-WB-007: 已知 agent → 正确分配模型
  - TC-WB-008: aegis_metadata 正确填充
  - TC-WB-009: failover 链在 LLM 错误时触发
  - TC-WB-010: failover 不修改全局方案
  - TC-WB-011: failover_enabled=False 时不重试

### 5.4 插件注册测试

- [x] 10. plugin_loader 单元测试
  - TC-PLUGIN-001: SUPPORTED_PLUGINS 包含 "agent_workbuddy"
  - TC-PLUGIN-002: `_initialize_agent_workbuddy_plugin()` 创建有效实例
  - TC-PLUGIN-003: `routing_plugin: agent_workbuddy` 正确加载
  - TC-PLUGIN-004: agent_workbuddy.yaml 缺失时方案表为空
  - TC-PLUGIN-005: 插件互斥验证

### 5.5 配置变更测试

- [x] 11. 配置变更单元测试
  - TC-CONFIG-001: 重启后加载新 agent_workbuddy.yaml → 方案正确重算
  - TC-CONFIG-002: 重启后加载新 capability_profiles.yaml → 方案正确重算
  - TC-CONFIG-003: 重启后加载新 models.yaml → 方案正确重算
  - TC-CONFIG-004: YAML 语法错误 → 启动失败，明确错误信息
  - TC-CONFIG-005: 原子替换验证
  - TC-CONFIG-006: （可选）宿主机环境 ConfigWatcher 触发重算

### 5.6 性能基准测试

- [x] 12. 性能测试
  - TC-PERF-WB-001: 方案生成延迟 < 2ms（20 个 Agent）
  - TC-PERF-WB-002: 请求分发延迟 < 0.1ms（HashMap lookup）
  - TC-PERF-WB-003: 方案内存占用 < 5KB
  - TC-PERF-WB-004: 1000 QPS 并发下分发无锁竞争、零错误

### 5.7 端到端集成测试

- [x] 13. 端到端测试（Mock LLM）
  - TC-E2E-WB-001: 完整启动 → 方案生成 → 请求分发 → 验证模型正确
  - TC-E2E-WB-002: WorkBuddy 请求格式（user 消息含 agent 字段）→ 正确路由
  - TC-E2E-WB-003: 响应包含正确 aegis_metadata
  - TC-E2E-WB-004: 配置变更 → 方案重算 → 新请求使用新方案
  - TC-E2E-WB-005: 插件切换 transaction ↔ agent_workbuddy 无副作用
  - TC-E2E-WB-006: Failover 场景 — 主模型失败 → 自动切换，全局方案不变
  - TC-E2E-WB-007: PII 脱敏 + agent_workbuddy 路由同时工作

### Phase 5 验证检查点
- [x] V5-1: 全部单元测试通过
- [x] V5-2: 全部集成测试通过
- [x] V5-3: 性能指标满足 NFR 要求


## Phase 6: 真实环境验收（Real LLM Integration）

> 前置条件：AegisRouter 完整启动，配置 `routing_plugin: agent_workbuddy`，接入真实 LLM API Key。

- [x] 14. 基础路由验证
  - TC-REAL-WB-001: intent_classifier (lightweight) → 路由到正确模型
  - TC-REAL-WB-002: reasoning_engine (strong_reasoning) → 路由到正确模型
  - TC-REAL-WB-003: heavy_analyst (override=gpt-5.6-sol) → 直接使用指定模型
  - TC-REAL-WB-004: 不同 Agent 路由到不同模型确认

- [x] 15. 异常与降级验证
  - TC-REAL-WB-005: 无 agent 字段请求 → fallback 模型
  - TC-REAL-WB-006: 未知 agent → fallback + 警告日志
  - TC-REAL-WB-007: Failover 场景验证

- [x] 16. PII + 路由联合验证
  - TC-REAL-WB-008: 中文 PII + agent_workbuddy 路由 → 脱敏 → 正确路由 → 还原
  - TC-REAL-WB-009: 英文 PII + agent_workbuddy 路由 → 正常处理

- [-] 17. 插件切换验证
  - TC-REAL-WB-010: agent_workbuddy → conversation 切换后对话级路由恢复
  - TC-REAL-WB-011: agent_workbuddy → transaction 切换后事务级路由恢复

### Phase 6 验证检查点
- [x] V6-1: 各 Agent 路由到正确的真实 LLM，返回有效响应
- [x] V6-2: override_model 在真实环境下生效
- [x] V6-3: PII 脱敏 + WorkBuddy 路由 + 响应还原完整工作
- [-] V6-4: Failover 在真实环境下正常触发
- [-] V6-5: 插件切换双向生效


## Phase 7: 文档

- [-] 18. 更新 README.md
  - 新增 "Agent-WorkBuddy 路由" 章节
  - 配置示例 + 使用说明
  - 与事务级路由的对比说明
  - WorkBuddy 客户端请求格式示例

### Phase 7 验证检查点
- [-] V7-1: README 包含 Agent-WorkBuddy 路由配置和使用说明
- [x] V7-2: 新用户按 README 配置可成功启动 agent_workbuddy 路由

## Notes

- 每个 Phase 结尾有验证检查点，确保增量验证
- 实现语言为 Python，与现有代码库一致
- 所有新文件遵循现有项目模式（routing_plan_store.py、template_plan_generator.py、transaction_router.py）
- AgentPlanStore 比 RoutingPlanStore 更简单（单键 vs 二元组键）
- Phase 6 真实环境验收需要接入实际 LLM API Key

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["3.1"] },
    { "id": 2, "tasks": ["4.1", "5.1", "5.2"] },
    { "id": 3, "tasks": ["6.1"] },
    { "id": 4, "tasks": ["7", "8", "9", "10", "11"] },
    { "id": 5, "tasks": ["12", "13"] },
    { "id": 6, "tasks": ["14", "15", "16", "17"] },
    { "id": 7, "tasks": ["18"] }
  ]
}
```
