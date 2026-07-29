# Requirements Document

## Introduction

Agent-WorkBuddy 是 AegisRouter 的第三个路由插件，专为 WorkBuddy 客户端设计。该插件采用单维度路由策略，仅以 agent 名称作为查表 key，从预计算方案中查找目标模型。系统启动时根据 `agent_workbuddy.yaml` 配置文件预计算路由方案表，运行时纯内存查表完成请求分发。

**与事务级路由的核心差异**：
- WorkBuddy 客户端无法在请求 metadata 中发送 `template` 字段
- 路由 key 从二维 `(template, agent)` 简化为一维 `agent`
- Agent 标识从 `role: "user"` 消息的 `agent` 字段提取（非 metadata）
- 不关心业务流程/模板编排，只看当前请求是哪个 Agent 发出的

## Glossary

- **AegisRouter**: 核心路由系统，负责将 LLM 请求分发到目标模型
- **Agent_WorkBuddy_Plugin**: Agent-WorkBuddy 路由插件，本需求文档描述的主体系统
- **AgentPlanStore**: 单维度内存查找表组件，存储 agent_name → model_name 映射
- **AgentPlanGenerator**: 方案生成器组件，启动时为所有 Agent 预计算最优模型分配
- **CapabilityProfileManager**: 已有的能力评分管理器，负责模型评分和约束过滤
- **ConfigWatcher**: 配置文件监听器，负责检测配置变更并触发热更新
- **Routing_Plan**: 预计算的 agent → model 映射方案
- **Capability_Profile**: 能力 Profile，描述 Agent 对模型能力的要求
- **Override_Model**: 管理员直接指定的模型，跳过自动评分
- **Failover_Chain**: 故障转移链，当主模型不可用时的备选模型列表
- **Fallback_Model**: 兜底模型，当无法确定路由目标时使用的默认模型

---

## Requirements

### Requirement 1: Agent 标识提取

**User Story:** 作为 WorkBuddy 客户端，我希望路由器能正确识别哪个 Agent 在发起请求，以便每个 Agent 被路由到最适合的模型。

#### Acceptance Criteria

1. 从最后一条 user 消息提取 — 遍历请求 messages，从最后一条 `role: "user"` 的消息中读取 `agent` 字段作为路由 key
2. 多条 user 消息取最后一条 — 当存在多条 `role: "user"` 消息时，使用最后一条的 `agent` 字段
3. metadata.agent 备选 — 若 user 消息中无 `agent` 字段，尝试从 `metadata.agent` 读取（兼容备选）
4. 缺失 agent 降级 — 若均无 agent 字段，使用 fallback 模型并发出 NO_AGENT 警告
5. Agent 名称校验 — 仅接受 `[a-zA-Z0-9_-]` 字符，防止注入
6. 非法名称降级 — 若 agent 名称包含非法字符，使用 fallback 模型并发出 INVALID_AGENT 警告

### Requirement 2: 方案预计算

**User Story:** 作为系统运维人员，我希望路由方案在启动时预计算完成，以便请求路由延迟最小化。

#### Acceptance Criteria

1. 启动时生成 — 系统启动时加载 `agent_workbuddy.yaml`、`models.yaml`、`capability_profiles.yaml`，为所有 Agent 一次性计算分配方案
2. Override 优先 — 当 Agent 定义了 `override_model` 时，直接使用指定模型，跳过评分
3. Profile 评分选模型 — 未定义 override_model 的 Agent，使用 CapabilityProfileManager 评分逻辑选择最优模型
4. 无候选降级 — 若无模型满足 Profile 硬约束，使用 fallback 模型并记录 NO_CANDIDATE 警告
5. Profile 不存在降级 — 若引用的 capability_profile 不存在，降级为 `medium` Profile 并记录 PROFILE_NOT_FOUND 警告
6. 重复 Agent 处理 — 配置文件中存在重复 agent 名称时，后定义的覆盖前面的，并记录 DUPLICATE_AGENT 警告
7. Override 校验 — `override_model` 值必须在 `models.yaml` 中已定义

### Requirement 3: 请求路由分发

**User Story:** 作为 WorkBuddy 客户端，我希望请求能基于 agent 标识路由到正确模型，以便每个 Agent 使用最适合其能力需求的模型。

#### Acceptance Criteria

1. 查表分发 — 有效 agent 且存在于方案表中时，直接设置 `data["model"]` 为预计算值
2. 未知 Agent 降级 — agent 不在方案表中时，使用 fallback 模型并发出 UNKNOWN_AGENT 警告
3. 响应 metadata — 路由完成后响应中注入 `aegis_metadata`，包含 agent 名、分配模型、路由插件标识 `agent_workbuddy`、警告列表
4. 分发延迟 — 纯内存查表，100 个 Agent 规模下分发延迟 < 0.1ms

### Requirement 4: 配置变更生效

**User Story:** 作为系统运维人员，我希望修改配置后能使新方案生效，以便更新 Agent 路由分配。

#### Acceptance Criteria

1. 重启生效 — 修改 `agent_workbuddy.yaml`、`capability_profiles.yaml` 或 `models.yaml` 后，重启 litellm 进程触发方案重新计算
2. 原子替换 — 方案重算后整体替换 AgentPlanStore 引用，无半成品状态
3. 语法错误保护 — YAML 语法错误时拒绝加载，进程启动失败并输出明确错误信息
4. ConfigWatcher 可选增强 — 在宿主机（非 Docker overlay fs）环境下，ConfigWatcher 可检测文件变更自动触发重算，但不作为核心依赖

### Requirement 5: 故障转移

**User Story:** 作为系统运维人员，我希望有故障转移机制，以便即使主模型不可用请求也能成功处理。

#### Acceptance Criteria

1. Agent 级重试 — LLM 调用失败时，沿 failover 链选下一个模型重试
2. 仅影响当次请求 — failover 不修改全局 AgentPlanStore
3. 可禁用 — 配置中关闭 failover 时不尝试替代模型

### Requirement 6: 插件注册与激活

**User Story:** 作为系统运维人员，我希望通过配置激活 Agent-WorkBuddy 插件，以便为 WorkBuddy 客户端选择合适的路由策略。

#### Acceptance Criteria

1. 配置激活 — `config.yaml` 中 `routing_plugin: agent_workbuddy` 时加载并初始化 AgentWorkbuddyCallback
2. 插件互斥 — agent_workbuddy 插件激活时，conversation 和 transaction 插件不参与路由
3. 配置缺失容错 — `agent_workbuddy.yaml` 不存在时正常启动，方案表为空，所有请求走 fallback
4. 插件切换安全 — 从 agent_workbuddy 切换到其他插件时完全卸载，不影响其他插件功能

### Requirement 7: AgentPlanStore 数据完整性

**User Story:** 作为开发者，我希望方案表维护数据完整性，以便路由结果一致且可预测。

#### Acceptance Criteria

1. O(1) 查表 — 按 agent 名称 O(1) 哈希查找
2. 确定性 — 相同配置输入永远生成相同方案
3. 并发一致性 — 方案表未被替换期间，同一 agent 的并发查询返回相同模型
4. 唯一键 — 方案表中 agent 名称不重复

### Requirement 8: 可观测性

**User Story:** 作为系统运维人员，我希望对路由决策和方案生成有可见性，以便监控和排查系统问题。

#### Acceptance Criteria

1. 启动日志 — 方案生成完成后输出汇总表（agent、模型、profile、评分）
2. 热更新日志 — 配置变更触发重算后输出新旧方案对比
3. 降级告警 — 走 fallback 或降级路径时记录告警日志，包含原因码（NO_AGENT、UNKNOWN_AGENT、PROFILE_NOT_FOUND、NO_CANDIDATE、INVALID_AGENT）

---

## 非功能性需求 (Non-Functional Requirements)

### NFR-1: 性能

- **NFR-1.1**: 请求分发延迟 ≤ 0.1ms — 纯内存 HashMap 查表
- **NFR-1.2**: 方案生成延迟 ≤ 2ms（20 个 Agent）— 启动/配置变更时计算
- **NFR-1.3**: 支持任意并发 — 分发逻辑无状态无锁
- **NFR-1.4**: 内存占用极低 — 全部方案 < 5KB

### NFR-2: 可靠性

- **NFR-2.1**: 无 Redis 依赖 — 方案在内存中持有
- **NFR-2.2**: 配置热更新安全 — 方案原子替换，不出现半成品
- **NFR-2.3**: 插件切换安全 — 切换时进行中请求不中断

### NFR-3: 可维护性

- **NFR-3.1**: 插件接口统一 — 继承 BaseRouterCallback 基类
- **NFR-3.2**: 共用评分引擎 — 复用 CapabilityProfileManager 和 models.yaml
- **NFR-3.3**: 共用 ConfigWatcher — 新增配置文件自动纳入监听
- **NFR-3.4**: 独立可测试 — AgentPlanGenerator 为纯函数，无外部依赖

### NFR-4: 兼容性

- **NFR-4.1**: API 兼容 — 不改变 OpenAI SDK 兼容格式
- **NFR-4.2**: 模型池兼容 — 复用 config.yaml 和 models.yaml
- **NFR-4.3**: Failover 链兼容 — 复用现有 failover 配置
- **NFR-4.4**: 向后兼容 — 不影响 conversation 和 transaction 插件的正常工作

---

## 约束条件

- Agent 标识从消息体的 `agent` 字段提取，不从 metadata.transaction 读取
- Profile 评分复用 CapabilityProfileManager，不引入新框架
- 方案在内存中持有，不依赖 Redis
- 插件切换通过配置控制，不需要重新部署
- 与事务级路由共享 models.yaml 和 capability_profiles.yaml，独立配置文件为 agent_workbuddy.yaml

---

## 配置文件清单与职责

| 文件 | 用途 | 谁来配 | 不配的默认行为 |
|------|------|--------|---------------|
| `config/config.yaml` | 模型连接信息 + `routing_plugin` 字段 | 用户必须配 | 无法连接任何模型 |
| `config/capability_profiles.yaml` | Profile 定义（评分权重、约束） | 系统内置，用户可选改 | 使用内置默认 Profile |
| `config/agent_workbuddy.yaml` | Agent 列表（agent → Profile 映射） | 用户按业务配 | 所有请求走 fallback 模型 |
| `config/models.yaml` | 模型能力参数 | 用户必须配 | 无法打分，所有 Agent 走 fallback |
| `config/route_config.yaml` | Failover 链 | 复用现有配置 | 无 failover，失败直接报错 |

---

## 验收标准摘要

1. 配置 `routing_plugin: agent_workbuddy` 后，WorkBuddy 路由生效，其他插件不生效
2. 系统启动时自动为所有 Agent 生成分配方案，日志可见详情
3. WorkBuddy 请求中 user 消息携带 `agent` 字段 → 直接按方案分发，无打分开销
4. Agent 代码无任何修改
5. 修改 models.yaml/profiles/agent_workbuddy.yaml 后方案自动重算
6. Agent LLM 调用失败时自动 failover，不影响全局方案
7. 切换回 `routing_plugin: conversation` 或 `transaction` 后原有功能完全恢复
8. PII 脱敏、合规检测在所有插件下均正常工作
9. 分发延迟 < 0.1ms
10. 缺失/非法/未知 agent 均正确降级到 fallback 模型
