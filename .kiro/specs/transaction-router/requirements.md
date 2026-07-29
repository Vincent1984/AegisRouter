# AegisRouter 事务级路由插件 — 需求规格

## 产品概述

事务级路由插件是 AegisRouter 的第二阶段路由策略，面向多 Agent/Skill 协作的业务流程场景。管理员预先定义业务流程模板（每个 Agent 的能力需求），系统在启动时或配置变更时一次性为所有模板中的所有 Agent 计算模型分配方案，之后请求直接按方案分发——零计算开销。

**核心理念**：
- 模型分配绑定在"业务流程模板 + Agent 名"上
- Supervisor 编排时注入路由上下文（template + agent），Agent 本身零修改
- 同一 Agent 在不同流程模板下可使用不同模型
- 与第一阶段对话级路由插件互斥可选

---

## 功能性需求 (Functional Requirements)

### FR-1: 插件化路由架构

- **FR-1.1**: 路由策略插件化 — 系统支持多种路由策略插件（对话级、事务级等），通过配置字段 `routing_plugin` 切换，同一时刻只有一个插件生效
- **FR-1.2**: 插件互斥 — 对话级路由插件和事务级路由插件不能同时工作，切换时旧插件完全卸载
- **FR-1.3**: 公共管道不受影响 — 无论选用哪个路由插件，PII 脱敏、合规检测、流式还原、Failover 等公共管道始终执行
- **FR-1.4**: 插件加载器 — 系统启动时根据配置自动加载对应的路由插件实例
- **FR-1.5**: 未来可扩展 — 架构设计允许后续添加更多路由插件，无需修改核心管道代码

### FR-2: 业务流程模板定义

- **FR-2.1**: 模板 YAML 配置 — 管理员通过 `config/transaction_templates.yaml` 声明业务流程模板
- **FR-2.2**: 模板内容 — 每个模板包含：名称、描述、Agent 列表（agent_name + capability_profile）
- **FR-2.3**: Agent 声明 — 每个 Agent 条目声明其名称标识和所需的能力 Profile
- **FR-2.4**: 模型覆盖 — 管理员可为特定 Agent 直接指定模型（`override_model`），跳过 Profile 自动选择，优先级最高
- **FR-2.4**: 模板配置变更生效 — 修改模板文件后需重启 litellm 进程使新配置生效（Docker 环境限制，inotify 不触发）
- **FR-2.6**: 模板校验 — 加载时校验模板引用的 Profile 是否存在，不存在则记录警告并降级为 `medium` Profile

### FR-3: 能力 Profile 体系

- **FR-3.1**: Profile 定义 — 每个 Profile 包含：评分权重（各维度的权重分配）、最低能力门槛、成本硬约束、上下文长度约束、偏好模型列表
- **FR-3.2**: 预定义 Profile — 系统内置至少 6 种 Profile：`lightweight`、`medium`、`strong_reasoning`、`code_specialist`、`long_context`、`heavy`
- **FR-3.3**: Profile 评分算法 — 复用第一阶段的 ModelScorer 归一化算法，权重由 Profile 动态指定
- **FR-3.4**: 硬约束过滤 — 不满足 Profile 硬约束的模型直接排除
- **FR-3.5**: 偏好模型加权 — Profile 可声明 `prefer_models` 列表，候选中命中偏好的模型优先选中
- **FR-3.6**: Profile 配置变更生效 — 修改后需重启 litellm 进程使变更生效

### FR-4: 模板级分配方案生成

- **FR-4.1**: 启动时生成 — 系统启动时为所有模板中的所有 Agent 一次性计算分配方案
- **FR-4.2**: 配置变更时重算 — 模板/Profile/models.yaml 任一变更后，重启 litellm 进程触发方案重新计算
- **FR-4.3**: Profile 驱动选模型 — 每个 Agent 根据其 capability_profile，用 Profile 权重对所有模型打分，选最优
- **FR-4.4**: 覆盖优先 — `override_model` 直接生效，不走 Profile 评分
- **FR-4.5**: 方案全局有效 — 同一模板同一 Agent 的分配方案对所有请求生效，不区分用户/session
- **FR-4.6**: 方案在内存持有 — 配置不变则永不过期
- **FR-4.7**: 无候选降级 — 若无模型满足硬约束，使用 fallback 模型并记录 NO_CANDIDATE 警告

### FR-5: 路由上下文注入与分发

- **FR-5.1**: Supervisor 注入 — Supervisor 编排时将 `template` + `agent` 注入到请求 metadata 中，Agent 本身不感知
- **FR-5.2**: Agent 零修改 — Agent 正常调用 LLM，不需要知道路由信息，不需要改代码
- **FR-5.3**: 查表分发 — AegisRouter 从 metadata 读取 `(template, agent)`，查内存方案表，直接分发到对应模型
- **FR-5.4**: 无需打分 — 分发时不执行任何评分计算
- **FR-5.5**: 未知模板处理 — 引用不存在的模板时返回 HTTP 400
- **FR-5.6**: 未知 Agent 处理 — Agent 不在模板定义中时使用 fallback 模型，记录 UNKNOWN_AGENT 警告
- **FR-5.7**: 非事务请求 — 未携带 transaction metadata 的请求使用 fallback 模型
- **FR-5.8**: 分发延迟极低 — 纯内存查表，无网络 IO

### FR-6: 步骤 Failover

- **FR-6.1**: Agent 级重试 — LLM 调用失败时，使用 failover 链中下一个模型重试
- **FR-6.2**: 复用 Failover 链 — 复用第一阶段的 failover 链配置
- **FR-6.3**: 不影响全局方案 — failover 只影响当次请求，不修改全局分配方案
- **FR-6.4**: 不回退流程 — Agent 失败只重试该 Agent，不影响其他 Agent

### FR-7: 客户端协议

- **FR-7.1**: 向后兼容 — 通过 metadata 扩展，不改变 `v1/chat/completions` 核心格式
- **FR-7.2**: 请求协议 — `metadata.transaction` 包含 `template`（模板名）和 `agent`（Agent 标识），均由 Supervisor 注入
- **FR-7.3**: 响应扩展 — `aegis_metadata` 返回：模板名、Agent 名、分配的模型、告警列表
- **FR-7.4**: Agent 无感 — Agent 不需要处理任何路由相关字段
- **FR-7.5**: 插件切换兼容 — 切换为对话级路由时 transaction metadata 被忽略

### FR-8: 审计与可观测性

- **FR-8.1**: 方案生成日志 — 记录启动/重算时各模板各 Agent 的分配结果
- **FR-8.2**: 分发日志 — 每次分发记录：模板名、Agent 名、分配模型、分发原因
- **FR-8.3**: 配置版本追踪 — 记录配置变更导致方案重算的时间戳和变更摘要
- **FR-8.4**: 告警事件 — NO_CANDIDATE、UNKNOWN_AGENT、AGENT_FAILOVER、PROFILE_NOT_FOUND

### FR-9: 管理员可见性

- **FR-9.1**: 方案查看 — 提供日志/接口输出当前所有模板的分配方案
- **FR-9.2**: 变更通知 — 配置热更新导致方案变化时输出新旧对比
- **FR-9.3**: 同一 Agent 跨模板对比 — 可查看同一 Agent 在不同模板下分配的不同模型


---

## 非功能性需求 (Non-Functional Requirements)

### NFR-1: 性能

- **NFR-1.1**: 请求分发延迟 ≤ 0.1ms — 纯内存 HashMap 查表
- **NFR-1.2**: 方案生成延迟 ≤ 5ms — 启动/配置变更时计算，不影响请求处理
- **NFR-1.3**: 支持任意并发 — 分发逻辑无状态无锁
- **NFR-1.4**: 内存占用极低 — 全部方案 < 10KB

### NFR-2: 可靠性

- **NFR-2.1**: 无 Redis 依赖 — 方案在内存中持有
- **NFR-2.2**: 配置热更新安全 — 方案原子替换，不出现半成品
- **NFR-2.3**: 插件切换安全 — 切换时进行中请求不中断

### NFR-3: 可维护性

- **NFR-3.1**: 插件接口统一 — 所有路由插件继承相同基类
- **NFR-3.2**: 共用评分引擎 — 复用 ModelScorer 和 models.yaml
- **NFR-3.3**: 共用 ConfigWatcher — 新增配置文件自动纳入监听
- **NFR-3.4**: 独立可测试 — 方案生成器为纯函数，无外部依赖

### NFR-4: 兼容性

- **NFR-4.1**: API 兼容 — 不改变 OpenAI SDK 兼容格式
- **NFR-4.2**: 模型池兼容 — 复用 config.yaml 和 models.yaml
- **NFR-4.3**: Failover 链兼容 — 复用第一阶段 failover 配置
- **NFR-4.4**: Agent 零侵入 — 不要求 Agent 修改任何代码

---

## 约束条件

- 路由上下文（template + agent）由 Supervisor 注入，Agent 无感知
- Profile 评分复用 ModelScorer，不引入新框架
- 方案在内存中持有，不依赖 Redis
- 插件切换通过配置控制，不需要重新部署

---

## 配置文件清单与职责

| 文件 | 用途 | 谁来配 | 不配的默认行为 |
|------|------|--------|---------------|
| `config/config.yaml` | 模型连接信息（URL、API Key）+ `routing_plugin` 字段 | 用户必须配 | 无法连接任何模型 |
| `config/capability_profiles.yaml` | Profile 定义（评分权重、约束） | 系统内置，用户可选改 | 使用内置 6 种默认 Profile |
| `config/transaction_templates.yaml` | 业务流程模板（Agent → Profile 映射） | 用户按业务配（提供示例参考） | 所有请求走 fallback 模型 |
| `config/models.yaml` | 模型能力参数（benchmark、上下文、价格） | 用户必须配（提供预填模板） | 无法打分，所有 Agent 走 fallback |
| `config/route_config.yaml` | Failover 链 | 复用第一阶段配置 | 无 failover，失败直接报错 |

### 配置默认行为规则

- **FR-CFG-1**: `capability_profiles.yaml` 不存在或为空时，系统使用内置默认 Profile（lightweight、medium、strong_reasoning、code_specialist、long_context、heavy）
- **FR-CFG-2**: `transaction_templates.yaml` 不存在或为空时，系统正常启动但无预计算方案，所有请求使用 fallback 模型
- **FR-CFG-3**: `models.yaml` 不存在或为空时，系统无法执行 Profile 打分，所有 Agent 使用 fallback 模型，并记录 NO_MODELS 警告
- **FR-CFG-4**: 系统提供 `models.yaml` 预填模板，包含常见模型（OpenAI、DeepSeek、Gemini、Claude、本地模型）的参数，用户按需保留/删除
- **FR-CFG-5**: 系统提供 `transaction_templates.yaml` 示例文件，包含 3-4 个常见业务流程模板作为参考

---

## 验收标准摘要

1. 配置 `routing_plugin: transaction` 后，事务级路由生效，对话级不生效
2. 系统启动时自动为所有模板生成分配方案，日志可见详情
3. Supervisor 注入 `{template, agent}` 的请求，直接按方案分发，无打分开销
4. 同一 Agent 在不同模板下分配不同模型
5. Agent 代码无任何修改
6. 修改 models.yaml/profiles/templates 后方案自动重算
7. Agent LLM 调用失败时自动 failover，不影响全局方案
8. 切换回 `routing_plugin: conversation` 后第一阶段功能完全恢复
9. PII 脱敏、合规检测两种插件下均正常
10. 分发延迟 < 0.1ms
