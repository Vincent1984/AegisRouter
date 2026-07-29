# AegisRouter 事务级路由插件 — 技术设计文档

## Overview

事务级路由插件（Transaction Router Plugin）是 AegisRouter 的第二阶段路由策略。与第一阶段（每次请求实时打分）不同，事务级路由在**系统启动时**为每个业务流程模板中的每个 Agent 预计算好模型分配，之后所有请求直接查表分发。

**核心架构决策**：
- 路由上下文（template + agent）由 **Supervisor 注入**，Agent 本身零修改
- 查表 key：`(template_name, agent_name) → model_name`
- 同一 Agent 在不同模板下可分配不同模型
- 方案绑定在模板上，不绑定在单次事务/请求上

**关键区别**：

| 维度 | 对话级路由（第一阶段） | 事务级路由（第二阶段） |
|------|----------------------|----------------------|
| 路由粒度 | 单次请求 | 模板 × Agent |
| 决策时机 | 每次请求实时打分 | 启动时/配置变更时预计算 |
| 分发方式 | 打分 → 区间匹配 | 查表 (template, agent) → model |
| 分发延迟 | ~10ms | < 0.1ms |
| 状态依赖 | Redis（会话锁定） | 纯内存 |
| Agent 侵入 | 无（单请求粒度） | 无（Supervisor 注入上下文） |
| 适用场景 | 独立对话请求 | 多 Agent 协作的业务流程 |

---

## Architecture

### 插件化架构

```
AegisRouter Core (LiteLLM + ClawVault + 基础管道)
    │
    ├── [Plugin Slot: Routing Strategy]
    │       ├── conversation_router (对话级)
    │       ├── transaction_router  (事务级)  ← 本文档
    │       └── (future plugins...)
    │
    └── 公共管道 (PII脱敏, 合规, 流式还原 — 始终生效)
```

### 调用链路

```
用户请求
    │
    ▼
Supervisor (知道 template + agent)
    │
    │  注入 metadata: {"transaction": {"template": "xxx", "agent": "yyy"}}
    │
    ▼
Agent (正常调 LLM，不感知路由信息)
    │
    │  请求带着 Supervisor 注入的 metadata
    │
    ▼
AegisRouter
    ├── [公共管道] 合规检测 + PII 脱敏
    ├── [事务路由] 读 metadata → 查表 → data["model"] = 结果
    └── [公共管道] 转发 → 响应还原
    │
    ▼
目标 LLM
```

### 方案预计算流程

```
系统启动（或配置变更后重启 litellm 进程）
    │
    ▼
加载: models.yaml + capability_profiles.yaml + transaction_templates.yaml
    │
    ▼
TemplatePlanGenerator.generate_all()
    │  对每个模板的每个 Agent:
    │  ├── 有 override_model? → 直接使用
    │  └── 否则: 加载 Profile → 对所有模型打分 → 过滤约束 → 选最优
    │
    ▼
RoutingPlanStore (内存 HashMap)
    │  key: (template_name, agent_name)
    │  value: model_name
    │
    ▼
就绪，接收请求
```


---

## Components and Interfaces

### 1. TransactionRouterCallback (主入口)

```python
class TransactionRouterCallback(BaseRouterCallback):
    """事务级路由插件 — 查表分发，极简逻辑"""
    
    def __init__(self, plan_store: RoutingPlanStore, fallback_model: str):
        self._plan_store = plan_store
        self._fallback_model = fallback_model
    
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # 1. 公共管道
        await self._execute_common_pipeline(data)
        
        # 2. 读取路由上下文 (Supervisor 注入)
        metadata = data.get("metadata", {})
        txn = metadata.get("transaction")
        
        if txn is None:
            data["model"] = self._fallback_model
            return
        
        template = txn.get("template")
        agent = txn.get("agent")
        
        # 3. 查表分发
        model = self._plan_store.get_model(template, agent)
        
        if model is None:
            logger.warning("UNKNOWN_AGENT: template=%s, agent=%s", template, agent)
            model = self._fallback_model
        
        data["model"] = model
```

### 2. TemplatePlanGenerator (方案生成器)

```python
class TemplatePlanGenerator:
    """纯函数：输入配置 → 输出方案表"""
    
    def __init__(self, profile_manager, models, fallback_model):
        self.profile_manager = profile_manager
        self.models = models
        self.fallback_model = fallback_model
    
    def generate_all(self, templates: dict[str, TemplateDef]) -> RoutingPlanStore:
        store = RoutingPlanStore()
        
        for tpl_name, tpl_def in templates.items():
            for agent_def in tpl_def.agents:
                model = self._select_model(agent_def)
                store.set_model(tpl_name, agent_def.name, model)
        
        return store
    
    def _select_model(self, agent_def: AgentDef) -> str:
        # 覆盖优先
        if agent_def.override_model:
            return agent_def.override_model
        
        profile = self.profile_manager.get_profile(agent_def.capability_profile)
        candidates = self.profile_manager.filter_by_constraints(self.models, profile)
        
        if not candidates:
            return self.fallback_model
        
        scored = [
            (m, self.profile_manager.score_model(m, profile))
            for m in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 偏好选择
        if profile.prefer_models:
            for m, s in scored:
                if m.name in profile.prefer_models:
                    return m.name
        
        return scored[0][0].name
```

### 3. RoutingPlanStore (内存查找表)

```python
class RoutingPlanStore:
    """线程安全的只读查找表，通过引用替换实现原子更新"""
    
    def __init__(self):
        self._table: dict[tuple[str, str], str] = {}
    
    def set_model(self, template: str, agent: str, model: str):
        self._table[(template, agent)] = model
    
    def get_model(self, template: str, agent: str) -> Optional[str]:
        return self._table.get((template, agent))
    
    def get_template_plan(self, template: str) -> dict[str, str]:
        """获取某模板的完整方案 {agent → model}"""
        return {
            agent: model
            for (t, agent), model in self._table.items()
            if t == template
        }
    
    def get_all_plans(self) -> dict[str, dict[str, str]]:
        """所有方案 {template → {agent → model}}"""
        result = {}
        for (tpl, agent), model in self._table.items():
            result.setdefault(tpl, {})[agent] = model
        return result
```

### 4. CapabilityProfileManager (Profile 评分)

```python
class CapabilityProfileManager:
    """Profile 加载、评分、约束过滤"""
    
    def get_profile(self, name: str) -> CapabilityProfile:
        if name not in self.profiles:
            logger.warning("Profile '%s' not found, fallback to 'medium'", name)
            return self.profiles["medium"]
        return self.profiles[name]
    
    def score_model(self, model: ModelInfo, profile: CapabilityProfile) -> float:
        """用 Profile 权重为模型打分 → [0, 1]"""
        score = 0.0
        for dim, weight in profile.scoring_weights.items():
            if dim == "cost_efficiency":
                cost = model.params.get("cost_per_1m_input", 0)
                score += weight * (1.0 - self._normalize(cost, 0, 20))
            elif dim == "context_window":
                val = model.params.get("context_window", 4096)
                score += weight * self._normalize(val, 4096, 2000000)
            else:
                val = model.params.get(dim)
                r = self.normalization.get(dim, [0, 100])
                score += weight * self._normalize(val, r[0], r[1])
        return max(0.0, min(1.0, score))
    
    def filter_by_constraints(self, models, profile) -> list:
        return [m for m in models if self._meets_constraints(m, profile)]
    
    def _meets_constraints(self, model, profile) -> bool:
        if self.score_model(model, profile) < profile.min_score_threshold:
            return False
        if model.params.get("cost_per_1m_input", 0) > profile.max_cost_per_1m_input:
            return False
        if profile.min_context_window:
            if model.params.get("context_window", 0) < profile.min_context_window:
                return False
        return True
```

### 5. 配置热更新集成

```python
# ConfigWatcher 新增监听:
#   - config/capability_profiles.yaml
#   - config/transaction_templates.yaml
#
# 任一文件变更 → 重新调用 TemplatePlanGenerator.generate_all()
#              → 原子替换 RoutingPlanStore 引用
#              → 日志输出新旧方案对比

def _on_config_changed(self, changed_files):
    new_store = self.generator.generate_all(self.templates)
    old_plans = self._plan_store.get_all_plans()
    new_plans = new_store.get_all_plans()
    
    # 日志输出差异
    self._log_plan_diff(old_plans, new_plans)
    
    # 原子替换
    self._plan_store = new_store
```


---

## Data Models

### 核心数据结构

```python
@dataclass
class CapabilityProfile:
    name: str
    description: str
    scoring_weights: dict[str, float]
    min_score_threshold: float = 0.0
    max_cost_per_1m_input: float = 60.0
    min_context_window: Optional[int] = None
    prefer_models: list[str] = field(default_factory=list)


@dataclass
class AgentDef:
    """模板中的 Agent 定义"""
    name: str                              # Agent 标识
    capability_profile: str                # Profile 名称
    override_model: Optional[str] = None   # 管理员直接指定（最高优先级）


@dataclass
class TemplateDef:
    """业务流程模板"""
    name: str
    description: str
    agents: list[AgentDef]
```

### 配置文件格式

#### config/transaction_templates.yaml

```yaml
templates:
  resume_screening:
    description: "简历筛选流程"
    agents:
      - name: intent_classifier
        capability_profile: lightweight
      - name: resume_parser
        capability_profile: long_context
      - name: skill_matcher
        capability_profile: strong_reasoning
      - name: compliance_checker
        capability_profile: medium

  code_review:
    description: "代码审查流程"
    agents:
      - name: code_analyzer
        capability_profile: code_specialist
      - name: issue_detector
        capability_profile: strong_reasoning
      - name: fix_suggester
        capability_profile: code_specialist

  supplier_evaluation:
    description: "供应商评估流程"
    agents:
      - name: data_collector
        capability_profile: lightweight
      - name: performance_scorer
        capability_profile: medium
      - name: compliance_checker          # 同一 Agent，不同模板
        capability_profile: medium
      - name: tier_determiner
        capability_profile: strong_reasoning

  custom_pipeline:
    description: "自定义流程（部分 Agent 手动指定模型）"
    agents:
      - name: analyzer
        capability_profile: medium
      - name: generator
        capability_profile: heavy
        override_model: gpt-5.6-sol       # 管理员覆盖
```

#### config/capability_profiles.yaml

```yaml
profiles:
  lightweight:
    description: "低延迟低成本，简单分类/意图识别"
    scoring_weights:
      benchmark_mmlu: 0.10
      benchmark_humaneval: 0.05
      benchmark_math: 0.05
      context_window: 0.05
      cost_efficiency: 0.75
    min_score_threshold: 0.0
    max_cost_per_1m_input: 0.5

  medium:
    description: "平衡质量和成本"
    scoring_weights:
      benchmark_mmlu: 0.25
      benchmark_humaneval: 0.15
      benchmark_math: 0.15
      context_window: 0.10
      cost_efficiency: 0.35
    min_score_threshold: 0.30
    max_cost_per_1m_input: 3.0

  strong_reasoning:
    description: "强推理，复杂逻辑/数学/分析"
    scoring_weights:
      benchmark_mmlu: 0.15
      benchmark_humaneval: 0.30
      benchmark_math: 0.35
      context_window: 0.05
      cost_efficiency: 0.15
    min_score_threshold: 0.60
    max_cost_per_1m_input: 20.0

  code_specialist:
    description: "代码专精"
    scoring_weights:
      benchmark_mmlu: 0.10
      benchmark_humaneval: 0.50
      benchmark_math: 0.15
      context_window: 0.10
      cost_efficiency: 0.15
    min_score_threshold: 0.50
    max_cost_per_1m_input: 10.0
    prefer_models: [codex-mini, gpt-5.5]

  long_context:
    description: "超长上下文"
    scoring_weights:
      benchmark_mmlu: 0.15
      benchmark_humaneval: 0.10
      benchmark_math: 0.10
      context_window: 0.50
      cost_efficiency: 0.15
    min_score_threshold: 0.35
    min_context_window: 500000
    max_cost_per_1m_input: 10.0

  heavy:
    description: "最强模型，复杂推理"
    scoring_weights:
      benchmark_mmlu: 0.30
      benchmark_humaneval: 0.25
      benchmark_math: 0.30
      context_window: 0.10
      cost_efficiency: 0.05
    min_score_threshold: 0.75
    max_cost_per_1m_input: 60.0
```

### 客户端协议

Supervisor 注入的请求：
```jsonc
{
  "messages": [{"role": "user", "content": "..."}],
  "metadata": {
    "transaction": {
      "template": "resume_screening",
      "agent": "resume_parser"
    }
  }
}
```

Agent 代码示例（零修改，metadata 由 Supervisor 透传）：
```python
# Agent 内部正常调 LLM，不感知路由
response = await openai_client.chat.completions.create(
    model="placeholder",  # 会被 AegisRouter 覆盖
    messages=[{"role": "user", "content": resume_text}],
    extra_body={"metadata": inherited_metadata}  # Supervisor 传下来的
)
```

响应扩展：
```jsonc
{
  "choices": [...],
  "aegis_metadata": {
    "template": "resume_screening",
    "agent": "resume_parser",
    "assigned_model": "gemini-2.5-pro",
    "routing_plugin": "transaction",
    "warnings": []
  }
}
```


---

## 方案生成示例

### 输入

- 模型池：11 个模型（local-7b, deepseek-v4-pro, claude-sonnet, gpt-5.2, gpt-5.4-mini, gpt-5.5, gpt-5.6-sol, codex-mini, gemini-2.5-flash, gemini-2.5-pro, gemini-3.1-pro）
- Profile：6 种
- 模板：4 个

### 系统启动日志

```
[INFO] Transaction Router: 方案生成完成

模板: resume_screening
  intent_classifier   → local-7b         (profile=lightweight, score=0.92)
  resume_parser       → gemini-2.5-pro   (profile=long_context, score=0.78)
  skill_matcher       → gpt-5.5          (profile=strong_reasoning, score=0.85)
  compliance_checker  → deepseek-v4-pro  (profile=medium, score=0.71)

模板: code_review
  code_analyzer       → codex-mini       (profile=code_specialist, score=0.82, preferred)
  issue_detector      → gpt-5.5          (profile=strong_reasoning, score=0.85)
  fix_suggester       → codex-mini       (profile=code_specialist, score=0.82, preferred)

模板: supplier_evaluation
  data_collector      → local-7b         (profile=lightweight, score=0.92)
  performance_scorer  → deepseek-v4-pro  (profile=medium, score=0.71)
  compliance_checker  → deepseek-v4-pro  (profile=medium, score=0.71)
  tier_determiner     → gpt-5.5          (profile=strong_reasoning, score=0.85)

模板: custom_pipeline
  analyzer            → deepseek-v4-pro  (profile=medium, score=0.71)
  generator           → gpt-5.6-sol      (override, 管理员指定)

同名 Agent 跨模板对比:
  compliance_checker:
    resume_screening     → deepseek-v4-pro  (medium)
    supplier_evaluation  → deepseek-v4-pro  (medium)  [相同]
```

### 配置变更日志

```
[INFO] 检测到 models.yaml 变更，重算所有模板方案...

模板: resume_screening 变化:
  skill_matcher: gpt-5.5 → gpt-5.2  (新模型评分更高)
  其他 Agent 不变

模板: code_review 无变化
模板: supplier_evaluation 无变化
模板: custom_pipeline 无变化 (generator 为 override，不受影响)
```

---

## Error Handling

| 场景 | 行为 |
|------|------|
| 引用不存在的模板 | HTTP 400: `{"error": "Template 'xxx' not found"}` |
| Agent 不在模板中 | fallback 模型 + UNKNOWN_AGENT 警告 |
| Profile 不存在 | 降级 medium + PROFILE_NOT_FOUND 警告 |
| 无模型满足约束 | fallback 模型 + NO_CANDIDATE 警告 |
| LLM 调用失败 | failover 链重试（仅当次请求） |
| 无 transaction metadata | fallback 模型转发 |
| 配置语法错误 | 拒绝加载，保持上一版方案 + CONFIG_ERROR |

---

## Testing Strategy

### 单元测试

| 组件 | 测试重点 |
|------|---------|
| TemplatePlanGenerator | 方案生成、Profile 评分、覆盖优先、约束过滤 |
| CapabilityProfileManager | 评分精度、约束逻辑、Profile 降级 |
| RoutingPlanStore | 查表、原子替换 |
| TransactionRouterCallback | 分发正确性、异常处理 |

### 集成测试

- 启动 → 方案生成 → 请求分发 → 验证模型正确
- 同一 Agent 不同模板 → 验证分配不同模型
- 配置热更新 → 验证方案自动重算
- 插件切换 → `conversation` ↔ `transaction` 无副作用
- Failover → LLM 错误时重试不影响全局方案

### 性能基准

| 指标 | 目标 |
|------|------|
| 请求分发延迟 | < 0.1ms |
| 方案生成 (10模板×5Agent) | < 5ms |
| 内存 | < 10KB |

---

## 项目工程结构

```
aegis_router/
├── callbacks/
│   ├── __init__.py
│   ├── base_router.py               # 新增: 插件基类
│   ├── smart_router.py              # 保留: 对话级
│   ├── transaction_router.py        # 新增: 事务级
│   ├── plugin_loader.py             # 新增: 插件加载器
│   ├── stream_rehydrator.py
│   ├── degradation.py
│   └── uds_pool.py
├── router/
│   ├── __init__.py
│   ├── rule_engine.py               # 对话级用
│   ├── model_classifier.py          # 对话级用
│   ├── model_scorer.py              # 共用
│   ├── route_resolver.py            # 对话级用
│   ├── config_watcher.py            # 共用（新增监听）
│   ├── template_plan_generator.py   # 新增
│   ├── capability_profiles.py       # 新增
│   └── routing_plan_store.py        # 新增
├── ...

config/
├── config.yaml                       # 修改: 新增 routing_plugin 字段
├── capability_profiles.yaml          # 新增 (系统内置默认值，用户可选改)
├── transaction_templates.yaml        # 新增 (用户按业务配，提供示例)
├── models.yaml                       # 不变 (用户必须配，提供预填模板)
├── route_config.yaml                 # 不变 (failover 链复用)
└── route_overrides.yaml              # 不变 (对话级专用)
```

---

## Correctness Properties

1. **确定性**: 相同配置永远生成相同方案
2. **一致性**: 同一 (template, agent) 并发请求路由到相同模型
3. **原子更新**: 配置变更时方案整体替换，无半成品
4. **覆盖优先级**: override_model > Profile 自动选择 > fallback
5. **Failover 隔离**: 仅影响当次请求，全局方案不变
6. **插件互斥**: 同一时刻只有一个路由插件工作
7. **Agent 无感**: Agent 代码不因路由插件存在而需修改

---

## 配置默认行为

### 各文件职责与缺省策略

| 文件 | 谁来配 | 缺省行为 |
|------|--------|---------|
| `capability_profiles.yaml` | 系统内置，用户可选改 | 不存在时使用代码内置的 6 种默认 Profile |
| `transaction_templates.yaml` | 用户按业务配 | 不存在时系统正常启动，所有请求走 fallback 模型 |
| `models.yaml` | 用户必须配 | 不存在时无法打分，所有 Agent 走 fallback + NO_MODELS 警告 |

### 内置 Profile 默认值

当 `capability_profiles.yaml` 不存在时，系统使用以下硬编码默认值：

```python
DEFAULT_PROFILES = {
    "lightweight": CapabilityProfile(
        name="lightweight",
        scoring_weights={"cost_efficiency": 0.75, "benchmark_mmlu": 0.10, ...},
        min_score_threshold=0.0,
        max_cost_per_1m_input=0.5,
    ),
    "medium": ...,
    "strong_reasoning": ...,
    "code_specialist": ...,
    "long_context": ...,
    "heavy": ...,
}
```

### models.yaml 预填模板

系统提供 `models.yaml.example`，包含常见模型参数（用户按需保留）：

```yaml
# models.yaml.example — 复制为 models.yaml，删掉你没有的模型
models:
  - name: local-7b
    litellm_model: ollama/qwen2.5-7b
    params:
      benchmark_mmlu: 65.0
      benchmark_humaneval: 45.0
      benchmark_math: 40.0
      context_window: 32000
      cost_per_1m_input: 0.0

  - name: deepseek-v4-pro
    litellm_model: deepseek/deepseek-v4-pro
    params:
      benchmark_mmlu: 90.2
      benchmark_humaneval: 88.5
      benchmark_math: 82.0
      context_window: 128000
      cost_per_1m_input: 0.27

  # ... 其他常见模型
```

### transaction_templates.yaml 示例

系统提供 `transaction_templates.yaml.example` 作为参考：

```yaml
# transaction_templates.yaml.example — 按你的业务流程修改
templates:
  example_pipeline:
    description: "示例流程"
    agents:
      - name: your_agent_name
        capability_profile: medium    # 可选: lightweight/medium/strong_reasoning/code_specialist/long_context/heavy
```

### 降级链路

```
请求到达
    │
    ├── 有 transaction metadata?
    │     ├── 有模板方案? → 按方案分发 ✓
    │     ├── 模板存在但 Agent 不在里面? → fallback + UNKNOWN_AGENT 警告
    │     └── 模板不存在? → HTTP 400
    │
    └── 无 transaction metadata? → fallback 模型
```

---

## 配置变更操作指南

当前版本不支持不重启热更新（Docker overlay fs 不触发 inotify 事件）。修改配置文件后需手动重启 litellm 进程。

### 修改步骤

**1. 修改配置文件**

可修改的文件：
- `config/transaction_templates.yaml` — 增删 Agent、修改 Profile 引用、修改 override_model
- `config/capability_profiles.yaml` — 修改 Profile 权重、约束参数
- `config/models.yaml` — 增删模型、修改参数、标记 `available: false`
- `config/config.yaml` — 切换 `routing_plugin`（conversation / transaction）

**2. 重启 litellm 进程**

```bash
# 本地开发（Docker Compose，config 已通过 volume 挂载）
# 改完本地 config/ 文件后直接重启进程，不需要重建镜像
docker exec aegis-router supervisorctl restart litellm

# K8s 环境（config 通过 ConfigMap 挂载）
# 修改 ConfigMap 后重启 Pod
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment aegis-router
```

**3. 验证生效**

```bash
# 发请求确认路由变化
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"any","messages":[{"role":"user","content":"test"}],"metadata":{"transaction":{"template":"YOUR_TEMPLATE","agent":"YOUR_AGENT"}}}'
```

检查响应中 `model` 字段是否为预期模型。

### 切换路由插件

```bash
# 1. 修改 config/config.yaml
routing_plugin: conversation   # 或 transaction

# 2. 重启
docker exec aegis-router supervisorctl restart litellm
```

### 标记模型不可用

在 `config/models.yaml` 中对应模型的 `params` 下添加：

```yaml
- name: some-model
  params:
    available: false   # 评分时跳过，不会被分配给任何 Agent
    ...
```

然后重启 litellm 进程。
