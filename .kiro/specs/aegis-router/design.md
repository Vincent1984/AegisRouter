# AegisRouter — 技术设计文档

## 1. 系统架构概览

### 1.1 物理拓扑

```
┌──────────────────── Docker Container ────────────────────┐
│                                                           │
│  Supervisor (PID 1)                                       │
│    ├── ClawVault 进程 → 监听 /var/run/clawvault.sock      │
│    └── LiteLLM Proxy → 监听 0.0.0.0:8000 (对外 TCP)      │
│                                                           │
│  ┌──────────────┐   UDS (IPC)    ┌───────────────────┐   │
│  │   LiteLLM    │◄═════════════►│    ClawVault       │   │
│  │  (网关主体)   │ clawvault.sock │  (安全伴生进程)    │   │
│  └──────┬───────┘                └───────────────────┘   │
│         │                                                 │
└─────────┼─────────────────────────────────────────────────┘
          │ HTTPS
          ▼
  ┌─────────────────────┐
  │  外部 LLM API 集群   │
  │  (OpenAI/DeepSeek/  │
  │   Claude/本地模型)   │
  └─────────────────────┘
```

### 1.2 请求生命周期管道

```
Client Request (OpenAI SDK Compatible)
    │
    ▼
[1] LiteLLM Gateway — 鉴权 + Rate Limit
    │
    ▼
[2] pre_call_hook → UDS → ClawVault.mask()
    │  ├── Presidio PII 检测 (NER + Regex)
    │  ├── 占位符替换
    │  ├── Redis 存储映射 (TTL=30min)
    │  └── Prompt Injection 检测
    │
    ▼
[3] 智能路由决策 (LiteLLM callback 内)
    │  ├── 规则前置 (<1ms): 寒暄词库命中 → 本地7B
    │  └── RouteLLM BERT 推理 (~8ms): score → 模型选择
    │
    ▼
[4] LiteLLM 转发 → 目标 LLM API
    │  └── Failover: 429/503 → 候选模型漂移
    │
    ▼
[5] post_call_hook → UDS → ClawVault.restore()
    │  ├── 响应合规检测
    │  └── 占位符还原 (Rehydration)
    │
    ▼
Client Response
```

---

## 2. 模块详细设计

### 2.1 网关接入层 (LiteLLM Proxy)

**技术选型**: LiteLLM Proxy (Python / FastAPI / uvicorn)

**职责**:
- 对外暴露 `v1/chat/completions` 端点
- API Key 鉴权
- Rate Limit (令牌桶算法，按 API Key 粒度)
- Custom Callbacks 注册与生命周期管理
- Failover 与模型池管理

**配置结构** (`config.yaml`):
```yaml
model_list:
  - model_name: local-7b
    litellm_params:
      model: ollama/qwen2-7b
      api_base: http://localhost:11434
  - model_name: deepseek-chat
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY
      api_base: https://api.deepseek.com
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: o1
    litellm_params:
      model: openai/o1
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: aegis_router.callbacks.smart_router_instance

general_settings:
  master_key: os.environ/AEGIS_MASTER_KEY
```

**多 Worker 部署**:
- uvicorn workers = CPU 核心数 (推荐 4-8)
- 每个 worker 独立维护 UDS 连接池到 ClawVault

---

### 2.2 ClawVault 安全伴生进程

**职责**: PII 脱敏、映射管理、占位符还原、合规检测

**通信协议**: Unix Domain Socket + JSON-RPC 2.0

```
请求格式:
{
  "jsonrpc": "2.0",
  "method": "mask" | "restore" | "check_compliance",
  "params": { ... },
  "id": "request-uuid"
}

响应格式:
{
  "jsonrpc": "2.0",
  "result": { ... },
  "id": "request-uuid"
}
```

**核心接口**:

| Method | 输入 | 输出 | 延迟目标 |
|--------|------|------|----------|
| `mask` | `{text, session_id, request_id}` | `{masked_text, entities_found: []}` | < 12ms |
| `restore` | `{text, request_id}` | `{restored_text}` | < 3ms |
| `restore_stream_chunk` | `{chunk, request_id, buffer_state}` | `{flushed_text, new_buffer_state}` | < 1ms |
| `check_compliance` | `{text, direction: "inbound"|"outbound"}` | `{passed: bool, violations: []}` | < 5ms |

**PII 检测引擎 (基于 Presidio)**:

内置 Recognizer:
- `PhoneRecognizer` (通用)
- `EmailRecognizer` (通用)
- `IpAddressRecognizer` (通用)
- `CreditCardRecognizer` (通用)

自定义中文 Recognizer:
- `ChinesePhoneRecognizer` — 正则: `1[3-9]\d{9}`
- `ChineseIdCardRecognizer` — 正则 + 校验位验证: `[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]`
- `ChineseNameRecognizer` — 基于 spaCy zh_core_web_trf NER 模型 + 百家姓前缀增强

**映射存储 (Redis)**:

```
Key:    aegis:pii:{session_id}:{request_id}
Value:  { "[PERSON_1]": "张三", "[PHONE_1]": "13800138000" }
TTL:    1800 seconds (30 min)

Key:    aegis:pii:session:{session_id}
Value:  { "[PERSON_1]": "张三" }  # 会话级持久映射
TTL:    3600 seconds (1 hour)
```

---

### 2.3 智能路由决策模块

#### 2.3.1 整体流程

```
prompt 进入
    │
    ▼
[规则前置] 寒暄检测 (<1ms)
    │ 命中 → 本地小模型，结束
    │ 未命中 ↓
    ▼
[RouteLLM 打分] prompt 难度评估 (~8ms)
    │ 输出 score [0, 1]
    ▼
[区间匹配] score 匹配模型能力区间
    │ 找出所有候选模型
    ▼
[重叠策略] 从候选中选择最终模型
    │
    ▼
路由完成，改写 target_model
```

#### 2.3.2 规则前置引擎

```python
TRIVIAL_PATTERNS = [
    "你好", "hello", "hi", "hey", "谢谢", "再见",
    "早上好", "晚上好", "good morning", ...
]

def is_trivial_chat(prompt: str) -> bool:
    if len(prompt) > config.trivial_max_length:  # 默认 30
        return False
    prompt_lower = prompt.lower().strip()
    return any(p in prompt_lower for p in TRIVIAL_PATTERNS)
```

#### 2.3.3 RouteLLM 集成（Prompt 难度打分）

- 模型: `routellm/mf` (Matrix Factorization，推荐) 或 `routellm/bert`
- 推理方式: 本地 ONNX 推理，不调用外部 API
- 输出: `strong_win_rate` float [0, 1]（越高 = 任务越难）
- 不改 RouteLLM 源码，仅调用其 `calculate_strong_win_rate()` 接口

**打分输入策略 (`score_input`)**:

路由打分可以使用脱敏后文本或原文，通过配置切换：

```yaml
routing:
  score_input: masked  # masked | original
```

| 模式 | 输入文本 | 适用场景 | 说明 |
|------|---------|---------|------|
| `masked` (默认) | 脱敏后的 prompt | 安全优先 | PII 不进入分类器缓存/日志，精度损失极小 |
| `original` | 原始 prompt | 精度优先 | 仅限本地推理（不出网），审计日志中不记录原文 |

**`original` 模式的执行流程**（并行优化）:
```
prompt 进入
    ├──► [脱敏] → masked_prompt (存 Redis)       ─┐
    └──► [路由打分] → score (用原文，本地推理)     ─┤ 并行执行
                                                   ↓
                              合并: 用 score 决定模型，用 masked_prompt 转发
```

安全约束：`original` 模式下，路由分类器必须为纯本地推理（不调用外部 API），且审计日志仅记录 prompt hash，不记录原文内容。

#### 2.3.4 模型能力自动评估引擎

**模型参数配置** (`config/models.yaml`):
```yaml
models:
  - name: local-7b
    litellm_model: ollama/qwen2-7b
    params:
      parameter_size_b: 7          # 参数量 (Billion)
      context_window: 32000        # 上下文长度
      benchmark_mmlu: 65.0         # MMLU 分数
      benchmark_humaneval: 45.0    # HumanEval 代码能力
      benchmark_math: 40.0         # MATH 数学能力
      cost_per_1m_input: 0.0       # 本地模型无费用
      cost_per_1m_output: 0.0
      latency_avg_ms: 200
      supports_streaming: true
      supports_function_call: false

  - name: deepseek-v3
    litellm_model: deepseek/deepseek-chat
    params:
      parameter_size_b: 671
      context_window: 128000
      benchmark_mmlu: 87.1
      benchmark_humaneval: 82.6
      benchmark_math: 75.3
      cost_per_1m_input: 0.27
      cost_per_1m_output: 1.10
      latency_avg_ms: 800
      supports_streaming: true
      supports_function_call: true

  - name: gemini-1.5-pro
    litellm_model: gemini/gemini-1.5-pro
    params:
      parameter_size_b: null
      context_window: 2000000
      benchmark_mmlu: 85.9
      benchmark_humaneval: 71.9
      benchmark_math: 67.7
      cost_per_1m_input: 1.25
      cost_per_1m_output: 5.00
      latency_avg_ms: 1000
      supports_streaming: true
      supports_function_call: true

  - name: gpt-4o
    litellm_model: openai/gpt-4o
    params:
      parameter_size_b: null
      context_window: 128000
      benchmark_mmlu: 88.7
      benchmark_humaneval: 90.2
      benchmark_math: 81.4
      cost_per_1m_input: 2.50
      cost_per_1m_output: 10.00
      latency_avg_ms: 600
      supports_streaming: true
      supports_function_call: true

  - name: o1
    litellm_model: openai/o1
    params:
      parameter_size_b: null
      context_window: 200000
      benchmark_mmlu: 91.8
      benchmark_humaneval: 94.2
      benchmark_math: 94.8
      cost_per_1m_input: 15.00
      cost_per_1m_output: 60.00
      latency_avg_ms: 3000
      supports_streaming: true
      supports_function_call: true
```

**评分权重配置** (`config/route_config.yaml`):
```yaml
scoring:
  weights:
    benchmark_mmlu: 0.25        # 通用知识
    benchmark_humaneval: 0.20   # 代码能力
    benchmark_math: 0.20        # 数学推理
    context_window: 0.10        # 上下文长度
    cost_efficiency: 0.25       # 性价比 (1 - 归一化成本)
  
  # 归一化边界 (用于将原始值映射到 [0,1])
  normalization:
    benchmark_mmlu: [50, 95]
    benchmark_humaneval: [30, 95]
    benchmark_math: [20, 95]
    context_window: [4096, 2000000]
    cost_per_1m_input: [0, 20]  # 用于计算 cost_efficiency = 1 - normalize(cost)

  # 区间容差: computed_score ± tolerance 生成 score_range
  range_tolerance: 0.15
```

**能力分数计算算法**:

```python
class ModelScorer:
    """模型能力评分引擎 — 算法可替换"""
    
    def __init__(self, weights: dict, normalization: dict, tolerance: float):
        self.weights = weights
        self.normalization = normalization
        self.tolerance = tolerance
    
    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        """归一化到 [0, 1]，超出范围则 clamp"""
        if value is None:
            return 0.5  # 未知参数取中位值
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def compute_score(self, params: dict) -> float:
        """加权归一化计算模型能力分数"""
        score = 0.0
        
        # 各 benchmark 维度
        for key in ["benchmark_mmlu", "benchmark_humaneval", "benchmark_math"]:
            if key in self.weights:
                norm_range = self.normalization[key]
                score += self.weights[key] * self.normalize(
                    params.get(key, None), norm_range[0], norm_range[1]
                )
        
        # 上下文长度
        if "context_window" in self.weights:
            norm_range = self.normalization["context_window"]
            score += self.weights["context_window"] * self.normalize(
                params.get("context_window", 4096), norm_range[0], norm_range[1]
            )
        
        # 性价比 (成本越低分越高)
        if "cost_efficiency" in self.weights:
            norm_range = self.normalization["cost_per_1m_input"]
            cost_norm = self.normalize(
                params.get("cost_per_1m_input", 0), norm_range[0], norm_range[1]
            )
            score += self.weights["cost_efficiency"] * (1.0 - cost_norm)
        
        return max(0.0, min(1.0, score))
    
    def compute_score_range(self, params: dict) -> tuple[float, float]:
        """计算能力分数并生成路由区间"""
        score = self.compute_score(params)
        return (
            max(0.0, score - self.tolerance),
            min(1.0, score + self.tolerance)
        )
```

**自动生成路由表**:
```python
def build_routing_table(models_config: list, scorer: ModelScorer, overrides: dict) -> list:
    """根据模型参数自动生成路由表"""
    tiers = []
    for model in models_config:
        name = model["name"]
        
        # 检查是否有人工覆盖
        if name in overrides:
            score_range = overrides[name]["score_range"]
            computed_score = scorer.compute_score(model["params"])
        else:
            computed_score = scorer.compute_score(model["params"])
            score_range = scorer.compute_score_range(model["params"])
        
        tiers.append({
            "name": name,
            "model": model["litellm_model"],
            "computed_score": computed_score,
            "score_range": score_range,
            "cost_per_1m_input": model["params"].get("cost_per_1m_input", 0),
            "overridden": name in overrides,
        })
    
    # 按 computed_score 排序
    tiers.sort(key=lambda t: t["computed_score"])
    return tiers
```

**人工覆盖** (`config/route_overrides.yaml`):
```yaml
# 管理员手动覆盖的模型区间（优先级高于自动计算）
overrides:
  gpt-4o:
    score_range: [0.50, 0.82]
    reason: "实测 GPT-4o 在中文场景表现优于 benchmark 预期"
  local-7b:
    score_range: [0.0, 0.18]
    reason: "限制本地模型只处理最简单的任务"
```

#### 2.3.5 路由匹配与重叠策略

```python
class RouteResolver:
    def __init__(self, tiers: list, strategy: str, fallback_model: str):
        self.tiers = tiers
        self.strategy = strategy
        self.fallback = fallback_model
        self._round_robin_idx = 0
        self._session_models = {}  # session_id → locked model
    
    def resolve(self, prompt_score: float, session_id: str = None, session_policy: str = "sticky") -> dict:
        """根据 prompt 难度分数，匹配模型并返回路由结果"""
        
        # 会话锁定检查
        if session_id and session_policy != "per_turn":
            locked = self._check_session_lock(session_id, prompt_score, session_policy)
            if locked:
                return locked
        
        # 找出所有区间覆盖该分数的模型
        candidates = [
            t for t in self.tiers
            if t["score_range"][0] <= prompt_score <= t["score_range"][1]
        ]
        
        if not candidates:
            selected_model = self.fallback
            reason = "no_match_fallback"
        elif len(candidates) == 1:
            selected_model = candidates[0]["model"]
            reason = "single_match"
        else:
            selected = self._apply_strategy(candidates)
            selected_model = selected["model"]
            reason = f"overlap_{self.strategy}"
        
        # 记录 session 锁定
        if session_id and session_policy != "per_turn":
            self._session_models[session_id] = {
                "model": selected_model,
                "tier_score": self._get_tier_score(selected_model),
            }
        
        return {"model": selected_model, "reason": reason}
    
    def _check_session_lock(self, session_id: str, prompt_score: float, policy: str) -> dict | None:
        """检查会话锁定策略"""
        prev = self._session_models.get(session_id)
        if prev is None:
            return None  # 第一轮，不锁定
        
        if policy == "sticky":
            # 始终使用第一轮选定的模型
            return {"model": prev["model"], "reason": "session_sticky"}
        
        elif policy == "escalate_only":
            # 计算当前应选模型的 tier_score
            current_model = self._resolve_without_lock(prompt_score)
            current_tier = self._get_tier_score(current_model)
            
            if current_tier >= prev["tier_score"]:
                # 升级：允许，并更新 session 记录
                self._session_models[session_id] = {
                    "model": current_model,
                    "tier_score": current_tier,
                }
                return {"model": current_model, "reason": "session_escalated"}
            else:
                # 降级：不允许，保持之前的模型
                return {"model": prev["model"], "reason": "session_no_downgrade"}
        
        return None
    
    def _apply_strategy(self, candidates: list) -> dict:
        if self.strategy == "lowest_cost":
            return min(candidates, key=lambda t: t["cost_per_1m_input"])
        elif self.strategy == "highest_capability":
            return max(candidates, key=lambda t: t["computed_score"])
        elif self.strategy == "round_robin":
            selected = candidates[self._round_robin_idx % len(candidates)]
            self._round_robin_idx += 1
            return selected
        elif self.strategy == "random":
            import random
            return random.choice(candidates)
        else:
            return candidates[0]
```

**会话路由配置**:
```yaml
# config/route_config.yaml
routing:
  session_policy: sticky  # sticky | per_turn | escalate_only
  
  # session 锁定过期时间（避免长期占用）
  session_lock_ttl_minutes: 60
```

**会话策略行为说明**:

| 策略 | 第1轮 | 第2轮(简单追问) | 第3轮(复杂追问) | 适用场景 |
|------|-------|----------------|----------------|---------|
| `sticky` | GPT-4o | GPT-4o | GPT-4o | 咨询/代码审计/连续对话 |
| `per_turn` | GPT-4o | DeepSeek | GPT-4o | 独立任务（翻译/分类） |
| `escalate_only` | DeepSeek | DeepSeek | GPT-4o | 成本敏感但不降质 |

#### 2.3.6 路由配置总览

```yaml
# config/route_config.yaml
routing:
  # 规则前置
  trivial:
    enabled: true
    max_length: 30
    patterns_file: ./patterns/trivial_chat.txt
    target_model: local-7b

  # RouteLLM 分类器
  classifier:
    type: mf           # mf | bert | sw_ranking
    model_path: null   # null = 使用默认预训练模型

  # 重叠策略
  overlap_strategy: lowest_cost

  # 兜底模型
  fallback_model: deepseek-v3

  # 评分配置
  scoring:
    weights:
      benchmark_mmlu: 0.25
      benchmark_humaneval: 0.20
      benchmark_math: 0.20
      context_window: 0.10
      cost_efficiency: 0.25
    normalization:
      benchmark_mmlu: [50, 95]
      benchmark_humaneval: [30, 95]
      benchmark_math: [20, 95]
      context_window: [4096, 2000000]
      cost_per_1m_input: [0, 20]
    range_tolerance: 0.15
```

**配置热更新**: 通过 watchdog 文件监听 `config/route_config.yaml`、`config/models.yaml`、`config/route_overrides.yaml`，变更时自动重新计算路由表，无需重启。

#### 2.3.7 自动生成的路由表（运行时产物）

系统启动时根据 `models.yaml` + `route_config.yaml` + `route_overrides.yaml` 自动计算生成以下路由表结构（内存中持有，配置变更时重新生成）：

```yaml
# 系统自动生成（可人工覆盖）
routing_table:
  auto_generated: true
  generated_at: "2026-07-16T10:00:00Z"

  tiers:
    - name: local-7b
      model: ollama/qwen2-7b
      computed_score: 0.15
      score_range: [0.0, 0.20]
      cost_per_1m_input: 0.0
      overridden: false

    - name: deepseek-v3
      model: deepseek/deepseek-chat
      computed_score: 0.42
      score_range: [0.10, 0.50]
      cost_per_1m_input: 0.27
      overridden: false

    - name: gemini-1.5-pro
      model: gemini/gemini-1.5-pro
      computed_score: 0.55
      score_range: [0.35, 0.65]
      cost_per_1m_input: 1.25
      overridden: false

    - name: gpt-4o
      model: openai/gpt-4o
      computed_score: 0.72
      score_range: [0.50, 0.82]       # 人工覆盖
      cost_per_1m_input: 2.50
      overridden: true

    - name: o1
      model: openai/o1
      computed_score: 0.91
      score_range: [0.75, 1.0]
      cost_per_1m_input: 15.00
      overridden: false

  overlap_strategy: lowest_cost
  fallback_model: deepseek-v3
```

**路由匹配示例**：

| prompt 难度分数 | 命中区间的模型 | 策略 lowest_cost 结果 |
|----------------|---------------|---------------------|
| 0.12 | local-7b, deepseek-v3 | local-7b (cost=0) |
| 0.35 | deepseek-v3, gemini-1.5-pro | deepseek-v3 (cost=0.27) |
| 0.55 | gemini-1.5-pro, gpt-4o | gemini-1.5-pro (cost=1.25) |
| 0.80 | gpt-4o, o1 | gpt-4o (cost=2.50) |
| 0.95 | o1 | o1 (单候选，直接命中) |

---

### 2.4 流式响应还原引擎

**核心问题**: SSE chunk 可能切割占位符（如 `[PER` + `SON_1]`）

**解决方案**: 带缓冲的流式占位符替换器

```python
class StreamRehydrator:
    PLACEHOLDER_PATTERN = re.compile(r'\[[A-Z]+_\d+\]')
    PARTIAL_PATTERN = re.compile(r'\[[A-Z_\d]*$')  # 尾部不完整的占位符
    
    def __init__(self, mapping: dict):
        self.mapping = mapping
        self.buffer = ""
    
    def process_chunk(self, chunk_text: str) -> str:
        """处理单个 chunk，返回可安全 flush 的文本"""
        self.buffer += chunk_text
        
        # 检查尾部是否有不完整的占位符
        partial_match = self.PARTIAL_PATTERN.search(self.buffer)
        
        if partial_match:
            # 尾部可能不完整，保留在 buffer
            safe_part = self.buffer[:partial_match.start()]
            self.buffer = self.buffer[partial_match.start():]
        else:
            # 没有不完整的占位符，全部可 flush
            safe_part = self.buffer
            self.buffer = ""
        
        # 还原 safe_part 中的完整占位符
        restored = self.PLACEHOLDER_PATTERN.sub(
            lambda m: self.mapping.get(m.group(), m.group()),
            safe_part
        )
        return restored
    
    def flush_remaining(self) -> str:
        """流结束时 flush 剩余 buffer"""
        restored = self.PLACEHOLDER_PATTERN.sub(
            lambda m: self.mapping.get(m.group(), m.group()),
            self.buffer
        )
        self.buffer = ""
        return restored
```

**集成到 LiteLLM Callback**:
```python
async def async_post_call_streaming_iterator_hook(
    self, user_api_key_dict, response, request_data
) -> AsyncGenerator[ModelResponseStream, None]:
    request_id = request_data.get("litellm_call_id")
    mapping = await redis.get_mapping(request_id)
    rehydrator = StreamRehydrator(mapping)
    
    async for chunk in response:
        content = chunk.choices[0].delta.content or ""
        restored = rehydrator.process_chunk(content)
        if restored:
            chunk.choices[0].delta.content = restored
            yield chunk
    
    # flush 残留 buffer
    remaining = rehydrator.flush_remaining()
    if remaining:
        final_chunk = create_final_chunk(remaining)
        yield final_chunk
```

---

### 2.5 安全合规拦截

**Prompt Injection 检测策略**:

1. **规则匹配** (fast path, < 1ms):
   - 检测已知攻击模式: `ignore previous instructions`, `you are now`, `system prompt`
   - 中文变体: `忽略之前的指令`, `你现在是`, `输出你的系统提示`

2. **特征评分** (second pass, < 3ms):
   - 异常长度检测 (prompt 中 system 指令比例过高)
   - 角色切换关键词密度
   - 编码混淆检测 (base64/unicode 转义)

**拦截模式配置**:
```yaml
compliance:
  mode: strict  # strict | interactive | permissive
  inbound_checks:
    - prompt_injection
    - sensitive_words
  outbound_checks:
    - harmful_content
  sensitive_words_file: ./rules/sensitive_words.txt
```

---

### 2.6 灾备容错

**Failover 链配置**:
```yaml
failover:
  enabled: true
  timeout_ms: 50
  chains:
    gpt-4o: [claude-3.5-sonnet, deepseek-chat]
    deepseek-chat: [gpt-4o-mini, local-7b]
    o1: [gpt-4o, claude-3.5-sonnet]
```

**降级策略**:

| 故障场景 | 行为 |
|----------|------|
| ClawVault 进程挂掉 | Bypass 脱敏，直通转发，记录 CRITICAL 告警 |
| Redis 不可用 | 拒绝需脱敏的请求，返回 HTTP 503 |
| RouteLLM 推理超时 (>15ms) | 默认路由到 deepseek-chat |
| 目标 LLM 返回 429 | 沿 Failover 链漂移到下一个模型 |
| 所有模型不可用 | 返回 HTTP 503 + 标准错误体 |

---

## 3. 数据流与存储

### 3.1 Redis 数据模型

| Key Pattern | Value | TTL | 用途 |
|-------------|-------|-----|------|
| `aegis:pii:{session}:{request}` | JSON 映射表 | 30min | 单次请求脱敏映射 |
| `aegis:pii:session:{session}` | JSON 映射表 | 1h | 会话级累积映射 |
| `aegis:ratelimit:{api_key}` | Counter | 60s | Rate Limit 计数 |
| `aegis:metrics:{date}:{api_key}` | Hash (tokens_in, tokens_out, cost) | 7d | 用量统计 |

### 3.2 审计日志 (stdout / file)

```json
{
  "ts": "2026-07-16T10:30:00Z",
  "event": "route_decision",
  "request_id": "uuid",
  "session_id": "uuid",
  "api_key_hash": "sha256...",
  "prompt_hash": "sha256...",
  "prompt_length": 156,
  "entities_detected": ["PERSON", "PHONE"],
  "route_score": 0.45,
  "target_model": "gpt-4o",
  "latency_mask_ms": 8.2,
  "latency_route_ms": 6.1
}
```

---

## 4. 部署架构

### 4.1 高可用多活架构

AegisRouter 采用**无状态网关 + 共享 Redis** 的多活部署模式，所有实例对等，任意实例故障不影响整体服务。

```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │  (Nginx/ALB/SLB)    │
                    └──────┬──────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  AegisRouter │ │  AegisRouter │ │  AegisRouter │
    │  Instance 1  │ │  Instance 2  │ │  Instance 3  │
    │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │
    │  │LiteLLM │  │ │  │LiteLLM │  │ │  │LiteLLM │  │
    │  │  :8000 │  │ │  │  :8000 │  │ │  │  :8000 │  │
    │  └───┬────┘  │ │  └───┬────┘  │ │  └───┬────┘  │
    │      │ UDS   │ │      │ UDS   │ │      │ UDS   │
    │  ┌───┴────┐  │ │  ┌───┴────┐  │ │  ┌───┴────┐  │
    │  │ClawVault│  │ │  │ClawVault│  │ │  │ClawVault│  │
    │  └────────┘  │ │  └────────┘  │ │  └────────┘  │
    └──────────────┘ └──────────────┘ └──────────────┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Redis Cluster (HA)   │
              │   (Sentinel / Cluster) │
              └────────────────────────┘
```

**多活设计要点**：

| 设计原则 | 实现方式 |
|----------|---------|
| 网关无状态 | 所有状态存 Redis，实例间不共享内存 |
| 对等部署 | 每个实例完全相同，无主从之分 |
| 水平扩缩 | 按 QPS/CPU 自动增减实例数 |
| 流量分发 | L4/L7 负载均衡，支持加权轮询 |
| 故障隔离 | 单实例挂掉，LB 自动摘除，其余实例接管 |
| 配置一致 | ConfigMap 统一分发，所有实例读同一份配置 |

### 4.2 Redis 高可用

Redis 作为 PII 映射表的存储，是整个系统唯一的有状态依赖，必须高可用：

```yaml
# 生产推荐: Redis Sentinel (3节点) 或 Redis Cluster (6节点)
redis:
  mode: sentinel          # standalone | sentinel | cluster
  sentinel:
    master_name: aegis-master
    nodes:
      - redis-sentinel-0:26379
      - redis-sentinel-1:26379
      - redis-sentinel-2:26379
    password: ${REDIS_PASSWORD}
  
  # 连接池配置
  pool:
    max_connections: 100
    min_idle: 10
    timeout_ms: 500
    retry_on_timeout: true
```

**Redis 故障场景处理**：

| 场景 | 行为 |
|------|------|
| Redis 主节点故障 | Sentinel 自动故障转移 (< 30s)，网关自动重连 |
| Redis 完全不可用 | 网关拒绝需脱敏的请求 (HTTP 503)，不脱敏的请求可 bypass |
| Redis 网络抖动 | 连接池自动重试，重试 3 次失败则降级 |

### 4.3 单容器内部结构（每个实例）

每个 AegisRouter 实例是一个 Docker 容器，内部由 Supervisor 管理两个进程：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 复制 vendor 源码（LiteLLM, RouteLLM, ClawVault）
COPY vendor/ ./vendor/

# 安装 vendor 依赖（本地源码安装）
RUN pip install --no-cache-dir -e ./vendor/litellm && \
    pip install --no-cache-dir -e ./vendor/routellm && \
    pip install --no-cache-dir -e ./vendor/clawvault

# 安装项目自身依赖（非 vendor 的第三方库）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 spaCy 中文模型
RUN python -m spacy download zh_core_web_trf

# 复制应用代码和配置
COPY aegis_router/ ./aegis_router/
COPY config/ ./config/
COPY patterns/ ./patterns/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Supervisor 配置
COPY supervisord.conf /etc/supervisord.conf

# 健康检查
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
```

**关键点**：
- vendor 源码通过 `pip install -e ./vendor/xxx` 以可编辑模式安装，既能本地修改又有正常的包导入路径
- `requirements.txt` 只声明非 vendor 的依赖（redis, spacy, presidio, watchdog 等）
- 构建顺序：先装 vendor（变化少）→ 再装 requirements（变化少）→ 最后 COPY 应用代码（变化多），利用 Docker 层缓存

**Supervisor 配置**:
```ini
[supervisord]
nodaemon=true

[program:clawvault]
command=python -m aegis_router.clawvault.server
priority=1
autorestart=true
startretries=5
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0

[program:litellm]
command=litellm --config /app/config/config.yaml --port 8000 --num_workers 4
priority=2
autorestart=true
startsecs=5
depends_on=clawvault
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
```

### 4.4 Kubernetes 多活部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aegis-router
  labels:
    app: aegis-router
spec:
  replicas: 3                    # 最少 3 实例
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0          # 滚动更新不中断服务
  selector:
    matchLabels:
      app: aegis-router
  template:
    metadata:
      labels:
        app: aegis-router
    spec:
      containers:
        - name: aegis-router
          image: aegis-router:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          envFrom:
            - secretRef:
                name: aegis-router-secrets
          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: aegis-router-config

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aegis-router-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aegis-router
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "800"     # 单实例 800 QPS 时开始扩容

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: aegis-router
spec:
  type: ClusterIP
  selector:
    app: aegis-router
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP

---
# k8s/pdb.yaml (Pod Disruption Budget)
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: aegis-router-pdb
spec:
  minAvailable: 2               # 任何时刻至少 2 个实例在线
  selector:
    matchLabels:
      app: aegis-router
```

### 4.5 多活部署验证清单

| 验证项 | 方法 |
|--------|------|
| 单实例宕机不中断 | Kill 1 个 Pod，确认请求无失败 |
| 滚动更新零停机 | `kubectl rollout` 期间持续压测 |
| Redis 主节点切换 | 手动触发 Sentinel failover，观察网关自动重连 |
| 扩容响应 | 模拟 QPS 飙升，确认 HPA 在 30s 内拉起新实例 |
| 缩容安全 | 确认缩容时 in-flight 请求不被中断（graceful shutdown） |

---

## 5. 项目工程结构

```
AegisRouter/
├── vendor/                           # 第三方开源源码 (vendored, 锁定版本)
│   ├── litellm/                      # LiteLLM 源码 (MIT License)
│   │   ├── litellm/                  # 核心包
│   │   ├── pyproject.toml
│   │   └── LICENSE
│   ├── routellm/                     # RouteLLM 源码 (Apache 2.0)
│   │   ├── routellm/
│   │   ├── pyproject.toml
│   │   └── LICENSE
│   └── clawvault/                    # ClawVault 源码 (MIT License)
│       ├── src/claw_vault/
│       ├── pyproject.toml
│       └── LICENSE
│
├── aegis_router/                     # 主应用包（我们的代码）
│   ├── __init__.py
│   ├── callbacks/                    # LiteLLM Custom Callbacks
│   │   ├── __init__.py
│   │   ├── smart_router.py           # 主回调类 (pre_call + post_call)
│   │   └── stream_rehydrator.py      # 流式还原引擎
│   ├── clawvault/                    # ClawVault 二次开发
│   │   ├── __init__.py
│   │   ├── server.py                 # UDS Server 主进程 (改写自 vendor)
│   │   ├── masker.py                 # PII 脱敏 (集成 Presidio)
│   │   ├── restorer.py               # 占位符还原 (新增)
│   │   ├── compliance.py             # 合规检测引擎 (扩展)
│   │   └── recognizers/              # 自定义 Presidio Recognizer (新增)
│   │       ├── __init__.py
│   │       ├── chinese_phone.py
│   │       ├── chinese_id_card.py
│   │       └── chinese_name.py
│   ├── router/                       # 智能路由模块
│   │   ├── __init__.py
│   │   ├── rule_engine.py            # 规则前置引擎
│   │   ├── model_classifier.py       # RouteLLM 推理封装 (调用 vendor/routellm)
│   │   ├── model_scorer.py           # 模型能力评分引擎
│   │   ├── route_resolver.py         # 区间匹配与重叠策略
│   │   └── config_watcher.py         # 配置热更新
│   ├── storage/                      # Redis 存储层
│   │   ├── __init__.py
│   │   └── redis_client.py           # 异步 Redis 操作封装
│   ├── observability/                # 可观测性
│   │   ├── __init__.py
│   │   ├── metrics.py                # 指标收集
│   │   └── audit_logger.py           # 审计日志
│   └── config.py                     # 全局配置管理
│
├── config/                           # 配置文件
│   ├── config.yaml                   # LiteLLM 模型池配置
│   ├── models.yaml                   # 模型参数声明
│   ├── route_config.yaml             # 路由阈值、权重、策略
│   ├── route_overrides.yaml          # 人工覆盖的模型区间
│   └── compliance_rules/             # 合规规则
│       ├── sensitive_words.txt
│       └── injection_patterns.yaml
│
├── patterns/                         # 规则匹配文件
│   └── trivial_chat.txt              # 寒暄词库
│
├── tests/                            # 测试
│   ├── __init__.py
│   ├── test_masker.py
│   ├── test_restorer.py
│   ├── test_stream_rehydrator.py
│   ├── test_router.py
│   ├── test_compliance.py
│   └── fixtures/
│       └── sample_prompts.json
│
├── k8s/                              # Kubernetes 配置
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   └── pdb.yaml
│
├── Dockerfile
├── supervisord.conf
├── requirements.txt                  # 仅声明非 vendor 的第三方依赖
├── pyproject.toml                    # 项目元信息 + vendor 本地依赖声明
├── LICENSE                           # 我们自己的产品协议
├── THIRD_PARTY_LICENSES.md           # 第三方开源协议声明
├── README.md
└── .env.example
```

---

## 6. 技术栈总结

| 层次 | 技术 | 角色 |
|------|------|------|
| 网关骨架 | LiteLLM Proxy | HTTP 代理、模型池、Failover、Callbacks |
| PII 检测 | Microsoft Presidio + spaCy zh_core_web_trf | NER + 正则检测 |
| 安全网关 | ClawVault (自研/魔改) | 脱敏、还原、合规检测 |
| 智能路由 | RouteLLM (BERT/MF, ONNX) | Prompt 复杂度评分 |
| 映射存储 | Redis | PII 映射表、Rate Limit、Metrics |
| 进程管理 | Supervisor | 容器内多进程管理 |
| IPC | Unix Domain Socket + JSON-RPC 2.0 | LiteLLM ↔ ClawVault |
| 部署 | Docker + Kubernetes | 容器化 + 弹性伸缩 |
