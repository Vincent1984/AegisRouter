# AegisRouter

<p align="center">
  <strong>智能安全 LLM 路由网关</strong> — 企业级大模型安全路由中间件
</p>

<p align="center">
  PII 脱敏保护 · 智能成本分流 · 安全合规拦截 · 高可用灾备 · OpenAI SDK 兼容
</p>

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [核心功能](#核心功能)
- [快速开始](#快速开始)
- [配置参考](#配置参考)
- [API 使用示例](#api-使用示例)
- [智能路由详解](#智能路由详解)
- [PII 脱敏详解](#pii-脱敏详解)
- [安全合规](#安全合规)
- [灾备容错](#灾备容错)
- [可观测性](#可观测性)
- [性能指标](#性能指标)
- [Kubernetes 部署](#kubernetes-部署)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目简介

AegisRouter 是一款企业级智能安全 LLM 路由网关，部署在应用层与 LLM API 之间，为企业提供：

- **隐私数据零泄露** — 基于 Microsoft Presidio 的 PII 实时脱敏，支持中文敏感信息（姓名、手机号、身份证号）
- **智能成本优化** — 自动评估 prompt 难度，将简单请求路由到低成本模型，复杂请求路由到强模型
- **安全合规拦截** — Prompt Injection 检测、敏感词过滤，阻断安全威胁
- **高可用无感切换** — Failover 链自动漂移，50ms 内完成模型切换
- **完全兼容** — 对外暴露标准 `v1/chat/completions` API，现有 OpenAI SDK 代码无需修改

---

## 系统架构

### 容器内部结构

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
  │   Gemini/本地模型)   │
  └─────────────────────┘
```

### 请求生命周期

```
Client Request (OpenAI SDK 兼容)
    │
    ▼
[1] LiteLLM Gateway — 鉴权 + Rate Limit
    │
    ▼
[2] pre_call_hook → UDS → ClawVault.mask()
    │  ├── Presidio PII 检测 (NER + Regex)
    │  ├── 占位符替换 (如 [PERSON_1], [PHONE_1])
    │  ├── Redis 存储映射 (TTL=30min)
    │  └── Prompt Injection 检测
    │
    ▼
[3] 智能路由决策 (LiteLLM Callback)
    │  ├── 规则前置 (<1ms): 寒暄词库命中 → 本地7B
    │  └── RouteLLM 推理 (~8ms): score → 模型能力区间匹配
    │
    ▼
[4] LiteLLM 转发 → 目标 LLM API
    │  └── Failover: 429/503 → 候选模型自动漂移
    │
    ▼
[5] post_call_hook → UDS → ClawVault.restore()
    │  ├── 响应合规检测
    │  └── 占位符还原 (Rehydration)
    │
    ▼
Client Response (PII 已还原的完整响应)
```

### 高可用多活部署

```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    └──────┬──────────────┘
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  AegisRouter │ │  AegisRouter │ │  AegisRouter │
    │  Instance 1  │ │  Instance 2  │ │  Instance 3  │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           └────────────────┼────────────────┘
                            ▼
              ┌────────────────────────┐
              │   Redis Cluster (HA)   │
              └────────────────────────┘
```

- 网关完全无状态，所有 PII 映射存 Redis
- 实例对等部署，支持 Kubernetes HPA 水平扩缩
- 单实例故障 LB 自动摘除，其余实例无感接管

---

## 核心功能

| 功能模块 | 说明 | 延迟开销 |
|----------|------|----------|
| PII 脱敏/还原 | 自动检测并替换敏感信息，响应时还原 | < 12ms |
| 智能路由 | prompt 难度评估 + 模型能力匹配，成本最优选择 | < 10ms |
| 合规拦截 | Prompt Injection + 敏感词，请求/响应双向检测 | < 5ms |
| 流式还原 | SSE 流式响应中跨 chunk 占位符安全还原 | < 1ms/chunk |
| Failover | 自动故障漂移，支持多级候选模型链 | < 50ms |
| 配置热更新 | YAML 文件变更自动生效，无需重启 | 0 (异步) |

---

## 快速开始

### 环境要求

- Python >= 3.11
- Redis >= 5.0
- Docker (生产部署)
- 至少一个 LLM API Key (OpenAI / DeepSeek / Gemini)

### 方式一：本地开发

```bash
# 1. 克隆项目
git clone <repo-url> && cd AegisRouter

# 2. 创建虚拟环境
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
# python -m venv .venv && .venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -e ".[dev]"

# 4. 安装 spaCy 中文模型 (中文 PII 识别需要)
python -m spacy download zh_core_web_trf

# 5. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Keys 和 Redis 地址

# 6. 启动 Redis (如果本地没有)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 7. 启动 ClawVault 安全进程
python -m aegis_router.clawvault.server &

# 8. 启动 LiteLLM 网关
litellm --config config/config.yaml --port 8000
```

### 方式二：Docker 部署 (推荐)

```bash
# 1. 构建镜像
docker build -t aegis-router:latest .

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入真实值

# 3. 启动 (需要外部 Redis)
docker run -d \
  --name aegis-router \
  -p 8000:8000 \
  --env-file .env \
  aegis-router:latest

# 4. 验证健康状态
curl http://localhost:8000/health
```

### 方式三：Docker Compose (含 Redis)

```yaml
# docker-compose.yaml
version: "3.8"
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  aegis-router:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 3s
      retries: 3

volumes:
  redis_data:
```

```bash
docker compose up -d
```

### 验证部署

```bash
# 健康检查
curl http://localhost:8000/health

# 发送测试请求
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

---

## 配置参考

AegisRouter 使用多个 YAML 配置文件，均位于 `config/` 目录：

### config/config.yaml — LiteLLM 模型池

定义可用的 LLM 后端和 Failover 策略：

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: deepseek-chat
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY
      api_base: https://api.deepseek.com

  - model_name: local-7b
    litellm_params:
      model: ollama/qwen2-7b
      api_base: http://localhost:11434

router_settings:
  routing_strategy: "simple-shuffle"
  num_retries: 2
  timeout: 30
  fallbacks:
    - gpt-4o: ["gemini-1.5-pro", "deepseek-chat"]
    - deepseek-chat: ["gpt-4o", "local-7b"]
```

### config/models.yaml — 模型能力参数

声明每个模型的 Benchmark 分数和成本参数，用于自动计算能力分数：

```yaml
models:
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
```

### config/route_config.yaml — 路由策略

控制智能路由的所有参数：

```yaml
routing:
  score_input: masked          # masked | original
  session_policy: sticky       # sticky | per_turn | escalate_only
  session_lock_ttl_minutes: 60 # 会话锁定过期时间
  trivial:
    enabled: true
    max_length: 30
    target_model: local-7b
  classifier:
    type: mf                   # mf | bert | sw_ranking
  overlap_strategy: lowest_cost
  fallback_model: deepseek-v3
  scoring:
    weights:
      benchmark_mmlu: 0.30
      benchmark_humaneval: 0.25
      benchmark_math: 0.25
      context_window: 0.10
      cost_efficiency: 0.10
    range_tolerance: 0.12
```

**会话路由策略 (`session_policy`)**:

| 策略 | 行为 | 适用场景 |
|------|------|---------|
| `sticky` (默认) | 第一轮路由后，同一 session 后续请求全走同一模型 | 连续对话（咨询/代码审计） |
| `per_turn` | 每轮独立打分独立路由 | 无关联的单次任务（翻译/分类） |
| `escalate_only` | 后续轮次只能升级模型（weak→strong），不能降级 | 平衡成本与质量 |

### config/route_overrides.yaml — 人工覆盖

管理员可手动覆盖自动计算的模型能力区间：

```yaml
overrides:
  gpt-4o:
    score_range: [0.50, 0.82]
    reason: "实测 GPT-4o 在中文场景表现优于 benchmark 预期"
  local-7b:
    score_range: [0.0, 0.18]
    reason: "限制本地模型只处理最简单的任务"
```

### config/compliance_rules/ — 合规规则

完整的合规规则配置体系，所有文件修改后自动热更新：

#### injection_patterns.yaml — 注入攻击模式

```yaml
# 特征评分配置（控制角色切换检测的灵敏度）
feature_scoring:
  min_hits: 3              # 最少命中关键词数量才触发
  density_threshold: 0.15  # 关键词密度阈值 (命中数/总词数)
  min_words: 5             # 低于此词数的短文本跳过检测

# 注入攻击模式列表
patterns:
  - id: INJ_001                        # 规则 ID (唯一)
    pattern: "ignore previous instructions"  # 匹配文本 (大小写不敏感)
    severity: high                     # 严重程度: high | medium | low
    description: "尝试覆盖系统指令"      # 人类可读描述

  - id: INJ_002
    pattern: "忽略之前的指令"
    severity: high
    description: "中文指令覆盖攻击"
```

#### role_switch_keywords.txt — 角色切换关键词

用于特征评分检测。当一条 prompt 中同时出现多个关键词且密度超过阈值时，判定为 Prompt Injection：

```text
# 英文关键词
ignore
forget
bypass
pretend
act as
you are now

# 中文关键词
忽略
假装
扮演
你现在是
```

#### sensitive_words.txt — 敏感词库

企业自定义的不允许发送到 LLM 的关键词，命中时根据模式执行拦截/告警：

```text
# 企业项目代号
Project Phoenix
星火计划

# 内部系统
天网监控系统
OA审批平台

# 财务敏感
Q4营收数据
利润率
IPO计划
```

#### 配置拦截模式

通过环境变量 `COMPLIANCE_MODE` 控制命中后的行为：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `strict` (默认) | 命中即拦截，返回 HTTP 400 | 生产环境 |
| `interactive` | 标记不通过，由上层决定处理 | 内部测试 |
| `permissive` | 记录日志但放行 | 开发调试 |

### 环境变量

完整的环境变量列表见 [`.env.example`](./.env.example)，按类别分组：

| 类别 | 关键变量 | 说明 |
|------|---------|------|
| 鉴权 | `AEGIS_MASTER_KEY` | 网关 API Key |
| LLM | `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` | 模型服务密钥 |
| Redis | `REDIS_URL` | PII 映射存储 |
| 路由 | `ROUTING_OVERLAP_STRATEGY` | 重叠策略选择 |
| 合规 | `COMPLIANCE_MODE` | 拦截模式 |
| 日志 | `LOG_LEVEL` | 日志级别 |

---

## API 使用示例

AegisRouter 完全兼容 OpenAI SDK，现有代码只需修改 `base_url` 即可接入。

### cURL

```bash
# 标准请求 (自动路由到最优模型)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $AEGIS_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "我叫张三，手机号13800138000，请帮我写一封邮件"}
    ],
    "stream": false
  }'

# 流式请求
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $AEGIS_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "解释量子计算的基本原理"}],
    "stream": true
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-master-key",       # AEGIS_MASTER_KEY
    base_url="http://localhost:8000/v1"  # AegisRouter 地址
)

# 标准调用 — PII 自动脱敏和还原，对调用方完全透明
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "我叫张三，身份证号是110101199001011234，帮我查下快递"}
    ]
)
print(response.choices[0].message.content)
# 响应中「张三」和身份证号已自动还原，LLM 只看到 [PERSON_1] 和 [ID_CARD_1]

# 流式调用
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首关于春天的诗"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Python (httpx)

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer sk-your-master-key"},
    json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(resp.json()["choices"][0]["message"]["content"])
```

---

## 智能路由详解

AegisRouter 的智能路由通过四步决策，将每个请求路由到成本最优的模型：

### 决策流程

```
prompt 进入
    │
    ▼
[1] 规则前置 — 寒暄检测 (<1ms)
    │ 命中寒暄词库 → 直接路由本地7B，结束
    │ 未命中 ↓
    ▼
[2] RouteLLM 打分 — prompt 难度评估 (~8ms)
    │ 本地 ONNX 推理，输出 score [0, 1]
    │ (0=极简单, 1=极复杂)
    ▼
[3] 区间匹配 — score 与模型能力区间比对
    │ 找出所有 score_range 覆盖该分数的模型
    ▼
[4] 重叠策略 — 多候选模型时按策略选择
    │ lowest_cost / highest_capability / round_robin / random
    ▼
路由完成 → 请求转发至目标模型
```

### 模型能力评分

系统根据 `models.yaml` 中的参数自动计算每个模型的能力分数：

| 评分维度 | 权重 | 说明 |
|---------|------|------|
| MMLU Benchmark | 25% | 通用知识能力 |
| HumanEval | 20% | 代码生成能力 |
| MATH | 20% | 数学推理能力 |
| Context Window | 10% | 上下文处理能力 |
| Cost Efficiency | 25% | 性价比 (成本越低分越高) |

### 路由匹配示例

| prompt 难度 | 命中模型 | lowest_cost 结果 | 原因 |
|------------|---------|-----------------|------|
| 0.12 | local-7b, deepseek-v3 | local-7b | 成本 $0 |
| 0.35 | deepseek-v3, gemini-1.5-pro | deepseek-v3 | 成本 $0.27/M |
| 0.55 | gemini-1.5-pro, gpt-4o | gemini-1.5-pro | 成本 $1.25/M |
| 0.80 | gpt-4o, o1 | gpt-4o | 成本 $2.50/M |
| 0.95 | o1 | o1 | 单候选直接命中 |

### 配置热更新

修改 `config/route_config.yaml`、`config/models.yaml` 或 `config/route_overrides.yaml` 后，系统通过文件监听自动重新计算路由表，无需重启服务。

---

## PII 脱敏详解

### 支持的实体类型

| 实体类型 | 占位符格式 | 示例 |
|---------|-----------|------|
| 人名 | `[PERSON_N]` | 张三 → [PERSON_1] |
| 手机号 | `[PHONE_N]` | 13800138000 → [PHONE_1] |
| 身份证号 | `[ID_CARD_N]` | 110101199001011234 → [ID_CARD_1] |
| 邮箱 | `[EMAIL_N]` | test@example.com → [EMAIL_1] |
| IP 地址 | `[IP_ADDRESS_N]` | 192.168.1.1 → [IP_ADDRESS_1] |
| 银行卡号 | `[CREDIT_CARD_N]` | 6222021234567890 → [CREDIT_CARD_1] |

### 中文 PII 识别

AegisRouter 针对中文场景定制了三个专用识别器：

- **ChinesePhoneRecognizer** — 正则匹配 `1[3-9]\d{9}` 格式
- **ChineseIdCardRecognizer** — 正则 + 校验位验证 (18 位身份证)
- **ChineseNameRecognizer** — spaCy `zh_core_web_trf` NER + 百家姓前缀增强

### 工作原理

```
用户输入: "我叫张三，手机号13800138000"
    │
    ▼ ClawVault.mask()
发送到 LLM: "我叫[PERSON_1]，手机号[PHONE_1]"
    │
    ▼ Redis 存储映射: { "[PERSON_1]": "张三", "[PHONE_1]": "13800138000" }
    │
    ▼ LLM 返回: "好的[PERSON_1]，您的手机号[PHONE_1]已记录"
    │
    ▼ ClawVault.restore()
返回用户: "好的张三，您的手机号13800138000已记录"
```

### 流式还原

SSE 流式响应中，占位符可能被切割到不同 chunk（如 `[PER` + `SON_1]`）。AegisRouter 使用带缓冲的 StreamRehydrator 确保跨 chunk 占位符安全还原，延迟 < 1ms/chunk。

### 数据安全

- Redis 映射 TTL = 30 分钟，到期自动物理擦除
- 网关本地不持久化任何明文 PII
- 审计日志仅记录 prompt hash，不记录原始内容
- 会话级映射支持多轮对话占位符一致性

---

## 安全合规

### Prompt Injection 检测

双层检测策略：

1. **规则匹配 (fast path, < 1ms)**
   - 已知攻击模式：`ignore previous instructions`, `you are now`, `system prompt`
   - 中文变体：`忽略之前的指令`, `你现在是`, `输出你的系统提示`

2. **特征评分 (second pass, < 3ms)**
   - 异常长度检测 (system 指令比例过高)
   - 角色切换关键词密度
   - 编码混淆检测 (base64/unicode 转义)

### 敏感词过滤

- 支持自定义敏感词库 (`config/compliance_rules/sensitive_words.txt`)
- 每行一个关键词，支持中英文
- 配置热更新，修改后实时生效

### 拦截模式

通过 `COMPLIANCE_MODE` 环境变量配置：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `strict` | 命中即拦截，返回 HTTP 400 | 生产环境（默认） |
| `interactive` | 提示客户端确认 | 内部测试环境 |
| `permissive` | 记录日志但放行 | 开发调试 |

### 双向检测

- **入站检测 (Inbound)**: Prompt Injection + 敏感词
- **出站检测 (Outbound)**: LLM 响应中的有害/违规内容

---

## 灾备容错

### Failover 链

每个模型可配置优先级排序的候选模型链：

```yaml
fallbacks:
  - gpt-4o: ["gemini-1.5-pro", "deepseek-chat"]
  - o1: ["gpt-4o", "deepseek-chat"]
  - deepseek-chat: ["gpt-4o", "local-7b"]
```

当主模型返回 429 (限流) / 503 (不可用) / Timeout 时，50ms 内自动漂移到下一候选模型，客户端无感知。

### 降级策略

| 故障场景 | 系统行为 |
|----------|---------|
| ClawVault 进程崩溃 | Bypass 脱敏直通转发，记录 CRITICAL 告警 |
| Redis 完全不可用 | 拒绝需脱敏的请求，返回 HTTP 503 |
| RouteLLM 推理超时 (>15ms) | 默认路由到 deepseek-chat |
| 目标 LLM 返回 429 | 沿 Failover 链漂移到下一模型 |
| 所有模型不可用 | 返回 HTTP 503 + 标准错误体 |

---

## 可观测性

### 健康检查

```bash
GET /health
# 200 OK — 服务正常
# 503 — 服务异常
```

### 审计日志

每次请求生成结构化 JSON 审计记录：

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

### 指标收集

Redis 中按维度聚合的用量统计：

| 指标 | Key Pattern | TTL |
|------|------------|-----|
| Token 消耗 | `aegis:metrics:{date}:{api_key}` | 7 天 |
| Rate Limit 计数 | `aegis:ratelimit:{api_key}` | 60 秒 |
| 路由决策分布 | 审计日志聚合 | — |

### 分步耗时打点

每次请求记录完整的处理链路耗时：
- `latency_mask_ms` — PII 脱敏耗时
- `latency_route_ms` — 路由决策耗时
- `latency_llm_ms` — LLM API 响应耗时
- `latency_restore_ms` — 占位符还原耗时

---

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 网关附加延迟 | ≤ 20ms | 脱敏 + 路由决策（不含 LLM 网络耗时） |
| 单实例 QPS | ≥ 1000 | 并发代理能力 |
| 规则引擎延迟 | < 1ms | 寒暄词匹配 |
| RouteLLM 推理 | < 10ms | 本地 ONNX 推理 |
| PII 脱敏 | < 12ms | Presidio NER + Regex |
| 占位符还原 | < 3ms | Redis 读取 + 替换 |
| 流式 chunk 还原 | < 1ms | 带缓冲的正则替换 |
| Failover 漂移 | < 50ms | 故障检测 + 请求重发 |

---

## Kubernetes 部署

项目提供完整的 Kubernetes 配置文件，位于 `k8s/` 目录：

```
k8s/
├── deployment.yaml    # Deployment (3+ 副本)
├── service.yaml       # ClusterIP Service
├── configmap.yaml     # 配置文件挂载
├── secret.yaml        # API Keys 等敏感信息
├── hpa.yaml           # 水平自动扩缩容
└── pdb.yaml           # Pod Disruption Budget
```

### 部署步骤

```bash
# 1. 创建 Namespace
kubectl create namespace aegis

# 2. 创建 Secret (API Keys)
kubectl create secret generic aegis-router-secrets \
  --from-env-file=.env \
  -n aegis

# 3. 创建 ConfigMap (YAML 配置)
kubectl create configmap aegis-router-config \
  --from-file=config/ \
  -n aegis

# 4. 部署服务
kubectl apply -f k8s/ -n aegis

# 5. 验证
kubectl get pods -n aegis
kubectl logs -f deployment/aegis-router -n aegis
```

### 关键配置

| 配置项 | 值 | 说明 |
|--------|---|------|
| replicas | 3 (最低) | 保证高可用 |
| HPA min/max | 3 / 20 | 自动扩缩范围 |
| CPU target | 70% | 扩容触发阈值 |
| QPS target | 800/实例 | 扩容触发阈值 |
| PDB minAvailable | 2 | 任何时刻至少 2 实例在线 |
| Rolling Update | maxSurge=1, maxUnavailable=0 | 滚动更新不中断 |

### 资源配额

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

### Redis 高可用

生产环境推荐使用 Redis Sentinel (3 节点) 或 Redis Cluster (6 节点)：

```bash
# 连接字符串示例 (Sentinel 模式)
REDIS_URL=redis+sentinel://sentinel-0:26379,sentinel-1:26379,sentinel-2:26379/aegis-master/0
```

---

## 项目结构

```
AegisRouter/
├── aegis_router/                     # 主应用包
│   ├── callbacks/                    # LiteLLM Custom Callbacks
│   │   ├── smart_router.py           # 主回调 (pre_call + post_call + 路由)
│   │   ├── stream_rehydrator.py      # 流式占位符还原引擎
│   │   ├── degradation.py            # 降级策略管理
│   │   └── uds_pool.py               # UDS 连接池
│   ├── clawvault/                    # ClawVault 安全伴生进程
│   │   ├── server.py                 # UDS Server (JSON-RPC 2.0)
│   │   ├── masker.py                 # PII 脱敏 (Presidio)
│   │   ├── restorer.py               # 占位符还原
│   │   ├── compliance.py             # 合规检测引擎
│   │   └── recognizers/              # 自定义 Presidio Recognizer
│   │       ├── chinese_phone.py       # 中国手机号
│   │       ├── chinese_id_card.py     # 中国身份证号
│   │       └── chinese_name.py        # 中文人名
│   ├── router/                       # 智能路由模块
│   │   ├── rule_engine.py            # 规则前置引擎 (寒暄检测)
│   │   ├── model_classifier.py       # RouteLLM 推理封装
│   │   ├── model_scorer.py           # 模型能力评分
│   │   ├── route_resolver.py         # 区间匹配 + 重叠策略
│   │   └── config_watcher.py         # 配置热更新 (watchdog)
│   ├── storage/                      # Redis 存储层
│   │   └── redis_client.py           # 异步 Redis 封装
│   ├── observability/                # 可观测性
│   │   ├── audit_logger.py           # 审计日志
│   │   └── metrics.py                # 指标收集
│   ├── config.py                     # 全局配置管理
│   └── health.py                     # 健康检查端点
│
├── config/                           # 配置文件
│   ├── config.yaml                   # LiteLLM 模型池 + Failover
│   ├── models.yaml                   # 模型能力参数
│   ├── route_config.yaml             # 路由权重、策略、阈值
│   ├── route_overrides.yaml          # 人工覆盖的模型区间
│   └── compliance_rules/             # 合规规则
│       ├── injection_patterns.yaml    # 注入攻击模式
│       └── sensitive_words.txt        # 敏感词库
│
├── patterns/                         # 规则匹配文件
│   └── trivial_chat.txt              # 寒暄词库
│
├── tests/                            # 测试套件
├── k8s/                              # Kubernetes 部署配置
├── Dockerfile                        # 容器镜像定义
├── supervisord.conf                  # Supervisor 进程管理
├── pyproject.toml                    # Python 项目配置
├── requirements.txt                  # 第三方依赖
└── .env.example                      # 环境变量模板
```

---

## 常见问题

### Q: 是否需要修改现有代码才能接入 AegisRouter？

不需要。AegisRouter 完全兼容 OpenAI SDK 的 `v1/chat/completions` API，只需将 `base_url` 指向 AegisRouter 地址即可。所有 PII 脱敏、路由决策、合规检测对调用方完全透明。

### Q: PII 脱敏会影响 LLM 的回答质量吗？

影响极小。占位符保持了语义结构（如 `[PERSON_1]说...` vs `张三说...`），LLM 仍能理解上下文关系。RouteLLM 打分默认使用脱敏后文本（`score_input: masked`），经测试对路由精度的影响 < 2%。

### Q: Redis 挂了会怎样？

Redis 不可用时，AegisRouter 会拒绝需要脱敏的请求（返回 HTTP 503），避免 PII 明文泄露到 LLM。生产环境建议使用 Redis Sentinel 或 Cluster 保证高可用。

### Q: 如何添加新的 LLM 模型？

1. 在 `config/config.yaml` 的 `model_list` 中添加模型配置
2. 在 `config/models.yaml` 中声明模型能力参数
3. (可选) 在 `config/route_overrides.yaml` 中手动覆盖评分区间
4. 系统自动检测配置变更并重新计算路由表，无需重启

### Q: 如何关闭某个安全功能？

- 关闭 PII 脱敏: 设置 `SKIP_PII_MASKING=true` (仅限开发环境)
- 关闭合规检测: 设置 `SKIP_COMPLIANCE_CHECK=true` (仅限开发环境)
- 宽松模式: 设置 `COMPLIANCE_MODE=permissive` (记录但不拦截)

### Q: 如何自定义敏感词库？

编辑 `config/compliance_rules/sensitive_words.txt`，每行一个敏感词，支持中英文混合。保存后系统自动热加载，无需重启。

### Q: 本地开发不想启动 Redis 怎么办？

设置 `SKIP_PII_MASKING=true` 可以跳过脱敏流程（不需要 Redis）。但注意此模式下 PII 数据会明文发送到 LLM，仅限本地开发使用。

### Q: 日志太多如何控制？

- 调整 `LOG_LEVEL=WARNING` 只输出警告以上级别
- 审计日志始终记录路由决策，通过 `AUDIT_LOG_OUTPUT` 控制输出方式

### Q: 如何监控路由决策分布？

审计日志中包含每次路由决策的完整信息（目标模型、难度分数、候选列表），可通过 ELK/Grafana Loki 聚合分析，了解各模型的流量分布和成本构成。

---

## 开发指南

### 运行测试

```bash
# 运行全部测试
pytest

# 带覆盖率
pytest --cov=aegis_router --cov-report=html

# 运行特定模块测试
pytest tests/test_masker.py
pytest tests/test_router.py
```

### 代码风格

```bash
# 格式化
black aegis_router/ tests/

# Lint
ruff check aegis_router/ tests/

# 类型检查
mypy aegis_router/
```

---

## 许可证

Proprietary — 未经授权不得分发或修改。

第三方组件许可证详见 [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md)。
