# 智能安全 LLM 路由网关需求设计文档 (Smart Secure LLM Router Gateway)

## 1. 业务背景与市场痛点 (Background & Pain Points)
随着生成式人工智能 (GenAI) 在企业内部的深度应用，多大语言模型 (Multi-LLM) 混合架构已成为必然趋势。然而，企业在落地大模型应用上面临着三个核心痛点：
* **隐私数据泄露风险：** 员工在向第三方公共大模型发送请求时，极易无意间夹带个人身份信息 (PII，如手机号、身份证、邮箱) 以及企业核心机密（代码、财务报表、未发布项目代号），这违反了 GDPR、CCPA 以及国内数据安全法规。
* **高昂的 Token 费用账单：** 绝大多数日常简单任务（如文本分类、翻译、常规摘要）无需调用最贵的前沿模型（如 GPT-4o-Pro、o1-Pro），但缺乏智能化手段在入口处做低成本分流，导致企业在大模型算力上产生严重资金浪费。
* **碎片化的 API 适配与高可用灾备：** 不同的 LLM 供应商标准不一，且第三方 API 极易发生网络超时、限流 (Rate Limit 429) 或服务崩溃。业务层代码如果直接对接模型 API，将导致灾备恢复逻辑异常繁重。

> **产品定位**
> 本项目致力于打造一款**具备“高防线”与“高智商”的企业大模型安全路由中间件**。通过在 API 入口层拦截，先实现本地隐私脱敏保护，再通过轻量级算法预测 Prompt 复杂度并进行最优性价比分流，最后交由统一网关底座执行高可用调度与结果还原，实现**安全合规、极速响应、Token 消耗最合理**。

---

## 2. 系统核心功能与非功能需求 (Product Requirements)

### 2.1 功能性需求 (FR)
| 模块名称 | 核心子需求 | 详细描述 |
| :--- | :--- | :--- |
| **1. 网关物理接入层** | 统一 OpenAI 格式适配 | 对外暴露完全兼容 OpenAI SDK 的统一 `v1/chat/completions` API 端点，支持流式传输 (Streaming)。 |
| **2. 隐私脱敏层** | 双向脱敏与数据还原 | 对输入的 PII 数据实施强脱敏（用占位符替代），建立并维持安全映射表（Redis），在响应返回前无感重填（Rehydration）。 |
| **3. 智能路由决策层** | 动态复杂度分类判定 | 利用分类器（15ms 内）对脱敏后 Prompt 评分，智能重写请求参数。复杂任务分发给强模型，反射级任务分发给廉价/弱模型。 |
| **4. 安全合规拦截层** | 双向安全护栏 | 在发送前和返回后进行越狱防御、敏感词过滤，拒绝非法、有害意图的 Prompt 发送至云端。 |
| **5. 灾备容错层** | 无感降级与负载均衡 | 自动捕获 429 等报错，并在 50ms 内将请求漂移到可用候选模型，保证高可用。 |

### 2.2 非功能性需求 (NFR)
* **延迟开销 (Latency)：** 隐私检测与智能路由算法在本地运行，其带来的网关层额外开销必须控制在 $T \le 20\text{ms}$（不含 LLM API 网络耗时）。
* **吞吐量 (Throughput)：** 网关作为统一入口，需支撑单实例至少 $1000+\text{ QPS}$ 的并发代理。
* **数据生命周期安全：** Redis 中的隐私映射表必须设置严格的 TTL（例如 30 分钟），到期物理擦除，网关本地不持久化任何用户明文数据。

---

## 3. 物理拓扑与架构分层设计 (Architecture & Pipelines)

网关采用**“先到网关作为物理入口，在网关生命周期管道内顺序执行插件”**的拓扑设计。下面展示完整的请求生命周期链条：
[ 客户端请求 Client ]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│             智能安全路由网关 (唯一物理入口)              │
│                                                        │
│  【第一步：LiteLLM 基础网关接收】                        │
│   - 网关接收 HTTP 请求 (API Gateway Base)               │
│   - 进行基础鉴权 (Auth) 与流量控制 (Rate Limit)          │
│   - 统一接口格式 (OpenAI Compatible)                    │
│       │                                                │
│       ▼                                                │
│  【第二步：ClawVault 隐私脱敏插件】                     │
│   - 拦截 Prompt，扫描并提取其中的 PII 隐私数据            │
│   - 将真实数据存入 Redis，用占位符 (如 [PERSON_1]) 替换   │
│       │                                                │
│       ▼                                                │
│  【第三步：智能路由决策插件 (RouteLLM)】                │
│   - 评估脱敏后 Prompt 的复杂度与语言                    │
│   - 算法决策：判定该任务是用 Weak 还是 Strong 模型        │
│   - 动态改写请求的目标模型参数 (如重写为 deepseek/gpt-4o)  │
│       │                                                │
│       ▼                                                │
│  【第四步：LiteLLM 转发与合规】                         │
│   - 执行内容合规检测 (Guardrails)                       │
│   - 将请求发送至最终选定的底层大模型                     │
│   - 若遇到 429/503 错误，网关执行自动漂移灾备 (Failover)  │
│       │                                                │
│       ▼                                                │
│  【第五步：双向还原引擎 (Rehydration)】                 │
│   - 拦截大模型返回的 Response                           │
│   - 从 Redis 读取映射，将占位符还原为真实隐私数据并返回   │
└────────────────────────────────────────────────────────┘

---

## 4. 开源矩阵与融合粘合策略 (Open-Source Integration)

为了缩短研发周期并降低系统维护成本，本项目采用“强强联合”的组装架构，利用三个主流开源项目各展所长：
* **骨架与底座 (LiteLLM)：** 负责网络层。提供极速基于 Python 编写的 FastAPI 代理服务，实现 OpenAI 格式的动态封装、熔断灾备以及 Custom Callbacks（自定义生命周期钩子）。
* **隐私脱敏层 (Microsoft Presidio / ClawVault 理念)：** 负责隐私脱敏。利用 Presidio 内置的高精度多语言命名实体识别 (NER) 模型，完成人名、手机号、IP地址的定位及敏感度降维。
* **智能大脑层 (RouteLLM / LLMRouter)：** 负责智能调度。内置经过微调的 BERT 及矩阵分解模型，对脱敏后的 Prompt 计算难度分数，并在强/弱模型配对（如 Claude 3.5 vs DeepSeek）中动态寻找最优解。

---

## 5. 落地指南与部署方案 (Implementation Blueprint)

### 第一步：配置统一大模型池 & 挂载回调钩子
创建 `config.yaml` 配置文件，声明后端接入的异构模型，并声明加载的自定义中间件插件：

```yaml
model_list:
  - model_name: gpt-4o  # 强模型/高成本模型候选
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: deepseek-chat  # 弱模型/高性价比模型候选
    litellm_params:
      model: deepseek/deepseek-chat
      api_key: os.environ/DEEPSEEK_API_KEY
      api_base: [https://api.deepseek.com](https://api.deepseek.com)

general_settings:
  # 注册我们下面编写的自定义插件包，挂载在 LiteLLM 请求生命周期中
  custom_extension_package: custom_callbacks
‘’‘

###  第二步：编写生命周期插件逻辑 (融合粘合剂)
在同级目录下编写 custom_callbacks.py，实现前置、后置两阶段的无缝嵌入逻辑：

Python
import litellm
from litellm.integrations.custom_logger import CustomLogger
import uuid

# 引入我们内部的隐私、智能分流插件逻辑
from vault_plugin import mask_private_data, restore_private_data
from router_plugin import predict_best_model

class SmartSecureRouter(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict, cache, start_time, response_obj, call_types, model, prompt, kwargs
    ):
        request_id = kwargs.get("litellm_call_id", str(uuid.uuid4()))
        original_prompt = prompt
        
        # 1. 物理接收后，首先进行隐私安全隔离
        safe_prompt = mask_private_data(original_prompt, request_id)
        
        # 2. 接着进行智能路由决策
        model_tier = predict_best_model(safe_prompt)
        target_model = "gpt-4o" if model_tier == "strong" else "deepseek-chat"
        
        # 3. 参数重写：改变实际发送的模型和已脱敏的 Prompt
        kwargs["model"] = target_model
        kwargs["messages"] = [{"role": "user", "content": safe_prompt}]
        
        return kwargs

    async def async_post_call_success_hook(
        self, user_api_key_dict, response_obj, start_time, end_time, cache
    ):
        request_id = response_obj.get("id", "default_id")
        raw_output = response_obj["choices"][0]["message"]["content"]
        
        # 4. 逆向脱敏重填：保障用户见到的数据完整性
        real_output = restore_private_data(raw_output, request_id)
        response_obj["choices"][0]["message"]["content"] = real_output
        return response_obj

# 注册单例实例
smart_router_instance = SmartSecureRouter()
litellm.callbacks = [smart_router_instance]

### 第三步：极速启动服务
在控制台配置好您的 API Key，并在包含上述代码和 config.yaml 的目录下运行以下命令启动：

Bash
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"

# 启动 LiteLLM Proxy 服务并加载配置
litellm --config config.yaml --port 8000


一、 物理拓扑设计：基于 Unix Socket 的无感伴生
在方案一中，网络通信由 TCP 变更为 Unix 域套接字（Unix Domain Socket, UDS）。由于不经过网络协议栈，两者的通信开销等同于内存拷贝。

ClawVault（伴生进程）：在启动时，不再监听 127.0.0.1:8081，而是创建并监听一个本地套接字文件（例如 /var/run/clawvault.sock）。

LiteLLM 插件（通过 Custom Callbacks 运行）：通过这个套接字文件与 ClawVault 进行超高速的 IPC 进程间通信。

Supervisor（进程守护器）：作为容器主进程，负责在启动时依次拉起 ClawVault 和 LiteLLM，并监控它们的健康状态。