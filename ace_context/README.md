# ace_context — ACE 核心实现（Task A）

training-free 的可逆长轨迹上下文编排，对应论文 *ACE: Pluggable Adaptive Context Elasticizer across Agents*（arXiv:2606.31564）在 AlphaApollo 中的实现。

## 包结构

```
ace_context/
├── llm.py          # LLM 客户端抽象：OpenAICompatClient (vLLM/SGLang) + MockClient (离线测试)
├── models.py       # 数据模型：StepRecord / Abstraction / Decision / AuditEntry
├── store.py        # A1 无损消息维护：raw 永不覆盖，abstraction 缓存自动失效，JSONL 持久化
├── abstraction.py  # A2 结构化摘要：LLM + 确定性 fallback
├── orchestrator.py # A3 弹性编排：LLM 给 relevance 分数，代码层裁剪；last-step invariant
├── packing.py      # A4 token 装箱：优先级贪心填充，真实 tokenizer 可插拔
├── policy.py       # 统一入口 ContextPolicy：full | truncate | summarize | elastic + 审计日志 + 离线 replay
└── tests/
    └── test_ace.py # A5 必测性质，10 项，零 GPU 零网络运行
```

## 快速开始

```bash
# 本地跑测试（无需 GPU / 网络 / 模型）
python tests/test_ace.py

# 本地跑一次完整 elastic 流程（mock）
python examples/quick_demo.py   # （见下）
```

## 接入 AlphaApollo（伪代码）

在每次 generation call 前：

```python
from ace_context import ContextPolicy, OpenAICompatClient

llm = OpenAICompatClient(
    base_url="http://localhost:8000/v1",   # vLLM / SGLang endpoint
    model="Qwen/Qwen2.5-3B-Instruct",
)
policy = ContextPolicy(
    mode="elastic",            # full | truncate | summarize | elastic
    max_input_tokens=4096,
    reserve_output_tokens=512,
    keep_last_raw=True,
    llm=llm,
)

# 每个新观测到达时
policy.record_step(trajectory_id, step_id, role, raw_messages, token_count)

# 每次生成前
messages, audit = policy.build(task, system_prompt, tool_schema, decision_step=step_id)
# messages -> 传给主 LLM；audit -> 写入审计日志
```

## 关键不变量（A5，已测试）

1. raw store 在任意编排后 byte-identical
2. 同一旧 step 可经历 RAW → DROP → RAW（raw 永远可恢复）
3. 最新一步始终 RAW
4. 构造视图不超 token budget
5. abstraction 缺失不丢最新状态
6. 无效 LLM JSON 触发相同确定性 fallback
7. full 模式与原版行为一致
8. save / load / replay 生成相同 context view

## AlphaApollo Task B integration

`TrajectoryCollector` now owns one `BatchContextRuntime` per rollout. In non-`full`
modes it records the raw action/tool-observation pair after `envs.step()`, then
builds a policy-selected `memory_context` immediately before the next main
`generate_sequences()` call. Environment memory, rewards, tools, and the normal
trajectory logger continue to receive raw values.

The abstraction and orchestration phases are submitted serially to the same
AlphaApollo rollout worker with greedy decoding. Every generated trajectory
step contains a `context_audit` object; optional `audit_dir` and `store_dir`
also persist JSONL files.

Example:

```bash
python -m alphaapollo.workflows.test \
  --config examples/configs/test_informal_math.yaml \
  context_policy.mode=elastic \
  context_policy.orchestration=llm \
  context_policy.max_input_tokens=4096 \
  context_policy.reserve_output_tokens=512 \
  context_policy.audit_dir=./outputs/ace/audit \
  context_policy.store_dir=./outputs/ace/store
```

Offline integrity replay does not initialize an environment or execute tools:

```bash
python -m ace_context.replay \
  --store outputs/ace/store/trajectories.jsonl \
  --audit outputs/ace/audit/context_audit.jsonl \
  --config examples/configs/test_informal_math.yaml \
  --output outputs/ace/replayed_views.jsonl
```
