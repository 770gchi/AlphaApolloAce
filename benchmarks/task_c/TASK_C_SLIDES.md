# ACE × AlphaApollo：可逆长轨迹上下文编排

> 10 分钟 slide 草稿。建议 9–10 页，以图表和流程图为主；括号中的内容可作为讲述提示，不必全部放到页面上。

---

## 1. 问题：固定预算下的长轨迹记忆（1 分钟）

### Slide 1 — 为什么 Full 与固定截断不够

长 horizon agent 会持续累积：

- 用户目标与约束；
- assistant action；
- tool/environment observation；
- 修正信息和失败记录。

当历史达到 8K–16K tokens，而模型输入预算只有 2K/4K 时：

- **Full**：依赖 backend 截断，关键早期证据可能被直接裁掉；
- **Truncate**：只保留最近窗口，无法处理 delayed evidence；
- **Summarize**：节省 token，但摘要一旦遗漏精确数值或 source，信息不可逆丢失。

核心问题：

> 能否在固定 token budget 下动态压缩上下文，同时保留将旧步骤恢复为原文的能力？

**讲述提示：** 这里的难点不是“把历史变短”，而是后期目标变化后，早期证据可能重新相关。固定窗口不知道未来，单一滚动摘要又可能丢掉精确细节。

---

## 2. 方法：ACE 的核心分离（约 1 分钟）

### Slide 2 — Raw truth 与 Context view 分离

```text
Environment / Tools
        │
        ▼
Lossless raw store ───────► reward / done / trajectory logger
        │
        ▼
Abstraction cache
        │
        ▼
Per-turn orchestration: RAW / ABSTRACT / DROP
        │
        ▼
Tokenizer-aware packing
        │
        ▼
Model-visible context view
```

- **Lossless store**：原始 action 和 observation 永不被摘要覆盖；
- **Abstraction**：为旧 step 保存可缓存的结构化压缩表示；
- **Elastic orchestration**：每轮重新选择每个历史 step 的表示；
- **Context view**：只改变送入模型的 prompt，不修改环境真实状态。

关键能力是可逆性：

```text
step 3: RAW → step 8: DROP → step 14: RAW
```

被临时丢弃的证据仍可从 lossless store 恢复。

---

## 3. 方法：必须守住的不变量（约 1 分钟）

### Slide 3 — Correctness before compression

1. **Raw truth invariant**  
   工具执行、reward、done 和完整 trajectory 始终读取原始数据。

2. **Latest-step invariant**  
   默认最近一个完整 action–observation block 必须保持 RAW。

3. **Budget invariant**  
   使用 Qwen tokenizer 对最终 chat-template prompt 精确计数：

   ```text
   actual_prompt_tokens <= prompt_budget
   actual_prompt_tokens + 512 <= 32768
   ```

4. **Full compatibility invariant**  
   `mode=full` 保持 AlphaApollo 原生消息与 backend 行为。

5. **Deterministic failure invariant**  
   非法 JSON、未知 step、缺失 abstraction 或管理调用失败时，进入确定性 heuristic fallback。

6. **Replay invariant**  
   store + audit 可以离线重建每一步 context view，并验证 message hash。

**讲述提示：** ACE 不是新的 memory truth source。它只是决定“这一次给模型看什么”。

---

## 4. 方法：每轮如何构造上下文（约 1 分钟）

### Slide 4 — Abstraction → Orchestration → Packing

每个 generation turn 串行执行：

1. 对 abstraction cache miss 的历史 step 批量生成摘要；
2. LLM 或 heuristic 为历史 step 选择 `RAW / ABSTRACT / DROP`；
3. 代码验证 step ID、类型和 latest-step invariant；
4. tokenizer-aware packer 按优先级裁剪；
5. 执行正式 main generation。

Packing 优先级：

```text
固定模板与当前问题
  > latest raw
  > 高相关 raw evidence
  > 高相关 abstraction
  > 低相关历史
```

同一个 Qwen2.5-3B-Instruct worker 串行承担摘要、编排和主生成：

- 不加载第二份模型；
- 不训练或修改权重；
- 管理 generation 不进入 environment、reward 或正式 response。

---

## 5. AlphaApollo 接线（2 分钟）

### Slide 5 — 只接入现有 rollout 主循环

主要集成点：

```text
envs.reset()
    └─ initialize trajectory-scoped ACE state

generate_sequences() 前
    └─ build context view for active trajectories

envs.step(action) 后
    └─ record raw action + raw observation

done / padding
    └─ no management request, no new context step
```

实现原则：

- 不复制 rollout engine；
- `EnvironmentManager.memory` 保持完整历史；
- `envs.step(text_actions)`、reward 和 success evaluator 语义不变；
- 仅替换 informal-math/delayed-evidence prompt 中的 memory view；
- batch 内按 `trajectory_id` 隔离 store、cache、decision 和 audit。

支持四种统一 policy：

| Policy | 行为 |
|---|---|
| Full | AlphaApollo 原生路径 |
| Truncate | 保留预算内最近完整 steps |
| Summarize | 旧历史滚动摘要，latest raw |
| Elastic | 每轮重新选择 RAW / ABSTRACT / DROP |

---

## 6. AlphaApollo 审计与可复现性（约 1 分钟）

### Slide 6 — 每一次压缩决策都可追踪

每个 decision audit 记录：

- trajectory ID、decision step 和 mode；
- budget、实际 prompt tokens、output reserve；
- 每个历史 step 的 type、score、reason 和 source hash；
- raw/abstract/drop token 数；
- abstraction/orchestration token、延迟、cache hit 和失败状态；
- fallback 原因；
- 最终 observation/message hash。

实验完整性检查：

- 12 个配置 × 30 条任务 = **360 trajectories**；
- 每条 15 个 rollout records，共 **5400 steps**；
- 所有配置 `exit_code=0`，无缺失 trajectory；
- 各策略收到的 scripted raw observations **byte-identical**；
- 所有非 Full prompt 均未超过 2048/4096 tokens；
- 非 Full 策略审计覆盖率为 **450/450 records per run**。

---

## 7. 实验设置（约 1 分钟）

### Slide 7 — Delayed-evidence Stress Benchmark

冻结 benchmark：

```text
5 categories × 2 length tiers × 3 cases = 30 tasks
```

类别：

1. exact-number；
2. source-sensitive；
3. correction；
4. tool-error-recovery；
5. hard-negative。

长度与压力：

| Tier | Full prompt | 相对 2K | 相对 4K |
|---|---:|---:|---:|
| Short | 8K–12K | 约 5× | 约 2.5× |
| Long | 12K–16K | 约 7× | 约 3.5× |

统一配置：

- Qwen2.5-3B-Instruct，BF16；
- temperature 0，seed 0，单样本；
- prompt budget 2048 / 4096；
- output reserve 512；
- 同一批 30 条 deterministic trajectories；
- Full-20K 作为不参与主排名的可解性 Oracle。

---

## 8. C2 主结果（约 1 分钟）

### Slide 8 — 压缩策略显著优于 backend 截断

| Policy | 2K accuracy | 4K accuracy |
|---|---:|---:|
| Full | 0.0% | 0.0% |
| Truncate | 0.0% | 30.0% |
| Summarize | **86.7%** | **80.0%** |
| Elastic-LLM | 73.3% | 70.0% |
| Full-20K Oracle | `93.3%` | `93.3%` |

Oracle 答错 2 条 hard-negative。仅统计 Oracle 可解的 28 条时：

| Policy | 2K | 4K |
|---|---:|---:|
| Full | 0.0% | 0.0% |
| Truncate | 0.0% | 32.1% |
| Summarize | **85.7%** | **78.6%** |
| Elastic-LLM | 75.0% | 71.4% |

观察：

- Full 在 2K/4K 下 30/30 都触发输入截断，并全部答错；
- Truncate-4K 能解决部分任务，但不能恢复早期证据；
- Elastic recovery rate：2K 为 96.7%，4K 为 100%；
- 当前 benchmark 上 **Summarize 高于 Elastic**，不能宣称 ACE 是最佳 accuracy policy；
- 但 Elastic 展示了任务要求的动态、可逆 context semantics。

---

## 9. C3 消融与归因（约 1 分钟）

### Slide 9 — 哪些设计真的重要

4K、全部 30 条：

| Configuration | Accuracy | Recovery |
|---|---:|---:|
| Elastic-LLM | **70.0%** | 100% |
| `keep_last_raw=false` | **0.0%** | 100% |
| RAW/DROP only | 46.7% | 96.7% |
| Heuristic orchestration | 46.7% | 100% |

Oracle 可解 28 条上的准确率分别为：

- Elastic-LLM：71.4%；
- no-last-raw：0.0%；
- RAW/DROP only：42.9%；
- heuristic：50.0%。

结论：

1. **Latest raw 是必要安全约束**：关闭后即使旧 evidence 被“恢复”，模型仍无法正确理解当前目标；
2. **ABSTRACT 有价值**：去掉 ABSTRACT 后下降约 23 个百分点；
3. **LLM orchestration 优于 heuristic**：在同一 4K budget 下提高约 21 个百分点；
4. **Recovery rate 不能单独代表成功**：no-last-raw 的 recovery 为 100%，但 accuracy 为 0%。

管理成本：

- Elastic-LLM 每个完整 run 约产生 1.06M management tokens；
- Summarize 约 0.50M；
- 管理成本明显，必须与 accuracy 和可逆性一起报告。

---

## 10. 局限与下一步（1 分钟）

### Slide 10 — What worked, what remains

当前局限：

- benchmark 是 deterministic scripted environment，不等同于开放式真实工具任务；
- 只有 30 条任务、一个 3B 模型和一个 seed，统计规模有限；
- Summarize 在本 benchmark 上优于 Elastic，说明当前 orchestrator 仍可能过度 drop、选择失误或产生摘要噪声；
- LLM management token 成本很高，尚未优化 cache/prefix reuse；
- recovery validator 只检查必要字段是否恢复，不能完全衡量模型是否正确使用该证据；
- 所有正式配置已统一 batch size 30；manifests 记录 wall time 和 GPU sampler，但只有单次运行且包含启动成本；
- 三个 Full 配置峰值约 18.5–18.8 GiB，尚未报告多次运行方差；
- timeline 绘图脚本已完成，并生成 Elastic 成功恢复与 no-last-raw 失败案例 PNG。

下一步：

1. 重复运行 latency/peak-memory 并报告均值与方差；
2. 增加 dependency-aware orchestration，显式追踪 source 与 tool dependency；
3. 降低管理调用频率，并量化 abstraction cache 收益；
4. 扩展到真实 AlphaApollo tool-use 问题；
5. 对 hard-negative 和错误摘要做定向 case study。

---

## 一句话结论

> ACE 已作为 training-free、batch-aware、可审计且可逆的 context policy 接入 AlphaApollo；实验验证了预算与 raw-truth 不变量，也证明 latest raw、abstraction 和 LLM orchestration 都有实际作用，但当前实现的管理成本较高，且 accuracy 尚未超过更简单的滚动摘要 baseline。

---

## Slide 制作建议

建议配图：

1. Slide 2：raw truth / context view 双通路架构图；
2. Slide 4：`RAW → DROP → RAW` 时间线示意；
3. Slide 8：2K/4K accuracy 分组柱状图；
4. Slide 9：C3 消融柱状图；
5. 附录：成功恢复与 no-last-raw 失败案例的 context timeline。

数据来源：

- `outputs/task_c/metrics/overall.csv`
- `outputs/task_c/metrics/combined_summary.csv`
- `outputs/task_c/metrics/combined_per_case.csv`
- `outputs/task_c/audit/`
- `outputs/task_c/manifests/`
