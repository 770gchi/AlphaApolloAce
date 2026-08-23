# ACE × AlphaApollo 实现详解与答辩备忘录

> 用途：解释本实验做了什么、AlphaApollo 原本有什么能力、修改后增加了什么、为什么这样设计，并准备可能的技术追问。内容以当前仓库实际实现和实验结果为准。

---

## 1. 一句话概括

AlphaApollo 原本已经具备 multi-turn rollout、环境 memory、工具执行、reward 和批量 vLLM generation。本项目没有重写这些组件，而是在每次主生成前增加一个 batch-aware ACE context policy：从独立的 lossless store 中为历史步骤选择 RAW、ABSTRACT 或 DROP，并用真实 Qwen tokenizer 在 2K/4K budget 内构造模型可见视图；环境、工具、reward 和完整日志始终保留 raw truth。

---

# 一、AlphaApollo 原本实现的功能

## 2. 原生 multi-turn rollout

原始流程：

```text
dataset prompt
  → EnvironmentManager.reset()
  → obs["text"]
  → preprocess_batch()
  → chat template + tokenizer
  → actor_rollout_wg.generate_sequences()
  → decode assistant action
  → EnvironmentManager.step(action)
  → tool/environment observation + reward + done
  → repeat
```

原框架已有：

- 批量 trajectory rollout；
- vLLM/FSDP worker；
- environment reset/step；
- action projection 和工具执行；
- `EnvironmentManager.memory`；
- reward、done、success evaluator；
- tokenizer、padding、position IDs；
- trajectory history 和最终输出保存。

因此 ACE 的接入原则是复用唯一主循环，而不是复制第二套 rollout engine。

## 3. EnvironmentManager 原有职责

以 informal-math 为例，它负责：

1. 保存 task；
2. 接收模型 action；
3. 将 action 转成工具可执行形式；
4. 执行 Python/RAG 等工具；
5. 获得下一条 observation；
6. 把 action/observation 写入 environment memory；
7. 将 task、工具说明和 memory 渲染成下一轮 user observation；
8. 返回 reward、done 和 info。

AlphaApollo 的环境历史通常不是独立的 OpenAI assistant/tool message 列表，而是被渲染进单个 user prompt 的 `memory_context`。这也是本项目选择“替换 memory view、保留原模板”的原因。

## 4. 原生 Full 的不足

原生路径近似为：

```text
完整 environment memory
  → prompt renderer
  → tokenizer/backend truncation
```

它没有：

- per-step RAW/ABSTRACT/DROP；
- abstraction cache；
- 被 DROP 的步骤重新恢复为 RAW 的能力；
- tokenizer-aware relevance packing；
- context policy 审计；
- lossless replay。

所以“环境保存了完整历史”并不意味着“模型在 2K/4K 输入中看到了完整历史”。

---

# 二、修改后的总体架构

## 5. Raw truth 与 Context view 分离

```text
model action
    │
    ▼
EnvironmentManager.step()
    │
    ├────────► EnvironmentManager.memory ──► tool/reward/done
    │
    └────────► ACE LosslessStore
                    │
                    ├─ abstraction cache
                    ├─ orchestration
                    └─ RAW / ABSTRACT / DROP
                              │
                              ▼
                    tokenizer-aware packing
                              │
                              ▼
                    model-visible context view
```

左侧保持 AlphaApollo 环境语义；右侧只决定本轮给模型看什么。ACE 不允许修改 observation truth、reward 或 ground truth。

## 6. 新增配置

```yaml
context_policy:
  mode: full
  max_input_tokens: 4096
  reserve_output_tokens: 512
  keep_last_raw: true
  abstraction_cache: true
  orchestration: llm
  truncate_window_steps: 4
  abstraction_max_tokens: 256
  orchestration_max_tokens: 512
  allowed_types: [raw, abstract, drop]
  model_context_window: 32768
  audit_dir: null
  store_dir: null
```

| Mode | 行为 |
|---|---|
| Full | 完全走 AlphaApollo 原生路径 |
| Truncate | 最近完整 steps 为 RAW，旧步骤 DROP |
| Summarize | latest RAW，旧步骤使用 abstraction |
| Elastic | 每轮重新选择 RAW/ABSTRACT/DROP |

每个 rollout batch 创建独立 `traj_uid` 并 reset runtime；done/padding 样本不再产生管理请求。

## 7. 为什么必须 batch-aware

不同 trajectory 的 step ID 都从 1 开始。若只按 step ID 存储，A-step-1 和 B-step-1 会覆盖或串轨迹。

实际索引为：

```text
(trajectory_id, step_id)
```

store、abstraction cache、orchestration decision、audit 都按 trajectory 隔离。

---

# 三、ACE 核心模块

## 8. LosslessStore

每个 `StepRecord` 保存：

- trajectory ID、step ID；
- role/type；
- 原始 assistant action；
- 原始 tool/environment observation；
- metadata、timestamp；
- source hash、token count；
- abstraction；
- schema version。

### 为什么已有 EnvironmentManager.memory 还要 store？

| Environment memory | ACE store |
|---|---|
| 服务环境和工具流程 | 服务 context policy |
| 原框架生命周期 | 可持久化和 replay |
| 不应被 ACE 修改 | 可附加摘要、hash、decision metadata |

两者内容有重叠，但所有权不同。用 ACE store 覆盖 environment memory 会污染工具执行和 reward。

### 如何保证 raw 不被摘要覆盖？

- raw 和 abstraction 是不同字段；
- append/get 使用 defensive copy；
- `put_abstraction()` 只写摘要字段；
- source hash 改变时旧摘要失效；
- JSONL 按 trajectory/step 稳定排序；
- 临时文件、fsync、atomic rename。

## 9. 为什么一个 step 保存 action + observation

完整交互单元是：

```text
assistant action → tool/environment observation
```

只保存 observation 会丢失工具名、输入、失败尝试与因果对应。tool-error-recovery 类任务尤其需要同时恢复“工具 + 输入 + 错误码”。

## 10. Abstraction

逐 step 结构化摘要包含：

- summary；
- facts；
- variables；
- tool calls；
- open subgoals；
- source step/hash；
- generator/version/token count。

逐 step 而非单一滚动摘要的原因：

1. 单步缓存；
2. 单步选择；
3. 单步 RAW ↔ ABSTRACT 可逆；
4. source 可追踪；
5. 某一步摘要失败不会污染整个历史。

无效 JSON、缺字段、空摘要、worker 异常或 cache stale 时，使用 deterministic fallback abstraction。

## 11. Elastic orchestration

每轮为全部历史选择：

- `RAW`：原始 action/observation；
- `ABSTRACT`：结构化摘要；
- `DROP`：本轮视图不包含。

LLM 提出 type、score 和 reason；代码验证：

- step ID 是否存在；
- 是否重复；
- type 是否允许；
- latest step 是否违规；
- abstraction 是否有效；
- 最终是否满足 budget。

不能让 LLM 自己做 token arithmetic，因为 chat template、special tokens 和 tokenizer 分词无法由自然语言可靠精算。

## 12. Latest-step invariant

默认最近一个完整 action–observation block 必须 RAW。它往往包含最新工具结果、错误、target switch 或 FINAL QUERY。

C3 结果：

| 配置 | 4K accuracy |
|---|---:|
| Elastic-LLM，保留 latest raw | 70.0% |
| `keep_last_raw=false` | 0.0% |

no-last-raw 的旧 evidence recovery 仍可达到 100%，但模型缺失当前目标而全部答错。这证明 recovery rate 必须与 current-state integrity 和 accuracy 联合解释。

## 13. Token-aware packing

实验语义：

```text
effective_prompt_budget
= min(data.max_prompt_length, context_policy.max_input_tokens)

actual_prompt_tokens <= effective_prompt_budget

actual_prompt_tokens + 512 <= 32768
```

2K/4K 已经是完整 prompt 上限，512 是额外 output reserve，不能再从 2K/4K 中重复扣除。

计数使用 Qwen tokenizer 对最终 chat-template prompt 计算，包含 system、task、tool instruction、history、special tokens 和 generation prompt。

Packing 优先级：

1. 固定模板和当前问题；
2. latest raw；
3. 高分 raw；
4. 高分 abstraction；
5. 低分历史。

固定 prompt 或 latest raw 自身超预算时抛出 `BudgetExceededError`，不允许 backend 静默截断。

---

# 四、AlphaApollo 接线改动

## 14. rollout_loop.py

在 `TrajectoryCollector` 中新增 batch runtime。

每轮主生成前：

1. 找 active trajectory IDs；
2. 批量准备 abstraction cache-miss 请求；
3. 同 worker 执行 abstraction；
4. 准备 orchestration 请求；
5. 同 worker执行 orchestration；
6. build views；
7. 仅替换 `generation_obs["text"]`；
8. 原有 `preprocess_batch → generate_sequences` 继续执行。

环境 step 后：

1. 读取模型原始 action；
2. 从 `next_obs["anchor"]` 读取 raw observation；
3. 调用 `record_interaction()`；
4. done/padding trajectory 不再追加记录。

## 15. render_context_view() 为什么必要

ACE 必须使用原环境 prompt template，只替换 `memory_context`。否则直接增加 Task 字段或伪造 assistant/tool roles 会改变 AlphaApollo 原提示结构，baseline 不再公平。

新增接口：

```python
render_context_view(index, memory_context, step_count)
```

原 `obs` 保留；另建 `generation_obs` 给 tokenizer/main generation。env.step、reward 和 raw logger 继续使用原始数据。

## 16. anchor 字段

环境返回：

```python
{
  "text": rendered_model_observation,
  "anchor": raw_environment_observation
}
```

不能从 `text` 反推 raw observation，因为其中混有任务模板和历史。anchor 保证：

- ACE store 保存 byte-identical raw；
- source hash 有意义；
- replay 可验证；
- 不同 policy raw observation 可比较。

12 个实验配置的 raw observations 已检查为完全一致。

## 17. 同模型管理调用

顺序严格串行：

```text
abstraction generation
  → orchestration generation
  → main generation
```

三者调用同一个 `actor_rollout_wg.generate_sequences()`：

- 不加载第二份权重；
- temperature 0；
- 管理输出不进入环境；
- 不参与 reward；
- 不成为正式 trajectory response。

## 18. 为什么修改 vLLM rollout

原 worker 固定使用全局 `rollout.response_length=512`。abstraction/orchestration 需要独立的较小 max tokens。

新增从 `DataProto.meta_info["response_length"]` 读取临时 override，并同步控制：

- vLLM `max_tokens`；
- response padding；
- log-prob padding。

未传 override 时仍走原配置，所以 main generation 保持原行为。

## 19. Full mode 为什么旁路 ACE

`mode=full` 时 runtime disabled：

```text
obs → preprocess_batch → generate_sequences
```

不做摘要、编排或 view rewrite。原因是严格保证 ACE off 与原生 AlphaApollo 相同，不用 ACE “模拟 Full”。

代价：Full 没有 RAW/ABSTRACT/DROP decision audit；但 structured records 已通过只读 observer 记录 tokenizer 截断点的 pre/post tokens、方向和丢失 token 数，不改变 Full prompt。

## 20. main_generation.py

增加 `trajectory_records`：

- trajectory ID；
- step ID；
- response；
- raw observation；
- prompt/response tokens；
- context audit。

同时把原先依赖 NumPy ragged object array 的 transpose 改为按 sample/data index 显式列表转置。不同 trajectory step 数可能不同，显式转置更稳定。

---

# 五、Task C Benchmark

## 21. 为什么新增 delayed-evidence 环境

普通数学题不一定稳定触发“早期证据暂时无关、后期重新相关”，难以量化 recovery。

新增 deterministic 环境：

1. 早期 reveal evidence；
2. 中间加入相似干扰、旧值或工具记录；
3. 后期 target switch；
4. FINAL QUERY；
5. regex 严格评分。

scripted observation 不依赖中间 action，确保所有 policy 看到完全相同 raw trajectory。最终 action 仍由模型生成并严格评分。

## 22. 为什么 JSON 不是预写的多轮 user/assistant messages

JSON 保存环境脚本，而真实交互在运行时形成：

```text
environment/user observation
assistant model action
environment/user observation
assistant model action
...
```

如果预写 assistant messages，就无法衡量不同 context policy 对模型 action 的影响。AlphaApollo 的环境接口本身也是逐轮返回 observation，而不是读取固定 chat transcript。

## 23. 数据规模

```text
5 categories × 2 tiers × 3 cases = 30
```

类别：

1. exact-number；
2. source-sensitive；
3. correction；
4. tool-error-recovery；
5. hard-negative。

长度：

| Tier | Full prompt |
|---|---:|
| Short | 8K–12K |
| Long | 12K–16K |

相对 2K 是约 5–7 倍压力，相对 4K 是约 2.5–3.5 倍；同时低于 32768，允许用 Full-20K Oracle 检查可解性。

## 24. Recovery 定义

Eligible：关键证据在 reveal 与 target switch 之间至少一次变为 DROP 或 ABSTRACT。

Sufficient recovery：

- target switch 后恢复 RAW；或
- 恢复 ABSTRACT 且 category validator 确认必要字段完整。

source-sensitive 要同时恢复 source 和 value；tool-error-recovery 要恢复 tool、failure input 和 error code。

---

# 六、实验运行与结果

## 25. C2/C3 矩阵

C2：

| Policy | 2K | 4K |
|---|---:|---:|
| Full | ✓ | ✓ |
| Truncate | ✓ | ✓ |
| Summarize | ✓ | ✓ |
| Elastic-LLM | ✓ | ✓ |

C3，4K：

- `keep_last_raw=false`；
- `allowed_types=[raw,drop]`；
- `orchestration=heuristic`。

另有 Full-20K Oracle。

统一使用本地 Qwen2.5-3B-Instruct、BF16、temperature 0、seed 0、n=1、output reserve 512。

## 26. 运行中解决的问题

- 使用本地完整模型：`/root/autodl-tmp/models/Qwen2.5-3B-Instruct`；
- 用 `RAY_ADDRESS=local` 避免 Ray 旧地址/启动超时；
- runner 增加 `--batch-size 30`，减少六批重复开销；
- runner 增加 `--resume`，根据 manifest、raw 和 summary 跳过已完成项；
- 修正 2K/4K 重复扣除 512 reserve；
- 修正 evaluator 将 `full_2k` 错当非 Full 的命名问题；
- 重新生成 Full 指标与 combined tables。

## 27. 主结果

全部 30 条：

| Policy | 2K | 4K |
|---|---:|---:|
| Full | 0.0% | 0.0% |
| Truncate | 0.0% | 30.0% |
| Summarize | 86.7% | 80.0% |
| Elastic-LLM | 73.3% | 70.0% |

Full-20K Oracle 为 93.3%，失败两条 hard-negative long。

只看 Oracle 可解的 28 条：

| Policy | 2K | 4K |
|---|---:|---:|
| Full | 0.0% | 0.0% |
| Truncate | 0.0% | 32.1% |
| Summarize | 85.7% | 78.6% |
| Elastic-LLM | 75.0% | 71.4% |

正确结论：

- backend truncation 在强 delayed-evidence 压力下失败；
- 可保留旧信息的策略明显优于滑动窗口；
- Elastic 的可逆 recovery 工作；
- 当前 benchmark 上 Summarize 高于 Elastic，不能声称 ACE accuracy 最优。

## 28. C3 结果

| 配置 | 全部 30 条 | Oracle 可解 28 条 |
|---|---:|---:|
| Elastic-LLM | 70.0% | 71.4% |
| no latest raw | 0.0% | 0.0% |
| RAW/DROP only | 46.7% | 42.9% |
| heuristic | 46.7% | 50.0% |

说明：

- latest raw 是必要安全约束；
- ABSTRACT 在有限预算下有价值；
- LLM orchestration 优于简单 heuristic；
- recovery 不能脱离 accuracy 单独解释。

## 29. 为什么 Summarize 更高

可能原因：

1. benchmark 中多数早期事实最终有用，统一摘要很适合；
2. 3B orchestrator relevance 判断不稳定；
3. Elastic 多一次 selection error 来源；
4. fallback 较多；
5. per-step abstraction 和 orchestration 均可能引入噪声；
6. benchmark 尚未充分制造多次目标切换和摘要容量竞争。

这不是需要隐藏的结果，而是说明复杂 policy 不会自动带来更高准确率。

---

# 七、验证证据

## 30. 已完成检查

- 12 个配置全部 exit code 0；
- 360 trajectories；
- 每条 15 records，共 5400 records；
- 非 Full 每个 run 450/450 audits；
- 12 个 policy 的 raw observations byte-identical；
- 所有非 Full prompt 未超过 2048/4096；
- 无空 trajectory；
- benchmark manifest/hash 冻结；
- deterministic environment、严格 validator、recovery、budget、allowed types 和 mock rollout 测试通过。

当前环境没有 pytest，因此测试文件通过 Python 直接运行并全部通过。

---

# 八、常见答辩问题

## Q1：这和 RAG 有什么区别？

RAG 通常检索外部文档；ACE 管理同一 agent trajectory 内已经发生的 action/tool observation。重点是 lossless 保存、表示切换、可逆恢复和严格 budget。

## Q2：为什么不直接使用 EnvironmentManager.memory？

环境 memory 服务工具/reward；ACE store 服务 orchestration/audit/replay。修改 environment memory 会让 policy 污染环境真实状态。

## Q3：两份 raw truth 会不一致吗？

通过 anchor 直接记录环境原始 observation，与 model action 组成原子 step。跨 12 个 policy 已验证 raw observations 零 mismatch，source hash 用于缓存一致性。

## Q4：为什么不直接删除 chat messages？

AlphaApollo 当前把历史渲染在单个 user observation 内。伪造 assistant/tool roles 会改变原 prompt。替换原 renderer 的 memory_context 是最小兼容改动。

## Q5：Full 为什么 2K/4K 都是 0？

Full prompt 为 8K–16K，原 backend 右截断会移除早期 evidence，只留下后部 FINAL QUERY。20K Oracle 达 93.3%，说明主要问题是证据不可见。

## Q6：为什么 4K Elastic 没有高于 2K？

只差 1/30，样本太小，不应解释为稳定趋势。更大 context 也可能带来更多 distractor。需要更多 seed/样本判断。

## Q7：为什么 Summarize 超过 Elastic？

当前任务对统一摘要友好，3B orchestrator 又会引入 selection error。项目证明的是 ACE 语义、安全接线和消融贡献，不是伪造“ACE 必须第一”。

## Q8：no-last recovery 100% 为什么 accuracy 0？

Recovery 只检查早期证据是否重新出现，不保证最新 query/state 完整。说明 recovery 必须结合 latest invariant 和 accuracy。

## Q9：LLM 输出非法 JSON 怎么办？

parse、schema、ID/type、invariant、budget 任一失败都触发同一 deterministic heuristic fallback，并记录 audit reason。

## Q10：abstraction 缺失怎么办？

优先生成 deterministic fallback abstraction；普通旧 step 必要时可降级 RAW 或 DROP。latest raw 永远由 packer 强制。

## Q11：如何证明 RAW→DROP→RAW 可逆？

Decision 只改变 view，不改 store。下轮选择 RAW 时直接读取原始 action/observation，source hash 可验证 byte identity。

## Q12：为什么 management tokens 很多？

Elastic 每轮可能有 abstraction、orchestration、main 三次生成。完整 run 约 1.06M management tokens，Summarize 约 0.50M。当前优先正确性，尚未优化调用频率和 prefix cache。

## Q13：是否真的只加载一个模型？

是。管理 batch 和主 batch 都调用同一个 rollout worker，时间上串行，不创建第二 endpoint，不训练权重。

## Q14：temperature 0 还是 LLM orchestration 吗？

是。LLM orchestration 指决策由模型生成；temperature 0 用于降低方差和便于复现。

## Q15：scripted observation 是否太人工？

它牺牲部分交互真实性，以保证 policy 间 raw trajectory 完全一致，适合作 context stress harness。它不能替代真实开放工具任务，需作为局限说明。

## Q16：Full truncation 如何统计？

当前用冻结 case 的 reference full tokens 大于 budget 判断必然 overflow，并记录实际 max prompt。它尚不是 backend 逐轮 pre/post length 与方向的完整 telemetry；未来应在截断点增加只读 observer。

## Q17：wall time 和显存能公平比较吗？

目前 12 个正式配置均已统一使用 batch size 30，manifests 记录 wall time 和 0.5 秒 GPU memory sampler。三个 Full 配置峰值约 18.5–18.8 GiB。仍需注意这些是单次运行且包含 Ray/模型启动时间，不能当作稳定 latency benchmark。

## Q18：Full 为什么没有 ACE audit？

为了保证 ACE off 行为不变，Full 完全旁路 ACE decision runtime，因此没有 RAW/ABSTRACT/DROP audit；它仍保存 structured records，并由只读 truncation observer 记录实际 pre/post tokens 和方向。

## Q19：是否修改训练或 reward？

没有 optimizer、backward、LoRA、SFT 或 RL。工具、reward、done 仍走原环境；权重只读。这是 training-free policy。

## Q20：最可靠的结论是什么？

- ACE 可以安全接入 AlphaApollo；
- raw truth 与 context view 成功分离；
- 非 Full 严格满足 budget；
- 历史可以可逆恢复；
- latest raw、ABSTRACT 和 LLM orchestration 都有贡献；
- 管理成本较高；
- Summarize 是当前更强的 accuracy baseline。

---

# 九、应主动披露的局限

1. synthetic deterministic benchmark；
2. 只有 30 条、一个模型、一个 seed；
3. Summarize 高于 Elastic；
4. management token 成本高；
5. Full 无 RAW/ABSTRACT/DROP decision audit，但已有逐轮 truncation telemetry；
6. latency 和 peak memory 只有单次运行，没有方差；
7. wall time 包含 Ray 和模型启动成本；
8. timeline 脚本已完成，并已生成成功与失败案例 PNG；
9. 没有 self-evolution 或 KV/prefix cache bonus；
10. 尚需真实 AlphaApollo tool-use benchmark 验证外部有效性。

---

## 30 秒总结

> AlphaApollo 原本已有完整 multi-turn rollout 和环境 memory，但长历史只能整体渲染后交给 backend 截断。我在主生成前加入 batch-aware ACE runtime：从独立 lossless store 对历史选择 RAW、ABSTRACT 或 DROP，并用真实 tokenizer 精确 packing；环境、工具和 reward 始终读取 raw truth。30 条 8K–16K delayed-evidence 任务显示，Full/固定截断在 2K/4K 压力下明显失败，Elastic 达到约 70% 并验证可逆 recovery；关闭 latest raw 后降到 0，去掉 abstraction 或改 heuristic 也下降。不过 Summarize 达到 80% 以上，说明当前 orchestrator 与管理成本仍需优化。
