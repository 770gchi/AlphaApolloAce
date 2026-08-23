# ACE Task B 快速说明

## 主要改动

- 在 AlphaApollo multi-turn generation 前接入 `full | truncate | summarize | elastic` context policy。
- environment、工具、reward 和完整 trajectory 始终使用原始历史；ACE 只改变模型输入视图。
- abstraction、orchestration 和主生成串行复用同一个 rollout worker。
- 支持 tokenizer 精确预算、latest step 保持 RAW、确定性 fallback、审计日志和离线 replay。

## 启动

```bash
cd /root/AlphaApollo

python -m alphaapollo.workflows.test \
  --config examples/configs/test_informal_math.yaml \
  context_policy.mode=elastic \
  context_policy.orchestration=llm \
  context_policy.max_input_tokens=4096 \
  context_policy.reserve_output_tokens=512 \
  context_policy.audit_dir=./outputs/ace/audit \
  context_policy.store_dir=./outputs/ace/store
```

将 `context_policy.mode` 改为 `full`、`truncate` 或 `summarize` 即可运行对应基线。

## 测试与 Replay

```bash
python ace_context/tests/test_runtime.py
python ace_context/tests/test_ace.py

python -m ace_context.replay \
  --store outputs/ace/store/trajectories.jsonl \
  --audit outputs/ace/audit/context_audit.jsonl \
  --config examples/configs/test_informal_math.yaml \
  --output outputs/ace/replayed_views.jsonl
```

每个 rollout step 的 `context_audit` 也会随 AlphaApollo trajectory 输出保存。
