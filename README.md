<h1 align="center">AlphaApollo × ACE</h1>

<p align="center"><b>Training-Free、可逆、可审计的长轨迹上下文编排</b></p>

本仓库在 AlphaApollo 唯一的 multi-turn rollout 主循环中集成 Adaptive Context Elasticizer（ACE）。系统在固定 token budget 下逐轮为历史交互选择 RAW、ABSTRACT 或 DROP；环境、工具、reward 和完整 trajectory 始终使用 raw truth，模型权重保持只读。

## 核心能力

- batch-aware lossless store 与逐 step abstraction cache；
- RAW / ABSTRACT / DROP 动态编排和 RAW → DROP → RAW 可逆恢复；
- latest-step RAW invariant 与 tokenizer-aware budget packing；
- abstraction、orchestration、main generation 串行复用同一个 worker；
- deterministic fallback、JSONL audit、持久化和离线 replay；
- 30 条 delayed-evidence benchmark、C2/C3 实验、Full telemetry 和 PNG 图表。

## 安装

~~~bash
git clone https://github.com/770gchi/AlphaApolloAce.git
cd AlphaApolloAce
conda create -n alphaapollo python=3.12 -y
conda activate alphaapollo
bash installation.sh
python -m pip install matplotlib
~~~

## AlphaApollo 集成点

| 文件 | 作用 |
|---|---|
| ace_context/ | store、abstraction、orchestration、packing、audit、replay |
| alphaapollo/core/generation/multi_turn_rollout/rollout_loop.py | generation 前构造 context view，env step 后记录 raw interaction |
| alphaapollo/core/environments/env_manager.py | 使用原模板渲染 policy-selected memory |
| alphaapollo/core/generation/verl/trainer/main_generation.py | structured trajectory、audit 和 Full truncation telemetry |
| alphaapollo/core/generation/verl/trainer/config/generation.yaml | context policy Hydra 配置 |
| alphaapollo/core/environments/delayed_evidence/ | Task C deterministic stress environment |

每轮顺序为 abstraction（cache miss）→ orchestration → 代码验证与 packing → main generation。ACE 不复制 rollout engine，也不加载第二份模型。

---

# Task C Benchmark

本目录实现 30 条 deterministic delayed-evidence 任务：5 类 × short/long × 3。C1、C2 与 C3 共用同一冻结数据集。
人工审阅请打开 `data/task_c/cases/`，其中每条任务对应一个格式化 JSON；`cases.jsonl` 仅供程序批量读取。

## 构建

```bash
python -m benchmarks.task_c.build \
  --output-dir data/task_c \
  --model /path/to/Qwen2.5-3B-Instruct
```

冻结 manifest：30 条，short 8K–12K、long 12K–16K，`cases_sha256=cbe9231d1715ce7d4fc80986f07fa35a1e08a5db0a6942a05feb9f9bc9b6f7bc`。

## 运行

```bash
# 先查看完整矩阵命令
python -m benchmarks.task_c.run_experiments --dry-run

# 运行 C2 + C3 + Full-20k Oracle
python -m benchmarks.task_c.run_experiments \
  --model-path /path/to/Qwen2.5-3B-Instruct

# 隔离运行一个样例，不覆盖正式 benchmark
python -m benchmarks.task_c.run_experiments \
  --case-id exact_number_01_short \
  --run elastic_heuristic_4k \
  --model-path /path/to/Qwen2.5-3B-Instruct
```

正式结果写入 `outputs/task_c/{raw,audit,store,metrics,manifests}`；单样例写入 `outputs/task_c_trial/`。2K/4K 是实际 prompt 上限，512 output tokens 单独预留。

## 评分与绘图

```bash
python -m benchmarks.task_c.evaluate --cases data/task_c/cases.jsonl --output outputs/task_c/raw/elastic_llm_4k.jsonl --policy elastic_llm_4k --budget 4096 --output-dir outputs/task_c/metrics/elastic_llm_4k
python -m benchmarks.task_c.plot_timeline --per-step outputs/task_c/metrics/elastic_llm_4k/per_step.jsonl --case-id exact_number_01_short --output outputs/task_c/plots/elastic_llm_4k_exact_number_01_short.png
```


## 架构

```text
Environment/tool raw truth ──► memory, reward, done, full logger
             │
             └─► LosslessStore ─► abstraction cache
                                      │
                                      ▼
                         RAW / ABSTRACT / DROP
                                      │
                                      ▼
                         tokenizer-aware packing
                                      │
                                      ▼
                         model-visible context view
```

ACE 只改变每次 generation 前的 context view，不覆盖 `EnvironmentManager.memory`，也不改变工具执行、reward 或 ground truth。摘要、编排和主生成串行复用同一个 rollout worker。

## Context policy 配置

| 字段 | 默认值 | 含义 |
|---|---:|---|
| `mode` | `full` | `full/truncate/summarize/elastic` |
| `max_input_tokens` | 4096 | 实际主模型 prompt 上限 |
| `reserve_output_tokens` | 512 | 单独预留的输出长度 |
| `keep_last_raw` | true | 强制最近完整 interaction 为 RAW |
| `abstraction_cache` | true | 按 source hash 缓存逐 step 摘要 |
| `orchestration` | `llm` | LLM 或 deterministic heuristic |
| `truncate_window_steps` | 4 | 固定窗口保留步数 |
| `abstraction_max_tokens` | 256 | 摘要最大输出 |
| `orchestration_max_tokens` | 512 | 编排最大输出 |
| `allowed_types` | all | 消融时可禁用 ABSTRACT 等类型 |
| `audit_dir/store_dir` | null | 审计和 lossless JSONL 路径 |

Budget 语义：

```text
actual_prompt_tokens <= max_input_tokens
actual_prompt_tokens + reserve_output_tokens <= model_context_window
```

## 一键运行与断点续跑

```bash
# 完整 C2 + C3 + Oracle；本机 Ray 建议显式使用 local
RAY_ADDRESS=local python -m benchmarks.task_c.run_experiments \
  --batch-size 30 \
  --model-path /path/to/Qwen2.5-3B-Instruct

# 中断后只运行缺失配置
RAY_ADDRESS=local python -m benchmarks.task_c.run_experiments \
  --resume --batch-size 30 \
  --model-path /path/to/Qwen2.5-3B-Instruct

# 只运行指定配置
RAY_ADDRESS=local python -m benchmarks.task_c.run_experiments \
  --run elastic_llm_4k --batch-size 30 \
  --model-path /path/to/Qwen2.5-3B-Instruct
```

## Full truncation telemetry

每个 structured trajectory record 包含：

```json
{
  "truncation_telemetry": {
    "pre_truncation_tokens": 14208,
    "post_truncation_tokens": 2048,
    "truncation_direction": "right",
    "truncated_tokens": 12160,
    "was_truncated": true
  }
}
```

这些字段在 tokenizer 截断点只读采集，不改变 Full 的 prompt 或 backend 行为。聚合 CSV 报告 truncation event count、累计丢失 tokens 和最大截断前长度。

## 生成全部图表

```bash
python -m benchmarks.task_c.plot_results \
  --metrics-dir outputs/task_c/metrics \
  --raw-dir outputs/task_c/raw \
  --output-dir outputs/task_c/plots
```

输出：

- `c2_accuracy.png`
- `c3_ablations.png`
- `token_costs_4k.png`
- `elastic_types_over_turn.png`
- `full_truncation_telemetry.png`
- 两个 case timeline PNG

两个详细案例见 [CASE_STUDIES.md](benchmarks/task_c/CASE_STUDIES.md)，slides 草稿见 [TASK_C_SLIDES.md](benchmarks/task_c/TASK_C_SLIDES.md)，实现与答辩说明见 [IMPLEMENTATION_AND_DEFENSE_NOTES.md](benchmarks/task_c/IMPLEMENTATION_AND_DEFENSE_NOTES.md)。

## 测试

```bash
python benchmarks/task_c/test_task_c.py
python benchmarks/task_c/test_alphaapollo_smoke.py
```

如果已安装 pytest：

```bash
python -m pytest ace_context/tests benchmarks/task_c -q
```

## 失败回退与 replay

- abstraction JSON 无效或字段缺失：生成 deterministic fallback abstraction；
- orchestration JSON/step/type 非法：使用 deterministic heuristic；
- fixed prompt 或 latest raw 超预算：抛出明确错误，不允许静默截断；
- store 采用 source hash 失效缓存；
- audit 保存最终 message hash、type decision、token 和 fallback；
- replay 不调用环境即可重建记录的 context view。

## 实验结果摘要

| Policy | 2K | 4K |
|---|---:|---:|
| Full | 0.0% | 0.0% |
| Truncate | 0.0% | 30.0% |
| Summarize | 86.7% | 80.0% |
| Elastic-LLM | 73.3% | 70.0% |

Full-20K Oracle 为 93.3%。当前 benchmark 上 Summarize 高于 Elastic；Elastic 的主要验证点是可逆 recovery、严格 budget、逐 step 审计和消融可解释性，而不是宣称所有条件下 accuracy 最优。


## 已生成图表

### C2 accuracy

![C2 accuracy](outputs/task_c/plots/c2_accuracy.png)

### C3 ablations

![C3 ablations](outputs/task_c/plots/c3_ablations.png)

### Token cost

![Token costs](outputs/task_c/plots/token_costs_4k.png)

### Elastic types over turns

![Elastic types](outputs/task_c/plots/elastic_types_over_turn.png)

### Full truncation telemetry

![Full truncation telemetry](outputs/task_c/plots/full_truncation_telemetry.png)

正式 telemetry 汇总：

| Run | Records | Truncated turns | Lost tokens | Max pre-truncation | Direction |
|---|---:|---:|---:|---:|---|
| Full-2K | 450 | 354 | 3,029,426 | 20,895 | right |
| Full-4K | 450 | 294 | 2,020,893 | 19,878 | right |
| Full-20K Oracle | 450 | 0 | 0 | 15,562 | none |

三个 Full 配置均以 `batch_size=30` 重跑。manifest 记录的 wall time 分别为 257.3s、348.5s 和 312.3s，峰值 GPU memory 分别为 19,210 MB、19,280 MB 和 18,992 MB。wall time 包含 Ray 和模型启动成本，且只有一次运行，因此不报告方差或显著性。


---

# AlphaApollo 原项目说明

以下保留上游 AlphaApollo 的安装、功能和引用信息。

<h1 align="center">
<b>AlphaApollo: A System for Deep Agentic Reasoning</b>
</h1>

<p align="center">
  <a href="https://alphaapollo.org/"><img src="https://img.shields.io/badge/Project%20Website-AlphaApollo"></a>
  <a href="https://arxiv.org/abs/2510.06261"><img src="https://img.shields.io/badge/arXiv-2510.06261-b31b1b"></a>
</p>


AlphaApollo is an agentic reasoning framework that orchestrates multiple models and tools to enable iterative, verifiable, and self-evolving reasoning. It supports a broad range of paradigms, including tool-integrated reasoning, agentic post-training (e.g., multi-turn supervised fine-tuning and reinforcement learning), and agentic self-evolution. The framework offers extensible environments and toolsets for easy customization, extension, and scalable deployment of agentic reasoning workflows.


## News
- [2026.01] We are excited to release AlphaApollo, an agentic LLM reasoning system for advanced reasoning.
- [2025.10] Our technical report is released; see [here](https://arxiv.org/abs/2510.06261v1) for details.


## Installation

```bash
conda create -n alphaapollo python==3.12 -y
conda activate alphaapollo

git clone https://github.com/tmlr-group/AlphaApollo.git
cd AlphaApollo

bash installation.sh
```

## Supported features
### [Agentic reasoning](https://alphaapollo.org/multi-turn-agentic-reasoning)
- Tool-integrated reasoning rollout with seamless environment interaction
- Dynamic memory updates for multi-turn reasoning

### [Agentic learning](https://alphaapollo.org/multi-turn-agentic-learning)
- Multi-turn supervised fine-tuning (SFT)
- Reinforcement learning algorithms: GRPO, PPO, DAPO, and more

### [Agentic self-evolution](https://alphaapollo.org/multi-round-agentic-evolution)
- Multi-round, multi-model solution refinement with shared state
- Iterative improvement via feedback and executable checks

### [Built-in tools](docs/core-modules/tools.md)
- Python interpreter
- Retrieval-Augmented Generation (RAG)
  

## Quick-start recipes

Detailed quick-start commands (including script entrypoints) are documented in [quick-start.md](docs/getting-started/quick-start.md).

Note: Before using the local RAG module, please follow [RAG Service Setup](docs/core-modules/tools.md#rag-service-setup).

### Agentic reasoning
```bash
# no-tool reasoning
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=math-ai/aime24
```
```bash
# tool-integrated reasoning
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=math-ai/aime24 \
  --env.informal_math.enable_python_code=true \
  --env.informal_math.enable_local_rag=false \
  --env.max_steps=4
```

Single-question evaluation:
```bash
# Select specific dataset samples (e.g., the 0th AIME test question) and test
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.module=alphaapollo.data_preprocess.prepare_custom_data \
  --preprocess.data_source=math-ai/aime24 \
  --preprocess.splits=test \
  --preprocess.sample_indices=0 \
  --data.path=./data/custom_data/test.parquet
```
```bash
# Directly evaluate a plain text question (not from a dataset)
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.module=alphaapollo.data_preprocess.prepare_single_question \
  --preprocess.question_text="What is the sum of integers from 1 to 1000?" \
  --preprocess.ground_truth="500500" \
  --data.path=./data/single_question/test.parquet
```

### Agentic learning
```bash
# multi-turn SFT
python3 -m alphaapollo.workflows.sft \
  --model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=AI-MO/NuminaMath-TIR
```
```bash
# multi-turn RL
python3 -m alphaapollo.workflows.rl \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=HuggingFaceH4/MATH-500 \
  --algorithm.adv_estimator=grpo
```

### Agentic self-evolution
> Before running the self-evolution scripts, make sure to serve the corresponding number of models.
```python 
python alphaapollo/utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "0,1" --port 8000 --model_id "qwen3_4b_inst"
```
```bash
# single-model evolution
python3 -m alphaapollo.workflows.evo \
  --preprocess.data_source=math-ai/aime24 \
  --run.dataset_name=aime24 \
  --policy_model_cfg.model_name=qwen3_4b_inst \
  --policy_model_cfg.base_url=http://localhost:8000/v1 \
  --verifier_cfg.model_name=qwen3_4b_inst \
  --verifier_cfg.base_url=http://localhost:8000/v1
```

## Code Structure

```text
+------------------------------------------------------------------+
| alphaapollo/data_preprocess                                      |
| (dataset preparation scripts)                                    |
+------------------------------------------------------------------+
                               |
                               V
+------------------------------------------------------------------+
| alphaapollo/core                                                 |
| (core code)                                                      |
|                                                                  |
|  +----------------------+              +----------------------+  |
|  | generation/          |              | tools/               |  |
|  |                      | <----------> | - python_code        |  |
|  |                      |              | - rag/               |  |
|  +----------------------+              +----------------------+  |
|              Λ                                                   |
|              |                                                   |
|              V                                                   |
|  +------------------------------------------------------------+  |
|  | environments/                                              |  |
|  | - informal_math_training/                                  |  |
|  | - informal_math_evolving/                                  |  |
|  | - memory/                                                  |  |
|  | - prompts/                                                 |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### Informal Math Environment (Training): 
- Environment package in [alphaapollo/core/environments/informal_math_training/](alphaapollo/core/environments/informal_math_training/)
- Prompts in [alphaapollo/core/environments/prompts/informal_math_training.py](alphaapollo/core/environments/prompts/informal_math_training.py)

### Informal Math Environment (Evolving): 
- Environment package in [alphaapollo/core/environments/informal_math_evolving/](alphaapollo/core/environments/informal_math_evolving/)
- Prompts in [alphaapollo/core/environments/prompts/informal_math_evolving.py](alphaapollo/core/environments/prompts/informal_math_evolving.py)

### Tools (for reference)
- Python Code implementation: [alphaapollo/core/tools/python_code.py](alphaapollo/core/tools/python_code.py)
- RAG implementation: [alphaapollo/core/tools/rag/](alphaapollo/core/tools/rag/)


## Acknowledgement
AlphaApollo is built upon the open-source projects [verl](https://github.com/volcengine/verl), [verl-agent](https://github.com/langfengQ/verl-agent/tree/master), [vllm](https://github.com/vllm-project/vllm), and [sglang](https://github.com/sgl-project/sglang). We sincerely thank the contributors of these projects for their valuable work and support.

## Cite
If you find **AlphaApollo** useful in your research, please consider citing our work:

```
@article{zhou2025alphaapollo,
  title = {{AlphaApollo}: A System for Deep Agentic Reasoning},
  author = {Zhou, Zhanke and Cao, Chentao and Feng, Xiao and Li, Xuan and Li, Zongze and Lu, Xiangyu and Yao, Jiangchao and Huang, Weikai and Cheng, Tian and Zhang, Jianghangfan and Jiang, Tangyu and Xu, Linrui and Zheng, Yiming and Miranda, Brando and Liu, Tongliang and Koyejo, Sanmi and Sugiyama, Masashi and Han, Bo},
  journal = {arXiv preprint arXiv:2510.06261},
  year = {2025}
}
```
