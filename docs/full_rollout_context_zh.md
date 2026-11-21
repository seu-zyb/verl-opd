# 完整学生 Rollout 作为教师上下文：实验需求与实现说明

学校 gpuB 四卡实验的最新提交入口、运行日志位置及环境排查说明见
[gpuB 启动与修复说明](../scripts/opd/README_zh.md)。

## 1. 本次实验要验证什么

使用官方 verl 的 on-policy distillation（OPD）流程，研究以下问题：

> 教师预先看到学生生成的一整段回答后，能否在保留纠错能力的同时，更好地保留学生合理的推理结构、措辞和表达方式？

- 学生模型：`Qwen/Qwen3-1.7B-Base`。
- 教师模型：`Qwen/Qwen3-14B`，参数冻结，只用于提供概率监督。
- 第一版支持纯文本、单轮问题，不支持工具调用或图像、音频、视频。
- 学生生成 rollout 后，仍使用这条原始 rollout 做训练，不替换成教师改写后的答案。
- 教师额外看到完整 rollout；学生的输入、原始 token 序列和 response mask 保持不变。
- 不给教师额外提供标准答案、参考推导、奖励分数或“学生答案正确/错误”的标签。

这里“完整”指本次实际采样得到的全部 response token。如果生成因长度上限而终止，并不代表语义上已完整解答，应在分析时区分这种情况。

## 2. 代码版本与仓库关系

实验分支：`codex/full-rollout-context`。

基于官方 **verl v0.8.0**，基线提交：

```text
7aed6b230776f963fa09509c10d9c3a767d1102c
```

你的 GitHub 仓库 `main` 在开发时是更新的官方代码，不能直接把它当作学校服务器已安装的 v0.8.0。本分支从 v0.8.0 建立，未把 main 的版本升级一起引入。后续同步应明确使用此实验分支。

GitHub 更新不会自动修改服务器文件；服务器需要 `git fetch` / `git pull`。已经运行的训练任务也不会自动加载新代码，应在任务结束后更新并重新启动。

## 3. 教师与学生分别看到什么

记问题为 `x`，学生完整回答为 `y`，当前位置之前的学生回答为 `y_<t`。

- 普通 OPD 教师：`p_T(token | x, y_<t)`。
- 本实验教师：`p_T(token | teacher_context(x, y), y_<t)`。
- 学生始终：`p_S(token | x, y_<t)`。

教师输入按照以下方式组织：

```text
User 消息：
    Problem:
    {原问题}

    === Student Attempt Begin ===
    {学生完整作答}
    === Student Attempt End ===

    以上是学生对该问题的一次完整作答，可能包含错误。
    请独立判断其正确性。在给出正确解答的前提下，
    尽量保留其中合理的推理思路、措辞和组织方式；
    对于错误或不成立的内容，应予以纠正。

Assistant 位置：
    {原学生 rollout 的 token 序列}
```

User/Assistant 是消息角色。完整作答在 User 中作为参考材料出现；Assistant 位置的同一条作答则用于计算各位置概率。填入原学生 token 不意味着将其作为正确标签，也不要求教师把它预测为高概率。

教师输入使用教师自己的聊天模板；Assistant 目标部分直接拼接原学生 token ID，不做“解码后重新分词”。仅在 User 引用材料中去除 EOS 等特殊控制 token。原问题来自 `raw_prompt`，没有把数据集的其他列拼进教师输入。若原问题有 system 说明，会标明来源并保存在这一个 User 消息中。

Qwen3 教师模板默认使用 `enable_thinking=false`，避免在 Assistant 开头插入一个未闭合的思考段。这是本实验的固定模板选择，并非证明关闭思考更优。它仍可能插入原生模板规定的空思考段，属于教师前缀的一部分，不属于蒸馏目标。

`Qwen3-1.7B-Base` 不应假定自带可用的聊天模板。示例启动脚本为学生显式提供简短的 Qwen ChatML 模板；三组实验统一使用它。该模板不是经过验证的最优 Base 模型提示方式，正式实验也应固定并记录这一选择。

## 4. 修改的文件

| 文件 | 修改目的 |
| --- | --- |
| `verl/workers/config/distillation.py` | 声明新配置；为教师前缀、原响应以及接口额外生成的一个 token 预留长度 |
| `verl/trainer/config/distillation/distillation.yaml` | 默认关闭功能，提供说明文字、前缀长度预算及教师模板参数 |
| `verl/utils/full_rollout_context.py` | 构造单 User 上下文、校验 token 编号兼容、映射概率位置 |
| `verl/experimental/teacher_loop/teacher_manager.py` | 加载教师 tokenizer、发送扩展后的请求、返回与学生序列对齐的概率 |
| `verl/experimental/agent_loop/agent_loop.py` | 传入原问题与学生 prompt 长度；拒绝不支持的多轮轨迹 |
| `examples/on_policy_distillation_trainer/run_full_rollout_context.sh` | 三组实验共用的启动入口，复用官方 FSDP/vLLM OPD 脚本 |
| `examples/on_policy_distillation_trainer/qwen3_base_chat_template.jinja` | 学生 Base 模型统一使用的输入模板 |
| `tests/utils/test_full_rollout_context_on_cpu.py` | CPU 回归测试，不下载模型，不申请 GPU |

没有修改蒸馏损失公式、学生生成结果或训练标签，也没有实现 PAST 的教师适应训练。当前启动脚本沿用官方示例的 `k1 + use_policy_gradient=True`，不把任务奖励加入学生训练损失。可验证任务的正确率仍可用于评估。

### 概率位置为何需要单独处理

verl 的 vLLM/SGLang 适配器返回的是已经左移一位的 prompt logprobs：数组位置 `i` 对应下一个 token，末尾补一个占位行。

设教师新前缀长度为 `C`、学生原前缀长度为 `P`、响应长度为 `R`：

```python
source = teacher_outputs[C - 1 : C + R - 1]
student_layout[P - 1 : P + R - 1] = source
```

不能简单取最后 `R` 行，否则会漏掉第一个响应 token 并包含末尾占位行。映射后的张量仍是 `P+R` 行，随后使用原有 padding 和 response mask。这个逻辑同时支持单 token 概率和 top-k 分布。

## 5. 配置项

```yaml
distillation:
  enabled: true
  full_rollout_context: true
  full_rollout_context_instruction: >-
    以上是学生对该问题的一次完整作答，可能包含错误。
    请独立判断其正确性。在给出正确解答的前提下，
    尽量保留其中合理的推理思路、措辞和组织方式；
    对于错误或不成立的内容，应予以纠正。
  full_rollout_context_max_prompt_length: 4096
  full_rollout_context_chat_template_kwargs:
    enable_thinking: false
```

- `full_rollout_context=false`：保持原教师请求行为。
- `full_rollout_context_instruction`：说明文字，允许设为空字符串进行额外消融。
- `full_rollout_context_max_prompt_length`：教师前缀预算，包括原问题、引用的完整作答、说明文字和聊天标记；**不包括**后面第二份用于计分的响应。
- 教师 `max_model_len` 至少应为“前缀预算 + 学生响应上限 + 1”。示例是 `4096 + 1024 + 1 = 5121`。
- 超过预算会明确报错，不会静默截断完整 rollout。可以增加预算和教师长度上限，或降低学生问题/回答上限。
- 教师和学生必须具有相同的 token→ID 映射，程序会检查。换成不兼容 tokenizer 的模型不能直接运行这套逐 token 蒸馏。

## 6. 学校服务器如何获取代码

建议新建 checkout，保留之前手工改过的 v0.8.0 目录：

```bash
cd /seu_nvme/home/fenglei/213241973/code
git clone --branch codex/full-rollout-context \
  https://github.com/seu-zyb/verl-opd.git verl-opd
cd verl-opd
conda activate verl-opd

# 让当前 checkout 优先于环境里之前安装的 verl。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import verl; print(verl.__file__)"
```

打印的路径应位于这个 `code/verl-opd/verl/` 目录。已有模型文件和 Conda 环境可以继续使用，不需要复制模型。现有依赖仍须在 H100 节点验证。

以后在此目录同步：

```bash
git status
git pull --ff-only origin codex/full-rollout-context
```

如有未提交修改，先保留并处理，再拉取；不要使用强制重置覆盖自己的代码。

## 7. 三组实验与启动方式

| `CONTEXT_MODE` | 含义 |
| --- | --- |
| `baseline` | 普通 OPD，没有完整学生作答上下文 |
| `full` | 完整作答 + 中性说明：“以上学生作答可能包含错误。请独立判断并解答原问题。” |
| `style` | 完整作答 + 第 3 节的纠错与保留表达说明 |

先做 **2 步连通性测试**，不能据此判断研究假设是否成立。

在已经分配到 GPU 的作业内，设置实际路径。以下占位路径必须替换：

```bash
cd /seu_nvme/home/fenglei/213241973/code/verl-opd
conda activate verl-opd
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export STUDENT_MODEL="/你的实际模型目录/Qwen3-1.7B-Base"
export TEACHER_MODEL="/你的实际模型目录/Qwen3-14B"
export TRAIN_FILE="/你的实际数据目录/train.parquet"
export VAL_FILE="/你的实际数据目录/test.parquet"

CONTEXT_MODE=baseline bash examples/on_policy_distillation_trainer/run_full_rollout_context.sh
CONTEXT_MODE=full bash examples/on_policy_distillation_trainer/run_full_rollout_context.sh
CONTEXT_MODE=style bash examples/on_policy_distillation_trainer/run_full_rollout_context.sh
```

三条命令按顺序运行即可，不要在同一份 GPU 资源中并发启动。数据须采用 verl 的 RL parquet 格式（如官方 GSM8K 预处理结果），不是随意改后缀的文本文件。保留官方评分需要的 `data_source`、`reward_model` 等字段；这些评分字段不会进入新增的教师上下文。

默认使用单节点 **1 张学生 GPU + 1 张独立教师 GPU，共 2 张 GPU**，不是总共 1 张。教师默认 GPU 内存利用率 0.7；实际是否足够取决于 H100 显存、上下文和并发，需要实测。若只有一张卡，不能直接用此资源配置运行。

`gpuB` 的作业需分配足够 GPU。现有“只申请 1 张 GPU 做环境检查”的脚本需要先调整资源数，才能运行这里的训练；具体 CPU 数由学校策略匹配。此启动脚本本身不提交 Slurm 作业。

常用环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TOTAL_TRAINING_STEPS` | 2 | 连通性测试步数 |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` | 8 / 8 | 小批量检查 |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 1024 / 1024 | 学生长度上限 |
| `TEACHER_PREFIX_BUDGET` | 4096 | 教师扩展前缀上限 |
| `NGPUS_PER_NODE` / `TEACHER_WORLD_SIZE` | 1 / 1 | 学生和教师独立 GPU 数 |
| `ROLLOUT_TP` / `TEACHER_TP` | 1 / 1 | 相应推理模型的张量并行度 |
| `SEED` | 2026 | 数据、学生训练的随机种子；推理后端沿用其默认种子设置 |
| `SAVE_FREQ` / `TEST_FREQ` | -1 / -1 | 默认不保存/不做周期验证；需正式评估时调整 |

正式实验需统一训练预算、数据顺序、采样设置和评估配置，并从相同的原始学生模型分别开始，不能接着上一组的 checkpoint 训练。还需设置正式的训练步数、验证频率和 checkpoint 保存频率。示例：

```bash
CONTEXT_MODE=style TOTAL_TRAINING_STEPS=100 TOTAL_EPOCHS=3 \
  SAVE_FREQ=20 TEST_FREQ=20 \
  bash examples/on_policy_distillation_trainer/run_full_rollout_context.sh
```

100 步只是启动参数示例，不是已确定的研究训练预算；实际可运行步数还受数据量和 epoch 数影响。额外 Hydra 参数可接在脚本后面。日志默认输出控制台，不要求登录 WandB。

## 8. 应该如何判断实验是否成功

不能仅因为 KL 更小、学生答案更相似，就认定方法有效：教师看到了整段答案，可能只是更容易复述学生，包括学生的错误。

建议先固定一批学生 rollout，分成原本正确/错误两类，对比不同教师上下文，再进行完整训练。重点观察：

1. **正确性**：独立验证集最终答案准确率，尤其是错误题有没有改善。
2. **纠错信号**：在人工标注的小样本中，错误关键 token 的概率变化、正确替代 token 的倾向；不能仅看全回答平均概率。
3. **表达保留**：对正确且合理的部分，检查推理组织、术语和措辞是否保留；可以用编辑距离、长度变化和盲评辅助，但文本相似不等于推理正确。
4. **代价**：教师上下文 token 数、吞吐、GPU 显存和训练耗时。
5. **消融**：`full` 对比 `baseline` 检查上下文作用；`style` 对比 `full` 检查风格指令作用。严格归因时还需控制教师聊天模板/思考前缀的差异，可加入同模板但不提供作答的对照。

当前提交提供训练接口、启动脚本与回归测试，**尚未实现自动风格评分或固定 rollout 的完整评估流水线，也没有模型效果结论**。正式比较应使用多个随机种子，并记录数据版本、模型版本及代码提交号。

## 9. 验证范围与限制

已在本地完成：

- CPU 测试：23 项通过。覆盖上下文构造、词表编号不匹配、空响应、多模态拒绝、预算检查、普通 OPD 不变、首尾 token 对齐、top-k 和 padding。
- 三种启动模式的实际命令参数经过 Hydra 配置组合检查，学生 Jinja 模板经过渲染检查。
- Python 格式/静态检查和 Bash 语法检查。

CPU 测试使用轻量 tokenizer 和推理客户端替身；运行真实的上下文、配置预算与管理器方法，不加载模型权重。它们**不代表** Qwen、vLLM、FlashAttention 或 H100 训练已验证。本地测试解释器为 Python 3.13；学校 Python 3.12 下仍需复跑。

复跑 CPU 测试：

```bash
python -m pytest -q tests/utils/test_full_rollout_context_on_cpu.py
```

部署前在实际分配到的 H100 节点检查 PyTorch/CUDA、FlashAttention 导入和模型路径，然后从 2 步测试开始。此前 fat01 上的 GLIBC 报错既不能证明 H100 也会失败，也不能证明 H100 环境已可用。

## 10. 相关研究与实现边界

完整 rollout 作为教师额外信息已有相近研究，例如 [SD-Zero](https://arxiv.org/abs/2604.12002) 和 [PAST](https://arxiv.org/abs/2608.08726)。这里不是这些论文的完整复现：本实验采用冻结的外部教师，不提供参考解答或对错提示，也不额外训练教师。

因此，不能仅把“加入完整 rollout”作为已经确认的新颖贡献；需要进一步通过相关工作比较和实验，研究冻结教师条件下纠错能力与学生表达保留之间的关系。

本次代码和文档由 AI 辅助编写；运行正式实验前请结合上述说明审阅改动。
