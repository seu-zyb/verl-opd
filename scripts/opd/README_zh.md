# gpuB 上的 OPD 实验：启动与本次修复

## 现在执行什么

在登录节点的仓库目录更新代码，提交一次任务。脚本会在分配到的 H100 上检查环境，通过后直接开始 100 步实验，不需要单独提交冒烟实验。

```bash
cd /seu_nvme/home/fenglei/213241973/code/verl-opd
git pull --ff-only origin codex/full-rollout-context
sbatch scripts/opd/run_style.slurm
```

启动入口是 **`scripts/opd/run_style.slurm`**。不要继续执行仓库外的旧 `opd_fix_bundle` 副本。仓库内 `server_debug` 下的两个旧提交脚本和检查程序已改成转发入口，统一调用维护中的脚本。

默认路径读取 `server_debug/full_rollout_paths.sh`，使用你上传的学生模型、教师模型及输出目录。兼容数据位于 `$DATA_ROOT/compat_datasets3/`。你此前已经验证这两个文件通过，正常不必再次转换。

只有兼容数据缺失时，才在已有 `verl-opd` 环境中执行：

```bash
conda activate verl-opd
python scripts/opd/prepare_gsm8k_compat.py \
  --data-dir /seu_nvme/home/fenglei/213241973/data/gsm8k
```

它保留原始文件，仅对新副本移除不兼容的 Hugging Face schema 元数据，并检查两份数据的全部内容相同。需要自定义路径配置时，可在提交前 `export OPD_PATHS_FILE=/你的配置文件路径`；该文件需包含 `PROJECT_ROOT`、`STUDENT_MODEL`、`TEACHER_MODEL`、`DATA_ROOT`、`OUTPUT_ROOT`。

## 已确认并修正的问题

1. **旧检查导入不存在的 worker。** 当前 `main_ppo.py` 使用 `verl.workers.engine_workers`；仓库已没有 `verl.workers.fsdp_workers`。检查改为导入实际 worker、FSDP engine 和训练入口。这修复了一个确定的后续导入错误，但不能据此声称已解释此前的 SIGSEGV。
2. **旧脚本仍指向仓库外的旧检查文件。** 新入口全部使用仓库内脚本和配置，避免更新 GitHub 后仍执行旧副本。
3. **原生崩溃缺少定位信息。** 每个检查子进程开启 `faulthandler`，打印阶段名，保留独立日志。组合导入失败后，才自动运行独立导入诊断及 `importtime` 跟踪，不需要再排一次队来收集这些信息。
4. **超时输出被丢弃。** 现在超时也保留已打印内容；非零退出、段错误和超时均阻止训练。独立导入成功不会掩盖组合导入失败。
5. **实验结果缺少统一记录。** 每个作业保存实际包版本、路径配置、提交编号、启动脚本、检查日志、训练日志和各次验证样本。开启训练前验证，每 20 步验证/保存，便于比较训练前后的学生能力。

保留 `set -eo pipefail`，不在 conda 激活期间启用 `set -u`；继续清除 AMD 可见设备变量，使用短 Ray 临时目录。模型和数据来自本地，启用 Hub/Datasets 离线模式。脚本不会安装、卸载或升级服务器依赖。

## 本次实验的含义

教师：Qwen3-14B；学生：Qwen3-1.7B-Base。gpuB 单节点 4 张卡、48 个 CPU 核，不设置学校禁止的 `--mem`。学生资源池 2 张卡，教师资源池 2 张卡，各自 TP=1；教师为两个单卡副本，而不是把单份教师拆到两张卡。

学生产生 rollout；教师的 user 上下文包含题目、完整学生作答和 style 指令。教师随后对**原始学生 token 序列**进行 teacher forcing，给出每个 token 的概率，并对齐到学生响应位置。教师不会把标准答案当作上下文，也不会训练自身参数。style 指令要求在正确性允许时保留学生的表达方式，并不是另一个模型。

训练沿用 `k1 + policy gradient`，不加入任务奖励；验证仍用 GSM8K 答案评分。随机种子 2026，batch/minibatch=8，prompt/response 上限各 1024，教师前缀预算 4096，总长度上限 5121。默认新实验不恢复旧 checkpoint。

训练前验证和训练后验证使用相同测试集、模板和贪心解码，先判断学生是否改善。要证明“完整 rollout 上下文”优于普通 OPD，后续还需相同条件的 baseline 对照；单次 style 训练不能单独证明这一点。

## 如何看结果

标准输出：仓库目录下的 `opd-style-作业号.out`。

完整结果目录：`$OUTPUT_ROOT/style/seed-2026-job-作业号/`。

| 文件 | 用途 |
| --- | --- |
| `environment-check.log` | 所有检查及完整错误输出；子进程 stderr 合并在这里 |
| `checks/check-*.log`、`checks/summary.json` | 每项检查的详细日志及通过/失败状态 |
| `checks/import-trace.log` | 组合导入失败时额外生成的导入顺序/崩溃信息 |
| `training.log` | 检查通过后实际训练输出 |
| `validation/` | 训练前及训练过程中的验证生成、答案和分数 |
| `package_versions.json`、`git-commit.txt`、`submitted.slurm`、`paths.sh` | 本次运行所用版本和配置；节点没有 Git 时不生成提交编号 |

**此前 `FAIL (exit -11)` 的具体底层依赖尚未确认。** 上传的版本列表与无堆栈旧日志不能定位哪个原生扩展发生非法内存访问。本次修改修复了确定的脚本错误并补齐一次排队内的诊断，不能宣称已修复该段错误或已完成 H100 训练。

核心版本 torch 2.8.0、torchaudio 2.8.0、torchvision 0.23.0、xformers 0.0.32.post1 符合 [vLLM 0.11.0 的 CUDA 依赖要求](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)。这不等于所有扩展的二进制兼容性已经通过。崩溃跟踪机制见 [Python 3.12 faulthandler 文档](https://docs.python.org/3.12/library/faulthandler.html)。

## 本地验证范围

运行 `pytest tests/utils/test_opd_startup_on_cpu.py tests/utils/test_full_rollout_context_on_cpu.py -q`：30 项通过，包括实际 Bash 启动命令的 Hydra 配置组合（style/full/baseline）、teacher/student token 对齐、异常/超时输出保留和 worker 路径存在性。Python 静态检查、格式检查和 Bash 语法检查通过。兼容数据转换复现了原有 List schema 错误并验证转换后内容不变、重复运行和坏数据拒绝。

这些检查在本地 CPU 环境完成，没有启动服务器上的模型引擎，也未验证 H100 多卡通信、完整模型显存占用和 optimizer step。
