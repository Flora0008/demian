# DeMiAn 层级控制恢复与 RunAI 资源说明

## 已确认的设计与执行顺序

用户已选择此前的恢复方案并授权本地修改、测试和提交。
目标是消除控制器失败导致的提前终止，不承诺尚未实测的成功率提升。

考虑过直接增加超时/重规划次数，但这会同时改变执行预算和模型调用量，难以归因。
本次选择最小恢复改动：保留 `phase_timeout=240`、`max_replans=2`，
完整任务的环境执行上限仍由 `--max-steps` 控制（composite 对照使用 1200）。

1. 先用实际 `run_episode` 函数的 CPU 回归测试复现提前终止。
2. 修改控制器和评测器：规划结束、规划故障或预算耗尽后回退完整任务，继续执行。
3. 修复非法规划输出一次；明确区分请求、模型输出和推理错误。
4. 检查旧 caption 隔离、环境成功/终止、同步与异步异常恢复，检查差异后提交。

## 修改逻辑和注释位置

- `openpi/scripts/eval/hierarchical_controller.py`
  - `stopped` 只表示层级控制停止，不表示仿真结束。
  - `enter_fallback` 幂等地切回原始完整任务、清空 caption、作废 pending 请求。
    正在运行的后台请求可能无法取消，但其结果不再被轮询或注入。
  - 重规划预算仍是整个 episode 的上限；耗尽后不再调用 Planner，使用剩余动作步数。
  - Caption 异常时继续当前阶段、无 caption；Verifier 连续两次请求/解析异常后回退。
    合法的 ongoing/uncertain 不计为请求异常，也不会自动获得成功。
  - 摘要新增 `fallback_reason`、`fallback_step`、`plan_finished`；回退状态为 `fallback`。
    原始技术错误仍保存在 `error`，不能仅凭该字段把 episode 算作失败。
- `openpi/scripts/eval/run_robocasa_eval.py`
  - 删除初始规划故障直接返回失败、控制器停止令 `done=True` 的逻辑。
  - 在动作块边界切换回完整任务；清空旧 caption，禁用回退阶段的旧 Instructor。
  - 环境 `success` 仍是成功依据；环境 done/truncated 和步数上限仍终止执行。
- `openpi/scripts/eval/qwen_hierarchy_server.py`
  - 不合法的计划输出最多生成两次：初次生成和一次带校验反馈的修复。
    不使用 `phases[:5]` 截断，避免删除必需步骤。空计划仍不合法。
  - 同一后台请求内修复，不在动作线程增加等待；HTTP 客户端超时仍适用。
    服务端在途生成不保证可被客户端取消，不能把一次修复当作无限重试。
  - 请求错误返回 400，非法模型输出返回 422，推理异常返回 500，带 `error_type`。
    客户端保留最多 8192 字节的错误响应，不记录请求图像。
- `openpi/scripts/eval/test_hierarchical_controller.py`
  - 使用实际控制器、评测循环和服务端结构化输出代码；仅替换重型模型生成/环境。
  - 这些测试验证控制流与协议，不验证视觉质量、真实物理或神经网络推理性能。

关键变更处已添加解释“为什么这样恢复”的代码注释。
本次不引入软/硬超时参数，也不改变置信度阈值；这些参数应在恢复版小规模对照后再调整。

## RunAI：哪些执行不需要 GPU？

| 工作 | GPU | 当前验证边界 |
|---|---:|---|
| 编辑、Git、语法检查、控制器和协议回归测试 | 0 | 本机 CPU 已执行，无需下载权重 |
| 分析已有 JSON/CSV、压缩备份 | 0 | 不运行模型 |
| 真实 RoboCasa 物理、相机渲染，模型服务在别处 | 客户端可尝试 0 | 需软件渲染；远端模型仍可能占 GPU，未做本次实机验证 |
| 当前三模型完整评测 | 建议共享 1 个显存足够的 GPU 设备 | 不需要每个服务各占一张；最低显存/份额尚未压测 |

没有 GPU 时，先申请学校允许的 CPU-only workload，运行下方回归即可；
不需要启动三个模型服务，也不需要为这些测试占用 GPU。CPU/RAM 的具体最小值未做资源压测。

### CPU-only 检查命令

在仓库根目录执行，Python 3.10+：

```bash
CUDA_VISIBLE_DEVICES="" python3 -m unittest discover \
  -s openpi/scripts/eval -p 'test_hierarchical_controller.py' -v
python3 -m py_compile \
  openpi/scripts/eval/hierarchical_controller.py \
  openpi/scripts/eval/qwen_hierarchy_server.py \
  openpi/scripts/eval/run_robocasa_eval.py
git diff --check
```

`py_compile` 不导入模型库或加载权重。测试用标准库解析并执行相关代码，
不要求安装 torch、transformers、MuJoCo 或 pytest。

### 为什么不直接承诺完整评测能在 CPU-only 上运行？

代码有 CPU 路径，但整条部署尚未验证：

- `qwen_hierarchy_server.py` 和已有 Instructor 支持 `--device cpu`；
  现有加载器使用 bfloat16，CPU 支持和速度需单独确认。
- `openpi/src/openpi/policies/policy_config.py` 无 CUDA 时选择 CPU；
  但动作模型仍执行 `torch.compile(..., mode="max-autotune")`，不能仅根据设备选择
  就认定所有算子、编译和推理均可在当前 CPU 镜像运行。
- 无 NVIDIA GPU 的相机渲染需要可用的软件 OpenGL，例如安装并配置 OSMesa；
  原来面向 GPU 的 EGL 配置不能直接当作 CPU-only 成功配置。
- 大模型 CPU 推理可能超过现有 HTTP 超时；即使模型最终输出，迟到的异步结果
  也会改变 caption 注入和验证时序。因此不能把全 CPU 运行视作原 GPU 实验的等价复现。

本次没有提供未经验证的全 CPU 模型启动承诺。完整 CPU 部署应先分别验证
VLA、Instructor、Planner/Verifier、渲染，再跑一个 episode 和测量耗时/内存。

### 最少要申请多少 GPU？

若要跑当前完整模型组合，先以三个服务共用一个可见 GPU 设备为目标。
这不是“必须申请整张卡”：若学校允许 GPU 份额且分配显存足够，可以使用份额；
具体份额取决于 GPU 型号及学校配置，不能给出通用的 0.25/0.5 下限。
之前同卡运行过只能证明那个配置可用，不能证明 24GB、32GB 等更小配置一定够用。
最低峰值显存本次未测量；不保证任意一张 GPU 都能装下三个模型。
量化、卸载或换小模型需要另外验证，并会改变实验条件。

## 恢复后的实验与提交边界

先用固定样本做原路径/恢复版配对小试：BreadAndCheese、SnackSorting、
CupcakeCleanup、DessertUpgrade 各 5 个环境样本。记录完整成功率、
回退后成功率、失败执行步数、回退原因、HTTP 错误及模型调用量。
参数选择后，再用未参与调参的种子验证。

结果必须写到新的实验目录，例如 `runs/paper-benchmark-recovery-v1`。
不要将旧批次的 `complete.json` 直接视为新版本已完成，否则断点恢复会跳过需要重测的样本。
需要重启已加载旧代码的三个相关进程后才能测试新版本；单纯服务器 `git pull` 不更新在途进程。

此次只修改本地源码、测试和本文档；没有修改服务器文件、启动 GPU 任务或改写旧结果。
本地环境缺少 pytest，尝试运行全仓库 `python3 -m pytest` 无法开始；
不能把本文件的定向回归称为整个仓库所有测试通过。
