# 来源与证据索引

核查日期：2026-11-03。正文的基础主线刻意固定到 R1 初版与 V3；后续材料只作有明确版本边界的扩展阅读。没有使用排行榜快照来证明今天的综合领先地位。

## 基础文献与仓库

| 编号 | 一手材料 | 本文使用位置 | 证据范围 |
|---|---|---|---|
| P1 | [DeepSeek-R1，arXiv v1](https://arxiv.org/html/2501.12948v1) | 02、05 | §2.2 的奖励与 R1-Zero；§2.3 的四阶段流程；§2.4 的蒸馏 |
| P2 | [DeepSeek-V3 Technical Report，arXiv v1](https://arxiv.org/html/2412.19437v1) | 04 | §2.1 模型架构及平衡策略；§2.2 MTP |
| P3 | [DeepSeekMath，arXiv v1](https://arxiv.org/html/2402.03300v1) | 03 | §4.1 的 GRPO、组内基线、裁剪与 KL；不是完整训练实现 |
| P4 | [Evaluating Large Language Models Trained on Code，v2](https://arxiv.org/abs/2107.03374v2) | 07 | pass@k 定义与有限样本估计；本文数据为构造示例 |
| G1 | [R1 固定提交](https://github.com/deepseek-ai/DeepSeek-R1/tree/0cf78561f1d51c84a21b2190626b21116d5c68bb) | 01、02、05 | 发布内容、模型来源、使用说明；不能据目录推断公司的全部内部资产 |
| G2 | [V3 固定提交](https://github.com/deepseek-ai/DeepSeek-V3/tree/9b4e9788e4a3a731f7567338ed15d3ec549ce03b) | 04、06 | 本文源码定位与配置的版本锚点 |

G1 所核查提交日期为 2025-04-09，G2 为 2025-08-27。它们是本文实际读取的代码锚点；不将连接器返回的提交称为所有访问渠道下的最新版本。正文关于后续项目的介绍来自另外读取的公开页面。

## 源码定位

| 文件与符号 | 定位 | 本文从代码确认的内容 |
|---|---|---|
| `inference/configs/config_671B.json` | [文件](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/configs/config_671B.json) | 61 层、前 3 层稠密、128 头、专家与 MLA 维度 |
| `MLA` | [model.py L396](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L396) | naive 与 absorb 路径；缓存类型与形状 |
| `Gate.forward` | [model.py L566](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L566) | 偏置影响选择，原始分数用于权重 |
| `MoE.forward` | [model.py L669](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L669) | 专家输出聚合、共享专家、分布式归并 |
| `Block` | [model.py L696](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L696) | 注意力与前馈层连接 |
| `Transformer.forward` | [model.py L773](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L773) | 从 token 输入到 logits |
| `sample` / `generate` | [generate.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/generate.py#L14) | 采样、生成循环与终止条件 |

## 工程组件与后续版本

以下页面在核查日读取，链接会随主分支或模型卡更新。本文仅据其公开说明介绍职责，不对这些项目作完整实现审计。

| 材料 | 阅读入口 | 本文使用的事实 |
|---|---|---|
| DeepEP | [官方仓库](https://github.com/deepseek-ai/DeepEP) | 专家并行通信组件 |
| DeepGEMM | [官方仓库](https://github.com/deepseek-ai/DeepGEMM) | GPU GEMM 与 grouped GEMM 等内核 |
| FlashMLA | [官方仓库](https://github.com/deepseek-ai/FlashMLA) | 高效注意力内核；当前主分支已面向更新版本 |
| DualPipe | [官方仓库](https://github.com/deepseek-ai/DualPipe) | 双向流水线并行与计算通信重叠 |
| V3.2 | [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | DSA、RL 与工具场景训练；Speciale 工具调用边界 |
| V4-Pro | [官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | 该卡介绍的 V4 预览架构、优化器与后训练路径 |
| deepseek-recipe | [官方仓库](https://github.com/deepseek-ai/deepseek-recipe) | 协议转换、prompt 编码与输出解析；需外部推理与工具执行 |
| deepseek-harness | [官方仓库](https://github.com/deepseek-ai/deepseek-harness) | Cordis 插件式应用框架、开发者预览状态 |
| 教学结构参考 | [learn-claude-code 中文版](https://github.com/shareAI-lab/learn-claude-code/blob/main/README-zh.md) | 递进主题、小实验与模型/Harness 分层的组织启发 |

## 我们实际验证了什么

- 读取官方 V3 配置、模型实现和生成代码，核对正文函数名称、行号和缓存维度。
- 在本机运行 5 个标准库脚本和 12 项单元测试，验证教学算术与工具循环边界。
- 核对本仓库本地 Markdown 文件链接和章节锚点。
- 没有下载或执行 DeepSeek 权重，没有进行 SFT/RL，没有运行 GPU 内核，没有测量模型准确率或推理吞吐量。

“我的理解”段落属于基于材料的分析；实验中的奖励、专家、候选答案与模型响应均为人工构造。任何数字如未明确标为论文报告值，都不能用于宣称 DeepSeek 的真实性能。

## 复用与署名

本文和演示代码为独立编写，没有打包上游源码、权重或论文。参考仓库的教学组织方式在正文中致谢。仓库 MIT 许可仅适用于本仓库原创内容，不重新许可第三方材料。
