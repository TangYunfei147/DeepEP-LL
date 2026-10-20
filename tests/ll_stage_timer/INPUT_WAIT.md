# Dispatch 输入与量化尾部等待

后续一包前瞻预取保留这两个阶段；TOKEN_INPUT_WAIT改为等待在当前包DMA后
记录的独立MTE2_S事件，避免误等下一包。TOKEN_INPUT的DeQue位置也随调度移动，
详见[INPUT_PREFETCH.md](INPUT_PREFETCH.md)。下文记录最初补点基线的同步形式。

本变更只在 `SendPacket` 的两处原有同步前后读取时间，不增加、删除或移动同步：

| ID | 阶段 | 原有指令 | 含义 |
|---|---|---|---|
| 64 | TOKEN_INPUT_WAIT | 非在线量化路径的 MTE2_S | Scalar 等输入路径完成时暴露的剩余等待 |
| 65 | TOKEN_QUANT_WAIT | 在线量化后、FreeTensor 后的 V_S | Scalar 等向量流水完成时暴露的剩余等待 |

预量化 FP8 属于前者；现场量化属于后者。V_S 也可能等待此前向量工作，
不能把整个区间认定为量化计算或输入 DMA 时长。打点本身有开销，
这些指标不能证明显式预取一定有效，最终用 timer OFF A/B 判断。

## 格式与兼容性

新采样 schema9，旧 0..63 ID 不变，追加两个 ID。trace容量192条不变，
metric容量66条，最多258条记录；每核仍72×8个int64（4608B）。
仍只保留每阶段前两次实际区间，全量累积 cycles/calls。
业务核心协议10000+S、包格式、S/R分工和缓冲预算不变。
源码指纹基线不更新：剥离新增探针后必须与原算法 token 摘要一致。

Python/native布局不一致时拒绝采集。全rank重装timer ON完整wheel并重新采集；
无需因此重建已校验SDK。旧schema7可离线分析，缺少新指标时输出数据不足。
新schema9使用原始JSON、CSV、Chrome trace，不导出只支持64阶段的参考
format5二进制；显式请求该导出时报错，不静默丢掉新阶段。
旧schema7仍支持原二进制导出。不将schema7/9诊断耗时作加速比，
同配置timer OFF正式性能仍可对比。

## 使用与结论

沿用README采集命令，诊断尾段建议至少3轮。单轮/两轮仍正常导出数值，
但自动结论为数据不足，不自动增加采样轮数。

在工程根目录：

```bash
python tests/ll_stage_timer/analyze.py results/你的运行目录 --op dispatch
```

每个case/路由报告目录增加：

- `input_wait_bottleneck.md`：中文结论与逐核依据。
- `input_wait_bottleneck.csv`：逐核占比、有效/缺失样本、ticks/call和候选标记。
- `summary.json` 内 `input_wait_bottleneck`：阈值、总体筛查与逐核结构化结果。

默认启发式规则：

1. 同核至少3个有效轮次；任何轮次缺记录、wait与TOKEN_INPUT的calls不一致，
   或等待超过同核SEND_TOKENS/KERNEL时，标为数据不足。
2. 至少半数轮次中，等待/SEND_TOKENS≥20%且等待/KERNEL≥10%，标为疑似瓶颈。
3. 有充分数据但未达阈值，标为未见明显证据，不等于不存在等待瓶颈。

比值先在同rank/轮/核计算，再汇总，不能用不同核的最大值相除。
`weighted_ticks_per_call`为总等待ticks/总calls，不假设时钟频率。
`critical_core_fraction`是该核KERNEL达到同rank同轮最长核90%的轮次比例，
帮助判断局部等待是否也落在慢核上，不证明跨rank全局关键路径。
总体提示任意核是否命中候选；不能解读为已确定整个dispatch的瓶颈原因。
combine过滤输出没有dispatch样本时明确无法判断。

本地CPU模型检查覆盖schema7兼容、schema9容量、缺失/不一致/少轮数、
两条同步路径、低等待与高等待，以及分析输出落盘；不替代CANN或NPU验收。
本次完整本地回归为462 passed、44 skipped、9 deselected。
跳过项为缺C++编译器和POSIX检查，未执行CANN编译或设备采样。
