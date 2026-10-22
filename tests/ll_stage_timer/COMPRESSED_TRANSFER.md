# 无损压缩传输与本地还原

本更新只改Python离线工具。kernel、探针、通信协议、收发顺序、打点组和原始采样量均不改。
完整一键命令见[PHASED_DIAGNOSIS.md](PHASED_DIAGNOSIS.md)。

## 工作流程

原始timer JSON → 正常分析 → 紧凑表保存完整解码字段 → LZMA/XZ → Base64 → 带校验的文本块。
默认整套多B/profile结果只发送一次，OFF汇总和active-capture完整耗时一起包含。
读取已有未压缩v1包时先无损恢复，再转换为压缩格式；不要求重采或重编wheel。

```bash
python tests/ll_stage_timer/diagnose.py --analyze-existing diagnostic-runs/<run-name>
```

读取新生成的`SEND_FIRST.txt`，发送其中列出的全部`suite_*_part_*.txt`。
不用额外发送每case概览、priority/full文件、PERFORMANCE_COMPARE或分析CSV，这些信息要么已包含，要么能由完整字段重新计算。
本地原始文件与详细报告保留。若采集尚不完整，概览标为INCOMPLETE并说明缺哪些内容，导出成功不等于采集完整。

## 每次只打印一块

```bash
python tests/ll_stage_timer/transfer_report.py show diagnostic-runs/<run-name>/transfer/<bundle-id> --group 1 --part 1
```

按索引递增part，然后继续后续group。命令只输出一块，没有附加stdout日志。
单case的`transfer/`使用`--part 1`，不加`--group`。
默认每块包含首尾封装、哈希、换行后仍≤60000字符和60000字节；CRLF按两个字节计算，满足65535限制。
Base64是ASCII，不受中文字符字节数影响。数据不是摘要或删减版，不要把Base64直接当可阅读报告。

## 接收端

复制完整BEGIN至END块，保存到一个或多个UTF-8文本文件；也可用`-`从stdin读取。

```bash
python tests/ll_stage_timer/transfer_report.py inspect copied_*.txt
python tests/ll_stage_timer/transfer_report.py restore copied_*.txt --output-dir restored_small_b
```

`inspect`检查已收到的每个传输包，列出其中缺少的part；完全没收到的group无法仅凭其余块判断，`restore`还会根据最终group的总数和前序哈希核验整套group。
支持乱序、完全相同的重复块、BOM、CRLF和Base64重新折行。不同版本的包、矛盾重复、截断或任意改动必须报错，不能猜测修复。
所有group及其真实区间先验证，再创建恢复目录；目录必须是新的，不覆盖旧文件。

还原结果：

- `suite.json`或`suite_index.json`列出group、case、profile、B和源完整性；完成标记在写入结束后生成。
- 各case的`context.json`、`decoded_samples.json`恢复所有已解码字段，包括精确整数ticks、calls、空值与缺失字段。
- `trace.json`保留所选轮的全部已记录区间，默认iteration0，可用`--trace-iteration`选择；其他轮仍在数据中。
- 完整样本重新生成`summary.json`和`communication_bottleneck.md`；源元数据缺少正确性/域身份时明确记录分析不可用，仍保留解码数据。
- `reports/`保留OFF summary、active-capture报告和原性能对照。多group时在最后一组的reports下。

恢复trace使用独立rank时钟原点，不进行跨rank对齐。按AIV/阶段分轨，KERNEL、INIT、TIMER_SETUP、父阶段各自可见；等待不会被过滤。
仅前两次真实区间的采集限制仍然存在，累计指标不能扩展成完整逐包时间线。`--no-render`只恢复数据而不生成trace/派生分析。

## 格式与资源限制

单case外层manifest为`ll-timer-transfer-v2`，样本payload为`ll-timer-samples-v2`；整套payload为`ll-timer-suite-v1`。
每块封装为`deepep-ll-text-v1`，包含压缩前后长度与SHA256、分块序号/总数、chunk SHA256。无需另传manifest即可解码。
整套按case边界以约64MiB未压缩数据分组，单个case不拆分；单包原始数据上限512MiB，超限明确报错。
还原两次遍历压缩组，每次只保留一个解压组，避免整个矩阵的解码对象同时驻留。LZMA字典内存上限256MiB。
所有组的原始样本都保留；划分传输组不改变样本归属。相同输入重复打包复用哈希目录，变更输入生成新目录，旧包不删除。

## 验证与收益

使用此前用户提供文件，只做字节压缩/还原测试（不是设备性能数据）：

| 输入 | 原字节数 | 新文本总字节，含封装/CRLF | 分块 | 减少 |
|---|---:|---:|---:|---:|
| communication_costs.csv | 114384 | 31106 | 1 | 72.81% |
| 新建 文本文档.txt | 132262 | 18470 | 1 | 86.04% |

两者均验证还原字节完全一致。实际矩阵的压缩率取决于数据，发送索引会给出该次的实际原始大小和传输大小。
测试覆盖多组矩阵、旧v1包、缺块/损坏/混包、CRLF边界、未知字段、大整数、恢复后真实trace与原解析器一致、目录带空格及写入失败不发布完成标记。
本地回归494通过、29项因缺少C++/POSIX环境跳过；无失败。
无需为这次离线更新重新进行CANN编译或NPU采样；没有设备性能改善的声明。

流程参考DJLRY/DeepEP `codex/megamoe-hierarchical-roce`固定提交
[`855f848`的trace_text/text_transfer](https://github.com/DJLRY/DeepEP/blob/855f848de7bc41e4cbe310cb74099fc8321b401e/experiments/megamoe/python/deep_ep_experimental/megamoe/timer/text_transfer.py)，
使用本分支已解码样本格式独立实现，未引入参考分支的设备计时器、GM逐事件写入或屏障。
