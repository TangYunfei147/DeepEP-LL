> 同一suite的摘要及必要补充累计最多发送10次。第一轮只发送一个≤60000字符及UTF-8字节（含CRLF）的SEND_FIRST摘要；后续仅按明确疑点定点提取。
> `--transfer-mode archive`已禁用。已有全量归档留在目标机，不发送；旧格式的本地还原能力继续保留。

# 必要补充的压缩传输与本地还原

当前新采集为schema15稀疏时间戳，**需全rank完整重编wheel并用新的run-name重新采集**；SDK不强制重建。收发逻辑与通信协议保持不变。
本页描述采集之后的纯离线摘要、必要补充和恢复，执行这些离线步骤本身不重编或启动设备。完整OFF+sparse多B命令见[PHASED_DIAGNOSIS.md](PHASED_DIAGNOSIS.md)。
旧schema7～14数据仍可离线读取；旧full/phase采样只用于历史复核，不转化为schema15数据。

## 工作流程

第一轮读取全部采样数据进行统计，只输出一份用于定位的摘要：

```bash
python tests/ll_stage_timer/diagnose.py --analyze-existing diagnostic-runs/<run-name>
```

只发送运行根目录的`SEND_FIRST.txt`。OFF/ON性能、插桩扰动、各B的准备/收发跨度及首次发送/Probe/ready核心数量已进入摘要，不单独发送性能CSV或`PERFORMANCE_COMPARE.txt`。
同一份sparse数据可以同时分析多个准备与等待疑点；包数、逐包均衡和ready/probe比例明确未采集。设备性能A/B应各自独立串行，不能混合多个实验的时间线。
若采集尚不完整，摘要标为INCOMPLETE并说明缺项；导出成功不等于采集完整。

收到分析后的明确补充需求，再选择对应样本。profile、B、rank、iteration必须显式给出；core可选。例如：

```bash
python tests/ll_stage_timer/triage_extract.py diagnostic-runs/<run-name> \
  --profile sparse --bs 1 --rank 0 --iteration 0 --core 30 \
  --output-dir diagnostic-runs/<run-name>/followup_sparse_b1
```

仅所选数据按“紧凑字段 → LZMA/XZ → Base64 → 带校验文本块”压缩，所选原始ticks、首次milestone及同核端点跨度无损。每次默认最多3块，且必须符合suite累计10次的剩余额度。`--max-parts`不能突破该累计限制。
只发送命令列出的新part文件；ledger、manifest、本地索引和未选中的数据都不发送。超限时要求缩小选择，不扩大文件数量或静默抽样。

额度在suite内持久保存，生成需发送的新内容时即预留。相同内容重复生成不再扣额，也不重复要求发送；改用另一个`--output-dir`不能重新获得10次。
首次接入已有suite时，旧`SEND_FIRST.txt`保守占用一份已生成摘要额度；新摘要内容不同时额外占用一份。只有必要的新内容才应继续占用剩余额度。
本地原始采样、详细报告及旧归档不删除；仅对已有数据执行本页离线流程不要求重采。升级到schema15新采集时则必须重编匹配wheel并重采。

## 每次只打印一块

```bash
python tests/ll_stage_timer/transfer_report.py show diagnostic-runs/<run-name>/followup_sparse_b1 --part 1
```

仅对该次命令列出的新part逐块打印；命令只输出一块，没有附加stdout日志。不要遍历旧整套归档。
默认每块包含首尾封装、哈希、换行后仍≤60000字符和60000字节；CRLF按两个字节计算，满足65535限制。
Base64是ASCII，不受中文字符字节数影响。它在明确选择范围内无损，不表示完整suite，也不是供直接阅读的统计摘要。

## 接收端

复制完整BEGIN至END块，保存到一个或多个UTF-8文本文件；也可用`-`从stdin读取。

```bash
python tests/ll_stage_timer/transfer_report.py inspect copied_*.txt
python tests/ll_stage_timer/transfer_report.py restore copied_*.txt --output-dir restored_small_b
```

`inspect`检查已收到的所选传输包，列出缺少的part；恢复这一次选择需要该包的完整part。文本封装已经包含校验信息，无需补发manifest或索引。
支持乱序、完全相同的重复块、BOM、CRLF和Base64重新折行。不同版本的包、矛盾重复、截断或任意改动必须报错，不能猜测修复。
所选包及其真实区间先验证，再创建恢复目录；目录必须是新的，不覆盖旧文件。旧多group归档的本地恢复仍按最终group记录的总数和前序哈希核验，但不要求通过聊天发送旧归档。

还原结果：

- `context.json`、`decoded_samples.json`恢复所选范围的解码字段，包括精确整数ticks、calls、空值与缺失字段；保留选择条件和部分视图标识。
- `trace.json`展示实际记录的首次milestone及明确命名的同核端点跨度；指定非零轮次提取时，恢复端用`--trace-iteration`选择对应轮次。
- 源元数据或覆盖不足时，派生分析明确标为不可用或部分结果，不将选中的rank/core视为完整suite。
- 旧完整归档的`reports/`等字段仍可本地还原；不需要为本次补充另传整套OFF/ON性能文件。

恢复trace使用独立rank时钟原点，不进行跨rank对齐。schema15保留真实首次时间戳，INIT/KERNEL及其他同核端点跨度明确分轨；没有逐包重复区间、包数、轮询计数或ready比例，不补零、不伪造逐包时间线。
旧schema13/14仍按原有前两次真实区间和累计指标恢复，累计指标不能扩展成完整时间线，也不能与schema15混合。
SEND_DONE是原有发送资源退休，可能晚于接收工作，不是纯网络完成时间。`--no-render`只恢复数据而不生成trace/派生分析。

## 格式与资源限制

单case外层manifest为`ll-timer-transfer-v2`，样本payload为`ll-timer-samples-v2`；整套payload为`ll-timer-suite-v1`。
每块封装为`deepep-ll-text-v1`，包含压缩前后长度与SHA256、分块序号/总数、chunk SHA256。无需另传manifest即可解码。
单包原始数据上限512MiB，LZMA字典内存上限256MiB；必要补充还同时受每次分块数及suite累计发送次数限制。
旧整套归档按case边界约64MiB分组，恢复时逐组解压以限制内存。这里仅说明格式兼容，不提供全量发送入口。
变更输入保留旧包；相同补充内容通过持久记录去重，不重复占用发送额度。

## 历史压缩验证（不是schema15设备验收）

使用此前用户提供文件，只做字节压缩/还原测试（不是设备性能数据）：

| 输入 | 原字节数 | 新文本总字节，含封装/CRLF | 分块 | 减少 |
|---|---:|---:|---:|---:|
| communication_costs.csv | 114384 | 31106 | 1 | 72.81% |
| 新建 文本文档.txt | 132262 | 18470 | 1 | 86.04% |

两者均验证还原字节完全一致。目标机完整suite的536261158字节虽压缩到52138287字节（约减少90.28%），仍需887块，因此已取消全量发送，改为必要摘要与定点补充。
实际选择的压缩率取决于数据，不能依赖压缩率保证发送次数，必须先检查真实分块数和累计剩余额度。
测试覆盖多组矩阵、旧v1包、缺块/损坏/混包、CRLF边界、未知字段、大整数、恢复后真实trace与原解析器一致、目录带空格及写入失败不发布完成标记。
此前压缩格式版本的本地回归494通过、29项因缺少C++/POSIX环境跳过；这不是本次累计额度改动的验证记录。
上述历史离线格式更新本身不要求CANN编译或NPU采样。当前schema15升级需重编并重采；本次26/27各一次SSH连接均超时，尚无schema15 CANN或设备验收，不声明设备性能改善。

流程参考DJLRY/DeepEP `codex/megamoe-hierarchical-roce`固定提交
[`855f848`的trace_text/text_transfer](https://github.com/DJLRY/DeepEP/blob/855f848de7bc41e4cbe310cb74099fc8321b401e/experiments/megamoe/python/deep_ep_experimental/megamoe/timer/text_transfer.py)，
使用本分支已解码样本格式独立实现，未引入参考分支的设备计时器、GM逐事件写入或屏障。
