# LL SystemCycle 逐核阶段耗时

独立分支 `kernel/ll-g4-systemcycle-profile`，从恢复后的基线
`78e9fb4d2686d73e8baf9dbdd633b9bf203fdd4d` 创建。业务算法沿用
`9027275b31a28608296b1b71c93fa6e2bb456db5`，保留 SHMEM PI=0/1 身份兼容。
不带入 shared-pull 或 Vector-tail 候选。诊断实现由 `profile.py` 的
`DIAGNOSTIC_REVISION` 固定，完整 OFF/ON wheel 使用同一源码及 SDK。
当前固定打点源码为 `36e2d6a94a5c9a51d4785bf88dc2e833d141d9e3`。

## 参考与适配

用户提供的两个参考仅用于打点语义及表格结构，原文件不修改、不提交到仓库：

| 参考 | SHA256 |
|---|---|
| `DeepEP-feat-dispatch-combine-systemcycle-profile.zip` | `d43f1a8070ff899feaf0396ddf38daaa850a505fbbc536e1a2d7ad8953cd4b04` |
| `EP64_DeepSeek_BS1_dispatch_PR44回退_rank乘1KB偏移_逐核耗时.xlsx` | `4bbfe376ba7893015661b6734127521eaf2affaaadc8567a7a70a694299e10d0` |

ZIP 的主打点位于 HT dispatch/combine/notify，LL 目录没有对应打点。
本分支借鉴其编译关闭读钟、每核独立记录、主时点差值、循环内累计及结束导出方式，
将阶段重新映射到当前 LL 控制流。没有移植 HT 算法、固定64核、notify全rank对齐屏障，
也不改变通信窗口地址。参考表文件名中的 rank×1KB 偏移不是本分支的通信改动。

默认测试沿用本聊天基线：EP16、B=M256、H3584、K8、E384、G4、S30/R26、HT OFF、MTE。
dispatch 输入为预量化 FP8，combine 实际是 BF16 输入/输出。
EP64/BS1 是参考表的原始配置，不会因文件名而自动替换当前测试配置。

## 打点范围与时间口径

schema18/profile9 `systemcycle` 仅允许 `observe`、stage3，同时支持 dispatch/combine。
每核1024B、512B对齐；前8word为身份/valid位，后40word为主时点，最后40×2word为
`sum_ticks/calls`。34项为累计耗时，6项为计数；计数不除以频率。

- 主阶段包括入口/布局/magic/计数与收发初始化、计数发送、发送准备与提交、R0计数等待、
  前缀计算/发布/加载、接收准备/循环/收尾、lane退休，以及combine准备、角色工作和最终排空。
- 重复细分涵盖q播种/更新、路由分类、输入等待、元数据、打包/提交、flag探测/归约/解码、
  payload依赖、relay/本地输出/pending更新，以及combine输入、缓冲依赖、回传、归约与清flag。
- 对同一核取 `end-start` 或循环累计。频率按用户确认的1GHz，`us=ticks/1000`，原始ticks保留。
- 没有新加 PIPE_ALL、MTE等待、跨核/跨rank屏障、UB缓冲、lane或event。
  原业务同步保留；OFF及ON未挂载诊断tensor的调用不读钟。
- 所有诊断GM写出位于 `WORK_DONE` 后。核内总耗时不包含最后导出，实际完整profiler task包含它。
- 父阶段与内部细分存在包含关系，不能逐列相加。发送函数返回/下发区间不代表MTE3已完成，
  累计细分没有逐次起止时间，不绘制伪造的连续时间线。
- 未发生/未记录的阶段留空，已有valid位的零计数或零时差保留0；不跨NPU相减绝对时钟。

表内“阶段说明”列出每一列的时点/原始word、单位、边界及包含关系。
`deep_ep/utils/ll_stage_timer_systemcycle.py` 保存完整的阶段字典。

## 目标机一键采样

在已有 DeepEP-LL 工程根目录，沿用已配置的16张卡、`MASTER_ADDR`、
`MASTER_PORT`、`SHMEM_PORT`，两个端口必须不同。先保留或提交自己的源码改动，
再切换到此独立分支。下面函数失败只返回，不退出当前交互shell：

```bash
run_systemcycle_profile() {
  git fetch origin || return
  git switch kernel/ll-g4-systemcycle-profile || return
  git pull --ff-only || return
  python tests/ll_systemcycle/profile.py \
    --devices "$ASCEND_RT_VISIBLE_DEVICES" \
    --run-prefix systemcycle-001 || return
}
if run_systemcycle_profile; then
  printf '%s\n' "$PWD/results/ll-systemcycle-profile/systemcycle-001/stage-times.xlsx"
  printf '%s\n' "$PWD/results/ll-systemcycle-profile/systemcycle-001/stage-report.txt"
else
  printf '%s\n' '失败：当前终端、日志和已有结果均已保留。'
fi
```

脚本构建两份完整OFF/ON wheel并装入独立venv，使用新缓存 `.tmp/ll-systemcycle-profile/`。
原SDK/环境/历史结果不覆盖；复用SDK仍完整核验PI身份、版本与所有产物哈希。
全部Host检查通过后才启动设备，启动前检查空闲，不停止其他进程。
每个rank实际加载的native/kernel/SDK哈希与构建清单对应。

默认seed为0/17/101。本基线ST三轮正确性后的实际测量路由是seed+2，表中分别记录请求seed与
实际路由seed（2/19/103），没有把其他分支的采样实现混入。
每seed先OFF、再ON：OFF为10预热、100采集、每rank尾50；ON为10普通预热、100普通稳定调用，
之后5轮真实挂载诊断的capture。稳定/采样/capture每轮Exp+scalar all_reduce；
正确性检查、CPU复制和导出在capture循环后进行。
两种算子均采集，完整task与内部阶段值分别统计，OFF/ON差值只是当前采样条件下的打点扰动。

追加 `--plan` 仅生成计划；`--build-only` 只完成两份构建和Host检查；
`--run-only` 复用已验证的两份构建。`--timer-iters 1|3|5` 控制capture次数，默认5。
首次只做小轮数诊断可用 `--seeds 0 --timer-iters 1`，不会改变OFF的10/100/尾50协议。
重复测试使用新的run-prefix，不覆盖已有目录。

## Excel与回传文件

自动输出 `stage-times.xlsx`，表格采用参考的深蓝表头、逐核宽表、三位小数和冻结身份列：

1. **性能对照**：每seed两算子的OFF完整task、实际ON capture完整task及扰动。
2. **Dispatch逐核耗时**：按seed、capture、rank、core排序，记录实际S/R及共享接收角色。
3. **Combine逐核耗时**：对应combine自己的主阶段与累计细分。
4. **阶段说明**：频率、实验身份、每列边界和原始word来源。
5. **原始Cycle**：全部128个原始word、每个阶段的精确差值/累计ticks和calls。

默认每seed展示最后一轮capture，其他capture仍完整留在原始JSON；这与参考表只展示一次
profile的阅读方式相近。绝对uint64 word保存为十进制文本，避免Excel约15位数字精度截断。
精确差值先用Python整数计算，耗时公式再引用“阶段说明!B2”换算，数量直接引用，不做除法。
导出不依赖目标机安装Excel或额外Python表格库。

完整表通常超过65535字节，留在目标机查看。**只回传 `stage-report.txt`**，
导出时按实际UTF-8字节限制到60000，数量1/10；超限拒绝，不能截断证据。
该报告保留全部阶段分seed/算子/初始S/R的统计：先对同rank/core的capture取均值，
再统计核间中位数、均值、最慢核，并列出有效覆盖数。缺失样本不补零。

需要导出全部capture、或仅重新导出时使用新文件名，避免覆盖：

```bash
python tests/ll_systemcycle/report.py \
  results/ll-systemcycle-profile/systemcycle-001 \
  --captures all \
  --output results/ll-systemcycle-profile/systemcycle-001/stage-report-all.txt \
  --xlsx results/ll-systemcycle-profile/systemcycle-001/stage-times-all.xlsx
```

## 本地验证与限制

已完成native OFF/ON mock、7模式入口、原152B ABI和53文件去探针业务token检查，
schema18及旧15/16/17兼容、真实ST命令与导出字段/路径逐项核对。
全部stage-timer回归522通过、19跳过（17缺CMake、1POSIX、1历史快照）。
另外runner44、报表47、XLSX18项CPU检查通过；这些批次不与其他回归数量相加。
最终新入口/报表/表格/解析器/ST合并回归168项通过，旧schema15/16/17报告101项通过。
合成输入覆盖全部16rank×56核、两算子及3seed×5capture，原始高位uint64与合法0均校验。
生成的五页工作簿已通过独立读取、表格引擎重算、公式错误扫描和逐页渲染检查。
修改频率参数时耗时正确重算、计数不变，原始uint64文本精确保持；检查后恢复原频率。
合成样例均明确标记“不是设备实测”，不提交模拟数据充当性能结果。

未连接远端、未执行CANN编译、NPU正确性或性能测量。目标机命令用于完成这些验证。
旧schema15/16/17的离线解析及历史固定采集入口保留；新采样须安装本分支匹配的完整wheel。
