# Dispatch 各发送阶段带宽

本实验按用户选择保留 `9027275b31a28608296b1b71c93fa6e2bb456db5` 的业务算法。
失败的 Vector-tail 已撤销。新的诊断源码由 `profile_bandwidth.py` 的
`DIAGNOSTIC_REVISION` 固定，OFF/ON 两份完整 wheel 使用同一源码、同一 SDK。
报告同时记录业务参考、实际诊断源码、runner、实际加载库和 SDK 的身份。

固定配置：EP16、B=M256、H3584、K8、E384、G4、S30/R26、HT OFF、MTE、
dispatch-only。ST 的 FP8 是预量化输入 tuple，INT32 打包 E8M0 scale；
本次不运行 combine，也不把输入等待称作在线量化耗时。

## 计数范围

| 阶段 | 发起位置 | 计数内容 |
|---|---|---|
| COUNT_CONTROL | R 核 SendExpertCntRange | 实际发布的 count 记录；一次 copy 可包含多个 32B 记录 |
| PREFIX_LOCAL | R0 CalRecvAndSetFlag | 本卡 counts 输出、共享前缀表、32B ready epoch，共三次 copy |
| REMOTE_FIRST_HOP | S 核 ConsumePacket，mask 非零 | 去重远端组首跳 packet |
| SOURCE_DIRECT | S 核 ConsumePacket，mask 为零 | 远端单目标 DIRECT packet |
| GROUP_LOCAL | R 核 ProcessGroupLocal/ConsumePacket | 本组逐专家初发 packet |
| RELAY | 接收侧 FIRST_HOP 转发循环 | 剩余专家的 onward packet，逐次 copy 计一次 |

`peer` 是目标 rank 不同于发起 rank；`self` 是写本卡通信槽。它不等同于同机/跨机。
wire 字节直接取 copy 的长度，包含 flags、header 和 padding；不计 GM stride 的间隙。
data payload 字节为有效 FP8 元素与 packed scale：当前每包 3584+112=3696B，
wire 每包 4096B。控制阶段的 payload 速率为 NA。源端初发和 relay 分别计数，
跨卡 relay 字节可以与跨卡初发字节相加，但不同并发阶段的时长不能相加为全局时间。

不计最终本地 output、接收读、清 flag、source mask/weight 本地输出。
这是指定发送阶段的搬运统计，不能称为整个 MTE3 的全部流量。
报告验证各 capture/condition 的逐 rank/core 字节、copy 数、lane 一致，
校验 count 所有权、prefix 三次写、包长及全局 data packet 总数 EP×B×K=32768。

## 两种打点模式与分母

- `observe`：不增加同步，只记录实际发起字节及首次/末次提交时间。
  提交跨度不是传输完成时间，不用于计算阶段通信带宽。
- `bandwidth-complete`：仅 stage3 dispatch 独立 capture。阶段开始前 MTE3_S
  排掉此前已排入的工作；在第一笔 copy 前记时；结束时 MTE3_S 确认此前提交完成，
  紧接着重复无新搬运的空控制。阶段有效时长从第一笔发起到完成确认，包含此区间内
  的打包、调度、下发和等待，也包含首笔 copy 尚未完成的生产者依赖。

首跳与 DIRECT 之间，completion 模式先 flush 尚未消费的最后一个预取包，
再关闭首跳区间、打开 DIRECT 区间、重新开始原两输入预取。
OFF 和 observe 保留原连续一包前瞻；不增加 input buffer、lane 或常驻 lane event。
源/控制阶段每核最多一个非空区间，relay 每次转发循环是一个区间，累计其真实时长。
仅有累计值和首尾时间，Chrome trace 不伪造逐包或连续独占 relay 时间线。

`阶段有效 GB/s = 实际字节 × frequency_hz / completed_elapsed_ticks / 1e9`。
peer 与 self 分开报告，分母仍是同一真实区间，可以包含这两类写。
前置排空和空控制单列，末尾排空包含在阶段时长内；不减去空控制制造校正值。
每 rank/core 先汇总多次 capture；报告给出阶段时长中位数、非零字节类核的速率中位数，
以及 `sum(bytes)/sum(core elapsed)`，后者是平均每核有效速率，不是 rank/link 聚合吞吐。
空阶段或零时长不生成带宽。

另给出 `同一 rank 的阶段总字节 / 该 rank 完整 profiler task 时间`，
再对全部 16 rank 求平均；这是完整任务时间归一化速率，零流量 rank 贡献零。
OFF 的分子来自已匹配的 observe 路由，分母为 OFF 的同 rank 尾50均值；
ON 分母来自每个实际挂载 capture 对应的完整 task，包含最后 timer 导出。
不跨 rank 相减时钟，不把最大单核阶段时长当作 rank 聚合窗口，不相加各核带宽。

completion 会改变并发和网络竞争，测得的是干预状态下的有效搬运速率，
不能直接赋给未打点基线，也不能当作物理链路线速或峰值达成率。
报告先展示 OFF/observe/completion 完整 task 扰动，保留所有上述解释限制。

## 目标机执行

在本分支工程根目录，沿用之前测试的16张卡与已配置的
`MASTER_ADDR`、`MASTER_PORT`、`SHMEM_PORT`；端口须显式设置且不同。
以下函数失败后返回，保留当前交互 shell 和日志：

```bash
run_send_bandwidth() {
  git pull --ff-only || return
  python tests/ll_control_opt/profile_bandwidth.py \
    --devices "$ASCEND_RT_VISIBLE_DEVICES" \
    --run-prefix send-bandwidth-001 || return
}
if run_send_bandwidth; then
  printf '%s\n' '完成：results/ll-send-bandwidth/send-bandwidth-001/bandwidth-report.txt'
else
  printf '%s\n' '失败：终端和日志已保留，请查看上方报错。'
fi
```

默认三 seed 0/17/101，实际测量路由为2/19/103。每个 seed 先 OFF，ON 的顺序为
observe/complete、complete/observe、observe/complete。
OFF 使用10预热、100采样、每 rank 尾50；每个 ON 条件10普通预热、100普通稳定调用，
之后5次实际挂载诊断的 profiler capture。稳定/采样/capture 每轮 Exp+scalar all_reduce；
输出正确性检查、CPU复制和 JSON 导出均在采样循环后。
普通稳定调用没有 completion 等待，所以五次 completion 是相同普通前导后的干预试次。

脚本自动准备 `.tmp/ll-send-bandwidth/` 下独立源码、完整 OFF/ON wheel 与两个 venv，
保留原安装和结果；两份 Host 导入/身份预检全部通过后才启动设备。
SDK 用现有工程内构建和标准身份校验，不静默重建不兼容 SDK。
各次启动前检查设备空闲，不停止其他进程；16 rank 实际加载库哈希再核验。
日志、缓存和原始数据均留在工程内。使用新 run-prefix 重跑，不覆盖部分失败结果。

同一命令追加 `--plan` 只列计划；`--build-only` 只构建两份完整 wheel 并做 Host 预检，
不启动设备；`--run-only` 复用已核验的两份构建。独立冷对象编译计时仍见
[BUILD_DISPATCH_SPLIT.md](../../docs/BUILD_DISPATCH_SPLIT.md) 的冷构建步骤。

## 回传与验证状态

只回传 `results/ll-send-bandwidth/send-bandwidth-001/bandwidth-report.txt`。
导出器按实际 UTF-8 字节拒绝超过65535的报告，不截断证据，交付数量为1/10。
原始 JSON、完整 profiler 数据及日志留在目标机。需要单独重新导出时用新文件名：

```bash
python tests/ll_control_opt/report_bandwidth.py \
  results/ll-send-bandwidth/send-bandwidth-001 \
  --output results/ll-send-bandwidth/send-bandwidth-001/bandwidth-report-002.txt
```

当前只进行本地 Python/C++ mock、业务 token、完整合成报告与 runner 检查。
本地回归覆盖879项通过、2项跳过；修正历史兼容夹具后相关用例已复测。
三seed、每条件五次capture的完整合成报告为14957 UTF-8字节，只生成1份回传文件。
未连接27或其他编译主机，未执行 CANN 编译、NPU 正确性或性能测量；
CPU 检查与合成报告不是设备带宽结果。实际数值由以上目标机命令生成。
schema17需匹配新完整wheel及新采样；schema15/16及其历史固定入口继续可用。
