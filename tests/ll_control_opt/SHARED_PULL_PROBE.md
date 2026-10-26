# 单机 shared-pull：独立分支及逐提交对比

当前实验分支是 `kernel/ll-g4-shared-pull`。原 `kernel/ll-g4-control-opt`
已通过追加撤销提交恢复基线代码，最新 SDK PI 兼容提交及固定带宽入口保留。
本实验尚未完成 CANN 编译、设备正确性或性能验证，不预设收益。

## 固定实验边界

单机 EP16，B=M256、H3584、K8、E384、G4、S30/R26，预量化 FP8 dispatch、
BF16 combine，HT OFF、完整 timer OFF wheel，所有版本使用同一已校验 SDK。
同一通信器按相同调用顺序在单 stream 上串行执行，combine 使用对应 dispatch
的 handle。这里的无额外栅栏证明不覆盖并发 stream、任意 shape 或不匹配的 handle。

## 每项改动一个提交

| 标签 | 提交 | 本项变化 |
|---|---|---|
| baseline | 78e9fb4 | 原分支恢复后的基线，保留 SDK PI 兼容；业务路径对应用户保留的9027275 |
| prefix-only | b2bcdbd | 各消费核独立 ready，prefix 读取完成后清自己的槽，包括 R0；仍为原传输 |
| pull-reference | d4c7c0a | 在独立分支启用原 shared-pull，实现可比较的起点 |
| no-fence | 2daff37 | 删除额外全16卡完成轮询、本卡 SyncAll 和相关计数读写 |
| prepare-once | fdc4335 | 同一首跳的 descriptor/flags 在成员循环外只构造一次 |
| async-flags | 10a4371 | flags 用 V→MTE3 发布，删除逐引用尾部 Scalar drain，保留 lane 退休 |
| logical-role | 68fe8d1 | 删除无消费者的 wire 改写，Terminal 用局部逻辑角色表达 |
| single-ready | 0d02c07 | descriptor32B+单独commit32B；第三引用mask贯穿双快照，DIRECT省去包头预读 |
| lean-validation | e359eeb | 固定构造协议下的逐引用魔数/地址检查仅保留在诊断构建 |

`commits-shared-pull-fast.json` 固定 baseline、prefix-only、最终版三条。
最终版对 prefix-only 的差值用于观察数据路径收益，绝对均值还须与 baseline 比较。
`commits-shared-pull-series.json` 固定全部九条，每项 reference 指向前一实测版本。
两个清单均保留40位提交，不使用浮动 HEAD；切勿按执行顺序替代 comparison reference。

历史 `commits-shared-pull.json` 的9027275/0c3c150/762b5c6仍固定原源码，仅供复现旧穿刺。
本分支不把旧实验的fence-control当最终方案；旧MTE3和带宽诊断入口亦保持历史版本。

## prefix ready 与生命周期

每核使用自己512B magic槽的+32位置存32B ready。R0先确认共享 prefix DMA完成，
再向实际消费核发布1。每个消费核完成 prefix MTE2读取后，将自己的 ready 清0。
去重路径有56个消费者，非去重只发布给后R核；R0不清其他核的槽，没有 readDone ACK。
clear源使用原readyBuf，async由既有最终drain退休，fallback在Reset前局部退休。
这增加本地 ready 发布/清零，必须通过 prefix-only 单独计入完整kernel成本。

共享首跳payload在单dispatch内不被修改，清理只触及flags。紧邻调用使用另一个bank。
中间是dispatch时，下一轮包括零计数的全来源发布保证前次读者退出后才可再次复用。
中间是匹配combine时，旧FIRST_HOP的source必须等其partial；partial依赖同G4 ready，
从而约束组外source的下一次覆盖。combine远端结果区为[2MiB,17MiB)，不覆盖从17MiB
开始的旧dispatch区；本卡stage覆盖受上次combine的组ready保护。不能只因读者在G4
就武断缩成4卡栅栏，也不能删除这些原有count/ready/partial依赖。

## 最终引用协议和保留依赖

每个引用先写32B descriptor，再确认其完成，最后写首flag槽的32B commit。
commit首float=-1，普通包仍使用完整的1/1.5块标志；VF同时输出ready/keep/reference
三个mask，保存在原32B结果空间。引用状态必须随A/B快照保存，不能依赖复用后的sum。
普通DIRECT按原方式读3840B；引用先读32B descriptor，再从首跳owner读取3840B。
同一lane的descriptor/flags在所有相关DMA读者退休前不可覆盖。最终仍清完整flags，
保证普通包与引用在后续调用互换时不存在残留通知。不增加活跃UB、lane或事件编号。

每个跨卡引用的请求数据字节从原穿刺288+3840=4128B降为64+3840=3904B；
原整包转发为4096B。这不含链路报文开销，也不代表端到端加速。

默认 `DEEPEP_LL_SHARED_PULL_VALIDATE_DESCRIPTOR=0`，生产快路径信任内部构造的owner/slot。
诊断编译可将此宏设1，检查magic、同组、范围与对齐；该构建不能冒充清单内默认版本。
Host shape/UB/Async准入、ABI及全rank能力/库哈希校验保留，它们不属于逐包Scalar热路径。
单ready默认核心协议为28000+S，prefix-only为25000+S；全rank必须重装各自匹配完整wheel。

## 目标机命令

保留已有16卡选择和MASTER_ADDR/MASTER_PORT/SHMEM_PORT环境。在原工程根执行：

```bash
run_shared_pull() {
  local kind="${1:-fast}"
  case "$kind" in fast|series) ;; *) return 2 ;; esac
  git fetch origin \
    refs/heads/kernel/ll-g4-control-opt:refs/remotes/origin/kernel/ll-g4-control-opt \
    refs/heads/kernel/ll-g4-shared-pull:refs/remotes/origin/kernel/ll-g4-shared-pull || return $?
  if git show-ref --verify --quiet refs/heads/kernel/ll-g4-shared-pull; then
    git switch kernel/ll-g4-shared-pull || return $?
  else
    git switch --track -c kernel/ll-g4-shared-pull origin/kernel/ll-g4-shared-pull || return $?
  fi
  git merge --ff-only origin/kernel/ll-g4-shared-pull || return $?
  python tests/ll_control_opt/suite.py \
    --commits "tests/ll_control_opt/commits-shared-pull-${kind}.json" \
    --devices "$ASCEND_RT_VISIBLE_DEVICES" \
    --num-nodes 1 --node-rank 0 --resume --send-cores 30 \
    --seeds 0 --rounds 2 --order alternating \
    --run-prefix "shared-pull-${kind}-001" || return $?
  python tests/ll_control_opt/export_report.py \
    "results/ll-control-opt/shared-pull-${kind}-001" || return $?
}
if run_shared_pull fast; then
  printf '%s\n' '完成：回传 results/ll-control-opt/shared-pull-fast-001/analysis-report.txt'
else
  printf '%s\n' '执行失败，保留终端、已有结果和日志。'
fi
```

逐提交测试将最后调用改为 `run_shared_pull series`，回传目录相应为
`results/ll-control-opt/shared-pull-series-001/analysis-report.txt`。
三版快测6次ST，逐项对比18次ST；均1个请求seed0（实际测量路由seed2）×2轮交替。
每次保留正确性检查、10预热、100采集和各rank尾50，Exp+scalar all_reduce协议不变。
完整dispatch与BF16 combine分别报告。正式扩展可用3seed，使用新run-prefix。

所有wheel在设备测试前准备，`--resume`只复用身份匹配的构建，不覆盖已有结果。
失败重跑使用新run-prefix。预览时给suite加 `--plan`，不运行exporter。
只回传对应的一个analysis-report.txt；导出器核验实际UTF-8字节数≤65535，原始数据留目标机。
36行逐项报告、扩展3seed的108行报告均有不丢行和文件大小检查。

## 验证边界

CPU检查覆盖真实prefix生产方法的延迟DMA、56核独立进度、变路由及D-C-D反例，
真实引用发布/加载方法的延迟DMA源生命周期，真实VF及双快照引用mask，
无额外fence的延迟网络/读者模型，以及Host/ST/SDK/报告身份校验。
这些结果不替代Bisheng/CANN编译、硬件内存序、设备正确性和性能验证。
不连接27或其他远端，不在本地启动NPU。

本次最终相关CPU回归199项通过，交付清单/报告8项通过；原分支恢复检查88项通过。计数分属不同检查集，不代表设备验证。
