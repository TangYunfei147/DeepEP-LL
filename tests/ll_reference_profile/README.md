# 当前 LL dispatch reference 打点

本次按用户要求，以 `kernel/ll-ndr-metadata-cores` 的
`c3d86406733fd28fe4e081e563eaf7e51321c18f` 为固定参考，完整替换
`kernel/ll-g4-systemcycle-profile` 原有打点和解析。
旧stage/expert、SystemCycle40时点、sparse、MTE3和bandwidth实现及兼容解析已删除，
当前只保留这一套reference dispatch代码。旧数据需使用Git历史版本读取。
wheel构建会精确清理暂存目录中的旧计时模块和字节码，包验证也拒绝夹带，
避免复用setuptools缓存时重新打入已删除代码；不清理源码、SDK或其他模块。

通信业务保持原MTE基线：固定56核、G4、S发送远端组、R计数/本组发送，随后共享接收。
没有迁入参考的M核、RDMA、QP、wire或SHMEM升级；SDK仍锁定8e3fc610，保留PI0/1兼容与完整校验。
combine完全移除打点，当前reference入口只采集dispatch。

## 与参考一致的打点方式

- 29项主阶段，按实际同核时点作整数差；主点一次GetSystemCycle加一次独占GM Scalar写。
- 接收等待开始读一次clock，有ready批次后再读两次并累计w33/w34；空扫描不反复读钟，
  包日志逐项记录身份/顺序时不读钟。父子和交错区间不能相加。
- 包记录最多128条，每条32B；只使用原分配后剩余UB，不减少lane或更改业务预算。
  空间不足时明确记录容量和截断，不把不完整记录当作完整任务顺序。
- 本核业务最终排空/CTRL恢复到w21，然后一次DMA导出包日志，w57包含这段诊断导出；
  w57后才发布身份/有效位并做cache写回。日志DMA需要的诊断依赖不进入业务w21区间。
- 未记录与合法0由valid位区分；不跨rank比较绝对时钟。频率默认1GHz，原始ticks保留。
- OFF或未挂载捕获tensor时不读钟、不分配日志。保留7TU、O2及原业务同步。

## 当前MTE路径的必要适配

当前格式为 **schema6，`ll-reference-cycles-mte-shared-v1`**，每核80×8 uint64＝5120B、512B对齐。
主word和packet区域的位置沿用参考；w22/w23记录实际S/R，w62为0，w63记录初始S/R角色。
两种角色都可以有发送与接收阶段。不能将此格式或数据冒充参考NDR schema5。

| 适配项 | 当前真实含义 |
|---|---|
| sender / send_drain_metadata | 当前Async发送函数提交/返回，不能称已有MTE3_S确认完成 |
| role_init | 当前顺序初始化按实际边界取差，具体公式见阶段说明 |
| recv_prefix_load | 当前先加载prefix再构建任务/缓冲，使用w30-w18；不套用参考的w30-w32 |
| count_drain、cumsum_join | 只在真实对应路径存在时记录；缺少独立工作时留空，不增等待凑阶段 |
| S包时序 | 实际ConsumePacket绑定packet身份，记录源packet数、本G4提交区间及lane；不绑定预取Push以免前后包错位 |
| RDMA细分 | 当前没有RDMA暂存/put/quiet，相关字段为不适用；外G4首跳不等于跨机RDMA |
| R包日志 | 所有实际接收核都可记录，当前同一任务核负责relay及本地输出；不是参考metadata版独立搬出owner分工 |

dispatch Host参数仅保留referenceProfile指针，结构144B；combine无诊断参数，结构88B。
内部 `_ref6` 符号及Host/Python布局校验拒绝旧对象混装，业务wire19000+S不变。

## 目标机运行

默认EP16、B=M256、H3584、K8、E384、G4、S30/R26、预量化FP8、HT OFF，
1次reference capture。沿用已有CANN、Torch/torch_npu环境与显式16卡设置。
`MASTER_ADDR` 应是本机/多机均可达的IPv4地址；HCCL、SHMEM及准备握手端口须不同。

在当前工程根目录执行，失败不会退出交互shell：

```bash
run_reference() {
  git fetch origin || return
  git switch kernel/ll-g4-systemcycle-profile || return
  git pull --ff-only || return
  bash tests/ll_reference_profile/one_click.sh \
    0 "$MASTER_ADDR" reference-001 \
    --devices "$ASCEND_RT_VISIBLE_DEVICES" \
    --reference-iters 1 || return
}
if run_reference; then
  printf '%s\n' "$PWD/results/ll-reference-profile/reference-001-reference/reference-analysis/stage-times.xlsx"
  printf '%s\n' "$PWD/results/ll-reference-profile/reference-001-reference/reference-analysis/return_report.md"
else
  printf '%s\n' '失败：当前终端、已有结果与日志已保留。'
fi
```

新入口使用当前干净分支源码构建独立 `.venv-ll-reference`，不再固定旧36e2d6a。
只初始化SHMEM及必要子模块，不拉CATLASS，不修改已有不兼容SDK；确需SDK重建须显式
`--rebuild-shmem`。`--skip-build` 核验当前源码、S/R、reference开关、CANN版本/timestamp、
SDK完整哈希和身份，不允许跳过首次更新的完整wheel构建。

`--plan` 只输出CPU计划，不建环境、不查询/启动设备；`--prepare-only` 仅构建和Host校验。
`--mode perf` 编译reference OFF并使用相同采样协议测完整dispatch task，不生成reference表。
两种模式不能混用同一旧wheel，切换模式应完整构建。日志均位于工程 `.logs/ll-reference/`。
每次执行前检查选中设备空闲，不停止任何进程；已有结果/日志不覆盖，请用新run名重跑。

多机时每机16卡，分别使用节点号 `0..N-1` 和相同 `--num-nodes N`、master地址、run名、端口及参数。
各机先完成Host/空闲检查，再通过准备身份握手放行；不在kernel内添加屏障。
各机源码路径可以不同，ST只广播run名字，结果在各自工程根保存。
可显式选择 `--tokens`、`--dtype`、`--capacity`、`--hidden`、`--topk`、`--experts`、
`--seed`、`--send-cores`，同一作业必须一致，原Host/shape/UB准入仍生效。

## 采样协议

与参考一致采用 `initial-reduce-exp100-batch-v3`：
批次初始all_reduce一次 → 预热 → 同步 → 100次Exp → 连续稳定/capture。
循环内没有Exp或采样all_reduce，Exp前导后不额外同步。
默认10预热、100稳定、1次reference；perf默认100采集、每rank最后50条。
正确性多轮后恢复请求seed及匹配handle，保存 `profile_seed=seed` 和完整 `route_sha256`。
当前MTE基线的路由生成算法不替换成NDR分支的其他路由实验。

这个协议与旧SystemCycle逐轮Exp+all_reduce不同，不能直接把新旧结果当同条件性能对照。
reference的核心统计是每rank每轮最长单核 `(w21-w0)`，包含打点指令且不含末尾导出，
不能称未打点完整kernel耗时。完整task必须另看perf模式的profiler结果。

## 解析与表格

只解析当前schema6，损坏数据、旧格式、混配置/源码/seed/路由/频率会拒绝。
默认允许完整本地16rank组并标明缺失远端；合并全域数据后用 `--require-all-ranks` 检查完整域。
单轮逐核默认选 `last`，多轮均值另存，不将均值包装成某一轮。

基础工作簿含采样汇总、逐核耗时、阶段说明、原始Cycle、S包时序、R包顺序、包记录覆盖、
包原始记录；run级合并工作簿还包含逐核均值、阶段调用次数、全部轮次明细。
多case独立保留并生成索引。原始uint64超过Excel精确位数时存文本，差值先在Python整数域计算。

自动生成 `reference-analysis/stage-times.xlsx` 与 `return_report.md`。
完整xlsx/CSV/JSON留目标机，**只回传run级单个 `return_report.md`（≤60000 UTF-8字节）**。
多case超出回传预算时明确列出省略case数量，完整case表仍保留，不逐个要求回传大文件。

只重新解析时无需CANN或设备，必须选择新的输出目录：

```bash
bash tests/ll_reference_profile/one_click.sh --analyze-only \
  results/ll-reference-profile/reference-001-reference \
  --output-dir results/ll-reference-profile/reference-001-reference/analysis-002 \
  --iteration last
```

## 验证边界

本地验证包含真实OFF/ON recorder和七模式Host selector、144/88B ABI、旧业务范围去探针对照、
稳定任务日志/截断/空核/paired probe模型、当前格式解析、ST采样顺序、跨节点路径及部署预检。
完整Excel通过标准读取、公式缓存和合并表测试，并用合成数据检查显示。
已完成461项完整CPU回归，21项因缺少CMake或未初始化本地SDK子模块跳过；
最后的字段、轮次列、部署身份和旧代码移除检查115项通过。以上批次存在交叉，不相加。
打包残留清理及真实setuptools打包检查88项通过，8项原有Linux执行/符号链接用例在Windows未运行。
合成工作簿11页通过独立表格引擎重算、错误扫描和渲染，频率修改后耗时正确重算；
精确uint64原文及公式缓存另由OOXML检查，完整合成回传报告5733字节。
合成数据不是设备测量，不提交原始测试产物。
本任务不连接27或其他主机，不执行CANN编译或NPU；真实编译、正确性和性能由目标机命令完成。
